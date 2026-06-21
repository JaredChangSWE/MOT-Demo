"""Eased PTZ controller that steers pan/tilt to keep a target centered.

Two behaviors make the camera motion demo-friendly:

* **Ease-in / ease-out** — instead of a raw proportional step, pan and tilt are
  driven by a critically-damped *SmoothDamp* filter. The camera accelerates
  gently from rest (ease-in) and decelerates smoothly as it approaches the
  target (ease-out), following a moving target without jerks.

* **Engage range (hysteresis)** — the camera only STARTS tracking once the
  normalized centering error exceeds `ctrl_engage_error`, and keeps tracking
  until the error falls below `ctrl_release_error`. With engage > release the
  camera holds still for small wobbles and won't chatter around the threshold.

Sign conventions: pan increases turning right (+X); tilt increases looking up
(+Z). Image v grows downward, so a target below center means tilt must decrease.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .params import Params


@dataclass
class ControlStatus:
    """Per-frame output from `PTZController.update`."""

    ex: float           # normalized horizontal error [-1, 1]
    ey: float           # normalized vertical error   [-1, 1]
    error_norm: float   # magnitude sqrt(ex^2 + ey^2)
    centered: bool      # True if target is within the release radius
    moved: bool         # True if a non-zero pan/tilt step was applied this frame
    engaged: bool       # True if the controller is actively tracking
    pan_delta_deg: float
    tilt_delta_deg: float


def _smooth_damp(current: float, target: float, velocity: float,
                 smooth_time: float, max_speed: float, dt: float) -> tuple[float, float]:
    """Critically-damped easing toward `target` (Unity-style SmoothDamp).

    Returns (new_value, new_velocity). Produces ease-in from rest and ease-out
    on arrival; `max_speed` caps the slew rate, `smooth_time` sets gentleness.
    """
    smooth_time = max(1e-4, smooth_time)
    omega = 2.0 / smooth_time
    x = omega * dt
    exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    change = current - target
    original_to = target
    max_change = max_speed * smooth_time
    change = max(-max_change, min(max_change, change))
    target = current - change
    temp = (velocity + omega * change) * dt
    velocity = (velocity - omega * temp) * exp
    new = target + (change + temp) * exp
    # Prevent overshoot past the target.
    if (original_to - current > 0.0) == (new > original_to):
        new = original_to
        velocity = (new - original_to) / dt if dt > 0 else 0.0
    return new, velocity


class PTZController:
    def __init__(self, params: Params | None = None) -> None:
        self.p = params or Params()
        self._vel_pan = 0.0    # rad/s
        self._vel_tilt = 0.0   # rad/s
        self._engaged = False

    @staticmethod
    def _pixel_to_world_angles(camera: Camera, u: float, v: float) -> tuple[float, float]:
        """World-space (pan, tilt) of the ray through image pixel (u, v)."""
        cx, cy = camera.frame_center
        focal = camera.focal
        xc = (u - cx) / focal
        yc = (cy - v) / focal           # image v grows down; camera-space y is up
        right, up, forward = camera._basis()
        ray = xc * right + yc * up + forward
        ray = ray / np.linalg.norm(ray)
        pan_t = math.atan2(float(ray[0]), float(ray[1]))
        tilt_t = math.asin(float(np.clip(ray[2], -1.0, 1.0)))
        return pan_t, tilt_t

    def _idle(self, camera: Camera, dt: float, ex: float = 0.0, ey: float = 0.0,
              error_norm: float = 0.0) -> ControlStatus:
        """Not tracking: ease any residual velocity to a smooth stop."""
        max_speed = math.radians(self.p.ctrl_max_speed_deg)
        pan0, tilt0 = camera.pan, camera.tilt
        camera.pan, self._vel_pan = _smooth_damp(
            camera.pan, camera.pan, self._vel_pan, self.p.ctrl_smooth_time, max_speed, dt)
        camera.tilt, self._vel_tilt = _smooth_damp(
            camera.tilt, camera.tilt, self._vel_tilt, self.p.ctrl_smooth_time, max_speed, dt)
        camera.clamp()
        moved = (camera.pan != pan0 or camera.tilt != tilt0)
        return ControlStatus(
            ex=ex, ey=ey, error_norm=error_norm,
            centered=error_norm <= self.p.ctrl_release_error,
            moved=moved, engaged=False,
            pan_delta_deg=math.degrees(camera.pan - pan0),
            tilt_delta_deg=math.degrees(camera.tilt - tilt0),
        )

    def update(self, camera: Camera, target_center: tuple[float, float] | None,
               dt: float | None = None) -> ControlStatus:
        """Steer `camera` so `target_center` (u, v pixels) eases toward center."""
        dt = self.p.dt if dt is None else dt

        if target_center is None:
            self._engaged = False
            return self._idle(camera, dt)

        u, v = target_center
        pan_t, tilt_t = self._pixel_to_world_angles(camera, u, v)
        # Wrap the pan error to [-pi, pi] so the camera always turns the SHORT
        # way (and the error doesn't blow up near the +/-180 wraparound).
        dpan = math.atan2(math.sin(pan_t - camera.pan), math.cos(pan_t - camera.pan))
        dtilt = tilt_t - camera.tilt
        ex = dpan / (camera.fovx / 2.0)
        ey = dtilt / (camera.fovy / 2.0)
        error_norm = math.hypot(ex, ey)

        # Hysteresis: engage past the engage radius, release inside the release radius.
        if not self._engaged and error_norm > self.p.ctrl_engage_error:
            self._engaged = True
        elif self._engaged and error_norm <= self.p.ctrl_release_error:
            self._engaged = False

        if not self._engaged:
            return self._idle(camera, dt, ex, ey, error_norm)

        # Engaged: ease pan/tilt toward the target. Use the wrapped short-way
        # goal for pan so SmoothDamp never takes the long route around.
        max_speed = math.radians(self.p.ctrl_max_speed_deg)
        pan0, tilt0 = camera.pan, camera.tilt
        pan_goal = camera.pan + dpan
        new_pan, self._vel_pan = _smooth_damp(
            camera.pan, pan_goal, self._vel_pan, self.p.ctrl_smooth_time, max_speed, dt)
        new_tilt, self._vel_tilt = _smooth_damp(
            camera.tilt, tilt_t, self._vel_tilt, self.p.ctrl_smooth_time, max_speed, dt)
        camera.pan, camera.tilt = new_pan, new_tilt
        camera.clamp()
        # Zero velocity on an axis that hit a mechanical limit (anti-windup).
        if camera.pan != new_pan:
            self._vel_pan = 0.0
        if camera.tilt != new_tilt:
            self._vel_tilt = 0.0

        pan_delta_deg = math.degrees(camera.pan - pan0)
        tilt_delta_deg = math.degrees(camera.tilt - tilt0)
        return ControlStatus(
            ex=ex, ey=ey, error_norm=error_norm,
            centered=error_norm <= self.p.ctrl_release_error,
            moved=(pan_delta_deg != 0.0 or tilt_delta_deg != 0.0),
            engaged=True,
            pan_delta_deg=pan_delta_deg, tilt_delta_deg=tilt_delta_deg,
        )
