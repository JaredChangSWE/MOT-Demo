"""Eased, velocity-based PTZ controller for the real ONVIF camera.

Ports the simulator controller's *feel* to real hardware. The sim drove absolute
pan/tilt angles of a virtual camera; a real ONVIF camera is driven by continuous
*velocity* commands with no usable absolute-angle feedback. So this works purely
in image space and outputs pan/tilt velocities in [-1, 1] for ContinuousMove,
while keeping the three behaviors that made the sim smooth:

* **SmoothDamp easing** — the commanded velocity eases in/out (no jerks) instead
  of stepping, via a critically-damped filter on the velocity itself.
* **Engage hysteresis** — start following once the normalized centering error
  exceeds ``ctrl_engage_error``; stop adjusting within ``ctrl_release_error``.
* **Feed-forward lead** — aim ``ctrl_lead_frames`` ahead along the target's
  image velocity so fast movers stay centered despite motor lag.

Sign: ONVIF +pan = right, +tilt = up. A target on the right (ex>0) -> pan +;
a target below center (ey>0, image y grows down) -> tilt - (look down).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from params import Params


@dataclass
class ControlStatus:
    ex: float               # normalized horizontal error [-1, 1]
    ey: float               # normalized vertical error [-1, 1]
    error_norm: float
    engaged: bool
    pan_vel: float          # commanded ONVIF pan velocity [-1, 1]
    tilt_vel: float


def _smooth_damp(current: float, target: float, velocity: float,
                 smooth_time: float, max_speed: float, dt: float) -> tuple[float, float]:
    """Critically-damped easing toward `target` (Unity-style SmoothDamp)."""
    smooth_time = max(1e-4, smooth_time)
    omega = 2.0 / smooth_time
    x = omega * dt
    exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    change = current - target
    max_change = max_speed * smooth_time
    change = max(-max_change, min(max_change, change))
    original_to = target
    target = current - change
    temp = (velocity + omega * change) * dt
    velocity = (velocity - omega * temp) * exp
    new = target + (change + temp) * exp
    if (original_to - current > 0.0) == (new > original_to):
        new = original_to
        velocity = (new - original_to) / dt if dt > 0 else 0.0
    return new, velocity


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class RTController:
    def __init__(self, params: Params) -> None:
        self.p = params
        self._pan_cmd = 0.0    # current eased velocity command
        self._tilt_cmd = 0.0
        self._accel_pan = 0.0  # SmoothDamp internal velocity (of the command)
        self._accel_tilt = 0.0
        self._engaged = False

    def reset(self) -> None:
        self._pan_cmd = self._tilt_cmd = 0.0
        self._accel_pan = self._accel_tilt = 0.0
        self._engaged = False

    def update(self, target_norm: tuple[float, float] | None,
               target_vel_norm: tuple[float, float], dt: float) -> ControlStatus:
        """target_norm = (cx, cy) in [0,1]; target_vel_norm = per-frame vel."""
        p = self.p
        if target_norm is None:
            self._engaged = False
            desired_pan = desired_tilt = 0.0
            ex = ey = error_norm = 0.0
        else:
            cx, cy = target_norm
            vx, vy = target_vel_norm
            aim_x = cx + p.ctrl_lead_frames * vx
            aim_y = cy + p.ctrl_lead_frames * vy
            ex = _clamp((aim_x - 0.5) * 2.0)
            ey = _clamp((aim_y - 0.5) * 2.0)
            error_norm = math.hypot(ex, ey)

            if not self._engaged and error_norm > p.ctrl_engage_error:
                self._engaged = True
            elif self._engaged and error_norm <= p.ctrl_release_error:
                self._engaged = False

            if self._engaged:
                desired_pan = _clamp(p.kp_pan * ex, -p.ctrl_max_speed, p.ctrl_max_speed)
                # image y grows down -> below center means look DOWN (-tilt)
                desired_tilt = _clamp(-p.kp_tilt * ey, -p.ctrl_max_speed, p.ctrl_max_speed)
            else:
                desired_pan = desired_tilt = 0.0

        # Ease the commanded velocity toward the desired velocity.
        self._pan_cmd, self._accel_pan = _smooth_damp(
            self._pan_cmd, desired_pan, self._accel_pan, p.ctrl_smooth_time, 10.0, dt)
        self._tilt_cmd, self._accel_tilt = _smooth_damp(
            self._tilt_cmd, desired_tilt, self._accel_tilt, p.ctrl_smooth_time, 10.0, dt)

        pan_out = -self._pan_cmd if p.invert_pan else self._pan_cmd
        tilt_out = -self._tilt_cmd if p.invert_tilt else self._tilt_cmd
        # Snap tiny residuals to zero so we can actually issue Stop.
        if abs(pan_out) < 0.02:
            pan_out = 0.0
        if abs(tilt_out) < 0.02:
            tilt_out = 0.0
        return ControlStatus(
            ex=ex, ey=ey, error_norm=error_norm, engaged=self._engaged,
            pan_vel=pan_out, tilt_vel=tilt_out,
        )
