"""Thin ONVIF PTZ wrapper.

Wraps ``onvif.ONVIFCamera`` so the control loop only sees ``move(pan, tilt)``
and ``stop()``. Uses ContinuousMove (velocity-based), which is the right
primitive for smooth tracking -- you command a direction/speed and later stop,
rather than nudging to absolute positions.
"""

from __future__ import annotations

import time

from onvif import ONVIFCamera

from applog import get_logger

_log = get_logger("ptz")


class PTZController:
    def __init__(self, ip: str, port: int, user: str, password: str) -> None:
        self._cam = ONVIFCamera(ip, port, user, password)
        self._media = self._cam.create_media_service()
        self._ptz = self._cam.create_ptz_service()
        profiles = self._media.GetProfiles()
        if not profiles:
            raise RuntimeError("Camera exposed no ONVIF media profiles.")
        self._profile_token = profiles[0].token
        self._moving = False

    def move(self, pan: float, tilt: float, timeout: float | None = None) -> None:
        """Command a continuous pan/tilt velocity in [-1, 1].

        ``timeout`` (seconds) is the ONVIF ContinuousMove Timeout: the device
        auto-stops after it if no fresh command arrives (a safety net). NOTE:
        this camera rejects sub-second durations (aborts the connection), so the
        timeout is sent as whole seconds, minimum 1s. Tight start/stop is handled
        by the explicit Stop() call, not this timeout.
        """
        request = {
            "ProfileToken": self._profile_token,
            "Velocity": {"PanTilt": {"x": pan, "y": tilt}},
        }
        if timeout is not None:
            # ISO-8601 whole-second duration; sub-second (PT0.4S) aborts on this
            # camera, PT1S works.
            request["Timeout"] = f"PT{max(1, round(timeout))}S"
        try:
            self._ptz.ContinuousMove(request)
            self._moving = True
        except Exception as exc:  # noqa: BLE001 - log ONVIF faults, keep running
            _log.error(f"ContinuousMove failed (pan={pan:.2f} tilt={tilt:.2f}): {exc}")

    def move_relative(self, dx: float, dy: float) -> None:
        """Move by a precise relative pan/tilt step (ONVIF RelativeMove).

        ``dx``/``dy`` are in the camera's normalized translation space (-1..1 =
        full travel), so the step is an exact angle regardless of latency -- the
        right primitive for small, repeatable steps like the Tapo app's 5-degree
        nudges. Self-completing; no Stop needed.
        """
        try:
            self._ptz.RelativeMove({
                "ProfileToken": self._profile_token,
                "Translation": {"PanTilt": {"x": dx, "y": dy}},
            })
        except Exception as exc:  # noqa: BLE001
            _log.error(f"RelativeMove failed (dx={dx:.3f} dy={dy:.3f}): {exc}")

    def get_position(self) -> tuple[float, float] | None:
        try:
            s = self._ptz.GetStatus({"ProfileToken": self._profile_token})
            return (float(s.Position.PanTilt.x), float(s.Position.PanTilt.y))
        except Exception:  # noqa: BLE001
            return None

    def wait_settled(self, timeout: float = 2.0, poll: float = 0.07,
                     eps: float = 0.0015) -> float:
        """Block until the gimbal stops moving (position stable for 2 polls).

        The camera doesn't report MoveStatus, so we watch the reported PanTilt
        position instead. This prevents RelativeMove commands from stacking up
        (which makes the camera coast past the target and oscillate). Returns the
        seconds waited.
        """
        t0 = time.time()
        time.sleep(0.1)  # let the move actually begin
        prev = self.get_position()
        stable = 0
        while time.time() - t0 < timeout:
            cur = self.get_position()
            if cur is not None and prev is not None \
                    and abs(cur[0] - prev[0]) < eps and abs(cur[1] - prev[1]) < eps:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            prev = cur
            time.sleep(poll)
        return time.time() - t0

    def stop(self, force: bool = False) -> None:
        if not self._moving and not force:
            return
        try:
            self._ptz.Stop(
                {"ProfileToken": self._profile_token, "PanTilt": True, "Zoom": False}
            )
        except Exception as exc:  # noqa: BLE001
            _log.error(f"Stop failed: {exc}")
        self._moving = False

    @property
    def is_moving(self) -> bool:
        return self._moving
