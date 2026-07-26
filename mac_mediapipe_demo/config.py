"""Configuration and tuning parameters.

Credentials are read from the environment (or an optional local ``.env`` file)
so they never get committed. Copy ``.env.example`` to ``.env`` and fill it in,
or export the variables in your shell:

    export TAPO_USER="your_user"
    export TAPO_PASS="your_pass"
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader (no external dependency).

    Only sets keys that are not already present in the environment, so a real
    shell export always wins over the file.
    """
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ---- Camera credentials (set in the ONVIF/Tapo app account) ----
    user: str = os.environ.get("TAPO_USER", "")
    password: str = os.environ.get("TAPO_PASS", "")

    # DUAL-camera device. Each lens has an HD and an SD substream:
    #   lens 1: stream1 (2304x1296 HD) / stream2 (1280x720 SD)
    #   lens 2: stream6 (2304x1296 HD) / stream7 (1280x720 SD)
    # Default to the SD substreams: two HD feeds over Wi-Fi saturate bandwidth
    # and cause lag/stutter. Set TAPO_CAM1_PATH=stream1 / TAPO_CAM2_PATH=stream6
    # for full HD if your network can carry it.
    cam1_path: str = os.environ.get("TAPO_CAM1_PATH", "stream2")
    cam2_path: str = os.environ.get("TAPO_CAM2_PATH", "stream7")
    # Which lens drives auto-tracking / PTZ (1 or 2). Default 2 (stream6),
    # matching the original script.
    track_cam: int = int(os.environ.get("TAPO_TRACK_CAM", "2"))
    rtsp_port: int = int(os.environ.get("TAPO_RTSP_PORT", "554"))

    # Optional: pin WS-Discovery to a specific local interface IP (your Mac's
    # LAN address, e.g. 10.0.0.165). Leave blank to auto-detect. Useful if a VPN
    # confuses interface selection.
    bind_ip: str = os.environ.get("TAPO_BIND_IP", "")

    # ---- Tracking control loop ----
    deadband: float = 0.15   # no movement while the target sits within the
                             # central 15% of the frame (prevents jitter)
    kp_pan: float = 0.30     # horizontal proportional gain
    kp_tilt: float = 0.20    # vertical proportional gain
    command_interval: float = 0.15  # min seconds between PTZ commands

    # Direction sign. ONVIF's convention (positive pan = right, positive tilt =
    # up) is not universal -- some cameras are wired the opposite way, which
    # makes the camera turn AWAY from the target and spin forever. If yours goes
    # the wrong way, flip these (env TAPO_INVERT_PAN / TAPO_INVERT_TILT = 1).
    invert_pan: bool = os.environ.get("TAPO_INVERT_PAN", "0") == "1"
    invert_tilt: bool = os.environ.get("TAPO_INVERT_TILT", "0") == "1"

    # Anti-runaway safety: if the camera moves continuously for this long without
    # the target re-entering the deadband, force a stop + brief cooldown. Caps
    # any wrong-direction / phantom-target chase instead of spinning endlessly.
    max_move_seconds: float = 2.5
    cooldown_seconds: float = 0.5

    # ---- MediaPipe pose ----
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6

    def rtsp_url(self, ip: str, path: str) -> str:
        return f"rtsp://{self.user}:{self.password}@{ip}:{self.rtsp_port}/{path}"

    @property
    def track_path(self) -> str:
        return self.cam2_path if self.track_cam == 2 else self.cam1_path

    def validate(self) -> None:
        if not self.user or not self.password:
            raise SystemExit(
                "Missing camera credentials. Set TAPO_USER and TAPO_PASS "
                "(see .env.example)."
            )


SETTINGS = Settings()
