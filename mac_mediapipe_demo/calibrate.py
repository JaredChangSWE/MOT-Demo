"""Interactive PTZ direction calibration.

ONVIF's velocity sign convention (positive pan = right, positive tilt = up) is
not honored by every camera. If yours is wired the opposite way, the tracker
steers *away* from the target and the camera spins endlessly. This tool nudges
the camera in the +pan and +tilt directions while showing the live view, asks
which way it actually moved, and prints (optionally writes) the correct
``TAPO_INVERT_PAN`` / ``TAPO_INVERT_TILT`` settings.

Run:
    python calibrate.py
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from config import SETTINGS
from discovery import discover_onvif_cameras
from ptz import PTZController
from stream import LatestFrameReader

NUDGE_SPEED = 0.25
NUDGE_SECONDS = 1.2


def _show_for(reader: LatestFrameReader, seconds: float, label: str) -> None:
    """Pump the preview window for ``seconds`` so the motion is visible."""
    end = time.time() + seconds
    while time.time() < end:
        frame = reader.read()
        if frame is not None:
            cv2.putText(
                frame, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 255), 2,
            )
            cv2.imshow("PTZ Calibration", frame)
        cv2.waitKey(30)


def _ask_yes(question: str) -> bool:
    while True:
        ans = input(f"{question} [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def main() -> None:
    SETTINGS.validate()
    cams = discover_onvif_cameras(bind_ip=SETTINGS.bind_ip or None)
    if not cams:
        print("No ONVIF cameras found.")
        return
    cam = cams[0]
    print(f"Using camera {cam['ip']}:{cam['port']}")

    ptz = PTZController(cam["ip"], cam["port"], SETTINGS.user, SETTINGS.password)
    # Calibrate against the lens that actually drives PTZ.
    reader = LatestFrameReader(SETTINGS.rtsp_url(cam["ip"], SETTINGS.track_path))

    # Let the stream warm up.
    print("Opening live view...")
    for _ in range(100):
        if reader.read() is not None:
            break
        time.sleep(0.1)

    invert_pan = invert_tilt = False
    try:
        input("\nReady. Press Enter to start the PAN test...")
        _show_for(reader, 0.5, "PAN test: watch the view")
        ptz.move(NUDGE_SPEED, 0.0)
        _show_for(reader, NUDGE_SECONDS, "PANNING (+) ...")
        ptz.stop()
        _show_for(reader, 0.4, "stopped")
        # +pan should pan the VIEW to the right (scene content slides left).
        if not _ask_yes("Did the VIEW pan to the RIGHT?"):
            invert_pan = True

        input("\nPress Enter to start the TILT test...")
        _show_for(reader, 0.5, "TILT test: watch the view")
        ptz.move(0.0, NUDGE_SPEED)
        _show_for(reader, NUDGE_SECONDS, "TILTING (+) ...")
        ptz.stop()
        _show_for(reader, 0.4, "stopped")
        # +tilt should tilt the VIEW up.
        if not _ask_yes("Did the VIEW tilt UP?"):
            invert_tilt = True
    finally:
        ptz.stop()
        reader.release()
        cv2.destroyAllWindows()

    print("\n==== Result ====")
    print(f"TAPO_INVERT_PAN={'1' if invert_pan else '0'}")
    print(f"TAPO_INVERT_TILT={'1' if invert_tilt else '0'}")

    if not (invert_pan or invert_tilt):
        print("Default directions are correct -- no changes needed.")
        return

    if _ask_yes("\nWrite these to your .env now?"):
        _write_env(invert_pan, invert_tilt)
        print("Written to .env. Re-run: python main.py")
    else:
        print("Add the lines above to your .env (or export them) before running main.py")


def _write_env(invert_pan: bool, invert_tilt: bool) -> None:
    env_path = Path(__file__).with_name(".env")
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    updates = {
        "TAPO_INVERT_PAN": "1" if invert_pan else "0",
        "TAPO_INVERT_TILT": "1" if invert_tilt else "0",
    }
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    for key, val in updates.items():
        if key not in seen:
            out.append(f"{key}={val}")
    env_path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
