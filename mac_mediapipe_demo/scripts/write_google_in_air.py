"""Air-writing 'Google' test script for ONVIF PTZ camera.

Traces out the letters G-o-o-g-l-e in mid-air using smooth continuous PTZ velocity control.

Run:
    python scripts/write_google_in_air.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Insert parent directory to Python path if executed directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from applog import setup_logging
from camera.discovery import discover_onvif_cameras
from camera.ptz import PTZController
from camera.ptz_worker import PTZCommandWorker
from config import SETTINGS


def draw_letter(worker: PTZCommandWorker, letter_name: str, strokes: list[tuple[float, float, float]]) -> None:
    print(f"\n>>> Writing letter '{letter_name}'...")
    for vx, vy, duration in strokes:
        if vx == 0.0 and vy == 0.0:
            worker.send_stop(force=True)
        else:
            worker.send_move(vx, vy, timeout=1.0)
        time.sleep(duration)
    worker.send_stop(force=True)
    time.sleep(0.4)  # Inter-letter pause


def main() -> None:
    setup_logging()
    SETTINGS.validate()

    print("=== PTZ Air-Writing Test: 'Google' ===")
    print("Discovering ONVIF camera...")
    cameras = discover_onvif_cameras(bind_ip=SETTINGS.bind_ip or None)
    if not cameras:
        print("Error: No ONVIF camera found!")
        sys.exit(1)

    cam = cameras[0]
    ip = cam["ip"]
    print(f"Connecting to camera {ip}:{cam['port']}...")
    try:
        ptz = PTZController(ip, cam["port"], SETTINGS.user, SETTINGS.password)
    except Exception as exc:
        print(f"ONVIF connection failed: {exc}")
        sys.exit(1)

    print("Connected successfully!")
    worker = PTZCommandWorker(ptz)
    worker.start()

    try:
        # Speed scaling factor
        S = 0.35

        # Letter 'G'
        strokes_G = [
            (-S, 0.0, 0.40),       # Top stroke left
            (0.0, -S, 0.45),       # Left wall down
            (S, 0.0, 0.40),        # Bottom curve right
            (0.0, S, 0.25),        # Right wall up
            (-S, 0.0, 0.20),       # Inward bar left
        ]

        # Letter 'o'
        strokes_O = [
            (-S, 0.0, 0.30),       # Top left
            (0.0, -S, 0.30),       # Down left
            (S, 0.0, 0.30),        # Bottom right
            (0.0, S, 0.30),        # Up right (close loop)
        ]

        # Letter 'g' (lowercase)
        strokes_g_lower = [
            (-S, 0.0, 0.25),       # Loop top left
            (0.0, -S, 0.25),       # Loop down
            (S, 0.0, 0.25),        # Loop bottom right
            (0.0, S, 0.25),        # Loop up
            (0.0, -S, 0.50),       # Stem down
            (-S, 0.0, 0.25),       # Hook left
            (0.0, S * 0.7, 0.15),  # Hook tip up
        ]

        # Letter 'l' (lowercase)
        strokes_L = [
            (0.0, -S * 1.1, 0.60),  # Straight down
            (S * 0.7, 0.0, 0.15),   # Small foot right
        ]

        # Letter 'e' (lowercase)
        strokes_E = [
            (S, 0.0, 0.25),        # Middle bar right
            (0.0, S, 0.20),        # Up arc
            (-S, 0.0, 0.25),       # Top bar left
            (0.0, -S, 0.40),       # Down curve left
            (S, 0.0, 0.25),        # Bottom bar right
        ]

        print("\nGet ready! PTZ Camera is starting to write 'Google' in mid-air in 2 seconds...")
        time.sleep(2.0)

        draw_letter(worker, "G", strokes_G)
        draw_letter(worker, "o", strokes_O)
        draw_letter(worker, "o", strokes_O)
        draw_letter(worker, "g", strokes_g_lower)
        draw_letter(worker, "l", strokes_L)
        draw_letter(worker, "e", strokes_E)

        print("\n=== Finished writing 'Google' in mid-air! ===")

    except KeyboardInterrupt:
        print("\nAir-writing interrupted by user.")
    finally:
        worker.send_stop(force=True)
        worker.stop_and_join()


if __name__ == "__main__":
    main()
