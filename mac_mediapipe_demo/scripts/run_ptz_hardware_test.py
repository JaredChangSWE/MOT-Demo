"""Real ONVIF hardware test script for PTZ preemption and smooth stop.

Runs a real hardware sequence on the physical PTZ camera:
1. Move right (pan=+0.4) -> Interrupted halfway (0.8s) by send_stop().
2. Move left (pan=-0.4)  -> Interrupted halfway (0.8s) by send_stop().
3. Rapid command sequence (+0.1, +0.2, +0.3 -> -0.4) -> Demonstrates preemption.

Run:
    python scripts/run_ptz_hardware_test.py
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


def main() -> None:
    setup_logging()
    SETTINGS.validate()

    print("=== ONVIF PTZ Real Camera Hardware Test ===")
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
    ptz_worker = PTZCommandWorker(ptz)
    ptz_worker.start()

    try:
        print("\n--- Test 1: Move FAR RIGHT (+0.4) & Stop Halfway ---")
        print(">> Sending move right (+0.4)...")
        ptz_worker.send_move(0.4, 0.0, 1.0)
        time.sleep(0.8)  # Move for 0.8s (halfway)

        print(">> Interrupting HALFWAY with send_stop()...")
        ptz_worker.send_stop(force=True)
        print("   Camera stopped!")
        time.sleep(1.5)

        print("\n--- Test 2: Move FAR LEFT (-0.4) & Stop Halfway ---")
        print(">> Sending move left (-0.4)...")
        ptz_worker.send_move(-0.4, 0.0, 1.0)
        time.sleep(0.8)  # Move for 0.8s (halfway)

        print(">> Interrupting HALFWAY with send_stop()...")
        ptz_worker.send_stop(force=True)
        print("   Camera stopped!")
        time.sleep(1.5)

        print("\n--- Test 3: Rapid Command Overwrite (Preemption) ---")
        print(">> Rapidly sending (+0.1 -> +0.2 -> +0.3 -> -0.4)...")
        ptz_worker.send_move(0.1, 0.0, 1.0)
        time.sleep(0.02)
        ptz_worker.send_move(0.2, 0.0, 1.0)
        time.sleep(0.02)
        ptz_worker.send_move(0.3, 0.0, 1.0)
        time.sleep(0.02)
        ptz_worker.send_move(-0.4, 0.0, 1.0)
        print(">> Latest command (-0.4) sent! Camera should immediately pan LEFT without waiting.")
        time.sleep(1.0)

        print(">> Stopping PTZ...")
        ptz_worker.send_stop(force=True)
        print("\n=== Real Camera Hardware Test Completed Successfully! ===")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        ptz_worker.stop_and_join()


if __name__ == "__main__":
    main()
