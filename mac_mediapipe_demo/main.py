"""Real dual-camera ONVIF auto-tracker with pluggable MOT + live tuning.

Threaded: a background worker does detect -> track -> select -> control -> PTZ,
while the GUI thread only reads frames, composes the view, and renders — so the
display stays smooth even when detection hiccups. Everything (both lens views,
the tuning sliders, a readable parameter panel, and live profiling) lives in one
window.

Run:
    python main.py                # dual view + tracking + controls
    python main.py --list-only    # list ONVIF cameras and exit
    python main.py --no-display   # headless (uses Params defaults)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from applog import get_logger, setup_logging
from camera import DiscoveredCamera, LatestFrameReader, PTZCommandWorker, PTZController, discover_onvif_cameras
from config import SETTINGS
from params import Params
from tracking import Profiler, SharedState, TrackingWorker
from ui import PANEL_W, VIEW_H, VIEW_W, ControlPanel
import ui.hud as hud

# Reuse the pluggable trackers from the sibling sim package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptz_tracking.trackers import TRACKER_NAMES  # noqa: E402


def _view(frame, label: str):
    """One camera lens scaled to the fixed view size, captioned."""
    if frame is None:
        img = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
    else:
        img = cv2.resize(frame, (VIEW_W, VIEW_H))
    cv2.rectangle(img, (0, 0), (VIEW_W, 26), (0, 0, 0), -1)
    cv2.putText(img, label, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    return img


def camera_row(frame1, frame2, label1: str, label2: str):
    """Two camera views side by side (top of the window)."""
    return np.hstack([_view(frame1, label1), _view(frame2, label2)])


def select_camera(cameras: list[DiscoveredCamera]) -> DiscoveredCamera:
    print(f"\nFound {len(cameras)} camera(s):")
    for i, cam in enumerate(cameras):
        print(f"  [{i}] IP: {cam['ip']}  (ONVIF port: {cam['port']})")
    if len(cameras) == 1:
        return cameras[0]
    choice = input(f"Select a camera [0-{len(cameras) - 1}] (default 0): ").strip()
    if choice.isdigit() and 0 <= int(choice) < len(cameras):
        return cameras[int(choice)]
    return cameras[0]


def run(show_display: bool) -> None:
    log_path = setup_logging()
    log = get_logger("main")
    log.info(f"=== run start (display={show_display}) === log: {log_path}")
    SETTINGS.validate()

    cameras = discover_onvif_cameras(bind_ip=SETTINGS.bind_ip or None)
    if not cameras:
        print(
            "\nNo ONVIF cameras found. Common causes:\n"
            "  1. macOS Local Network permission for your terminal is OFF "
            "(the usual cause of 'No route to host' above) -- enable it in "
            "System Settings > Privacy & Security > Local Network, then quit "
            "and reopen the terminal.\n"
            "  2. Camera and Mac are on different subnets, or ONVIF is disabled."
        )
        sys.exit(1)

    cam = select_camera(cameras)
    ip = cam["ip"]

    params = Params()
    params.invert_pan = SETTINGS.invert_pan
    params.invert_tilt = SETTINGS.invert_tilt

    print(f"\nConnecting to dual-camera device {ip} ...")
    print(f"  Camera 1: {SETTINGS.cam1_path}   Camera 2: {SETTINGS.cam2_path}")
    print(f"  Auto-tracking lens: camera {SETTINGS.track_cam} ({SETTINGS.track_path})")

    log.info(f"device {ip}: cam1={SETTINGS.cam1_path} cam2={SETTINGS.cam2_path} "
             f"track_cam={SETTINGS.track_cam} invert_pan={SETTINGS.invert_pan} "
             f"invert_tilt={SETTINGS.invert_tilt}")
    try:
        ptz = PTZController(ip, cam["port"], SETTINGS.user, SETTINGS.password)
    except Exception as exc:  # noqa: BLE001
        log.error(f"ONVIF connection failed: {exc}")
        print(f"ONVIF connection failed (check username/password): {exc}")
        sys.exit(1)
    log.info("ONVIF connected")

    print("Starting streams + background tracking worker (press 'q' to quit)...")
    reader1 = LatestFrameReader(SETTINGS.rtsp_url(ip, SETTINGS.cam1_path), name="cam1")
    reader2 = LatestFrameReader(SETTINGS.rtsp_url(ip, SETTINGS.cam2_path), name="cam2")
    track_is_cam2 = SETTINGS.track_cam == 2

    def get_track_frame():
        return reader2.latest() if track_is_cam2 else reader1.latest()

    ptz_worker = PTZCommandWorker(ptz)
    ptz_worker.start()

    shared = SharedState()
    prof = Profiler()
    worker = TrackingWorker(params, ptz_worker, get_track_frame, shared, prof)
    worker.start()

    win = "PTZ Tracker"
    label1 = f"Camera 1 ({SETTINGS.cam1_path})"
    label2 = f"Camera 2 ({SETTINGS.cam2_path})"
    panel = None
    if show_display:
        # One window: camera views on top, custom control panel below. AUTOSIZE
        # so mouse coordinates map 1:1 to the drawn sliders.
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(win, 20, 20)
        panel = ControlPanel(params, TRACKER_NAMES, win)
        cv2.setMouseCallback(win, panel.on_mouse)

    disp_t0 = time.time()
    disp_frames = 0
    disp_fps = 0.0
    log_t0 = time.time()
    try:
        while True:
            frame1 = reader1.read()
            frame2 = reader2.read()

            state = shared.read()
            meta = {
                "tracker_name": state["tracker_name"] or params.tracker_name,
                "latency_ms": state["det_latency_ms"] + state["trk_latency_ms"],
                "engage_error": params.ctrl_engage_error,
                "release_error": params.ctrl_release_error,
            }
            overlay_frame = frame2 if track_is_cam2 else frame1
            if overlay_frame is not None:
                hud.draw(overlay_frame, state["tracks"], state["followed_id"],
                         state["status"], meta)

            disp_frames += 1
            if time.time() - disp_t0 >= 1.0:
                disp_fps = disp_frames / (time.time() - disp_t0)
                disp_frames = 0
                disp_t0 = time.time()

            if show_display:
                row = camera_row(frame1, frame2, label1, label2)
                ctrl = panel.render(
                    extra=f"display {disp_fps:4.1f}fps  worker {state['worker_fps']:4.1f}fps"
                          f"   keys: q=quit  w=worker win  p=flip pan  t=flip tilt")
                cv2.imshow(win, np.vstack([row, ctrl]))

                if params.show_worker_window:
                    wf = state.get("worker_frame")
                    if wf is not None:
                        hud.draw(wf, state["tracks"], state["followed_id"],
                                 state["status"], meta)
                        wf_view = _view(wf, f"PTZ Worker Frame (Cam {SETTINGS.track_cam} Synced)")
                        cv2.imshow("PTZ Worker View", wf_view)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("w"):
                    params.show_worker_window = not params.show_worker_window
                    if not params.show_worker_window:
                        try:
                            cv2.destroyWindow("PTZ Worker View")
                        except Exception:
                            pass
                    log.info(f"toggled show_worker_window={params.show_worker_window}")
                elif key == ord("p"):
                    worker.flip_pan()   # manually flip pan direction (and lock)
                elif key == ord("t"):
                    worker.flip_tilt()  # manually flip tilt direction (and lock)

            # Periodic profiling log (answers "where is it stuck?").
            if time.time() - log_t0 >= 2.0:
                log_t0 = time.time()
                snap = prof.snapshot()
                parts = [f"display={disp_fps:.1f}fps",
                         f"worker={state['worker_fps']:.1f}fps"]
                for stage in ("detect", "track", "select+ctrl", "ptz", "worker_loop"):
                    if stage in snap:
                        avg, mx = snap[stage]
                        parts.append(f"{stage}={avg:.1f}/{mx:.1f}ms")
                print("[profile] " + "  ".join(parts))
    except KeyboardInterrupt:
        pass
    except Exception:  # noqa: BLE001 - record any GUI-thread crash
        import traceback
        log.error("GUI loop failed:\n" + traceback.format_exc())
    finally:
        print("\nStopping worker, motors, and releasing resources...")
        log.info(f"shutdown; cam1 stats={reader1.stats()} cam2 stats={reader2.stats()}")
        worker.stop_and_join()
        ptz_worker.stop_and_join()
        reader1.release()
        reader2.release()
        if show_display:
            cv2.destroyAllWindows()
        log.info("=== run end ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-camera ONVIF MOT auto-tracker")
    parser.add_argument("--list-only", action="store_true",
                        help="Discover and list ONVIF cameras, then exit.")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless: no window (Params defaults).")
    parser.add_argument("--timeout", type=float, default=3.0,
                        help="WS-Discovery wait time in seconds (default 3).")
    args = parser.parse_args()

    if args.list_only:
        cams = discover_onvif_cameras(timeout=args.timeout,
                                      bind_ip=SETTINGS.bind_ip or None)
        if not cams:
            print("No ONVIF cameras found.")
            return
        for i, c in enumerate(cams):
            print(f"  [{i}] {c['ip']}:{c['port']}  ({c['xaddr']})")
        return

    run(show_display=not args.no_display)


if __name__ == "__main__":
    main()
