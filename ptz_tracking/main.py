"""PTZ auto-tracking demo — orchestration loop.

Pipeline per frame:
    world.step -> detector.detect -> tracker.update -> pick/keep target
    -> render (at detection pose) -> controller.update (eased pan/tilt) -> overlay

Run interactively:
    python -m ptz_tracking.main
Run headless (no window; writes an mp4 and prints metrics):
    python -m ptz_tracking.main --headless --duration 40 --record out.mp4

Everything tunable lives in `Params` and is exposed as a CLI flag (see --help):
people count/speed, camera FOV/limits, detection noise, tracker smoothing, and
the easing/engage-range of the auto-tracking controller.

Keys (interactive): q quit | space pause | r re-target | n toggle detection noise
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from .camera import Camera
from .controller import PTZController
from .detector import GroundTruthDetector
from .overlay import draw_overlay
from .params import Params
from .renderer import render
from .trackers import Track, make_tracker, TRACKER_NAMES
from .world import World


def _pick_target(tracks: list[Track]) -> int | None:
    """First successfully recognized person: confirmed track with smallest id."""
    confirmed = [t for t in tracks if t.confirmed]
    if not confirmed:
        return None
    return min(t.track_id for t in confirmed)


def _find_track(tracks: list[Track], track_id: int | None) -> Track | None:
    if track_id is None:
        return None
    for t in tracks:
        if t.track_id == track_id:
            return t
    return None


class Demo:
    def __init__(self, params: Params | None = None, seed: int = 0):
        self.p = params or Params()
        self.seed = seed
        self.world = World(self.p, seed=seed)
        self.camera = Camera(self.p)
        self.detector = GroundTruthDetector(self.p, seed=seed + 1)
        self.tracker = make_tracker(self.p.tracker_name, self.p)
        self.controller = PTZController(self.p)

        self.target_id: int | None = None
        self.lost_frames = 0
        self.paused = False
        self.latency_ms = 0.0   # EMA-smoothed tracker.update() latency

    def rebuild_world(self) -> None:
        """Recreate the scene (e.g. after the people count changed via the UI)."""
        self.world = World(self.p, seed=self.seed)
        self.target_id = None
        self.lost_frames = 0

    def set_tracker(self, name: str) -> None:
        """Switch tracking algorithm at runtime (resets IDs/target)."""
        self.p.tracker_name = name
        self.tracker = make_tracker(name, self.p)
        self.target_id = None
        self.lost_frames = 0
        self.latency_ms = 0.0

    def step(self, fps: float | None = None) -> tuple[np.ndarray, float, str]:
        """Advance one frame. Returns (annotated_frame, target_error_norm, state)."""
        if not self.paused:
            self.world.step()

        detections = self.detector.detect(self.camera, self.world)

        # Render at the detection pose FIRST: bounding boxes line up with the
        # figures, and appearance-based trackers (DeepSORT / BoT-SORT) can crop
        # the frame. The camera only rotates later in the control step.
        frame = render(self.camera, self.world)

        # Time the tracker so the HUD can report its per-frame latency.
        t0 = time.perf_counter()
        tracks = self.tracker.update(detections, frame=frame, camera=self.camera)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.latency_ms = 0.85 * self.latency_ms + 0.15 * dt_ms if self.latency_ms else dt_ms

        # Target acquisition / maintenance.
        if self.target_id is None:
            self.target_id = _pick_target(tracks)
            self.lost_frames = 0

        target = _find_track(tracks, self.target_id)
        # Only steer toward a target that was actually detected recently. A track
        # that is merely coasting (the person walked out of view) extrapolates
        # off-screen, so chasing it would fling the camera around — hold instead.
        active = target is not None and target.time_since_update <= 2
        if self.target_id is None:
            state, target_center = "SEARCHING", None
        elif active:
            self.lost_frames = 0
            # Lead the target: aim ahead along its image-space velocity so a
            # fast mover stays centered despite the camera's smooth lag.
            cx, cy = target.center
            vx, vy = target.velocity
            lead = self.p.ctrl_lead_frames
            target_center = (cx + vx * lead, cy + vy * lead)
            state = "TRACKING"
        else:
            self.lost_frames += 1
            target_center = None
            state = "TARGET LOST"
            if self.lost_frames > self.p.ctrl_target_lost_grace:
                self.target_id = None  # re-acquire next frame

        status = self.controller.update(self.camera, target_center)
        if active:
            state = "CENTERED" if status.centered else ("TRACKING" if status.engaged else "IN RANGE")

        draw_overlay(frame, tracks, self.target_id, self.camera, status, state,
                     fps=fps, tracker_name=self.p.tracker_name, latency_ms=self.latency_ms)
        return frame, status.error_norm, state

    # -- key handling ------------------------------------------------------
    def handle_key(self, key: int) -> bool:
        """Returns False if the demo should quit."""
        if key in (ord("q"), 27):  # q or ESC
            return False
        if key == ord(" "):
            self.paused = not self.paused
        elif key == ord("r"):
            self.target_id = None
            self.lost_frames = 0
        elif key == ord("n"):
            self.detector.noisy = not self.detector.noisy
        return True


def run_interactive(demo: Demo, show_ui: bool = True) -> None:
    win = "PTZ Auto-Tracking Demo"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    panel = None
    if show_ui:
        try:
            from .controls import ControlPanel
            panel = ControlPanel(demo)
        except cv2.error as e:
            print(f"[warn] control panel unavailable ({e}); running without UI sliders.")
    last = time.time()
    fps = float(demo.p.fps)
    frame_ms = int(1000 / demo.p.fps)
    while True:
        if panel is not None:
            panel.apply()
        frame, _, _ = demo.step(fps=fps)
        now = time.time()
        dt = now - last
        last = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        cv2.imshow(win, frame)
        key = cv2.waitKey(frame_ms) & 0xFF
        if not demo.handle_key(key):
            break
    cv2.destroyAllWindows()


def run_headless(demo: Demo, frames: int, record: str | None) -> int:
    writer = None
    if record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(record, fourcc, demo.p.fps,
                                 (demo.p.image_w, demo.p.image_h))
    errors: list[float] = []
    target_ids: list[int | None] = []
    states: list[str] = []
    for _ in range(frames):
        frame, err, state = demo.step(fps=float(demo.p.fps))
        errors.append(err)
        target_ids.append(demo.target_id)
        states.append(state)
        if writer is not None:
            writer.write(frame)
    if writer is not None:
        writer.release()

    # --- metrics ---
    on_target = [e for e, s in zip(errors, states)
                 if s in ("TRACKING", "CENTERED", "IN RANGE")]
    distinct = sorted({t for t in target_ids if t is not None})
    secs = frames / demo.p.fps
    print(f"duration              : {secs:.1f}s ({frames} frames @ {demo.p.fps}fps)")
    print(f"distinct target ids   : {len(distinct)}  {distinct if len(distinct) <= 12 else distinct[:12] + ['...']}")
    if on_target:
        print(f"on-target error       : first={on_target[0]:.3f} "
              f"min={min(on_target):.3f} mean={np.mean(on_target):.3f} max={max(on_target):.3f}")
    print(f"frames CENTERED       : {states.count('CENTERED')}")
    print(f"frames TRACKING       : {states.count('TRACKING')}")
    print(f"frames IN RANGE       : {states.count('IN RANGE')}")
    print(f"frames SEARCHING/LOST : {states.count('SEARCHING')}/{states.count('TARGET LOST')}")
    if record:
        print(f"video written         : {record}")
    # Success: a target was held and the camera spent meaningful time on it, and
    # it never permanently lost the target for a long stretch.
    on_count = states.count("CENTERED") + states.count("TRACKING") + states.count("IN RANGE")
    ok = bool(distinct) and on_count > frames * 0.6
    print(f"RESULT                : {'PASS' if ok else 'FAIL'}  "
          f"(on-target {on_count}/{frames} = {100*on_count/frames:.0f}%)")
    return 0 if ok else 1


def build_params(args: argparse.Namespace) -> Params:
    """Construct Params from parsed CLI args (only override what was given)."""
    return Params(
        image_w=args.width, image_h=args.height, fps=args.fps,
        num_people=args.people, world_half=args.world_size,
        speed_scale=args.speed_scale, exclusion_radius=args.exclusion,
        person_speed_min=args.speed_min, person_speed_max=args.speed_max,
        camera_x=args.camera_x, camera_y=args.camera_y, camera_z=args.camera_z,
        fovy_deg=args.fovy, pan_limit_deg=args.pan_limit,
        pan_unlimited=not args.limit_pan,
        tilt_limit_low_deg=args.tilt_min, tilt_limit_high_deg=args.tilt_max,
        det_noisy=not args.no_noise, det_bbox_jitter_px=args.jitter,
        det_drop_prob=args.drop,
        trk_iou_threshold=args.iou, trk_max_age=args.max_age,
        trk_min_hits=args.min_hits, trk_trail_len=args.trail,
        trk_match_dist_factor=args.match_dist, bbox_smoothing=args.bbox_smoothing,
        ctrl_smooth_time=args.smooth_time, ctrl_max_speed_deg=args.max_speed,
        ctrl_engage_error=args.engage, ctrl_release_error=args.release,
        ctrl_target_lost_grace=args.lost_grace, ctrl_lead_frames=args.lead,
        tracker_name=args.tracker,
    )


def main() -> int:
    d = Params()  # defaults
    ap = argparse.ArgumentParser(
        description="PTZ camera auto-tracking demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    run = ap.add_argument_group("run")
    run.add_argument("--headless", action="store_true",
                     help="run without a window; print metrics")
    run.add_argument("--frames", type=int, default=0,
                     help="frames to run headless (overridden by --duration)")
    run.add_argument("--duration", type=float, default=30.0,
                     help="headless run length in SECONDS")
    run.add_argument("--record", type=str, default=None, help="write an mp4 here")
    run.add_argument("--seed", type=int, default=0, help="world RNG seed")
    run.add_argument("--no-ui", action="store_true",
                     help="hide the on-screen slider control panel")

    world = ap.add_argument_group("world / people")
    world.add_argument("--people", type=int, default=d.num_people)
    world.add_argument("--speed-scale", type=float, default=d.speed_scale,
                       help="multiply walking speed (e.g. 10 for 10x faster)")
    world.add_argument("--speed-min", type=float, default=d.person_speed_min)
    world.add_argument("--speed-max", type=float, default=d.person_speed_max)
    world.add_argument("--world-size", type=float, default=d.world_half,
                       help="half-size of the square walkable area (m)")
    world.add_argument("--exclusion", type=float, default=d.exclusion_radius,
                       help="keep-out radius around the camera (m)")

    cam = ap.add_argument_group("camera")
    cam.add_argument("--camera-x", type=float, default=d.camera_x)
    cam.add_argument("--camera-y", type=float, default=d.camera_y)
    cam.add_argument("--camera-z", type=float, default=d.camera_z, help="camera height (m)")
    cam.add_argument("--width", type=int, default=d.image_w)
    cam.add_argument("--height", type=int, default=d.image_h)
    cam.add_argument("--fps", type=int, default=d.fps)
    cam.add_argument("--fovy", type=float, default=d.fovy_deg, help="vertical FOV (deg)")
    cam.add_argument("--limit-pan", action="store_true",
                     help="enable the mechanical pan limit (default: unlimited 360° rotation)")
    cam.add_argument("--pan-limit", type=float, default=d.pan_limit_deg,
                     help="pan limit in degrees (only used with --limit-pan)")
    cam.add_argument("--tilt-min", type=float, default=d.tilt_limit_low_deg)
    cam.add_argument("--tilt-max", type=float, default=d.tilt_limit_high_deg)

    det = ap.add_argument_group("detection")
    det.add_argument("--no-noise", action="store_true", help="clean detections")
    det.add_argument("--jitter", type=float, default=d.det_bbox_jitter_px,
                     help="bbox noise std-dev (px)")
    det.add_argument("--drop", type=float, default=d.det_drop_prob,
                     help="per-person missed-detection probability")

    trk = ap.add_argument_group("tracker")
    trk.add_argument("--tracker", choices=TRACKER_NAMES, default=d.tracker_name,
                     help="multi-object tracking algorithm")
    trk.add_argument("--iou", type=float, default=d.trk_iou_threshold)
    trk.add_argument("--max-age", type=int, default=d.trk_max_age)
    trk.add_argument("--min-hits", type=int, default=d.trk_min_hits)
    trk.add_argument("--trail", type=int, default=d.trk_trail_len)
    trk.add_argument("--match-dist", type=float, default=d.trk_match_dist_factor)
    trk.add_argument("--bbox-smoothing", type=float, default=d.bbox_smoothing,
                     help="EMA weight for bbox/trail; lower = smoother (less jitter)")

    ctl = ap.add_argument_group("auto-tracking controller")
    ctl.add_argument("--smooth-time", type=float, default=d.ctrl_smooth_time,
                     help="ease-curve time (s); larger = gentler start/stop")
    ctl.add_argument("--max-speed", type=float, default=d.ctrl_max_speed_deg,
                     help="max pan/tilt speed (deg/s)")
    ctl.add_argument("--lead", type=float, default=d.ctrl_lead_frames,
                     help="aim this many frames ahead of the target (keeps fast targets centered)")
    ctl.add_argument("--engage", type=float, default=d.ctrl_engage_error,
                     help="start tracking once normalized error exceeds this")
    ctl.add_argument("--release", type=float, default=d.ctrl_release_error,
                     help="stop tracking once normalized error drops below this")
    ctl.add_argument("--lost-grace", type=int, default=d.ctrl_target_lost_grace,
                     help="frames to coast before re-acquiring a lost target")

    args = ap.parse_args()
    params = build_params(args)
    demo = Demo(params, seed=args.seed)

    if args.headless:
        frames = args.frames if args.frames > 0 else int(args.duration * params.fps)
        return run_headless(demo, frames, args.record)
    run_interactive(demo, show_ui=not args.no_ui)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
