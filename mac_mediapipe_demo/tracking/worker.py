"""Background tracking worker + shared state + a tiny profiler.

Splits the heavy work (detect -> track -> select -> control -> PTZ) onto its own
thread so the GUI thread only does read + compose + imshow and stays smooth even
when detection hiccups. The GUI reads the latest results from ``SharedState``.

The worker OWNS the detector, tracker, selector and controller, and reconciles
them against the live ``Params`` each iteration (so control-panel changes to the
tracker / detector are applied on the worker thread, not the GUI thread).
"""

from __future__ import annotations

import math
import threading
import time
import traceback
import sys
from collections import deque
from pathlib import Path

from applog import get_logger
from camera.ptz_worker import PTZCommandWorker
from detectors import make_detector
from params import Params

# Ensure sibling ptz_tracking package is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ptz_tracking.trackers import make_tracker

from tracking.rt_controller import ControlStatus, RTController
from tracking.target_select import TargetSelector

_log = get_logger("worker")


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class Profiler:
    """Rolling per-stage timing, in milliseconds."""

    def __init__(self, window: int = 120) -> None:
        self._samples: dict[str, deque] = {}
        self._window = window
        self.lock = threading.Lock()

    def add(self, stage: str, ms: float) -> None:
        with self.lock:
            dq = self._samples.setdefault(stage, deque(maxlen=self._window))
            dq.append(ms)

    def snapshot(self) -> dict[str, tuple[float, float]]:
        """stage -> (avg_ms, max_ms) over the rolling window."""
        with self.lock:
            return {
                s: (sum(dq) / len(dq), max(dq))
                for s, dq in self._samples.items() if dq
            }


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.tracks: list = []
        self.followed_id: int | None = None
        self.status: ControlStatus = ControlStatus(0, 0, 0, False, 0.0, 0.0)
        self.tracker_name: str = ""
        self.det_latency_ms: float = 0.0
        self.trk_latency_ms: float = 0.0
        self.worker_fps: float = 0.0
        self.worker_frame: np.ndarray | None = None

    def publish(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def read(self) -> dict:
        with self.lock:
            return {
                "tracks": list(self.tracks),
                "followed_id": self.followed_id,
                "status": self.status,
                "tracker_name": self.tracker_name,
                "det_latency_ms": self.det_latency_ms,
                "trk_latency_ms": self.trk_latency_ms,
                "worker_fps": self.worker_fps,
                "worker_frame": self.worker_frame.copy() if self.worker_frame is not None else None,
            }


class TrackingWorker(threading.Thread):
    def __init__(self, params: Params, ptz_worker: PTZCommandWorker, get_frame,
                 shared: SharedState, profiler: Profiler) -> None:
        super().__init__(daemon=True)
        self.p = params
        self.ptz_worker = ptz_worker
        self.get_frame = get_frame
        self.shared = shared
        self.prof = profiler
        self.running = True

        self.detector = make_detector(params)
        self.tracker = make_tracker(params.tracker_name, params)
        self.selector = TargetSelector(params)
        self.controller = RTController(params)
        self._cur_tracker = params.tracker_name
        self._smooth_center = None

        self._last_seq = -1
        self._cur_seq = -1
        self._last_infer = 0.0
        self._prev_followed: int | None = None
        # Continuous-velocity command state (smooth ContinuousMove control).
        self._cmd_vx = 0.0            # current eased velocity command
        self._cmd_vy = 0.0
        self._sent_vx = 0.0           # last velocity actually sent to the camera
        self._sent_vy = 0.0
        self._last_sent_t = 0.0
        # Auto direction calibration (windowed): while moving + target still,
        # compare error over ~0.5s; shrank=correct(lock), grew=wrong(flip).
        self._pan_ok = False
        self._tilt_ok = False
        self._cal_t0: float | None = None
        self._cal_ex0 = 0.0
        self._cal_ey0 = 0.0

    # -- reconcile live Params changes (owned on this thread) --------------
    def _reconcile(self) -> None:
        if self.p.tracker_name != self._cur_tracker:
            _log.info(f"switch tracker {self._cur_tracker} -> {self.p.tracker_name}")
            self.tracker = make_tracker(self.p.tracker_name, self.p)
            self._cur_tracker = self.p.tracker_name
            self.selector.reset()
            self.controller.reset()
            self._smooth_center = None
            self.ptz_worker.send_stop()

        want = "pose" if self.p.det_mode == 1 else "face"
        if self.detector.mode != want:
            _log.info(f"switch detector {self.detector.mode} -> {want}")
            self.detector.close()
            self.detector = make_detector(self.p)
        else:
            self.detector.maybe_rebuild()

    def _smoothed(self, cx: float, cy: float) -> tuple[float, float]:
        a = self.p.bbox_smoothing
        if self._smooth_center is None:
            self._smooth_center = (cx, cy)
        else:
            sx, sy = self._smooth_center
            self._smooth_center = (sx + a * (cx - sx), sy + a * (cy - sy))
        return self._smooth_center

    def flip_pan(self) -> None:
        self.p.invert_pan = not self.p.invert_pan
        self._pan_ok = True  # lock: the user is authoritative, stop auto-cal
        _log.info(f"manual: invert_pan={self.p.invert_pan} (locked)")

    def flip_tilt(self) -> None:
        self.p.invert_tilt = not self.p.invert_tilt
        self._tilt_ok = True
        _log.info(f"manual: invert_tilt={self.p.invert_tilt} (locked)")

    def _verify_direction(self, status: ControlStatus, still: bool, now: float) -> None:
        """Windowed sign check for continuous motion: while actively moving an
        axis AND the target is fairly still, compare the error to what it was
        ~0.5s ago. Shrank => direction correct (lock). Grew => wrong (flip)."""
        moving = abs(self._cmd_vx) > 0.03 or abs(self._cmd_vy) > 0.03
        if not still or not moving:
            self._cal_t0 = None  # can't trust this window
            return
        if self._cal_t0 is None:
            self._cal_t0, self._cal_ex0, self._cal_ey0 = now, status.ex, status.ey
            return
        if now - self._cal_t0 < 0.5:
            return
        m = 0.05
        if not self._pan_ok and abs(self._cmd_vx) > 0.03:
            if abs(status.ex) < abs(self._cal_ex0) - m:
                self._pan_ok = True
                _log.info("auto-cal: pan direction OK (locked)")
            elif abs(status.ex) > abs(self._cal_ex0) + m:
                self.p.invert_pan = not self.p.invert_pan
                _log.info(f"auto-cal: pan WRONG -> invert_pan={self.p.invert_pan}")
        if not self._tilt_ok and abs(self._cmd_vy) > 0.03:
            if abs(status.ey) < abs(self._cal_ey0) - m:
                self._tilt_ok = True
                _log.info("auto-cal: tilt direction OK (locked)")
            elif abs(status.ey) > abs(self._cal_ey0) + m:
                self.p.invert_tilt = not self.p.invert_tilt
                _log.info(f"auto-cal: tilt WRONG -> invert_tilt={self.p.invert_tilt}")
        self._cal_t0 = None  # start a fresh window

    def _drive_ptz(self, status: ControlStatus) -> str:
        """Two-tier bang-bang control for this camera's QUANTIZED velocity.

        The Tapo's ContinuousMove velocity is not proportional (measured: cmds
        0.05..0.30 all move at the same ~0.18/s), so easing/tapering the velocity
        does nothing -- it can't slow near center and overshoots. Instead:
          * far from center  -> FAST tier (keeps up with a walking person)
          * near (but outside the stop box) -> SLOW tier (gentle final approach)
          * inside the stop box -> STOP (coast is tiny ~0.016, so it's accurate)
        No easing (the camera ignores magnitude); we just pick a tier per axis.
        """
        p = self.p
        now = time.time()
        rel = p.ctrl_release_error

        def axis_cmd(e: float) -> float:
            a = abs(e)
            if a < rel:                       # inside the stop box -> hold
                return 0.0
            speed = p.ptz_fast_speed if a >= p.ptz_far_error else p.ptz_slow_speed
            return math.copysign(speed, e)

        if status.engaged:
            tx = axis_cmd(status.ex)
            ty = axis_cmd(-status.ey)         # ey>0 (below) -> look down
        else:
            tx = ty = 0.0
        if p.invert_pan:
            tx = -tx
        if p.invert_tilt:
            ty = -ty

        self._cmd_vx, self._cmd_vy = tx, ty
        status.pan_vel, status.tilt_vel = tx, ty

        # Inside the box / not engaged -> stop (once, on the moving->stopped edge).
        if tx == 0.0 and ty == 0.0:
            if self._sent_vx != 0.0 or self._sent_vy != 0.0:
                self.ptz_worker.send_stop()
                self._sent_vx = self._sent_vy = 0.0
            return "STOP (in box / not engaged)"

        # Resend when the tier changed or to refresh before the ONVIF Timeout.
        changed = (abs(tx - self._sent_vx) > 0.05 or abs(ty - self._sent_vy) > 0.05)
        stale = now - self._last_sent_t > max(0.5, p.ptz_timeout - 0.3)
        if changed or stale:
            self.ptz_worker.send_move(tx, ty, p.ptz_timeout)
            self._sent_vx, self._sent_vy = tx, ty
            self._last_sent_t = now
        return f"MOVE vx={tx:+.2f} vy={ty:+.2f} tier"

    def run(self) -> None:
        _log.info(f"worker started (detector={self.detector.mode}, "
                  f"tracker={self._cur_tracker}, inference_fps={self.p.inference_fps})")
        fps_t0 = time.time()
        frames = 0
        prof_t0 = time.time()
        while self.running:
            try:
                if self._step():
                    frames += 1
            except Exception:  # keep the thread alive; record what broke
                _log.error("worker step failed:\n" + traceback.format_exc())
                time.sleep(0.1)
                continue

            # Worker-rate + periodic profiling to the log file.
            if time.time() - fps_t0 >= 1.0:
                wfps = frames / (time.time() - fps_t0)
                frames = 0
                fps_t0 = time.time()
                self.shared.publish(worker_fps=wfps)
            if time.time() - prof_t0 >= 5.0:
                prof_t0 = time.time()
                snap = self.prof.snapshot()
                stats = "  ".join(
                    f"{s}={snap[s][0]:.1f}/{snap[s][1]:.1f}ms"
                    for s in ("detect", "track", "select+ctrl", "ptz")
                    if s in snap)
                _log.info(f"profile worker_fps={self.shared.read()['worker_fps']:.1f}  {stats}")
        _log.info("worker loop exited")

    def _step(self) -> bool:
        """One inference cycle; returns True if it actually ran (new frame)."""
        now = time.monotonic()
        # Throttle ML inference / decisions to inference_fps (display is separate).
        min_dt = 1.0 / max(0.5, self.p.inference_fps)
        if now - self._last_infer < min_dt:
            time.sleep(0.003)
            return False
        frame, seq = self.get_frame()
        if frame is None or seq == self._last_seq:
            time.sleep(0.003)
            return False
        self._last_seq = seq
        self._cur_seq = seq
        self._last_infer = now

        loop_t0 = time.perf_counter()
        self._reconcile()
        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        det_ms = (time.perf_counter() - t0) * 1000
        self.prof.add("detect", det_ms)

        t0 = time.perf_counter()
        tracks = self.tracker.update(detections, frame=frame, camera=None)
        trk_ms = (time.perf_counter() - t0) * 1000
        self.prof.add("track", trk_ms)

        t0 = time.perf_counter()
        followed = self.selector.select(tracks, w, h)
        # Only DRIVE the camera from a target actually observed THIS frame. A
        # coasting (Kalman-extrapolated) track has no real face behind it -- never
        # chase a phantom, or the camera pans forever after the face is gone.
        fresh = followed is not None and followed.time_since_update == 0
        if fresh:
            cx, cy = followed.center
            scx, scy = self._smoothed(cx / w, cy / h)
            target_norm = (scx, scy)
            target_vel = (followed.velocity[0] / w, followed.velocity[1] / h)
        else:
            self._smooth_center = None
            target_norm, target_vel = None, (0.0, 0.0)
        status = self.controller.update(target_norm, target_vel, self.p.dt)
        if fresh:
            # Verify direction only when the target is fairly still, so the
            # person's own motion isn't mistaken for a wrong camera direction.
            still = (target_vel[0] ** 2 + target_vel[1] ** 2) ** 0.5 < 0.02
            self._verify_direction(status, still, time.time())
        self.prof.add("select+ctrl", (time.perf_counter() - t0) * 1000)

        # Log target-lock and motion-state transitions (concise, not per-frame).
        fid = self.selector.followed_id
        if fid != self._prev_followed:
            if fid is None:
                _log.info(f"target lost ({len(tracks)} tracks visible)")
            else:
                _log.info(f"target locked ID {fid} "
                          f"(dets={len(detections)} tracks={len(tracks)})")
            self._prev_followed = fid

        t0 = time.perf_counter()
        reason = self._drive_ptz(status)
        self.prof.add("ptz", (time.perf_counter() - t0) * 1000)

        # Full per-decision debug line (to the log file) so behavior can be
        # diagnosed without seeing the screen: what was detected, where the
        # followed box is, whether it exceeds the engage ("judgment") box, and
        # what we decided to do (and why).
        if followed is not None:
            x1, y1, x2, y2 = followed.bbox
            # Use the track's own center (target_norm is None while coasting).
            cxn, cyn = followed.center[0] / w, followed.center[1] / h
            box = (f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                   f"c=({cxn:.2f},{cyn:.2f})")
        else:
            box = "box=NONE"
        beyond = status.error_norm > self.p.ctrl_engage_error
        _log.debug(
            f"dets={len(detections)} trks={len(tracks)} tgt="
            f"{fid if fid is not None else '--'} fresh={int(fresh)} {box} "
            f"ex={status.ex:+.2f} ey={status.ey:+.2f} err={status.error_norm:.2f} "
            f"beyond_box={beyond} eng={status.engaged} "
            f"inv=({int(self.p.invert_pan)},{int(self.p.invert_tilt)}) "
            f"cal=({int(self._pan_ok)},{int(self._tilt_ok)}) -> {reason}"
        )

        self.shared.publish(
            tracks=tracks, followed_id=fid, status=status,
            tracker_name=self.p.tracker_name,
            det_latency_ms=det_ms, trk_latency_ms=trk_ms,
            worker_frame=frame.copy(),
        )
        self.prof.add("worker_loop", (time.perf_counter() - loop_t0) * 1000)
        return True

    def stop_and_join(self) -> None:
        _log.info("stopping worker")
        self.running = False
        self.join(timeout=1.0)
        self.ptz_worker.send_stop()
        self.detector.close()
