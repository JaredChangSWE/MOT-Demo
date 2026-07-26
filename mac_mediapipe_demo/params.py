"""Tunable parameters for the real-camera PTZ auto-tracker.

A single `Params` object is threaded through the detector, the multi-object
tracker, the target selector, and the controller, so the whole pipeline can be
retuned live from the on-screen control panel (see controls_rt.py) without
editing code.

It intentionally includes every ``trk_*`` field the pluggable trackers in
``ptz_tracking.trackers`` read, so those trackers can be reused verbatim, plus
the real-camera detection / control / target-selection fields.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Params:
    # -- detection ---------------------------------------------------------
    # 0 = face (BlazeFace, fast + accurate for who's in frame), 1 = pose.
    det_mode: int = 0
    # Detection score threshold (both detectors). Higher = fewer false/weak hits.
    det_min_confidence: float = 0.7
    # Run detection on a copy downscaled to this width (px) for speed; boxes are
    # mapped back to full resolution. 0 = full frame.
    det_downscale_width: int = 960
    # Cap detection/tracking/control decisions to this rate (Hz). The display
    # still shows every camera frame; only the ML inference + PTZ decisions run
    # at this cadence -- 5 fps is plenty for tracking and keeps CPU low.
    inference_fps: float = 10.0  # higher = smoother/timelier velocity updates
    # pose-only knobs:
    det_num_poses: int = 4          # max simultaneous people (pose mode)
    det_min_visibility: float = 0.5  # landmark visibility to bound a box
    det_min_landmarks: int = 8       # min visible landmarks to accept a person
    min_tracking_confidence: float = 0.5
    # Display mode: True = open a separate window showing the exact frame read & processed by PTZ worker
    show_worker_window: bool = True

    # -- tracker (consumed by ptz_tracking.trackers) -----------------------
    tracker_name: str = "SORT"       # one of TRACKER_NAMES
    trk_iou_threshold: float = 0.2
    trk_max_age: int = 30
    trk_min_hits: int = 3
    trk_trail_len: int = 30
    trk_match_dist_factor: float = 1.5
    trk_high_thresh: float = 0.6
    trk_low_thresh: float = 0.1
    trk_new_thresh: float = 0.5
    trk_appearance_weight: float = 0.4
    trk_appearance_gate: float = 0.35
    trk_ocm_weight: float = 0.2
    # EMA smoothing of the followed target's center fed to the controller
    # (1.0 = raw / jittery, lower = steadier).
    bbox_smoothing: float = 0.35

    # -- target selection --------------------------------------------------
    # 0 = largest (closest) person, 1 = nearest to frame center.
    follow_mode: int = 0
    # Frames to keep chasing / waiting for a lost target before re-selecting.
    ctrl_target_lost_grace: int = 30

    # -- controller: SMOOTH continuous-velocity control of ONVIF PTZ -------
    # The camera is driven with ContinuousMove (velocity), NOT discrete steps, so
    # the motor runs continuously like a normal gimbal. Velocity is proportional
    # to the centering error (fast when far, slows to a smooth stop near center),
    # and the commanded velocity is eased so detection noise doesn't jerk it.
    deadband: float = 0.12           # legacy alias; engage/release drive behavior
    kp_pan: float = 0.5              # velocity per unit of error-beyond-stop-box
    kp_tilt: float = 0.4             # velocity per unit of error-beyond-stop-box
    ctrl_smooth_time: float = 0.30   # velocity easing time (s); higher = smoother
    ctrl_max_speed: float = 0.4      # cap on |ONVIF velocity| in [0, 1]
    # Wider hold band: start following only past a clear drift, and stop
    # adjusting well before dead center so it settles instead of hunting.
    ctrl_engage_error: float = 0.5   # start following once face is this far off-center
    ctrl_release_error: float = 0.25  # stop once face is within this of center
    # Lead defaults OFF on real hardware: velocity from a near-stationary face is
    # noisy and, with camera ego-motion, feed-forward drives oscillation. Raise
    # it only to help fast lateral movers.
    ctrl_lead_frames: float = 0.0    # aim this many frames ahead along target vel
    command_interval: float = 0.12   # min seconds between ONVIF commands
    # ONVIF ContinuousMove Timeout, WHOLE SECONDS (this camera rejects sub-second
    # durations). Safety auto-stop if commands stop arriving; tight start/stop is
    # done by the explicit Stop() call, not this. 1s is a good default.
    ptz_timeout: float = 1.0

    # RELATIVE-MOVE stepping (default actuation): instead of a velocity command
    # (ContinuousMove, whose real distance = velocity x time + network latency,
    # impossible to make reliably small), each decision issues one ONVIF
    # RelativeMove -- a precise fixed-angle step, exactly like the Tapo app's
    # 5-degree steps. Step size scales with the centering error, bounded by
    # min/max, with a minimum time between steps. The normalized translation
    # space is -1..1 over the full pan range (~360 deg), so 0.03 ~= 5 deg.
    # Smaller default steps = smoother motion (each settles faster too). Raise
    # step_gain/step_max via the panel if you need to catch fast movers.
    step_gain: float = 0.035    # normalized step per unit of centering error
    step_min: float = 0.015     # smallest step (~2.5 deg) -- gentle
    step_max: float = 0.06      # largest single step (~11 deg)
    step_interval: float = 0.10  # small floor; wait_settled() does the real pacing
    # After a step's motion has settled (position-polled), require this many fresh
    # detections before the next step. 1 is enough because wait_settled already
    # guarantees the move physically finished.
    step_settle_frames: int = 1

    # PULSE / continuous-move params (legacy fallback, not used by RelativeMove).
    # Pulse length scales with error: strong catch-up when the target is far,
    # tiny steps near center (no overshoot). Weak pulses can't keep up with a
    # moving person, so gain/max are generous; near-center stays small.
    pulse_gain: float = 0.35   # seconds of movement per unit of centering error
    pulse_min: float = 0.03
    pulse_max: float = 0.22

    # -- direction / safety ------------------------------------------------
    invert_pan: bool = False
    invert_tilt: bool = False
    max_move_seconds: float = 3.0    # anti-runaway: force stop after this long
    cooldown_seconds: float = 0.5

    def __post_init__(self) -> None:
        self.dt: float = 1.0 / 30.0
