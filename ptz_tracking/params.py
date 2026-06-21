"""Tunable parameters for the PTZ auto-tracking demo.

A single `Params` object is threaded through every component (world, camera,
detector, tracker, controller) so the whole demo can be reconfigured from the
command line without editing code. Defaults mirror the constants in `config.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config


@dataclass
class Params:
    # -- display -----------------------------------------------------------
    image_w: int = config.IMAGE_W
    image_h: int = config.IMAGE_H
    fps: int = config.FPS

    # -- world / people ----------------------------------------------------
    num_people: int = config.NUM_PEOPLE
    world_half: float = config.WORLD_HALF
    speed_scale: float = 2.5                        # multiplies base walking speed
    person_speed_min: float = config.PERSON_SPEED_RANGE[0]
    person_speed_max: float = config.PERSON_SPEED_RANGE[1]
    # No-go disc around the camera: people may not enter within this radius (m),
    # so nobody ever ends up right on top of the lens.
    exclusion_radius: float = 3.0

    # -- camera ------------------------------------------------------------
    # Fixed mounting point. Default is the CENTER of the scene so the PTZ camera
    # can pan around to follow people on any side of it.
    camera_x: float = 0.0
    camera_y: float = 0.0
    camera_z: float = 2.6
    fovy_deg: float = config.CAMERA_FOVY_DEG
    # Pan is UNLIMITED by default: the camera can rotate continuously (360°+) to
    # follow a target all the way around. pan_limit_deg only applies when
    # pan_unlimited is False.
    pan_unlimited: bool = True
    pan_limit_deg: float = 170.0
    tilt_limit_low_deg: float = -60.0              # look down at people close to the keep-out ring
    tilt_limit_high_deg: float = 25.0

    # -- detection ---------------------------------------------------------
    det_noisy: bool = True
    det_bbox_jitter_px: float = config.DET_BBOX_JITTER_PX
    det_drop_prob: float = config.DET_DROP_PROB

    # -- tracker -----------------------------------------------------------
    trk_iou_threshold: float = config.TRK_IOU_THRESHOLD
    trk_max_age: int = config.TRK_MAX_AGE
    trk_min_hits: int = config.TRK_MIN_HITS
    trk_trail_len: int = config.TRK_TRAIL_LEN
    trk_match_dist_factor: float = config.TRK_MATCH_DIST_FACTOR
    # EMA weight for new measurements when smoothing the displayed bbox/trail.
    # 1.0 = raw (no smoothing / max jitter), lower = smoother & less jittery.
    bbox_smoothing: float = 0.35
    # Which multi-object tracker to run. One of TRACKER_NAMES (see trackers pkg).
    tracker_name: str = "SORT"
    # ByteTrack-style two-stage confidence thresholds.
    trk_high_thresh: float = 0.6      # detections >= this are "high" (stage 1)
    trk_low_thresh: float = 0.1       # detections in [low, high) used in stage 2
    trk_new_thresh: float = 0.7       # min confidence to spawn a brand-new track
    # Appearance fusion (DeepSORT / BoT-SORT). Weight blends appearance cost with
    # IoU cost; gate is the max cosine distance allowed for an appearance match.
    trk_appearance_weight: float = 0.4
    trk_appearance_gate: float = 0.35
    # OC-SORT observation-centric momentum weight (direction-consistency cost).
    trk_ocm_weight: float = 0.2

    # -- controller (ease-curve tracking + engage range) -------------------
    # SmoothDamp easing: larger smooth_time => gentler ease-in / ease-out.
    ctrl_smooth_time: float = 0.35                 # seconds
    ctrl_max_speed_deg: float = 180.0              # cap on pan/tilt speed (deg/s)
    # Hysteresis "tracking range": the camera only STARTS moving once the
    # normalized centering error exceeds `engage`, and keeps tracking until the
    # error falls below `release`. engage > release avoids chattering.
    ctrl_engage_error: float = 0.15
    ctrl_release_error: float = 0.04
    ctrl_target_lost_grace: int = config.CTRL_TARGET_LOST_GRACE
    # Feed-forward lead: aim this many frames AHEAD of the target along its
    # image-space velocity, so a fast-moving target stays centred despite the
    # camera's (smooth) lag. 0 = aim at the current position.
    ctrl_lead_frames: float = 8.0

    def __post_init__(self) -> None:
        # Derived timestep; keep in sync with fps.
        self.dt: float = 1.0 / float(self.fps)
