"""On-screen control panel (OpenCV trackbars) for live parameter tuning.

Creates a "PTZ Controls" window with sliders bound to the live `Params` object.
Because every component holds a reference to the same `Params` instance, moving a
slider takes effect on the next frame. Structural changes (people count) trigger
a world rebuild.

Trackbars are integer-only, so each slider stores `value = param / scale`; on
read we multiply back by `scale` and clamp to `[smin, smax]`.
"""

from __future__ import annotations

import cv2

from .trackers import TRACKER_NAMES

_TRACKER_LABEL = "tracker 0-4"

# (label, Params attribute, slider_min, slider_max, scale, structural)
SPECS: list[tuple[str, str, int, int, float, bool]] = [
    ("people",           "num_people",             1,  30, 1.0,  True),
    ("speed x0.1",       "speed_scale",            0, 200, 0.1,  False),
    ("exclude x0.1m",    "exclusion_radius",       0,  80, 0.1,  False),
    ("world x0.5m",      "world_half",             4,  40, 0.5,  False),
    ("smooth_t x0.01s",  "ctrl_smooth_time",       1, 150, 0.01, False),
    ("max_spd deg/s",    "ctrl_max_speed_deg",    30, 600, 1.0,  False),
    ("lead frames",      "ctrl_lead_frames",       0,  30, 1.0,  False),
    ("engage x0.01",     "ctrl_engage_error",      1, 100, 0.01, False),
    ("release x0.01",    "ctrl_release_error",     0, 100, 0.01, False),
    ("box_smooth x0.01", "bbox_smoothing",         5, 100, 0.01, False),
    ("jitter px",        "det_bbox_jitter_px",     0,  12, 1.0,  False),
    ("pan_lim deg",      "pan_limit_deg",         10, 175, 1.0,  False),
    ("lost_grace",       "ctrl_target_lost_grace", 0, 150, 1.0,  False),
]


class ControlPanel:
    WIN = "PTZ Controls"

    def __init__(self, demo) -> None:
        self.demo = demo
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, 460, 600)
        # Algorithm selector: 0=SORT, 1=ByteTrack, 2=DeepSORT, 3=OC-SORT, 4=BoT-SORT
        try:
            ti = TRACKER_NAMES.index(demo.p.tracker_name)
        except ValueError:
            ti = 0
        cv2.createTrackbar(_TRACKER_LABEL, self.WIN, ti, len(TRACKER_NAMES) - 1,
                           lambda _v: None)
        for label, attr, smin, smax, scale, _structural in SPECS:
            init = int(round(getattr(demo.p, attr) / scale))
            init = max(smin, min(smax, init))
            # Some OpenCV builds need a callback; a no-op is fine (we poll).
            cv2.createTrackbar(label, self.WIN, init, smax, lambda _v: None)
            cv2.setTrackbarMin(label, self.WIN, smin)
        self._prev_people = demo.p.num_people
        self._prev_tracker_idx = ti

    def apply(self) -> None:
        """Read every slider and push values into the live Params (+ rebuild)."""
        p = self.demo.p
        for label, attr, smin, smax, scale, _structural in SPECS:
            v = max(smin, min(smax, cv2.getTrackbarPos(label, self.WIN)))
            value = v * scale
            if isinstance(getattr(p, attr), int) and float(value).is_integer():
                value = int(value)
            setattr(p, attr, value)
        if p.num_people != self._prev_people:
            self.demo.rebuild_world()
            self._prev_people = p.num_people
        # Tracker selector (string, handled specially — switching resets IDs).
        ti = max(0, min(len(TRACKER_NAMES) - 1,
                        cv2.getTrackbarPos(_TRACKER_LABEL, self.WIN)))
        if ti != self._prev_tracker_idx:
            self.demo.set_tracker(TRACKER_NAMES[ti])
            self._prev_tracker_idx = ti
