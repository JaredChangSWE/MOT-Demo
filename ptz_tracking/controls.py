"""On-screen control panel (OpenCV trackbars) for live parameter tuning.

Opens a "PTZ Controls" window whose sliders are bound to the live `Params`
object — every component holds the same `Params`, so moving a slider takes
effect on the next frame. Structural changes (people count) rebuild the scene.

Trackbars are integer-only, so each slider stores ``value = param / scale`` and
we multiply back on read. The label states the unit and any scale factor, and a
full legend is printed to the console when the panel opens.
"""

from __future__ import annotations

import cv2

from .trackers import TRACKER_NAMES

_TRACKER_LABEL = "Tracker (0-4)"

# (label, Params attribute, slider_min, slider_max, scale, structural)
# Labels spell out the unit; `scale` keeps the on-screen integer intuitive.
SPECS: list[tuple[str, str, int, int, float, bool]] = [
    # --- scene -------------------------------------------------------------
    ("People in scene",         "num_people",             1,   30, 1.0,  True),
    ("Walk speed (%)",          "speed_scale",            0, 1000, 0.01, False),
    ("Keep-out radius (m)",     "exclusion_radius",       0,   15, 1.0,  False),
    ("Scene half-size (m)",     "world_half",             4,   40, 1.0,  False),
    # --- camera follow behaviour ------------------------------------------
    ("Camera ease time (1/100 s)", "ctrl_smooth_time",    1,  100, 0.01, False),
    ("Max pan speed (deg/s)",   "ctrl_max_speed_deg",    30,  600, 1.0,  False),
    ("Aim lead (frames ahead)", "ctrl_lead_frames",       0,   30, 1.0,  False),
    ("Start-follow error (%)",  "ctrl_engage_error",      1,  100, 0.01, False),
    ("Stop-follow error (%)",   "ctrl_release_error",     0,  100, 0.01, False),
    ("Re-acquire delay (frames)", "ctrl_target_lost_grace", 0, 150, 1.0, False),
    # --- detection / tracking ---------------------------------------------
    ("Box steadiness (1=raw)",  "bbox_smoothing",         5,  100, 0.01, False),
    ("Detection jitter (px)",   "det_bbox_jitter_px",     0,   12, 1.0,  False),
    ("Missed-detection (%)",    "det_drop_prob",          0,   50, 0.01, False),
]

# One-line plain-English explanation per control (printed as a legend).
_HELP: dict[str, str] = {
    "num_people":            "how many people walk in the scene",
    "speed_scale":           "walking speed; 100 = normal, 250 = 2.5x, 800 = very fast",
    "exclusion_radius":      "no-walk circle around the camera (people can't get closer)",
    "world_half":            "half-width of the square floor area",
    "ctrl_smooth_time":      "camera easing; lower = snappier follow, higher = gentler (20 = 0.20s)",
    "ctrl_max_speed_deg":    "ceiling on how fast the camera can pan/tilt",
    "ctrl_lead_frames":      "aim this many frames AHEAD of the target (helps keep fast movers centered)",
    "ctrl_engage_error":     "camera STARTS following once the target drifts this far off-center (15 = 15%)",
    "ctrl_release_error":    "camera STOPS adjusting once the target is within this of center (4 = 4%)",
    "ctrl_target_lost_grace":"frames to keep chasing a lost target before picking a new one",
    "bbox_smoothing":        "box/trail smoothing; lower = steadier (less jitter), 100 = raw",
    "det_bbox_jitter_px":    "simulated detector noise added to box edges",
    "det_drop_prob":         "simulated chance a person is missed each frame (5 = 5%)",
}


class ControlPanel:
    WIN = "PTZ Controls"

    def __init__(self, demo) -> None:
        self.demo = demo
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, 540, 680)

        # Tracking algorithm selector (0..4 -> TRACKER_NAMES).
        try:
            ti = TRACKER_NAMES.index(demo.p.tracker_name)
        except ValueError:
            ti = 0
        cv2.createTrackbar(_TRACKER_LABEL, self.WIN, ti, len(TRACKER_NAMES) - 1,
                           lambda _v: None)

        for label, attr, smin, smax, scale, _structural in SPECS:
            init = int(round(getattr(demo.p, attr) / scale))
            init = max(smin, min(smax, init))
            cv2.createTrackbar(label, self.WIN, init, smax, lambda _v: None)
            cv2.setTrackbarMin(label, self.WIN, smin)

        self._prev_people = demo.p.num_people
        self._prev_tracker_idx = ti
        self._print_legend()

    def _print_legend(self) -> None:
        print("\n=== PTZ Controls (drag sliders in the 'PTZ Controls' window) ===")
        names = "  ".join(f"{i}={n}" for i, n in enumerate(TRACKER_NAMES))
        print(f"  {_TRACKER_LABEL:<28} tracking algorithm: {names}")
        for label, attr, _smin, _smax, _scale, _s in SPECS:
            print(f"  {label:<28} {_HELP.get(attr, '')}")
        print("=" * 64 + "\n")

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
        # Tracker selector (switching resets IDs / target).
        ti = max(0, min(len(TRACKER_NAMES) - 1,
                        cv2.getTrackbarPos(_TRACKER_LABEL, self.WIN)))
        if ti != self._prev_tracker_idx:
            self.demo.set_tracker(TRACKER_NAMES[ti])
            self._prev_tracker_idx = ti
