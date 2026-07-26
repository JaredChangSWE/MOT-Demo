"""Custom control panel rendered at the BOTTOM of the single window.

OpenCV trackbars can only live at the top of a window, so instead this draws its
own compact sliders into an image (placed below the camera views) and handles
mouse drags to edit the shared ``Params``. Each control shows a big label, its
current value, a small slider, and a one-line description.

Layout is fixed so mouse coordinates map 1:1 (the window uses WINDOW_AUTOSIZE).
"""

from __future__ import annotations

import cv2
import numpy as np

from params import Params

# Camera row geometry (each lens), so the panel knows its y-offset.
VIEW_W = 640
VIEW_H = 360
PANEL_W = VIEW_W * 2          # 1280
_COLS = 3
_COL_W = PANEL_W // _COLS     # ~426
_HEADER_H = 30
_CELL_H = 46
_ROWS = 7
PANEL_H = _HEADER_H + _ROWS * _CELL_H + 12

_TRACKER_ATTR = "__tracker__"

# (attr, label, description, min, max, is_int)
# Direction is handled AUTOMATICALLY (auto-cal), so no manual invert sliders.
# Column layout is 3 wide x 7 tall; the most-used controls come first so they
# land in the left/middle columns.
SPECS: list[tuple[str, str, str, float, float, bool]] = [
    # column 1 -- detection / target
    (_TRACKER_ATTR, "Tracker", "algorithm: SORT / ByteTrack / DeepSORT / OC-SORT / BoT-SORT", 0, 4, True),
    ("det_mode", "Detector", "0 = face (fast), 1 = full-body pose", 0, 1, True),
    ("det_min_confidence", "Confidence", "higher = fewer false detections", 0.05, 1.0, False),
    ("inference_fps", "Inference FPS", "how often it detects & decides", 1, 30, True),
    ("follow_mode", "Follow", "0 = biggest person, 1 = nearest to center", 0, 1, True),
    ("bbox_smoothing", "Steadiness", "center smoothing (lower = steadier)", 0.05, 1.0, False),
    ("ctrl_target_lost_grace", "Re-acquire", "frames to keep a briefly-lost target", 0, 150, True),
    # column 2 -- smooth motion + the follow box (Start/Stop-follow)
    ("kp_pan", "Pan speed", "how fast it pans per error (higher = faster)", 0.1, 3.0, False),
    ("kp_tilt", "Tilt speed", "how fast it tilts per error", 0.1, 3.0, False),
    ("ctrl_max_speed", "Max speed", "top motor speed (lower = gentler)", 0.05, 1.0, False),
    ("ctrl_smooth_time", "Smoothness", "velocity easing; higher = smoother", 0.05, 1.0, False),
    ("ctrl_engage_error", "Start-follow", "start moving once face is THIS far off-center", 0.02, 1.0, False),
    ("ctrl_release_error", "Stop-follow", "stop once face is within THIS of center", 0.0, 0.5, False),
    ("ctrl_lead_frames", "Aim lead", "aim ahead for fast movers (0 = off)", 0, 30, True),
    # column 3 -- advanced
    ("det_downscale_width", "Detect width", "detection image width (lower = faster)", 320, 1280, True),
    ("trk_iou_threshold", "IoU match", "box overlap needed to keep the same ID", 0.05, 1.0, False),
    ("trk_max_age", "Track max age", "frames a track survives while unseen", 1, 120, True),
    ("trk_min_hits", "Track min hits", "detections before a track is confirmed", 1, 10, True),
]
_ATTR_ALIAS: dict[str, str] = {}


class ControlPanel:
    def __init__(self, params: Params, tracker_names: list[str], win: str) -> None:
        self.p = params
        self.names = tracker_names
        self.win = win
        self._hit: list[tuple[int, int, int, int]] = []  # (spec_idx, tx0, tx1, ty_canvas)

    # -- value get/set ----------------------------------------------------
    def _get(self, attr: str):
        if attr == _TRACKER_ATTR:
            try:
                return self.names.index(self.p.tracker_name)
            except ValueError:
                return 0
        return getattr(self.p, _ATTR_ALIAS.get(attr, attr))

    def _set(self, spec, frac: float) -> None:
        attr, _lbl, _desc, lo, hi, is_int = spec
        val = lo + frac * (hi - lo)
        if attr == _TRACKER_ATTR:
            self.p.tracker_name = self.names[int(round(val))]
            return
        if is_int:
            val = int(round(val))
        setattr(self.p, _ATTR_ALIAS.get(attr, attr), val)

    def _value_text(self, spec) -> str:
        attr, _lbl, _desc, _lo, _hi, is_int = spec
        if attr == _TRACKER_ATTR:
            return self.p.tracker_name
        if attr == "det_mode":
            return "face" if self.p.det_mode == 0 else "pose"
        if attr == "follow_mode":
            return "biggest" if self.p.follow_mode == 0 else "center"
        if attr in ("invert_pan", "invert_tilt"):
            return "ON" if self._get(attr) else "off"
        v = self._get(attr)
        return f"{int(v)}" if is_int else f"{float(v):.2f}"

    # -- rendering --------------------------------------------------------
    def render(self, extra: str = ""):
        img = np.full((PANEL_H, PANEL_W, 3), 28, dtype=np.uint8)
        cv2.putText(img, "CONTROLS  (drag a slider to adjust)", (12, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 200, 255), 1)
        if extra:
            cv2.putText(img, extra, (PANEL_W - 430, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 120), 1)

        self._hit = []
        for i, spec in enumerate(SPECS):
            attr, label, desc, lo, hi, _is_int = spec
            col, row = i // _ROWS, i % _ROWS
            x0 = col * _COL_W
            y0 = _HEADER_H + row * _CELL_H

            cv2.putText(img, label, (x0 + 12, y0 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1)
            cv2.putText(img, self._value_text(spec), (x0 + _COL_W - 96, y0 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (140, 255, 170), 1)

            tx0, tx1, ty = x0 + 12, x0 + _COL_W - 16, y0 + 28
            cv2.line(img, (tx0, ty), (tx1, ty), (80, 80, 80), 4)
            v = self._get(attr)
            frac = 0.0 if hi == lo else max(0.0, min(1.0, (float(v) - lo) / (hi - lo)))
            kx = int(tx0 + frac * (tx1 - tx0))
            cv2.line(img, (tx0, ty), (kx, ty), (90, 170, 250), 4)
            cv2.circle(img, (kx, ty), 6, (120, 200, 255), -1)

            cv2.putText(img, desc, (x0 + 12, y0 + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)
            # Hit region in CANVAS coords (panel sits below the camera row).
            self._hit.append((i, tx0, tx1, VIEW_H + ty))
        return img

    # -- mouse handling (registered on the window) ------------------------
    def on_mouse(self, event, x, y, flags, _param=None) -> None:
        dragging = (event == cv2.EVENT_LBUTTONDOWN
                    or (event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON)))
        if not dragging:
            return
        for i, tx0, tx1, ty in self._hit:
            if tx0 - 8 <= x <= tx1 + 8 and abs(y - ty) <= 12:
                frac = (x - tx0) / float(max(1, tx1 - tx0))
                self._set(SPECS[i], max(0.0, min(1.0, frac)))
                return
