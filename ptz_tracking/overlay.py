"""Overlay / HUD rendering for the PTZ auto-tracking demo.

Draws all annotations on top of an already-rendered camera frame:
  - Bounding boxes + ID labels for every track
  - Highlighted target track
  - Motion trails (trajectory polylines)
  - Center crosshair at the camera's optical axis
  - Error vector from frame center to the locked target
  - Semi-transparent status/HUD panel (top-left)
  - Controls hint bar (bottom)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

from . import config

if TYPE_CHECKING:
    from .camera import Camera
    from .controller import ControlStatus
    from .trackers.base import Track

# ---------------------------------------------------------------------------
# Color palette (BGR)
# ---------------------------------------------------------------------------
_COLOR_TRACK      = (80, 200, 80)       # green-ish for regular tracks
_COLOR_TARGET     = (0, 220, 255)       # bright yellow/cyan for locked target
_COLOR_TRAIL      = (60, 160, 60)       # dimmer green for regular trails
_COLOR_TARGET_TRAIL = (0, 190, 240)     # brighter for target trail
_COLOR_CROSSHAIR  = (220, 220, 220)     # near-white crosshair
_COLOR_ERR_OK     = (60, 220, 60)       # green: error within deadzone
_COLOR_ERR_BAD    = (30, 100, 255)      # orange-red: adjusting
_COLOR_HUD_BG     = (20, 20, 20)        # very dark background for panel
_COLOR_LABEL_BG   = (20, 20, 20)        # label background
_COLOR_WHITE      = (255, 255, 255)
_COLOR_HINT       = (200, 200, 200)     # controls-hint bar text

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_PLAIN = cv2.FONT_HERSHEY_PLAIN

# Font scales
_FS_LABEL   = 0.40   # small label on track box
_FS_TARGET  = 0.48   # slightly larger for the locked target
_FS_HUD     = 0.46   # HUD panel lines
_FS_HINT    = 0.38   # bottom controls-hint bar

_TH = 1  # general font thickness


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ipt(x: float, y: float) -> tuple[int, int]:
    """Convert floats to an integer pixel coordinate tuple."""
    return (int(round(x)), int(round(y)))


def _clamp_pt(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    """Clamp a point to image bounds and return as (int, int)."""
    return (int(np.clip(round(x), 0, w - 1)),
            int(np.clip(round(y), 0, h - 1)))


def _text_size(text: str, font: int, scale: float, thickness: int
               ) -> tuple[int, int]:
    """Return (width, height) of rendered text (height excludes descender)."""
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    return w, h


def _draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    font: int,
    scale: float,
    color: tuple[int, int, int],
    bg: tuple[int, int, int] = _COLOR_LABEL_BG,
    thickness: int = _TH,
    alpha: float = 0.75,
) -> None:
    """Draw text with a semi-opaque filled-rect background."""
    tw, th = _text_size(text, font, scale, thickness)
    pad = 2
    x1 = max(0, x - pad)
    y1 = max(0, y - th - pad)
    x2 = min(img.shape[1] - 1, x + tw + pad)
    y2 = min(img.shape[0] - 1, y + pad)

    # Semi-transparent background via addWeighted on a sub-region.
    roi = img[y1:y2 + 1, x1:x2 + 1]
    if roi.size > 0:
        overlay = roi.copy()
        overlay[:] = bg
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
        img[y1:y2 + 1, x1:x2 + 1] = roi

    cv2.putText(img, text, (x, y), font, scale, color, thickness,
                cv2.LINE_AA)


def _draw_crosshair(
    img: np.ndarray,
    cx: float,
    cy: float,
    size: int = 14,
    color: tuple[int, int, int] = _COLOR_CROSSHAIR,
) -> None:
    """Draw a crosshair + small circle at (cx, cy)."""
    h, w = img.shape[:2]
    px, py = int(round(cx)), int(round(cy))
    half = size // 2

    # Horizontal arm
    x0 = max(0, px - half)
    x1 = min(w - 1, px + half)
    cv2.line(img, (x0, py), (x1, py), color, 1, cv2.LINE_AA)

    # Vertical arm
    y0 = max(0, py - half)
    y1 = min(h - 1, py + half)
    cv2.line(img, (px, y0), (px, y1), color, 1, cv2.LINE_AA)

    # Small center circle
    cv2.circle(img, (px, py), 3, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_overlay(
    frame: np.ndarray,
    tracks: list[Track],
    target_id: int | None,
    camera: Camera,
    status: ControlStatus,
    state: str,
    fps: float | None = None,
    tracker_name: str | None = None,
    latency_ms: float | None = None,
) -> np.ndarray:
    """Draw all HUD annotations onto *frame* in place and return it.

    Parameters
    ----------
    frame:
        BGR image to annotate (modified in place).
    tracks:
        All currently alive tracks from the tracker.
    target_id:
        The track_id of the locked target, or None if no target is selected.
    camera:
        Live Camera instance; used for pan/tilt readout and frame_center.
    status:
        Latest ControlStatus from PTZController.update().
    state:
        Human-readable state string, e.g. "LOCKED", "SEARCHING".
    fps:
        Current frames-per-second estimate, or None to omit the FPS line.

    Returns
    -------
    frame (the same ndarray, now annotated).
    """
    img_h, img_w = frame.shape[:2]
    cx, cy = camera.frame_center

    # Build a quick lookup for the target track (may not exist among tracks).
    target_track: Track | None = None
    for trk in tracks:
        if trk.track_id == target_id:
            target_track = trk
            break

    # -----------------------------------------------------------------------
    # 1 & 2. Bounding boxes + labels (non-target first, target on top)
    # -----------------------------------------------------------------------
    for trk in tracks:
        is_target = (trk.track_id == target_id)
        x1, y1, x2, y2 = trk.bbox
        ix1, iy1 = _ipt(x1, y1)
        ix2, iy2 = _ipt(x2, y2)
        tcx, tcy = trk.center

        if is_target:
            box_color = _COLOR_TARGET
            thickness = 3
            label = f"TARGET ID{trk.track_id} {trk.confidence:.2f}"
            font_scale = _FS_TARGET
            # Filled center circle
            cv2.circle(frame, _ipt(tcx, tcy), 5, box_color, -1, cv2.LINE_AA)
        else:
            box_color = _COLOR_TRACK
            thickness = 1
            label = f"ID{trk.track_id} {trk.confidence:.2f}"
            font_scale = _FS_LABEL
            # Small dot at center
            cv2.circle(frame, _ipt(tcx, tcy), 3, box_color, -1, cv2.LINE_AA)

        # Clamp corners to image
        ix1c = int(np.clip(ix1, 0, img_w - 1))
        iy1c = int(np.clip(iy1, 0, img_h - 1))
        ix2c = int(np.clip(ix2, 0, img_w - 1))
        iy2c = int(np.clip(iy2, 0, img_h - 1))

        cv2.rectangle(frame, (ix1c, iy1c), (ix2c, iy2c), box_color,
                      thickness, cv2.LINE_AA)

        # Label: place just above the top-left corner
        label_x = int(np.clip(ix1, 0, img_w - 1))
        label_y = int(np.clip(iy1 - 4, 10, img_h - 1))
        _draw_label(frame, label, label_x, label_y,
                    _FONT, font_scale, box_color)

    # -----------------------------------------------------------------------
    # 3. Motion trails (polylines along .trajectory)
    # -----------------------------------------------------------------------
    for trk in tracks:
        traj = trk.trajectory
        if len(traj) < 2:
            continue

        is_target = (trk.track_id == target_id)
        trail_color = _COLOR_TARGET_TRAIL if is_target else _COLOR_TRAIL
        trail_thick = 2 if is_target else 1

        pts = np.array(
            [_clamp_pt(p[0], p[1], img_w, img_h) for p in traj],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=trail_color,
                      thickness=trail_thick, lineType=cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # 4. Center crosshair
    # -----------------------------------------------------------------------
    _draw_crosshair(frame, cx, cy, size=20, color=_COLOR_CROSSHAIR)

    # -----------------------------------------------------------------------
    # 5. Error vector (frame_center -> target center)
    # -----------------------------------------------------------------------
    if target_track is not None:
        tcx, tcy = target_track.center
        err_color = _COLOR_ERR_OK if status.centered else _COLOR_ERR_BAD
        p1 = _clamp_pt(cx, cy, img_w, img_h)
        p2 = _clamp_pt(tcx, tcy, img_w, img_h)
        cv2.arrowedLine(frame, p1, p2, err_color, 2,
                        cv2.LINE_AA, tipLength=0.15)

    # -----------------------------------------------------------------------
    # 6. Status HUD panel (top-left, semi-transparent)
    # -----------------------------------------------------------------------
    hud_lines: list[str] = []
    if tracker_name is not None:
        lat = f"  ({latency_ms:.2f} ms)" if latency_ms is not None else ""
        hud_lines.append(f"TRACKER: {tracker_name}{lat}")
    hud_lines += [
        f"STATE : {state}",
        f"TARGET: {target_id if target_id is not None else '--'}",
        f"TRACKS: {len(tracks)}",
        (f"PAN   : {math.degrees(camera.pan):+.1f}  "
         f"TILT: {math.degrees(camera.tilt):+.1f}"),
        (f"ERR   : {status.error_norm:.2f}  "
         f"{'CENTERED' if status.centered else 'ADJUSTING'}"),
    ]
    if fps is not None:
        hud_lines.append(f"FPS   : {fps:.1f}")

    # Measure the widest line to size the panel.
    pad = 8
    line_h = 18   # pixels per line (approx)
    max_tw = max(
        _text_size(ln, _FONT, _FS_HUD, _TH)[0] for ln in hud_lines
    )
    panel_w = max_tw + pad * 2
    panel_h = len(hud_lines) * line_h + pad * 2

    # Draw semi-transparent background rectangle.
    px1, py1 = 8, 8
    px2 = px1 + panel_w
    py2 = py1 + panel_h
    px2 = min(px2, img_w - 1)
    py2 = min(py2, img_h - 1)

    roi = frame[py1:py2, px1:px2]
    if roi.size > 0:
        bg_overlay = roi.copy()
        bg_overlay[:] = _COLOR_HUD_BG
        cv2.addWeighted(bg_overlay, 0.72, roi, 0.28, 0, roi)
        frame[py1:py2, px1:px2] = roi

    # Draw a thin border around the panel.
    cv2.rectangle(frame, (px1, py1), (px2, py2), (100, 100, 100), 1)

    # Draw each text line.
    for i, line in enumerate(hud_lines):
        tx = px1 + pad
        ty = py1 + pad + (i + 1) * line_h - 3
        ty = min(ty, img_h - 2)
        cv2.putText(frame, line, (tx, ty), _FONT, _FS_HUD,
                    _COLOR_WHITE, _TH, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # Controls hint bar at the bottom
    # -----------------------------------------------------------------------
    hint = "q quit | space pause | r re-target | n noise"
    hint_tw, hint_th = _text_size(hint, _FONT, _FS_HINT, _TH)
    hint_x = (img_w - hint_tw) // 2
    hint_y = img_h - 8

    # Background bar
    bar_y1 = max(0, hint_y - hint_th - 4)
    bar_y2 = img_h
    roi_hint = frame[bar_y1:bar_y2, 0:img_w]
    if roi_hint.size > 0:
        hint_bg = roi_hint.copy()
        hint_bg[:] = _COLOR_HUD_BG
        cv2.addWeighted(hint_bg, 0.60, roi_hint, 0.40, 0, roi_hint)
        frame[bar_y1:bar_y2, 0:img_w] = roi_hint

    cv2.putText(frame, hint, (hint_x, hint_y), _FONT, _FS_HINT,
                _COLOR_HINT, _TH, cv2.LINE_AA)

    return frame
