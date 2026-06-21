"""Render a 3D scene (ground plane + walking people) into an OpenCV BGR image.

This module produces the raw camera frame as seen through the PTZ camera.
It does NOT draw any detection boxes, track IDs, or HUD elements — those are
added by a separate overlay module downstream.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from . import config
from .camera import Camera, BBox, visible_people
from .world import World, Person

# ---------------------------------------------------------------------------
# Colour palette for the scene
# ---------------------------------------------------------------------------
_BG_WALL_COLOR = (45, 38, 35)       # dark brownish-grey wall / ceiling fill
_GROUND_FILL_COLOR = (55, 52, 48)   # slightly lighter dark floor fill
_GRID_COLOR = (90, 85, 80)          # dim grey grid lines
_SHADOW_COLOR = (30, 28, 26)        # very dark ellipse shadow under feet
_EXCLUSION_COLOR = (70, 90, 220)    # reddish keep-out ring around the camera

# How many sample points to use when drawing a projected world line.
_LINE_SAMPLES = 40
# Coordinate clamp to avoid extreme values slowing cv2.polylines.
_COORD_CLAMP = 5000


def _clamp_coord(v: float, lo: float = -_COORD_CLAMP, hi: float = _COORD_CLAMP) -> float:
    return max(lo, min(hi, v))


def _project_segment(
    camera: Camera,
    p0: np.ndarray,
    p1: np.ndarray,
    n_samples: int = _LINE_SAMPLES,
) -> list[tuple[int, int] | None]:
    """Sample a 3D segment densely and return projected (u, v) pairs that are
    valid (not behind near plane) and not wildly outside the image frame.
    Invalid samples become None so callers can break the polyline there."""
    pts: list[tuple[int, int] | None] = []
    for i in range(n_samples + 1):
        t = i / n_samples
        world_pt = p0 + t * (p1 - p0)
        result = camera.project_point(world_pt)
        if result is None:
            pts.append(None)
            continue
        u, v, _ = result
        # Keep points that are within a generous margin of the frame.
        if (u < -_COORD_CLAMP or u > camera.image_w + _COORD_CLAMP
                or v < -_COORD_CLAMP or v > camera.image_h + _COORD_CLAMP):
            pts.append(None)
            continue
        pts.append((int(round(_clamp_coord(u))), int(round(_clamp_coord(v)))))
    return pts


def _draw_polyline_skipping_nones(
    img: np.ndarray,
    pts: list[tuple[int, int] | None],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw connected segments, breaking whenever a None appears in pts."""
    segment: list[tuple[int, int]] = []
    for pt in pts:
        if pt is None:
            if len(segment) >= 2:
                arr = np.array(segment, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [arr], False, color, thickness, cv2.LINE_AA)
            segment = []
        else:
            segment.append(pt)
    if len(segment) >= 2:
        arr = np.array(segment, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [arr], False, color, thickness, cv2.LINE_AA)


def _draw_exclusion_zone(img: np.ndarray, camera: Camera) -> None:
    """Draw the camera keep-out disc as a ring on the ground (z=0)."""
    r = camera.p.exclusion_radius
    if r <= 0:
        return
    cam_xy = camera.pos[:2]
    n = 72
    pts: list[tuple[int, int] | None] = []
    for i in range(n + 1):
        a = 2.0 * math.pi * i / n
        wp = np.array([cam_xy[0] + r * math.cos(a), cam_xy[1] + r * math.sin(a), 0.0])
        result = camera.project_point(wp)
        if result is None:
            pts.append(None)
            continue
        u, v, _ = result
        if abs(u) > 2 * _COORD_CLAMP or abs(v) > 2 * _COORD_CLAMP:
            pts.append(None)
            continue
        pts.append((int(round(_clamp_coord(u))), int(round(_clamp_coord(v)))))
    _draw_polyline_skipping_nones(img, pts, _EXCLUSION_COLOR, thickness=2)


def _draw_ground_grid(img: np.ndarray, camera: Camera) -> None:
    """Draw a 1-metre grid on the z=0 ground plane with correct perspective."""
    half = camera.p.world_half
    steps = int(2 * half)  # one line per metre

    # x-constant lines (run along the Y axis)
    for ix in range(steps + 1):
        x = -half + ix
        p0 = np.array([x, -half, 0.0])
        p1 = np.array([x,  half, 0.0])
        pts = _project_segment(camera, p0, p1)
        _draw_polyline_skipping_nones(img, pts, _GRID_COLOR)

    # y-constant lines (run along the X axis)
    for iy in range(steps + 1):
        y = -half + iy
        p0 = np.array([-half, y, 0.0])
        p1 = np.array([ half, y, 0.0])
        pts = _project_segment(camera, p0, p1)
        _draw_polyline_skipping_nones(img, pts, _GRID_COLOR)


def _horizon_v(camera: Camera) -> int:
    """Estimate the image row of the horizon line (where z=0 plane's infinity
    projects to).  We project a distant ground point and use that v-coordinate
    as an approximation, clamped to the image bounds.
    """
    far_pt = np.array([0.0, 1e6, 0.0])
    result = camera.project_point(far_pt)
    if result is None:
        return camera.image_h // 2
    _, v, _ = result
    return int(np.clip(round(v), 0, camera.image_h))


def _fill_background(img: np.ndarray, camera: Camera) -> None:
    """Fill the image with wall/ceiling colour above the horizon and a slightly
    different floor fill below it, giving a strong floor-vs-wall read before
    the grid is drawn.
    """
    horizon = _horizon_v(camera)
    # Wall / ceiling portion (above horizon)
    img[:horizon, :] = _BG_WALL_COLOR
    # Floor portion (below horizon)
    img[horizon:, :] = _GROUND_FILL_COLOR


# ---------------------------------------------------------------------------
# Person drawing helpers
# ---------------------------------------------------------------------------

def _safe_int(v: float) -> int:
    return int(round(max(-_COORD_CLAMP, min(_COORD_CLAMP, v))))


def _darker(color: tuple[int, int, int], factor: float = 0.45) -> tuple[int, int, int]:
    return (
        int(color[0] * factor),
        int(color[1] * factor),
        int(color[2] * factor),
    )


def _draw_person(
    img: np.ndarray,
    person: Person,
    bbox: BBox,
    camera: Camera,
) -> None:
    """Draw a stylised standing figure that fills the projected bounding box.

    Anatomy (top to bottom inside bbox):
      - head circle         (~12 % of bbox height)
      - torso rectangle     (rounded shoulders, ~46 % of width)
      - two legs            (down to feet)

    All sizes are derived from the bbox so near/far scale naturally.
    """
    x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
    bw, bh = bbox.width, bbox.height

    # Skip truly degenerate bboxes.
    if bw < 3 or bh < 3:
        return

    cx = 0.5 * (x1 + x2)
    color = person.color
    outline = _darker(color, 0.35)
    skin = (100, 160, 210)  # neutral warm skin-ish tone (BGR)

    # ---- proportions ----
    head_r = max(2.0, bh * 0.12)         # head radius as fraction of bbox height
    head_cy = y1 + head_r + bh * 0.01    # a tiny margin from top

    torso_top = head_cy + head_r + bh * 0.02
    torso_bot = y1 + bh * 0.58
    torso_hw = bw * 0.46                  # half-width of torso

    legs_top = torso_bot
    leg_hw = bw * 0.22
    leg_gap = bw * 0.04                   # small gap between legs

    # ---- ground shadow (semi-transparent ellipse below feet) ----
    shadow_cx = _safe_int(cx)
    shadow_cy = _safe_int(y2 + bh * 0.03)
    shadow_rx = max(1, _safe_int(bw * 0.48))
    shadow_ry = max(1, _safe_int(bh * 0.04))
    overlay = img.copy()
    cv2.ellipse(
        overlay,
        (shadow_cx, shadow_cy),
        (shadow_rx, shadow_ry),
        0, 0, 360,
        _SHADOW_COLOR, -1, cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # ---- torso (filled rect + rounded shoulder cap) ----
    tx1 = _safe_int(cx - torso_hw)
    tx2 = _safe_int(cx + torso_hw)
    ty1 = _safe_int(torso_top)
    ty2 = _safe_int(torso_bot)
    cv2.rectangle(img, (tx1, ty1), (tx2, ty2), color, -1, cv2.LINE_AA)
    shoulder_r = max(1, _safe_int(torso_hw * 0.8))
    cv2.ellipse(img, (_safe_int(cx), ty1), (shoulder_r, shoulder_r),
                0, 180, 360, color, -1, cv2.LINE_AA)

    # ---- legs ----
    # Left leg
    ll_x1 = _safe_int(cx - leg_hw * 2 - leg_gap)
    ll_x2 = _safe_int(cx - leg_gap)
    ll_y1 = _safe_int(legs_top)
    ll_y2 = _safe_int(y2)
    cv2.rectangle(img, (ll_x1, ll_y1), (ll_x2, ll_y2), color, -1, cv2.LINE_AA)
    # Right leg
    rl_x1 = _safe_int(cx + leg_gap)
    rl_x2 = _safe_int(cx + leg_hw * 2 + leg_gap)
    rl_y1 = _safe_int(legs_top)
    rl_y2 = _safe_int(y2)
    cv2.rectangle(img, (rl_x1, rl_y1), (rl_x2, rl_y2), color, -1, cv2.LINE_AA)

    # ---- head ----
    hcx = _safe_int(cx)
    hcy = _safe_int(head_cy)
    hr = max(2, _safe_int(head_r))
    cv2.circle(img, (hcx, hcy), hr, skin, -1, cv2.LINE_AA)

    # ---- outlines for contrast ----
    ot = max(1, int(round(bw * 0.04)))  # outline thickness scales with size
    cv2.rectangle(img, (tx1, ty1), (tx2, ty2), outline, ot, cv2.LINE_AA)
    cv2.rectangle(img, (ll_x1, ll_y1), (ll_x2, ll_y2), outline, ot, cv2.LINE_AA)
    cv2.rectangle(img, (rl_x1, rl_y1), (rl_x2, rl_y2), outline, ot, cv2.LINE_AA)
    cv2.circle(img, (hcx, hcy), hr, outline, ot, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render(camera: Camera, world: World) -> np.ndarray:
    """Render the camera's view of the 3D scene into an HxWx3 uint8 BGR image.

    Draws (in order):
      1. Background fill (wall above horizon, floor below).
      2. Perspective ground grid.
      3. People (far-to-near, painter's order) as simple stylised figures.

    Returns an image of shape (camera.image_h, camera.image_w, 3).
    No HUD, no bounding boxes, no detection annotations are drawn here.
    """
    img = np.zeros((camera.image_h, camera.image_w, 3), dtype=np.uint8)

    # 1. Background
    _fill_background(img, camera)

    # 2. Ground grid + camera keep-out ring
    _draw_ground_grid(img, camera)
    _draw_exclusion_zone(img, camera)

    # 3. People (far->near so near people occlude far ones)
    for person, bbox in visible_people(camera, world):
        _draw_person(img, person, bbox, camera)

    return img
