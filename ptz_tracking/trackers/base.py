"""Shared building blocks for the pluggable trackers.

Every tracker (SORT / DeepSORT / ByteTrack / OC-SORT / BoT-SORT) consumes the
same `Detection` list and returns the same public `Track` objects, so the rest
of the demo (overlay, main loop, target lock) never changes.

This module provides the pieces the trackers share:
  * `Track`               - the public per-object output (what overlay reads).
  * `KalmanBoxTracker`    - a constant-velocity Kalman filter on a box, with
                            track bookkeeping (id, hits, age, trajectory,
                            velocity, appearance feature). Most trackers just
                            implement an association policy on top of this.
  * association helpers   - `iou`, `iou_batch`, `linear_assignment`.
  * appearance helpers    - `appearance_feature`, `feature_cosine_distance`.

Appearance note: DeepSORT / BoT-SORT normally use a deep CNN re-ID embedding.
To stay dependency-light, `appearance_feature` returns an HSV colour histogram
of the box crop. It is a faithful *structural* stand-in for the embedding, not a
learned descriptor — labelled as such wherever it is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

try:  # optimal assignment if available; otherwise greedy fallback
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Public track object (the contract consumed by overlay.py and main.py)
# ---------------------------------------------------------------------------

@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]      # (x1, y1, x2, y2) pixels
    confidence: float
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    hits: int = 1
    time_since_update: int = 0
    confirmed: bool = False
    velocity: tuple[float, float] = (0.0, 0.0)   # center px/frame

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


# ---------------------------------------------------------------------------
# Geometry / association helpers
# ---------------------------------------------------------------------------

def iou(a: tuple[float, float, float, float],
        b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def iou_batch(boxes_a: list, boxes_b: list) -> np.ndarray:
    """IoU matrix of shape (len(a), len(b))."""
    m = np.zeros((len(boxes_a), len(boxes_b)), dtype=float)
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            m[i, j] = iou(a, b)
    return m


def linear_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment. Returns (row, col) pairs. Uses the Hungarian
    algorithm when scipy is present, otherwise a greedy nearest fallback."""
    if cost_matrix.size == 0:
        return []
    if _HAS_SCIPY:
        rows, cols = linear_sum_assignment(cost_matrix)
        return list(zip(rows.tolist(), cols.tolist()))
    # Greedy fallback: take lowest-cost cells first, one per row/col.
    order = np.dstack(np.unravel_index(
        np.argsort(cost_matrix, axis=None), cost_matrix.shape))[0]
    used_r: set[int] = set()
    used_c: set[int] = set()
    out: list[tuple[int, int]] = []
    for r, c in order:
        r, c = int(r), int(c)
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        out.append((r, c))
    return out


def distance_fallback(tracks: list, track_idx: list[int],
                      det_list: list, det_idx: list[int], params
                      ) -> list[tuple[int, int]]:
    """Velocity-scaled center-distance association for leftovers.

    The primary (IoU / appearance) stage fails for FAST or direction-changing
    targets: the Kalman-predicted box no longer overlaps the new detection so
    IoU drops to 0 and the track would be lost (then a new ID spawns). This
    second pass re-links such tracks by center distance, with a gate that grows
    with the track's speed, comparing the detection to BOTH the predicted center
    and the last observed center (the latter wins on a sudden reversal).

    Shared by all trackers. Returns (track_index_into_tracks, det_index_into_det_list).
    """
    matches: list[tuple[int, int]] = []
    if not track_idx or not det_idx:
        return matches
    cands: list[tuple[float, int, int]] = []
    for ti in track_idx:
        t = tracks[ti]
        x1, y1, x2, y2 = t.bbox
        size = max(x2 - x1, y2 - y1, 1.0)
        speed = math.hypot(*t.velocity)
        tsu = min(max(1, t.time_since_update), 3)   # cap so long-lost tracks don't over-reach
        gate = params.trk_match_dist_factor * size + 2.0 * speed * tsu
        pcx, pcy = t.center
        lo = getattr(t, "last_observation", None)
        if lo is not None:
            locx, locy = 0.5 * (lo[0] + lo[2]), 0.5 * (lo[1] + lo[3])
        else:
            locx, locy = pcx, pcy
        for di in det_idx:
            dcx, dcy = det_list[di].center
            dist = min(math.hypot(pcx - dcx, pcy - dcy),
                       math.hypot(locx - dcx, locy - dcy))
            if dist <= gate:
                cands.append((dist, ti, di))
    cands.sort()  # nearest first
    ut, ud = set(track_idx), set(det_idx)
    for _dist, ti, di in cands:
        if ti in ut and di in ud:
            matches.append((ti, di))
            ut.discard(ti)
            ud.discard(di)
    return matches


def associate(cost_matrix: np.ndarray, max_cost: float
              ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Assign rows<->cols minimizing cost, rejecting pairs with cost > max_cost.

    Returns (matches, unmatched_rows, unmatched_cols)."""
    n_r, n_c = cost_matrix.shape
    if n_r == 0 or n_c == 0:
        return [], list(range(n_r)), list(range(n_c))
    matches: list[tuple[int, int]] = []
    matched_r: set[int] = set()
    matched_c: set[int] = set()
    for r, c in linear_assignment(cost_matrix):
        if cost_matrix[r, c] <= max_cost:
            matches.append((r, c))
            matched_r.add(r)
            matched_c.add(c)
    unmatched_r = [r for r in range(n_r) if r not in matched_r]
    unmatched_c = [c for c in range(n_c) if c not in matched_c]
    return matches, unmatched_r, unmatched_c


# ---------------------------------------------------------------------------
# Appearance helpers (colour-histogram stand-in for a CNN re-ID embedding)
# ---------------------------------------------------------------------------

def appearance_feature(frame: np.ndarray | None,
                       bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    """Normalized HSV colour histogram of the box crop (a lightweight, dependency
    -free stand-in for a deep re-ID embedding). Returns None if no crop."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2])); y2 = min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def feature_cosine_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Cosine distance in [0, 1]; 1.0 (max) if either feature is missing."""
    if a is None or b is None:
        return 1.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Kalman box tracker (constant-velocity), SORT-style state
# ---------------------------------------------------------------------------

def _bbox_to_z(bbox: tuple[float, float, float, float]) -> np.ndarray:
    """(x1,y1,x2,y2) -> measurement [cx, cy, s(area), r(aspect)]."""
    x1, y1, x2, y2 = bbox
    w = max(1e-3, x2 - x1)
    h = max(1e-3, y2 - y1)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w * h, w / h], dtype=float)


def _x_to_bbox(x: np.ndarray) -> tuple[float, float, float, float]:
    """state [cx, cy, s, r, ...] -> (x1,y1,x2,y2)."""
    cx, cy, s, r = x[0], x[1], max(1e-3, x[2]), max(1e-3, x[3])
    w = float(np.sqrt(s * r))
    h = float(s / w) if w > 0 else 0.0
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class KalmanBoxTracker:
    """A single track backed by a 7-state constant-velocity Kalman filter
    (state = [cx, cy, s, r, vcx, vcy, vs]). Carries all the bookkeeping a
    tracker needs; trackers add only an association policy on top.

    `appearance` (when frames are available) is kept as an EMA-smoothed feature
    so DeepSORT / BoT-SORT can match on it.
    """

    def __init__(self, bbox, confidence, track_id, params,
                 feature: np.ndarray | None = None):
        self.id = track_id
        self.p = params
        # --- Kalman matrices ---
        self.dim_x, self.dim_z = 7, 4
        self._F = np.eye(7)
        for i in range(3):
            self._F[i, i + 4] = 1.0
        self._H = np.zeros((4, 7))
        self._H[:4, :4] = np.eye(4)
        self._P = np.eye(7)
        self._P[4:, 4:] *= 1000.0   # high uncertainty on initial velocities
        self._P *= 10.0
        self._Q = np.eye(7)
        self._Q[4:, 4:] *= 0.01
        self._Q[-1, -1] *= 0.01
        self._R = np.eye(4)
        self._R[2:, 2:] *= 10.0
        self._x = np.zeros(7)
        self._x[:4] = _bbox_to_z(bbox)

        # --- bookkeeping ---
        self.confidence = float(confidence)
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.confirmed = (params.trk_min_hits <= 1)
        self.trajectory: list[tuple[float, float]] = [self.center]
        self.feature = feature
        self.last_observation: tuple[float, float, float, float] | None = bbox
        self._prev_center = self.center

    # -- state access ------------------------------------------------------
    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return _x_to_bbox(self._x)

    @property
    def center(self) -> tuple[float, float]:
        b = _x_to_bbox(self._x)
        return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self._x[4]), float(self._x[5]))

    # -- Kalman steps ------------------------------------------------------
    def predict(self) -> tuple[float, float, float, float]:
        # Guard against negative area blowing up.
        if self._x[2] + self._x[6] <= 0:
            self._x[6] = 0.0
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox, confidence=None, feature: np.ndarray | None = None) -> None:
        """Kalman correction with a new detection box (+ optional appearance)."""
        prev = self.center
        z = _bbox_to_z(bbox)
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(7) - K @ self._H) @ self._P

        self.hits += 1
        self.time_since_update = 0
        self.last_observation = bbox
        if confidence is not None:
            self.confidence = float(confidence)
        if feature is not None:
            self.feature = self._ema_feature(self.feature, feature)
        if self.hits >= self.p.trk_min_hits:
            self.confirmed = True
        self._append_trajectory()
        self._prev_center = prev

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _ema_feature(old, new, alpha: float = 0.9):
        if old is None:
            return new
        if new is None:
            return old
        f = alpha * old + (1.0 - alpha) * new
        n = np.linalg.norm(f)
        return f / n if n > 0 else f

    def _append_trajectory(self) -> None:
        self.trajectory.append(self.center)
        if len(self.trajectory) > self.p.trk_trail_len:
            self.trajectory = self.trajectory[-self.p.trk_trail_len:]

    def to_track(self) -> Track:
        return Track(
            track_id=self.id,
            bbox=self.bbox,
            confidence=self.confidence,
            trajectory=self.trajectory,
            hits=self.hits,
            time_since_update=self.time_since_update,
            confirmed=self.confirmed,
            velocity=self.velocity,
        )
