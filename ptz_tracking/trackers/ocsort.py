"""OC-SORT (Observation-Centric SORT) multi-object tracker.

Implements the key ideas from:

    Cao et al., "Observation-Centric SORT: Rethinking SORT for Robust
    Multi-Object Tracking", CVPR 2023.

OC-SORT is a motion-only tracker (no appearance features) that improves on
vanilla SORT through three observation-centric mechanisms:

  * **OCM** (Observation-Centric Momentum) — a direction-consistency cost
    added to the IoU cost so that the assignment penalises physically
    implausible direction changes.
  * **OCR** (Observation-Centric Recovery) — a second association stage that
    matches still-unmatched tracks against remaining detections using IoU
    computed from the track's *last real observation* rather than the Kalman
    prediction (which may have drifted during occlusion).
  * **ORU** (Observation-centric Re-Update) — in the full paper the Kalman
    filter is back-filled along a virtual trajectory between the last
    observation and the recovered detection. Here we apply a single
    ``update()`` as a faithful simplification.

Each frame:
  1. Predict all tracks forward via Kalman filter.
  2. **Stage 1** — OCM-augmented IoU association between predicted tracks
     and detections.
  3. **Stage 2** — OCR recovery: unmatched tracks re-associate against
     remaining detections using ``last_observation`` IoU.
  4. Spawn new tracks for remaining unmatched detections.
  5. Prune tracks that have not been observed for ``trk_max_age`` frames.
"""

from __future__ import annotations

import numpy as np

from .base import Track, KalmanBoxTracker, iou, iou_batch, associate, distance_fallback


# ---------------------------------------------------------------------------
# Angle-cost helper (OCM)
# ---------------------------------------------------------------------------

def _angle_cost(
    direction_a: tuple[float, float],
    direction_b: tuple[float, float],
) -> float:
    """Normalised angular inconsistency between two 2-D direction vectors.

    Returns ``(1 - cos(Δθ)) / 2`` which maps to:
      - 0.0 when the vectors point in the same direction,
      - 0.5 when perpendicular,
      - 1.0 when opposite.

    If either vector has zero length (undefined direction), returns 0.0 so that
    the OCM term does not penalise brand-new or stationary tracks.
    """
    ax, ay = direction_a
    bx, by = direction_b
    len_a = np.hypot(ax, ay)
    len_b = np.hypot(bx, by)
    if len_a < 1e-9 or len_b < 1e-9:
        return 0.0
    cos_theta = (ax * bx + ay * by) / (len_a * len_b)
    # Clamp for floating-point safety.
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return (1.0 - cos_theta) / 2.0


def _box_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Centre point of an (x1, y1, x2, y2) bounding box."""
    return (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))


# ---------------------------------------------------------------------------
# OC-SORT tracker
# ---------------------------------------------------------------------------

class OcSortTracker:
    """OC-SORT: motion-only, two-stage observation-centric tracker."""

    def __init__(self, params) -> None:
        self.p = params
        self.tracks: list[KalmanBoxTracker] = []
        self._next_id: int = 1

    # ------------------------------------------------------------------
    def _track_motion_direction(self, trk: KalmanBoxTracker) -> tuple[float, float]:
        """Recent motion direction of *trk*, derived from its trajectory.

        Uses the last two trajectory points if available; otherwise falls back
        to the Kalman velocity estimate.  Returns (dx, dy) — may be (0, 0) for
        a brand-new / stationary track.
        """
        if len(trk.trajectory) >= 2:
            x0, y0 = trk.trajectory[-2]
            x1, y1 = trk.trajectory[-1]
            return (x1 - x0, y1 - y0)
        vx, vy = trk.velocity
        return (vx, vy)

    # ------------------------------------------------------------------
    def _build_ocm_cost(
        self,
        tracks: list[KalmanBoxTracker],
        detections: list,
    ) -> np.ndarray:
        """OCM-augmented cost matrix: IoU cost + trk_ocm_weight * angle_cost.

        Any pair whose raw IoU < ``trk_iou_threshold`` gets its cost set to a
        prohibitively large value (hard gate) so it will never be matched.
        """
        n_t = len(tracks)
        n_d = len(detections)
        iou_matrix = iou_batch(
            [t.bbox for t in tracks],
            [d.bbox for d in detections],
        )
        cost = np.full((n_t, n_d), 1e5, dtype=float)

        for i, trk in enumerate(tracks):
            motion_dir = self._track_motion_direction(trk)
            last_obs_center = (
                _box_center(trk.last_observation)
                if trk.last_observation is not None
                else trk.center
            )
            for j, det in enumerate(detections):
                # Hard IoU gate: reject pair outright.
                if iou_matrix[i, j] < self.p.trk_iou_threshold:
                    continue
                # Direction from last observation to candidate detection.
                det_center = det.center
                candidate_dir = (
                    det_center[0] - last_obs_center[0],
                    det_center[1] - last_obs_center[1],
                )
                ac = _angle_cost(motion_dir, candidate_dir)
                cost[i, j] = (1.0 - iou_matrix[i, j]) + self.p.trk_ocm_weight * ac

        return cost

    # ------------------------------------------------------------------
    def update(
        self,
        detections: list,
        frame: np.ndarray | None = None,
        camera=None,
    ) -> list[Track]:
        """Run one OC-SORT cycle and return the current set of live tracks.

        Parameters
        ----------
        detections : list[Detection]
            New detections for the current frame.
        frame : ndarray | None
            The rendered frame (unused by OC-SORT; accepted for API compat).
        camera :
            Camera object (unused by OC-SORT; accepted for API compat).
        """

        # 1. Predict every existing track forward.
        for trk in self.tracks:
            trk.predict()

        if not self.tracks or not detections:
            matches_s1: list[tuple[int, int]] = []
            unmatched_trks = list(range(len(self.tracks)))
            unmatched_dets = list(range(len(detections)))
        else:
            # ---------------------------------------------------------------
            # STAGE 1 — OCM-augmented IoU association.
            # ---------------------------------------------------------------
            cost_s1 = self._build_ocm_cost(self.tracks, detections)
            max_cost_s1 = (1.0 - self.p.trk_iou_threshold) + self.p.trk_ocm_weight
            matches_s1, unmatched_trks, unmatched_dets = associate(
                cost_s1, max_cost_s1,
            )

        # Update matched tracks from stage 1.
        for t_idx, d_idx in matches_s1:
            det = detections[d_idx]
            self.tracks[t_idx].update(det.bbox, det.confidence)

        # ---------------------------------------------------------------
        # STAGE 2 — OCR (Observation-Centric Recovery).
        #
        # For tracks that are STILL unmatched, re-attempt association
        # using IoU between each track's *last real observation* (not the
        # Kalman prediction) and the remaining detections. This exploits
        # the insight that a track's last seen position is often a better
        # cue than a drifted KF state after one or more missed frames.
        # ---------------------------------------------------------------
        if unmatched_trks and unmatched_dets:
            remaining_trks = [self.tracks[i] for i in unmatched_trks]
            remaining_dets = [detections[i] for i in unmatched_dets]

            # Build last-observation boxes, falling back to predicted bbox
            # when a track has never been observed (should not happen in
            # practice but handles degenerate init).
            last_obs_boxes = [
                t.last_observation if t.last_observation is not None else t.bbox
                for t in remaining_trks
            ]
            iou_matrix_s2 = iou_batch(
                last_obs_boxes,
                [d.bbox for d in remaining_dets],
            )
            cost_s2 = 1.0 - iou_matrix_s2
            max_cost_s2 = 1.0 - self.p.trk_iou_threshold

            matches_s2, still_unmatched_trks, still_unmatched_dets = associate(
                cost_s2, max_cost_s2,
            )

            # ORU note: the full OC-SORT paper back-fills the Kalman filter
            # along a virtual trajectory between the last observation and the
            # re-found detection (Observation-centric Re-Update). Here we
            # apply a single .update() on re-match as a simplification.
            for rt_idx, rd_idx in matches_s2:
                det = remaining_dets[rd_idx]
                remaining_trks[rt_idx].update(det.bbox, det.confidence)

            # Remap local indices back to the original lists.
            unmatched_dets = [unmatched_dets[i] for i in still_unmatched_dets]
        # (If one or both lists were empty, unmatched_dets stays as-is.)

        # 2c. Fast-motion recovery: velocity-scaled distance fallback for any
        #     tracks the OCM/OCR stages missed (large jumps / direction flips).
        fb_trks = [i for i, t in enumerate(self.tracks) if t.time_since_update > 0]
        fb = distance_fallback(self.tracks, fb_trks, detections, unmatched_dets, self.p)
        for ti, di in fb:
            self.tracks[ti].update(detections[di].bbox, detections[di].confidence)
        fb_dets = {di for _, di in fb}
        unmatched_dets = [di for di in unmatched_dets if di not in fb_dets]

        # 3. Spawn new tracks for remaining unmatched detections.
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_trk = KalmanBoxTracker(
                bbox=det.bbox,
                confidence=det.confidence,
                track_id=self._next_id,
                params=self.p,
            )
            self._next_id += 1
            self.tracks.append(new_trk)

        # 4. Delete stale tracks (not seen for too long).
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.p.trk_max_age
        ]

        # 5. Return public Track objects for all surviving tracks.
        return [t.to_track() for t in self.tracks]
