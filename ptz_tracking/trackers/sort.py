"""Classic SORT (Simple Online and Realtime Tracking) multi-object tracker.

Implements the single-stage, motion-only association policy from:

    Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.

Each frame:
  1. Predict all existing tracks forward via Kalman filter.
  2. Build an IoU cost matrix between predicted tracks and new detections.
  3. Solve the assignment (Hungarian) with a minimum-IoU gate.
  4. Update matched tracks, spawn new tracks for unmatched detections,
     and prune stale tracks that have not been observed recently.

No appearance features are used — association is purely geometric.
"""

from __future__ import annotations

import numpy as np

from .base import Track, KalmanBoxTracker, iou_batch, associate, distance_fallback


class SortTracker:
    """SORT: motion-only, single-stage IoU tracker."""

    def __init__(self, params) -> None:
        self.p = params
        self.tracks: list[KalmanBoxTracker] = []
        self._next_id: int = 1

    # ------------------------------------------------------------------
    def update(
        self,
        detections: list,
        frame: np.ndarray | None = None,
        camera=None,
    ) -> list[Track]:
        """Run one SORT cycle and return the current set of live tracks.

        Parameters
        ----------
        detections : list[Detection]
            New detections for the current frame.
        frame : ndarray | None
            The rendered frame (unused by SORT; accepted for API compat).
        camera :
            Camera object (unused by SORT; accepted for API compat).
        """

        # 1. Predict every existing track forward (advances KF, bumps
        #    age & time_since_update).
        for trk in self.tracks:
            trk.predict()

        # 2. Build IoU cost matrix and solve assignment.
        if self.tracks and detections:
            iou_matrix = iou_batch(
                [t.bbox for t in self.tracks],
                [d.bbox for d in detections],
            )
            cost_matrix = 1.0 - iou_matrix
            max_cost = 1.0 - self.p.trk_iou_threshold
            matches, unmatched_trks, unmatched_dets = associate(
                cost_matrix, max_cost,
            )
        else:
            matches = []
            unmatched_trks = list(range(len(self.tracks)))
            unmatched_dets = list(range(len(detections)))

        # 3. Update matched tracks with the assigned detection.
        for t_idx, d_idx in matches:
            det = detections[d_idx]
            self.tracks[t_idx].update(det.bbox, det.confidence)

        # 3b. Fast-motion recovery: re-link leftover tracks to leftover
        #     detections by velocity-scaled center distance (IoU may be 0 for
        #     fast / direction-changing targets).
        fb_trks = [i for i, t in enumerate(self.tracks) if t.time_since_update > 0]
        fb = distance_fallback(self.tracks, fb_trks, detections, unmatched_dets, self.p)
        for ti, di in fb:
            self.tracks[ti].update(detections[di].bbox, detections[di].confidence)
        fb_dets = {di for _, di in fb}
        unmatched_dets = [di for di in unmatched_dets if di not in fb_dets]

        # 4. Create new tracks for unmatched detections.
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

        # 5. Delete stale tracks (not seen for too long).
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.p.trk_max_age
        ]

        # 6. Return public Track objects for all surviving tracks.
        return [t.to_track() for t in self.tracks]
