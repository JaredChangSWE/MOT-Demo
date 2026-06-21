"""DeepSORT multi-object tracker (Kalman + appearance + matching cascade).

Implements the association policy from:

    Wojke et al., "Simple Online and Realtime Tracking with a Deep
    Association Metric", ICIP 2017.

Key differences from SORT:
  * An *appearance descriptor* (see honesty note below) lets the tracker
    re-identify people after brief occlusions.
  * A **matching cascade** gives priority to tracks that were seen recently,
    reducing ID switches when people overlap.
  * A fallback IoU stage catches unconfirmed tracks and recently-missed tracks
    that the cascade did not absorb.

Honesty note on appearance:
  The ``appearance_feature`` used here is a normalised HSV colour histogram —
  a lightweight, dependency-free **stand-in** for the deep CNN re-ID embedding
  that the original DeepSORT paper uses.  It captures coarse colour similarity
  but lacks the discriminative power of a learned descriptor.  The per-track
  feature gallery is simplified to a single EMA-smoothed vector maintained
  inside ``KalmanBoxTracker.update``.
"""

from __future__ import annotations

import numpy as np

from .base import (
    Track,
    KalmanBoxTracker,
    iou,
    iou_batch,
    associate,
    distance_fallback,
    appearance_feature,
    feature_cosine_distance,
)


class DeepSortTracker:
    """DeepSORT: Kalman + appearance matching cascade + IoU fallback."""

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
        """Run one DeepSORT cycle and return the current set of live tracks.

        Parameters
        ----------
        detections : list[Detection]
            New detections for the current frame.
        frame : ndarray | None
            The rendered frame.  When provided, appearance features (HSV
            colour histograms — *not* CNN embeddings; see module docstring)
            are extracted for each detection and used in the matching
            cascade.  When ``None``, appearance distance defaults to 1.0
            and matching falls back to IoU only.
        camera :
            Camera object (unused; accepted for API compatibility).
        """

        # -- 1. Extract appearance features for every detection -----------
        feats: list[np.ndarray | None] = [
            appearance_feature(frame, det.bbox) for det in detections
        ]

        # -- 2. Predict every existing track forward ----------------------
        for trk in self.tracks:
            trk.predict()

        # -- 3. Matching cascade (the DeepSORT signature) -----------------
        #    Prioritise tracks that were seen most recently.
        #    Level 1 = just missed one frame, level 2 = missed two, etc.
        #    Only *confirmed* tracks enter the cascade; unconfirmed ones go
        #    straight to the IoU fallback stage.

        # Index pools: which detections / tracks are still available.
        remaining_det_indices: list[int] = list(range(len(detections)))
        cascade_matched: list[tuple[int, int]] = []  # (track_idx, det_idx)
        cascade_consumed_tracks: set[int] = set()

        for level in range(1, self.p.trk_max_age + 1):
            # Tracks eligible at this cascade level.
            level_track_indices: list[int] = [
                i for i, t in enumerate(self.tracks)
                if t.confirmed and t.time_since_update == level
                and i not in cascade_consumed_tracks
            ]

            if not level_track_indices or not remaining_det_indices:
                continue

            # Build appearance cost matrix (rows=tracks, cols=detections).
            n_t = len(level_track_indices)
            n_d = len(remaining_det_indices)
            cost = np.zeros((n_t, n_d), dtype=float)

            for ri, ti in enumerate(level_track_indices):
                trk = self.tracks[ti]
                for ci, di in enumerate(remaining_det_indices):
                    # Appearance (cosine) cost.
                    app_dist = feature_cosine_distance(trk.feature, feats[di])

                    # Loose spatial gate: forbid pairs with negligible overlap
                    # so the appearance metric cannot link distant boxes.
                    iou_val = iou(trk.bbox, detections[di].bbox)
                    if iou_val < 0.1:
                        cost[ri, ci] = 1e5
                    else:
                        cost[ri, ci] = app_dist

            matches, _unmatched_t, unmatched_d = associate(
                cost, max_cost=self.p.trk_appearance_gate,
            )

            # Record matches (translate back to global indices).
            for m_t, m_d in matches:
                g_t = level_track_indices[m_t]
                g_d = remaining_det_indices[m_d]
                cascade_matched.append((g_t, g_d))
                cascade_consumed_tracks.add(g_t)

            # Shrink the remaining detection pool.
            matched_det_local = {m_d for _, m_d in matches}
            remaining_det_indices = [
                remaining_det_indices[ci]
                for ci in range(n_d)
                if ci not in matched_det_local
            ]

        # -- 4. IoU fallback stage ----------------------------------------
        #    Remaining unconfirmed tracks + cascade-level-1 confirmed tracks
        #    that were NOT consumed by the cascade are matched against the
        #    remaining detections using pure IoU.

        iou_track_indices: list[int] = [
            i for i, t in enumerate(self.tracks)
            if i not in cascade_consumed_tracks
            and (not t.confirmed or t.time_since_update == 1)
        ]

        if iou_track_indices and remaining_det_indices:
            iou_matrix = iou_batch(
                [self.tracks[i].bbox for i in iou_track_indices],
                [detections[j].bbox for j in remaining_det_indices],
            )
            cost_iou = 1.0 - iou_matrix
            max_cost_iou = 1.0 - self.p.trk_iou_threshold

            matches_iou, _, unmatched_dets_iou = associate(
                cost_iou, max_cost=max_cost_iou,
            )

            for m_t, m_d in matches_iou:
                g_t = iou_track_indices[m_t]
                g_d = remaining_det_indices[m_d]
                cascade_matched.append((g_t, g_d))

            # Update remaining detections after IoU stage.
            matched_det_local_iou = {m_d for _, m_d in matches_iou}
            remaining_det_indices = [
                remaining_det_indices[ci]
                for ci in range(len(remaining_det_indices))
                if ci not in matched_det_local_iou
            ]

        # -- 5. Update matched tracks -------------------------------------
        matched_track_set: set[int] = set()
        for t_idx, d_idx in cascade_matched:
            det = detections[d_idx]
            self.tracks[t_idx].update(det.bbox, det.confidence, feature=feats[d_idx])
            matched_track_set.add(t_idx)

        # -- 5b. Fast-motion recovery: velocity-scaled distance fallback for
        #        leftover tracks vs leftover detections (IoU/appearance may both
        #        fail for fast / turning targets).
        fb_trks = [i for i, t in enumerate(self.tracks) if t.time_since_update > 0]
        fb = distance_fallback(self.tracks, fb_trks, detections,
                               remaining_det_indices, self.p)
        for ti, di in fb:
            self.tracks[ti].update(detections[di].bbox, detections[di].confidence,
                                   feature=feats[di])
        fb_dets = {di for _, di in fb}
        remaining_det_indices = [di for di in remaining_det_indices if di not in fb_dets]

        # -- 6. Create new tracks for unmatched detections ----------------
        for d_idx in remaining_det_indices:
            det = detections[d_idx]
            new_trk = KalmanBoxTracker(
                bbox=det.bbox,
                confidence=det.confidence,
                track_id=self._next_id,
                params=self.p,
                feature=feats[d_idx],
            )
            self._next_id += 1
            self.tracks.append(new_trk)

        # -- 7. Delete stale tracks (not seen for too long) ---------------
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.p.trk_max_age
        ]

        # -- 8. Return public Track objects -------------------------------
        return [t.to_track() for t in self.tracks]
