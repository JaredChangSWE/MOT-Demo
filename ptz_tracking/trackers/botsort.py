"""BoT-SORT tracker — ByteTrack two-stage association with appearance fusion
and camera-motion compensation (CMC).

BoT-SORT extends ByteTrack in two ways:

  1. **Appearance fusion** — stage-1 association blends a cosine-distance
     appearance cost with the IoU cost so tracks survive longer occlusions
     where spatial overlap alone would fail.

  2. **Camera-motion compensation (CMC)** — before computing IoU costs the
     predicted boxes are shifted by the ego-motion of the camera, so that a
     pan/tilt move does not look like every track suddenly drifting off-screen.

Honesty note on CMC: the real BoT-SORT paper estimates global motion from
image features (ORB keypoints + RANSAC homography).  Here we use the *known*
camera pan/tilt angles, which gives exact ego-motion for this simulator and is
a legitimate simplification.

Honesty note on appearance: ``appearance_feature`` returns an HSV colour
histogram, not a learned CNN re-ID embedding.  It is a faithful structural
stand-in — the fusion maths are identical.
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


class BotSortTracker:
    """Multi-object tracker using BoT-SORT association policy.

    Parameters
    ----------
    params : Params
        Shared parameter object.  Relevant fields:

        * ``trk_high_thresh``       — confidence split between HIGH / LOW.
        * ``trk_low_thresh``        — floor; detections below this are discarded.
        * ``trk_new_thresh``        — minimum confidence to spawn a new track.
        * ``trk_iou_threshold``     — IoU gate for stage-1 association.
        * ``trk_max_age``           — frames before an unmatched track is deleted.
        * ``trk_min_hits``          — consecutive hits to confirm a track.
        * ``trk_appearance_weight`` — blend weight for appearance vs IoU cost.
        * ``trk_appearance_gate``   — max cosine distance for appearance match.
    """

    def __init__(self, params) -> None:
        self.p = params
        self.tracks: list[KalmanBoxTracker] = []
        self._next_id: int = 1
        self._prev_pan: float | None = None
        self._prev_tilt: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: list,
        frame: np.ndarray | None = None,
        camera=None,
    ) -> list[Track]:
        """Run one frame of BoT-SORT association.

        Parameters
        ----------
        detections : list[Detection]
            Raw detections for this frame (each has ``.bbox``,
            ``.confidence``).
        frame : ndarray | None
            Current image, used for appearance feature extraction.
        camera : Camera | None
            Camera state with ``.pan``, ``.tilt``, ``.focal`` for CMC.

        Returns
        -------
        list[Track]
            Public track objects for this frame.
        """

        # -- 0. Appearance features for every detection ----------------------
        features: list[np.ndarray | None] = []
        for det in detections:
            features.append(appearance_feature(frame, det.bbox))

        # -- 1. Predict every track (once per frame) -------------------------
        for trk in self.tracks:
            trk.predict()

        # -- 2. Camera-motion compensation (CMC) -----------------------------
        #  Compute pixel shift of the SCENE due to camera rotation since the
        #  previous frame.  Small-angle approximation: the scene shifts
        #  opposite to the camera's angular motion, scaled by focal length.
        #
        #  Honesty: real BoT-SORT uses ORB+RANSAC on the image; we use the
        #  known camera angles — exact ego-motion for this simulator.
        dx: float = 0.0
        dy: float = 0.0
        if camera is not None and self._prev_pan is not None and self._prev_tilt is not None:
            dpan = camera.pan - self._prev_pan
            dtilt = camera.tilt - self._prev_tilt
            dx = -dpan * camera.focal
            dy = dtilt * camera.focal

        # Build shifted predicted boxes for IoU computation (do NOT mutate KF).
        shifted_boxes: list[tuple[float, float, float, float]] = []
        for trk in self.tracks:
            x1, y1, x2, y2 = trk.bbox
            shifted_boxes.append((x1 + dx, y1 + dy, x2 + dx, y2 + dy))

        # -- 3. Split detections by confidence --------------------------------
        high_dets: list[int] = []       # indices into `detections`
        low_dets: list[int] = []
        for i, det in enumerate(detections):
            if det.confidence >= self.p.trk_high_thresh:
                high_dets.append(i)
            elif det.confidence >= self.p.trk_low_thresh:
                low_dets.append(i)
            # detections below trk_low_thresh are silently discarded

        # -- 4. Stage 1: HIGH dets vs ALL tracks (appearance + IoU fusion) ----
        stage1_unmatched_trk_idx: list[int]     # indices into self.tracks
        unmatched_high_det_idx: list[int]       # indices into high_dets

        if self.tracks and high_dets:
            n_trk = len(self.tracks)
            n_det = len(high_dets)

            # Build the fused cost matrix.
            cost = np.zeros((n_trk, n_det), dtype=float)
            for ti in range(n_trk):
                trk_feat = self.tracks[ti].feature
                trk_shifted = shifted_boxes[ti]
                for di_local, di_global in enumerate(high_dets):
                    det = detections[di_global]
                    det_feat = features[di_global]

                    # Appearance component.
                    app_dist = feature_cosine_distance(trk_feat, det_feat)

                    # IoU component (use shifted predicted box).
                    iou_val = iou(trk_shifted, det.bbox)
                    iou_cost = 1.0 - iou_val

                    # Weighted fusion.
                    w = self.p.trk_appearance_weight
                    fused = w * app_dist + (1.0 - w) * iou_cost

                    # Hard gate: reject if IoU is too low OR appearance
                    # distance exceeds the gate (when both features exist).
                    gated = False
                    if iou_val < self.p.trk_iou_threshold:
                        gated = True
                    if (trk_feat is not None and det_feat is not None
                            and app_dist > self.p.trk_appearance_gate):
                        gated = True

                    cost[ti, di_local] = 1e5 if gated else fused

            matches_s1, unmatched_trk_s1, unmatched_det_s1_local = associate(
                cost, max_cost=1.0,
            )

            # Keep only matches that survived the hard gate (cost < 1e5).
            valid_matches_s1: list[tuple[int, int]] = []
            actually_matched_trk: set[int] = set()
            actually_matched_det_local: set[int] = set()
            for t_idx, d_local in matches_s1:
                if cost[t_idx, d_local] < 1e5:
                    valid_matches_s1.append((t_idx, d_local))
                    actually_matched_trk.add(t_idx)
                    actually_matched_det_local.add(d_local)

            # Update matched tracks.
            for t_idx, d_local in valid_matches_s1:
                di_global = high_dets[d_local]
                det = detections[di_global]
                self.tracks[t_idx].update(
                    det.bbox,
                    confidence=det.confidence,
                    feature=features[di_global],
                )

            # Recompute unmatched sets (associate may have matched some that
            # the hard gate later rejected).
            stage1_unmatched_trk_idx = [
                i for i in range(n_trk) if i not in actually_matched_trk
            ]
            unmatched_high_det_idx = [
                d_local for d_local in range(n_det)
                if d_local not in actually_matched_det_local
            ]
        else:
            stage1_unmatched_trk_idx = list(range(len(self.tracks)))
            unmatched_high_det_idx = list(range(len(high_dets)))

        # -- 5. Stage 2: LOW dets vs STILL-UNMATCHED tracks (IoU only) -------
        #    The BYTE trick — recover tracks using low-score boxes with a
        #    looser IoU gate (require IoU >= 0.5).
        if stage1_unmatched_trk_idx and low_dets:
            remaining_shifted = [shifted_boxes[i] for i in stage1_unmatched_trk_idx]
            low_boxes = [detections[i].bbox for i in low_dets]
            iou_mat_s2 = iou_batch(remaining_shifted, low_boxes)
            cost_s2 = 1.0 - iou_mat_s2
            max_cost_s2 = 1.0 - 0.5  # require IoU >= 0.5

            matches_s2, _unmatched_rem, _unmatched_low = associate(
                cost_s2, max_cost_s2,
            )

            for rem_idx, d_local in matches_s2:
                real_trk_idx = stage1_unmatched_trk_idx[rem_idx]
                di_global = low_dets[d_local]
                det = detections[di_global]
                self.tracks[real_trk_idx].update(
                    det.bbox,
                    confidence=det.confidence,
                    feature=None,  # low-confidence dets: skip appearance update
                )

        # -- 5b. Fast-motion recovery: velocity-scaled distance fallback for
        #        leftover tracks vs leftover HIGH detections (global indices).
        fb_trks = [i for i, t in enumerate(self.tracks) if t.time_since_update > 0]
        leftover_global = [high_dets[dl] for dl in unmatched_high_det_idx]
        fb = distance_fallback(self.tracks, fb_trks, detections, leftover_global, self.p)
        for ti, gd in fb:
            det = detections[gd]
            self.tracks[ti].update(det.bbox, det.confidence, feature=features[gd])
        fb_global = {gd for _, gd in fb}
        unmatched_high_det_idx = [dl for dl in unmatched_high_det_idx
                                  if high_dets[dl] not in fb_global]

        # -- 6. Spawn new tracks from unmatched HIGH detections ---------------
        for d_local in unmatched_high_det_idx:
            di_global = high_dets[d_local]
            det = detections[di_global]
            if det.confidence >= self.p.trk_new_thresh:
                self.tracks.append(
                    KalmanBoxTracker(
                        bbox=det.bbox,
                        confidence=det.confidence,
                        track_id=self._next_id,
                        params=self.p,
                        feature=features[di_global],
                    ),
                )
                self._next_id += 1

        # -- 7. Prune dead tracks (exceeded max_age without update) -----------
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.p.trk_max_age
        ]

        # -- 8. Update previous camera pose for next frame's CMC -------------
        if camera is not None:
            self._prev_pan = camera.pan
            self._prev_tilt = camera.tilt

        # -- 9. Return public Track objects -----------------------------------
        return [t.to_track() for t in self.tracks]
