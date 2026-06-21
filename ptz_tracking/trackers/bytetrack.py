"""ByteTrack tracker — two-stage BYTE association (motion/IoU only).

ByteTrack splits detections into HIGH and LOW confidence bins and runs two
rounds of IoU association:

  Stage 1 — match HIGH-confidence detections to ALL tracks.
  Stage 2 — match LOW-confidence detections to the tracks that remain
             unmatched after stage 1 (with a looser IoU gate).

This "BYTE trick" recovers tracks that would otherwise be lost when the
detector briefly produces a low-score box (occlusion, blur, etc.).

Note: with this demo's idealised ground-truth detector, confidence is sampled
in [0.70, 0.98] so almost every box lands above ``trk_high_thresh`` (0.6).
The two-stage mechanism is nevertheless implemented faithfully and will
activate if a noisier detector is plugged in.
"""

from __future__ import annotations

import numpy as np

from .base import Track, KalmanBoxTracker, iou_batch, associate, distance_fallback


class ByteTrackTracker:
    """Multi-object tracker using the BYTE two-stage association policy.

    Parameters
    ----------
    params : Params
        Shared parameter object.  Relevant fields:

        * ``trk_high_thresh``  — confidence split between HIGH / LOW.
        * ``trk_low_thresh``   — floor; detections below this are discarded.
        * ``trk_new_thresh``   — minimum confidence to spawn a new track.
        * ``trk_iou_threshold`` — IoU gate for stage-1 association.
        * ``trk_max_age``      — frames before an unmatched track is deleted.
        * ``trk_min_hits``     — consecutive hits to confirm a track.
    """

    def __init__(self, params) -> None:
        self.p = params
        self.tracks: list[KalmanBoxTracker] = []
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: list,
        frame: np.ndarray | None = None,
        camera=None,
    ) -> list[Track]:
        """Run one frame of ByteTrack association.

        Parameters
        ----------
        detections : list[Detection]
            Raw detections for this frame (each has ``.bbox``,
            ``.confidence``).
        frame : ndarray | None
            Current image (unused — ByteTrack is motion-only).
        camera : Camera | None
            Camera state (unused).

        Returns
        -------
        list[Track]
            Public track objects for this frame.
        """

        # -- 0. Split detections by confidence ----------------------------
        high_dets: list = []
        low_dets: list = []
        for d in detections:
            if d.confidence >= self.p.trk_high_thresh:
                high_dets.append(d)
            elif d.confidence >= self.p.trk_low_thresh:
                low_dets.append(d)
            # detections below trk_low_thresh are silently discarded

        # -- 1. Predict every track (once per frame) ----------------------
        for trk in self.tracks:
            trk.predict()

        # -- 2. Stage 1: HIGH detections vs ALL tracks --------------------
        stage1_unmatched_trk_idx: list[int]  # indices into self.tracks

        if self.tracks and high_dets:
            trk_boxes = [t.bbox for t in self.tracks]
            high_boxes = [d.bbox for d in high_dets]
            iou_mat = iou_batch(trk_boxes, high_boxes)
            cost = 1.0 - iou_mat
            max_cost_s1 = 1.0 - self.p.trk_iou_threshold

            matches_s1, unmatched_trk_s1, _unmatched_det_s1 = associate(
                cost, max_cost_s1,
            )

            # Update matched tracks with their high-confidence detection.
            for t_idx, d_idx in matches_s1:
                det = high_dets[d_idx]
                self.tracks[t_idx].update(det.bbox, confidence=det.confidence)

            stage1_unmatched_trk_idx = unmatched_trk_s1
            unmatched_high_det_idx: list[int] = _unmatched_det_s1
        else:
            # No tracks or no high detections — everything is unmatched.
            stage1_unmatched_trk_idx = list(range(len(self.tracks)))
            unmatched_high_det_idx = list(range(len(high_dets)))

        # -- 3. Stage 2: LOW detections vs STILL-UNMATCHED tracks ---------
        #    The BYTE trick — recover tracks using low-score boxes with a
        #    looser IoU gate (require IoU >= 0.5).
        if stage1_unmatched_trk_idx and low_dets:
            remaining_tracks = [self.tracks[i] for i in stage1_unmatched_trk_idx]
            rem_boxes = [t.bbox for t in remaining_tracks]
            low_boxes = [d.bbox for d in low_dets]
            iou_mat_s2 = iou_batch(rem_boxes, low_boxes)
            cost_s2 = 1.0 - iou_mat_s2
            max_cost_s2 = 1.0 - 0.5  # require IoU >= 0.5

            matches_s2, _unmatched_rem, _unmatched_low = associate(
                cost_s2, max_cost_s2,
            )

            for rem_idx, d_idx in matches_s2:
                real_trk_idx = stage1_unmatched_trk_idx[rem_idx]
                det = low_dets[d_idx]
                self.tracks[real_trk_idx].update(
                    det.bbox, confidence=det.confidence,
                )

        # -- 3c. Fast-motion recovery: velocity-scaled distance fallback for
        #     leftover tracks vs leftover HIGH detections (IoU may be 0 for
        #     fast / turning targets).
        fb_trks = [i for i, t in enumerate(self.tracks) if t.time_since_update > 0]
        fb = distance_fallback(self.tracks, fb_trks, high_dets,
                               unmatched_high_det_idx, self.p)
        for ti, dl in fb:
            det = high_dets[dl]
            self.tracks[ti].update(det.bbox, det.confidence)
        fb_dets = {dl for _, dl in fb}
        unmatched_high_det_idx = [dl for dl in unmatched_high_det_idx
                                  if dl not in fb_dets]

        # -- 4. Spawn new tracks from unmatched HIGH detections -----------
        #    Only high-confidence detections above trk_new_thresh may
        #    initialise new tracks.  LOW detections never spawn tracks.
        for d_idx in unmatched_high_det_idx:
            det = high_dets[d_idx]
            if det.confidence >= self.p.trk_new_thresh:
                self.tracks.append(
                    KalmanBoxTracker(
                        bbox=det.bbox,
                        confidence=det.confidence,
                        track_id=self._next_id,
                        params=self.p,
                    ),
                )
                self._next_id += 1

        # -- 5. Prune dead tracks (exceeded max_age without update) -------
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.p.trk_max_age
        ]

        # -- 6. Return public Track objects --------------------------------
        return [t.to_track() for t in self.tracks]
