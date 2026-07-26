"""Pick which track the camera should follow.

Locks onto one track id and stays on it (so the camera doesn't flip between
people every frame). When the followed track dies, it keeps waiting for
``ctrl_target_lost_grace`` frames — during which the track may still be coasting
on its Kalman prediction — before re-selecting a new target by the follow mode:

    0 = largest (nearest) person   1 = nearest to frame center
"""

from __future__ import annotations

from params import Params


class TargetSelector:
    def __init__(self, params: Params) -> None:
        self.p = params
        self.followed_id: int | None = None
        self.lost_frames: int = 0

    def reset(self) -> None:
        self.followed_id = None
        self.lost_frames = 0

    @staticmethod
    def _area(track) -> float:
        x1, y1, x2, y2 = track.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _pick(self, tracks, frame_w: int, frame_h: int):
        confirmed = [t for t in tracks if t.confirmed] or tracks
        if not confirmed:
            return None
        if self.p.follow_mode == 1:
            fx, fy = frame_w / 2.0, frame_h / 2.0
            return min(
                confirmed,
                key=lambda t: (t.center[0] - fx) ** 2 + (t.center[1] - fy) ** 2,
            )
        return max(confirmed, key=self._area)

    def select(self, tracks, frame_w: int, frame_h: int):
        by_id = {t.track_id: t for t in tracks}

        if self.followed_id is not None and self.followed_id in by_id:
            followed = by_id[self.followed_id]
            # Still a live observation? reset the grace counter.
            if followed.time_since_update == 0:
                self.lost_frames = 0
            else:
                self.lost_frames += 1
            if self.lost_frames <= self.p.ctrl_target_lost_grace:
                return followed  # keep following (may be coasting)

        # No current target, or grace expired -> choose a fresh one.
        pick = self._pick(tracks, frame_w, frame_h)
        self.followed_id = pick.track_id if pick is not None else None
        self.lost_frames = 0
        return pick
