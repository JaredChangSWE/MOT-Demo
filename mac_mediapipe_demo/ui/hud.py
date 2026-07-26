"""Tracking overlay / HUD for the tracked-lens window.

Draws every track (box, id, motion trail), highlights the followed target, shows
the engage/deadband box, and a status panel (tracker name, per-frame tracker
latency, live track count, follow state, centering error, PTZ velocity).
"""

from __future__ import annotations

import cv2

_ID_COLORS = [
    (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36),
]


def _color(track_id: int):
    return _ID_COLORS[track_id % len(_ID_COLORS)]


def draw(frame, tracks, followed_id, status, meta: dict) -> None:
    h, w = frame.shape[:2]

    # Follow boxes: outer = Start-follow (engage), inner = Stop-follow (release).
    # A face outside the start box begins tracking; the camera stops once the
    # face is inside the stop box.
    def _box(e: float, color, label: str) -> None:
        x1, y1 = int(w * (0.5 - e / 2)), int(h * (0.5 - e / 2))
        x2, y2 = int(w * (0.5 + e / 2)), int(h * (0.5 + e / 2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    _box(meta["engage_error"], (0, 180, 255), "start")   # orange
    _box(meta.get("release_error", 0.0), (0, 230, 0), "stop")  # green

    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        followed = t.track_id == followed_id
        col = (0, 255, 0) if followed else _color(t.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if followed else 1)
        label = f"ID {t.track_id}" + (" *" if followed else "")
        cv2.putText(frame, label, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        # Motion trail.
        pts = t.trajectory
        for i in range(1, len(pts)):
            cv2.line(frame, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                     (int(pts[i][0]), int(pts[i][1])), col, 1)

    # Followed target crosshair.
    if followed_id is not None:
        for t in tracks:
            if t.track_id == followed_id:
                cx, cy = int(t.center[0]), int(t.center[1])
                cv2.drawMarker(frame, (cx, cy), (0, 255, 0),
                               cv2.MARKER_CROSS, 22, 2)
                break

    # Status panel.
    if followed_id is None:
        state = "SEARCHING"
    elif status.engaged:
        state = "ENGAGED"
    else:
        state = "IN RANGE"
    lines = [
        f"Tracker: {meta['tracker_name']}   {meta['latency_ms']:.2f} ms",
        f"Tracks: {len(tracks)}   Target: "
        + ("--" if followed_id is None else f"ID {followed_id}"),
        f"State: {state}   err: {status.error_norm:.2f}",
        # ex>0 = face is RIGHT of center; pan cmd should be >0 to chase it (unless
        # inverted). ey>0 = face BELOW center. If cmd sign disagrees, calibrate.
        f"ex {status.ex:+.2f}  ey {status.ey:+.2f}",
        f"PTZ vel  pan {status.pan_vel:+.2f}  tilt {status.tilt_vel:+.2f}",
    ]
    y = 24
    for ln in lines:
        cv2.putText(frame, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4)
        cv2.putText(frame, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 1)
        y += 26
