"""Multi-person detector built on MediaPipe Pose (Tasks API).

Bridges the real camera to the pluggable multi-object trackers: it runs
``PoseLandmarker`` with ``num_poses > 1`` and turns each detected person into a
``Detection`` (pixel bbox + confidence) — exactly the contract the trackers in
``ptz_tracking.trackers`` consume. This keeps the demo dependency-light (no
YOLO), consistent with the simulator's detector interface.

Confidence is the mean visibility of the person's landmarks (a reasonable
stand-in for a detector score, used by ByteTrack's high/low gating).
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from detectors.base import Detection
from params import Params

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_MODEL_PATH = Path(__file__).resolve().parents[1] / "pose_landmarker_lite.task"


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        print(f"Downloading pose model (once) -> {_MODEL_PATH.name} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


class PoseDetector:
    mode = "pose"

    def __init__(self, params: Params) -> None:
        self.p = params
        self._num_poses = params.det_num_poses
        self._conf = params.det_min_confidence
        self._start = time.monotonic()
        self._last_ts = -1
        self._make()

    def _next_ts(self) -> int:
        ts = int((time.monotonic() - self._start) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def _make(self) -> None:
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=max(1, self._num_poses),
            min_pose_detection_confidence=self.p.det_min_confidence,
            min_tracking_confidence=self.p.min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def maybe_rebuild(self) -> None:
        """num_poses / confidence are fixed at construction; rebuild on change."""
        if (self.p.det_num_poses != self._num_poses
                or abs(self.p.det_min_confidence - self._conf) > 1e-6):
            self._num_poses = self.p.det_num_poses
            self._conf = self.p.det_min_confidence
            self._landmarker.close()
            self._make()

    def detect(self, frame_bgr) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        # Landmarks are normalized, so we can run inference on a downscaled copy
        # for speed and still map back with the full (w, h).
        small = frame_bgr
        target_w = self.p.det_downscale_width
        if target_w and w > target_w:
            small = cv2.resize(frame_bgr, (target_w, int(h * target_w / w)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._next_ts())

        dets: list[Detection] = []
        for person in result.pose_landmarks or []:
            vis = [l for l in person if l.visibility > self.p.det_min_visibility]
            if len(vis) < self.p.det_min_landmarks:
                continue
            xs = [l.x for l in vis]
            ys = [l.y for l in vis]
            x1, x2 = min(xs) * w, max(xs) * w
            y1, y2 = min(ys) * h, max(ys) * h
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            conf = sum(l.visibility for l in vis) / len(vis)
            dets.append(Detection(bbox=(x1, y1, x2, y2), confidence=float(conf)))
        return dets

    def close(self) -> None:
        self._landmarker.close()
