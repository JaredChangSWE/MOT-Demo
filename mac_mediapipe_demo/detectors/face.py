"""Face detector built on MediaPipe Face Detection (BlazeFace, Tasks API).

More accurate and much faster than full-body pose for "who is in frame", which
is what we want to keep centered. Emits the same ``Detection`` bbox contract the
trackers consume. Confidence is the detector score, gated by
``params.det_min_confidence`` (raise it to reject weak/false faces).

For speed, detection runs on a downscaled copy of the frame and the boxes are
mapped back to full resolution.

Note: the short-range BlazeFace model is tuned for faces within ~2 m of the
camera; stand closer if distant faces are missed.
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
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
_MODEL_PATH = Path(__file__).resolve().parents[1] / "blaze_face_short_range.tflite"


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        print(f"Downloading face model (once) -> {_MODEL_PATH.name} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


class FaceDetector:
    mode = "face"

    def __init__(self, params: Params) -> None:
        self.p = params
        self._conf = params.det_min_confidence
        self._start = time.monotonic()
        self._last_ts = -1
        self._make()

    def _next_ts(self) -> int:
        # MediaPipe VIDEO mode requires strictly increasing timestamps; sub-ms
        # loops can collide, so force monotonic.
        ts = int((time.monotonic() - self._start) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def _make(self) -> None:
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=self._conf,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def maybe_rebuild(self) -> None:
        """The confidence threshold is baked in at construction; rebuild if it
        changed on the slider."""
        if abs(self.p.det_min_confidence - self._conf) > 1e-6:
            self._conf = self.p.det_min_confidence
            self._detector.close()
            self._make()

    def detect(self, frame_bgr) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        scale = 1.0
        small = frame_bgr
        target_w = self.p.det_downscale_width
        if target_w and w > target_w:
            scale = w / float(target_w)
            small = cv2.resize(frame_bgr, (target_w, int(h / scale)))

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, self._next_ts())

        dets: list[Detection] = []
        for d in result.detections or []:
            score = d.categories[0].score if d.categories else 0.0
            if score < self.p.det_min_confidence:
                continue
            bb = d.bounding_box
            x1 = bb.origin_x * scale
            y1 = bb.origin_y * scale
            x2 = (bb.origin_x + bb.width) * scale
            y2 = (bb.origin_y + bb.height) * scale
            dets.append(Detection(bbox=(x1, y1, x2, y2), confidence=float(score),
                                  class_name="face"))
        return dets

    def close(self) -> None:
        self._detector.close()
