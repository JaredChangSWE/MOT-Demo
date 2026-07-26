"""Shared detection contract and detector factory."""

from __future__ import annotations

from dataclasses import dataclass

from params import Params

# 0 = face (BlazeFace), 1 = pose (full-body landmarks)
DET_MODE_NAMES = ["face", "pose"]


@dataclass
class Detection:
    """One detected object in image space (full-resolution pixels)."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str = "person"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def make_detector(params: Params):
    if params.det_mode == 1:
        from detectors.pose import PoseDetector
        return PoseDetector(params)
    from detectors.face import FaceDetector
    return FaceDetector(params)
