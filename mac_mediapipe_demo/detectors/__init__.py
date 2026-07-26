"""AI detector package (Face & Pose detectors)."""

from detectors.base import DET_MODE_NAMES, Detection, make_detector
from detectors.face import FaceDetector
from detectors.pose import PoseDetector

__all__ = ["Detection", "DET_MODE_NAMES", "make_detector", "FaceDetector", "PoseDetector"]
