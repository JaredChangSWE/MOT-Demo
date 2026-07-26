"""Detector factory: choose face or pose detection from the live params.

Both detectors expose the same interface —
``detect(frame) -> list[Detection]``, ``maybe_rebuild()``, ``close()``, and a
``mode`` string — so the rest of the pipeline is detector-agnostic.
"""

from __future__ import annotations

from params import Params

# 0 = face (BlazeFace), 1 = pose (full-body landmarks)
DET_MODE_NAMES = ["face", "pose"]


def make_detector(params: Params):
    if params.det_mode == 1:
        from pose_detector import PoseDetector
        return PoseDetector(params)
    from face_detector import FaceDetector
    return FaceDetector(params)
