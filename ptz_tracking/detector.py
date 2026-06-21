"""Object detection for the demo.

`GroundTruthDetector` derives detections directly from the simulator's true
geometry (the camera projection), optionally adding realistic noise: small bbox
jitter, varying confidence, and occasional missed detections. This keeps the
demo reliable, since a COCO-trained YOLO will not detect rendered figures.

`Detection` is the shared contract consumed by the tracker. A real
`YOLODetector` could implement the same `detect(camera, world) -> list[Detection]`
interface as a drop-in replacement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .camera import Camera, visible_people
from .params import Params
from .world import World


@dataclass
class Detection:
    """One detected object in image space."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) pixels
    confidence: float
    class_name: str = "person"
    gt_pid: int | None = None  # ground-truth person id (debug only; tracker ignores it)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


class GroundTruthDetector:
    def __init__(self, params: Params | None = None, noisy: bool | None = None,
                 seed: int | None = 1):
        self.p = params or Params()
        self.noisy = self.p.det_noisy if noisy is None else noisy
        self.rng = np.random.default_rng(seed)

    def detect(self, camera: Camera, world: World) -> list[Detection]:
        dets: list[Detection] = []
        for person, bbox in visible_people(camera, world):
            if self.noisy and self.rng.random() < self.p.det_drop_prob:
                continue
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            if self.noisy:
                j = self.p.det_bbox_jitter_px
                x1 += self.rng.normal(0, j)
                y1 += self.rng.normal(0, j)
                x2 += self.rng.normal(0, j)
                y2 += self.rng.normal(0, j)
            conf = float(self.rng.uniform(*config.DET_CONF_RANGE)) if self.noisy else 0.99
            dets.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                class_name="person",
                gt_pid=person.pid,
            ))
        return dets
