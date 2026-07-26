"""Shared detection contract consumed by the multi-object trackers."""

from __future__ import annotations

from dataclasses import dataclass


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
