"""Pluggable multi-object trackers, selectable at runtime.

All trackers share the same interface:
    tracker.update(detections, frame=None, camera=None) -> list[Track]
and return the same public `Track` objects (see `base.Track`), so the demo's
overlay / target-lock logic is identical regardless of which one is selected.

Use `make_tracker(name, params)` to construct one; `TRACKER_NAMES` is the
ordered list shown in the UI selector.
"""

from __future__ import annotations

from .base import Track
from .botsort import BotSortTracker
from .bytetrack import ByteTrackTracker
from .deepsort import DeepSortTracker
from .ocsort import OcSortTracker
from .sort import SortTracker

# Ordered for the UI slider (index -> name).
TRACKER_NAMES = ["SORT", "ByteTrack", "DeepSORT", "OC-SORT", "BoT-SORT"]

_REGISTRY = {
    "SORT": SortTracker,
    "ByteTrack": ByteTrackTracker,
    "DeepSORT": DeepSortTracker,
    "OC-SORT": OcSortTracker,
    "BoT-SORT": BotSortTracker,
}


def make_tracker(name: str, params):
    """Construct the tracker named `name` (case-insensitive)."""
    key = next((k for k in _REGISTRY if k.lower() == name.lower()), None)
    if key is None:
        raise ValueError(f"unknown tracker {name!r}; choose from {TRACKER_NAMES}")
    return _REGISTRY[key](params)


__all__ = ["Track", "TRACKER_NAMES", "make_tracker"]
