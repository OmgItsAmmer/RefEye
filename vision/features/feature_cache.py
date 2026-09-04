"""Bounded cache of per-frame CV features.

This is the "reuse prior work" mechanism from architecture.md section 4.4 and
section 16: the background pipeline continuously produces cheap features so
that pressing the analysis hotkey does not restart everything from zero.

Storage is bounded by frame count (section 16: "Feature storage must remain
bounded"). Entries hold small scalars and short tuples only — never images,
never tensors, never model activations that would pin GPU memory.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from core.domain.models import FrameFeatures, TrackObservation


@dataclass
class CachedFrame:
    """Everything the refinement stage needs about one frame."""

    features: FrameFeatures
    tracks: list[TrackObservation] = field(default_factory=list)
    scene_cut: bool = False
    segment: int = 0

    @property
    def frame_id(self) -> int:
        return self.features.frame_id


class FeatureCache:
    def __init__(self, max_frames: int = 900):
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self._max_frames = max_frames
        self._items: OrderedDict[int, CachedFrame] = OrderedDict()
        self._lock = threading.RLock()

    def put(self, entry: CachedFrame) -> None:
        with self._lock:
            self._items[entry.frame_id] = entry
            self._items.move_to_end(entry.frame_id)
            while len(self._items) > self._max_frames:
                self._items.popitem(last=False)

    def get(self, frame_id: int) -> CachedFrame | None:
        with self._lock:
            return self._items.get(frame_id)

    def get_range(self, start_frame_id: int, end_frame_id: int) -> list[CachedFrame]:
        with self._lock:
            return [
                entry
                for frame_id, entry in self._items.items()
                if start_frame_id <= frame_id <= end_frame_id
            ]

    def latest(self) -> CachedFrame | None:
        with self._lock:
            if not self._items:
                return None
            return next(reversed(self._items.values()))

    def covered_frame_ids(self) -> tuple[int, int] | None:
        with self._lock:
            if not self._items:
                return None
            keys = list(self._items.keys())
            return min(keys), max(keys)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
