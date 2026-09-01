"""Bounded ring buffers backing the encoded and decoded rolling windows.

See architecture.md section 12. All buffers must have explicit limits and
automatic eviction — never grow unbounded during a long session.
"""

from __future__ import annotations

from collections import deque

from core.domain.models import FramePacket
from core.interfaces.video import EncodedPacket


class DecodedRingBuffer:
    """Bounded buffer of recently decoded FramePackets, indexed by frame_id."""

    def __init__(self, max_frames: int):
        self._max_frames = max_frames
        self._items: deque[FramePacket] = deque(maxlen=max_frames)
        self._by_frame_id: dict[int, FramePacket] = {}

    def append(self, item: FramePacket) -> None:
        if len(self._items) == self._max_frames:
            evicted = self._items[0]
            self._by_frame_id.pop(evicted.frame_id, None)
        self._items.append(item)
        self._by_frame_id[item.frame_id] = item

    def get_by_frame(self, frame_id: int) -> FramePacket | None:
        return self._by_frame_id.get(frame_id)

    def get_window(self, end_timestamp_ms: int, duration_ms: int) -> list[FramePacket]:
        start_ms = end_timestamp_ms - duration_ms
        return [f for f in self._items if start_ms <= f.timestamp_ms <= end_timestamp_ms]

    def get_range(self, start_frame_id: int, end_frame_id: int) -> list[FramePacket]:
        return [f for f in self._items if start_frame_id <= f.frame_id <= end_frame_id]

    def __len__(self) -> int:
        return len(self._items)


class EncodedRingBuffer:
    """Bounded buffer of recent encoded packets for high-quality reconstruction."""

    def __init__(self, max_packets: int):
        self._max_packets = max_packets
        self._items: deque[EncodedPacket] = deque(maxlen=max_packets)

    def append(self, item: EncodedPacket) -> None:
        self._items.append(item)

    def get_range(self, start_index: int, end_index: int) -> list[EncodedPacket]:
        return list(self._items)[start_index:end_index]

    def __len__(self) -> int:
        return len(self._items)
