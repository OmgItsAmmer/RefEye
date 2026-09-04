"""Bounded rolling buffers. See architecture.md section 12.

Two logical buffers:
  * EncodedRingBuffer — recent compressed packets, cheap to hold, used to
    reconstruct original-quality frames around a selected candidate.
  * DecodedRingBuffer — recent decoded analysis-resolution frames, for
    instant access during a triggered analysis.

Both are hard-bounded by BOTH count and duration. Memory must never grow
without limit over a full match (architecture.md section 36).
"""

from __future__ import annotations

import threading
from collections import deque

from core.domain.models import FramePacket
from core.interfaces.video import EncodedPacket


def _frame_nbytes(frame: FramePacket) -> int:
    return 0 if frame.image is None else frame.image.nbytes


def estimate_frame_bytes(width: int, height: int, channels: int = 3) -> int:
    """Bytes one decoded BGR frame occupies — used to project buffer cost."""
    return width * height * channels


class DecodedRingBuffer:
    """Thread-safe bounded buffer of decoded FramePackets.

    Written by the ingest worker, read by the UI and analysis threads, so
    every operation takes the lock. Reads return the internal FramePacket
    objects; callers must treat them as immutable (never mutate `.image`).
    """

    def __init__(
        self,
        max_frames: int,
        max_duration_ms: int | None = None,
        max_bytes: int | None = None,
    ):
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")

        self._max_frames = max_frames
        self._max_duration_ms = max_duration_ms
        self._max_bytes = max_bytes
        self._items: deque[FramePacket] = deque()
        self._by_frame_id: dict[int, FramePacket] = {}
        self._total_bytes = 0
        self._lock = threading.RLock()

    def append(self, item: FramePacket) -> None:
        with self._lock:
            self._items.append(item)
            self._by_frame_id[item.frame_id] = item
            self._total_bytes += _frame_nbytes(item)
            self._evict_locked()

    def _evict_locked(self) -> None:
        while len(self._items) > self._max_frames:
            self._drop_oldest_locked()

        if self._max_duration_ms is not None and self._items:
            newest_ms = self._items[-1].timestamp_ms
            while (
                len(self._items) > 1
                and newest_ms - self._items[0].timestamp_ms > self._max_duration_ms
            ):
                self._drop_oldest_locked()

        # The byte cap is the bound that actually protects RAM: a frame count
        # means nothing without knowing the resolution behind it.
        if self._max_bytes is not None:
            while len(self._items) > 1 and self._total_bytes > self._max_bytes:
                self._drop_oldest_locked()

    def _drop_oldest_locked(self) -> None:
        evicted = self._items.popleft()
        self._by_frame_id.pop(evicted.frame_id, None)
        self._total_bytes -= _frame_nbytes(evicted)
        # Do NOT mutate `evicted.image` here. get_window()/get_range() hand
        # out these exact FramePacket objects (not copies) to callers such as
        # a triggered analysis, which can still be iterating them on another
        # thread seconds after this eviction runs. Nulling .image in place
        # was corrupting frames out from under an in-flight analysis — a
        # frame that passed an `image is not None` check moments earlier
        # would go empty mid-batch and crash downstream (e.g. cv2.resize on
        # a None/empty array). Simply removing our own references to the
        # object here is enough: once no other thread holds it, Python's GC
        # reclaims the pixel data on its own — no explicit null needed, and
        # no risk of yanking it out from under a concurrent reader.

    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def get_by_frame(self, frame_id: int) -> FramePacket | None:
        with self._lock:
            return self._by_frame_id.get(frame_id)

    def get_window(self, end_timestamp_ms: int, duration_ms: int) -> list[FramePacket]:
        start_ms = end_timestamp_ms - duration_ms
        with self._lock:
            return [f for f in self._items if start_ms <= f.timestamp_ms <= end_timestamp_ms]

    def get_range(self, start_frame_id: int, end_frame_id: int) -> list[FramePacket]:
        with self._lock:
            return [f for f in self._items if start_frame_id <= f.frame_id <= end_frame_id]

    def latest(self) -> FramePacket | None:
        with self._lock:
            return self._items[-1] if self._items else None

    def duration_ms(self) -> int:
        with self._lock:
            if len(self._items) < 2:
                return 0
            return self._items[-1].timestamp_ms - self._items[0].timestamp_ms

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._by_frame_id.clear()
            self._total_bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class EncodedRingBuffer:
    """Thread-safe bounded buffer of recent encoded packets.

    Holds `data` bytes plus PTS/keyframe metadata so a high-quality clip can
    be reconstructed later. The native `raw` handle is deliberately dropped on
    insert — keeping thousands of live PyAV packet objects alive would pin
    decoder memory we do not control.
    """

    def __init__(self, max_packets: int, max_bytes: int | None = None):
        if max_packets <= 0:
            raise ValueError("max_packets must be positive")

        self._max_packets = max_packets
        self._max_bytes = max_bytes
        self._items: deque[EncodedPacket] = deque()
        self._total_bytes = 0
        self._lock = threading.RLock()

    def append(self, item: EncodedPacket) -> None:
        stored = EncodedPacket(
            data=item.data,
            pts=item.pts,
            dts=item.dts,
            is_keyframe=item.is_keyframe,
            stream_index=item.stream_index,
            size_bytes=item.size_bytes or len(item.data),
            raw=None,
        )

        with self._lock:
            self._items.append(stored)
            self._total_bytes += stored.size_bytes

            while len(self._items) > self._max_packets:
                self._drop_oldest_locked()

            if self._max_bytes is not None:
                while len(self._items) > 1 and self._total_bytes > self._max_bytes:
                    self._drop_oldest_locked()

    def _drop_oldest_locked(self) -> None:
        evicted = self._items.popleft()
        self._total_bytes -= evicted.size_bytes

    def get_range(self, start_pts: int, end_pts: int) -> list[EncodedPacket]:
        """Packets covering [start_pts, end_pts], extended back to the last
        keyframe at or before start_pts so the range is independently decodable.
        """
        with self._lock:
            items = list(self._items)

        selected = [p for p in items if p.pts is not None and start_pts <= p.pts <= end_pts]
        if not selected:
            return []

        preceding_keyframes = [
            p for p in items if p.is_keyframe and p.pts is not None and p.pts <= start_pts
        ]
        if preceding_keyframes:
            anchor = preceding_keyframes[-1]
            selected = [
                p
                for p in items
                if p.pts is not None and anchor.pts <= p.pts <= end_pts
            ]

        return selected

    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._total_bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
