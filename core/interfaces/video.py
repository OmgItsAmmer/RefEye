"""Video input / buffer interfaces. See architecture.md sections 10 and 12."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class StreamInfo:
    width: int
    height: int
    fps: float
    codec: str
    time_base: tuple[int, int]
    source_id: str


@dataclass
class EncodedPacket:
    data: bytes
    pts: int | None
    dts: int | None
    is_keyframe: bool
    stream_index: int
    size_bytes: int = 0
    # Opaque handle to the adapter's native packet object (e.g. an av.Packet).
    # ONLY the decoder paired with the producing VideoInput may touch this;
    # nothing in vision/, ai/, analysis/, or apps/ may depend on its type.
    # It exists so we can decode without re-muxing raw bytes back into a
    # container, while `data`/`size_bytes` stay available for buffer accounting.
    raw: object | None = None


class VideoInput(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read_packet(self) -> EncodedPacket | None: ...
    def get_stream_info(self) -> StreamInfo: ...


class RecentVideoBuffer(Protocol):
    def append(self, item) -> None: ...
    def get_by_frame(self, frame_id: int): ...
    def get_window(self, end_timestamp_ms: int, duration_ms: int) -> list: ...
    def get_range(self, start_frame_id: int, end_frame_id: int) -> list: ...
