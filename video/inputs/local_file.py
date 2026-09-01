"""Local video file input adapter — the primary VideoInput for M1.

Wraps PyAV so callers only depend on the VideoInput protocol, never on `av`
directly (keeps the source swappable per architecture.md section 10).
"""

from __future__ import annotations

import av

from core.errors.exceptions import StreamUnavailableError
from core.interfaces.video import EncodedPacket, StreamInfo
from observability.logging.setup import get_logger

logger = get_logger(__name__)


class LocalFileInput:
    def __init__(self, path: str, loop: bool = True, source_id: str = "local_file"):
        self._path = path
        self._loop = loop
        self._source_id = source_id
        self._container: av.container.InputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None

    def start(self) -> None:
        try:
            self._container = av.open(self._path)
            self._stream = self._container.streams.video[0]
        except (av.error.FFmpegError, IndexError) as exc:
            raise StreamUnavailableError(f"Could not open local file: {self._path}") from exc
        logger.info("stream_connected", source=self._source_id, path=self._path)

    def stop(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
        logger.info("stream_disconnected", source=self._source_id)

    def read_packet(self) -> EncodedPacket | None:
        if self._container is None:
            raise StreamUnavailableError("read_packet called before start()")

        try:
            packet = next(self._container.demux(self._stream))
        except StopIteration:
            if self._loop:
                self._container.seek(0)
                return self.read_packet()
            return None

        return EncodedPacket(
            data=bytes(packet),
            pts=packet.pts,
            dts=packet.dts,
            is_keyframe=bool(packet.is_keyframe),
            stream_index=packet.stream_index,
        )

    def get_stream_info(self) -> StreamInfo:
        if self._stream is None:
            raise StreamUnavailableError("get_stream_info called before start()")

        return StreamInfo(
            width=self._stream.width,
            height=self._stream.height,
            fps=float(self._stream.average_rate) if self._stream.average_rate else 0.0,
            codec=self._stream.codec_context.name,
            time_base=(self._stream.time_base.numerator, self._stream.time_base.denominator),
            source_id=self._source_id,
        )
