"""Local video file input adapter — the primary VideoInput for M1.

Wraps PyAV so callers depend only on the VideoInput protocol, never on `av`
directly (architecture.md section 10). Capture-card / RTSP / NDI adapters
slot in behind the same protocol without touching downstream code.
"""

from __future__ import annotations

from pathlib import Path

import av

from core.config.paths import resolve
from core.errors.exceptions import StreamUnavailableError
from core.interfaces.video import EncodedPacket, StreamInfo
from observability.logging.setup import get_logger

logger = get_logger(__name__)


class LocalFileInput:
    def __init__(self, path: str, loop: bool = True, source_id: str = "local_file"):
        self._path = path
        self._loop = loop
        self._source_id = source_id
        self._container = None
        self._stream = None
        self._demux_iter = None

    # -- VideoInput protocol ------------------------------------------------

    def start(self) -> None:
        file_path = resolve(self._path)
        if not file_path.exists():
            raise StreamUnavailableError(f"Video file not found: {file_path}")

        try:
            self._container = av.open(str(file_path))
            self._stream = self._container.streams.video[0]
            self._stream.thread_type = "AUTO"
        except IndexError as exc:
            self._cleanup()
            raise StreamUnavailableError(f"File contains no video stream: {file_path}") from exc
        except av.FFmpegError as exc:
            self._cleanup()
            raise StreamUnavailableError(f"Could not open video file: {file_path}") from exc

        self._demux_iter = self._container.demux(self._stream)

        logger.info(
            "stream_connected",
            source=self._source_id,
            path=str(file_path),
            codec=self._stream.codec_context.name,
        )

    def stop(self) -> None:
        self._cleanup()
        logger.info("stream_disconnected", source=self._source_id)

    def read_packet(self) -> EncodedPacket | None:
        """Return the next encoded packet, or None at end of stream.

        When `loop` is enabled the file rewinds instead of ending — the
        timestamp normalizer treats the PTS jump as a discontinuity, so the
        application timeline stays monotonic across the wrap.
        """
        if self._container is None or self._demux_iter is None:
            raise StreamUnavailableError("read_packet called before start()")

        while True:
            try:
                packet = next(self._demux_iter)
            except StopIteration:
                if not self._loop:
                    return None
                self._rewind()
                continue
            except av.FFmpegError as exc:
                raise StreamUnavailableError(f"Demux failed on {self._path}") from exc

            # PyAV emits a final flush packet with no data at end of stream.
            if packet.size == 0:
                continue

            return EncodedPacket(
                data=bytes(packet),
                pts=packet.pts,
                dts=packet.dts,
                is_keyframe=bool(packet.is_keyframe),
                stream_index=packet.stream_index,
                size_bytes=packet.size,
                raw=packet,
            )

    def get_stream_info(self) -> StreamInfo:
        if self._stream is None:
            raise StreamUnavailableError("get_stream_info called before start()")

        rate = self._stream.average_rate or self._stream.guessed_rate
        time_base = self._stream.time_base

        return StreamInfo(
            width=self._stream.codec_context.width,
            height=self._stream.codec_context.height,
            fps=float(rate) if rate else 25.0,
            codec=self._stream.codec_context.name,
            time_base=(time_base.numerator, time_base.denominator),
            source_id=self._source_id,
        )

    # -- internals ----------------------------------------------------------

    def _rewind(self) -> None:
        self._container.seek(0, stream=self._stream)
        self._demux_iter = self._container.demux(self._stream)
        logger.debug("stream_looped", source=self._source_id)

    def _cleanup(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:  # noqa: BLE001 — closing must never raise upward
                pass
        self._container = None
        self._stream = None
        self._demux_iter = None
