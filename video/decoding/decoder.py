"""Frame decoding: EncodedPacket -> FramePacket on a normalized timeline.

Runs on the ingest worker thread, never on the UI thread
(architecture.md section 4.1).

Frames are downscaled once, here, to the configured analysis resolution. Every
downstream consumer (preview, detection, tracking) shares that single resized
copy rather than each doing its own resize.
"""

from __future__ import annotations

import cv2

from core.domain.models import FramePacket
from core.errors.exceptions import StreamUnavailableError
from core.interfaces.video import EncodedPacket, StreamInfo
from observability.logging.setup import get_logger
from video.timestamps.normalizer import TimestampNormalizer

logger = get_logger(__name__)


class FrameDecoder:
    """Decodes packets from a PyAV-backed VideoInput into FramePackets.

    Kept separate from the input adapter so the same decoder serves any
    source that produces PyAV packets (file, RTSP, capture card).
    """

    def __init__(
        self,
        stream_info: StreamInfo,
        target_width: int,
        target_height: int,
        discontinuity_threshold_ms: int = 2000,
    ):
        self._stream_info = stream_info
        self._target_width = target_width
        self._target_height = target_height
        self._normalizer = TimestampNormalizer(
            time_base=stream_info.time_base,
            discontinuity_threshold_ms=discontinuity_threshold_ms,
        )
        self._decode_errors = 0

    @property
    def normalizer(self) -> TimestampNormalizer:
        return self._normalizer

    @property
    def decode_error_count(self) -> int:
        return self._decode_errors

    def decode(self, packet: EncodedPacket) -> list[FramePacket]:
        """Decode one encoded packet into zero or more FramePackets.

        Returns an empty list for packets that produce no output (B-frame
        reordering, or a packet the decoder cannot use). A single bad packet
        must never take down the stream (architecture.md section 48).
        """
        native = packet.raw
        if native is None:
            raise StreamUnavailableError(
                "EncodedPacket carries no native handle; decoder requires a PyAV-backed input"
            )

        try:
            av_frames = native.decode()
        except Exception as exc:  # noqa: BLE001 — decoder errors are expected, log and skip
            self._decode_errors += 1
            logger.warning(
                "decoder_error",
                error=str(exc),
                pts=packet.pts,
                total_errors=self._decode_errors,
            )
            return []

        results: list[FramePacket] = []
        for av_frame in av_frames:
            results.append(self._to_frame_packet(av_frame, packet.is_keyframe))
        return results

    def flush(self, native_stream) -> list[FramePacket]:
        """Drain frames still held inside the decoder at end of stream."""
        try:
            av_frames = list(native_stream.decode(None))
        except Exception:  # noqa: BLE001
            return []
        return [self._to_frame_packet(f, is_keyframe=False) for f in av_frames]

    def _to_frame_packet(self, av_frame, is_keyframe: bool) -> FramePacket:
        image = av_frame.to_ndarray(format="bgr24")

        if (image.shape[1], image.shape[0]) != (self._target_width, self._target_height):
            image = cv2.resize(
                image,
                (self._target_width, self._target_height),
                interpolation=cv2.INTER_AREA,
            )

        ts = self._normalizer.normalize(av_frame.pts)

        if ts.is_discontinuity:
            logger.info(
                "stream_discontinuity_detected",
                frame_id=ts.frame_id,
                segment=self._normalizer.segment_index,
            )

        return FramePacket(
            frame_id=ts.frame_id,
            pts=ts.pts,
            timestamp_ms=ts.timestamp_ms,
            capture_timestamp_ms=ts.capture_timestamp_ms,
            width=self._target_width,
            height=self._target_height,
            source_id=self._stream_info.source_id,
            image=image,
            is_keyframe=is_keyframe,
        )
