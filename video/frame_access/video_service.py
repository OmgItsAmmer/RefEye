"""VideoService — owns ingest, decode, and the rolling buffers.

Runs a single background ingest thread. Contains no Qt imports: the UI
subscribes via plain callbacks, which the desktop layer converts into Qt
signals (architecture.md section 33, STACK.md section 6).

Threading contract:
  * `start()` / `stop()` are called from the UI thread.
  * The ingest thread is the ONLY writer to the buffers.
  * `get_recent_clip()` / `get_frame()` are safe to call from any thread.
  * Callbacks fire on the ingest thread — implementations must not block.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from core.config.schema import AppSettings
from core.domain.models import FramePacket
from core.errors.exceptions import AnalysisWindowUnavailableError, StreamUnavailableError
from core.interfaces.ai import AnalysisClip
from core.interfaces.video import StreamInfo
from observability.logging.setup import get_logger
from video.buffering.ring_buffer import (
    DecodedRingBuffer,
    EncodedRingBuffer,
    estimate_frame_bytes,
)
from video.decoding.decoder import FrameDecoder
from video.inputs.local_file import LocalFileInput

logger = get_logger(__name__)


class StreamState(str, Enum):
    STOPPED = "stopped"
    OPENING = "opening"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class VideoStats:
    """Snapshot of ingest health, for the diagnostics panel."""

    decoded_fps: float = 0.0
    decoded_frames: int = 0
    dropped_frames: int = 0
    decode_errors: int = 0
    buffer_frames: int = 0
    buffer_duration_ms: int = 0
    buffer_bytes: int = 0
    encoded_packets: int = 0
    encoded_bytes: int = 0
    source_fps: float = 0.0
    resolution: tuple[int, int] = field(default=(0, 0))


class VideoService:
    def __init__(
        self,
        settings: AppSettings,
        on_frame: Callable[[FramePacket], None] | None = None,
        on_state_change: Callable[[StreamState, str | None], None] | None = None,
    ):
        self._settings = settings
        self._on_frame = on_frame
        self._on_state_change = on_state_change

        buffer_cfg = settings.buffer
        self._decoded_buffer = DecodedRingBuffer(
            max_frames=buffer_cfg.max_decoded_frames,
            max_duration_ms=buffer_cfg.decoded_buffer_seconds * 1000,
            max_bytes=buffer_cfg.max_decoded_megabytes * 1024 * 1024,
        )
        self._encoded_buffer = EncodedRingBuffer(max_packets=buffer_cfg.max_encoded_packets)

        self._input: LocalFileInput | None = None
        self._decoder: FrameDecoder | None = None
        self._stream_info: StreamInfo | None = None

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state = StreamState.STOPPED
        self._state_lock = threading.Lock()

        self._stats = VideoStats()
        self._stats_lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    @property
    def state(self) -> StreamState:
        with self._state_lock:
            return self._state

    @property
    def stream_info(self) -> StreamInfo | None:
        return self._stream_info

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.debug("video_service_already_running")
            return

        self._stop_event.clear()
        self._set_state(StreamState.OPENING)

        self._thread = threading.Thread(
            target=self._run,
            name="video-ingest",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        self._set_state(StreamState.STOPPED)

    # -- frame access (any thread) -----------------------------------------

    def get_frame(self, frame_id: int, quality: str = "analysis") -> FramePacket | None:
        """Fetch a buffered frame by id.

        Only 'analysis' quality is served in M1. Reconstructing an
        original-resolution frame from the encoded buffer lands with the
        review UI in a later phase; asking for it now fails loudly rather
        than silently returning a downscaled frame.
        """
        if quality != "analysis":
            raise NotImplementedError(
                f"Frame quality '{quality}' is not available yet (M1.1 serves analysis frames only)"
            )
        return self._decoded_buffer.get_by_frame(frame_id)

    def get_recent_clip(
        self,
        end_timestamp_ms: int,
        duration_ms: int,
        request_id: str = "",
    ) -> AnalysisClip:
        frames = self._decoded_buffer.get_window(end_timestamp_ms, duration_ms)

        if not frames:
            raise AnalysisWindowUnavailableError(
                f"No buffered frames in window ending at {end_timestamp_ms}ms "
                f"(duration {duration_ms}ms)"
            )

        return AnalysisClip(
            request_id=request_id,
            start_frame_id=frames[0].frame_id,
            end_frame_id=frames[-1].frame_id,
            start_timestamp_ms=frames[0].timestamp_ms,
            end_timestamp_ms=frames[-1].timestamp_ms,
            frames=frames,
        )

    def latest_frame(self) -> FramePacket | None:
        return self._decoded_buffer.latest()

    def stats(self) -> VideoStats:
        with self._stats_lock:
            snapshot = VideoStats(**vars(self._stats))
        snapshot.buffer_frames = len(self._decoded_buffer)
        snapshot.buffer_duration_ms = self._decoded_buffer.duration_ms()
        snapshot.buffer_bytes = self._decoded_buffer.total_bytes()
        snapshot.encoded_packets = len(self._encoded_buffer)
        snapshot.encoded_bytes = self._encoded_buffer.total_bytes()
        if self._decoder is not None:
            snapshot.decode_errors = self._decoder.decode_error_count
        return snapshot

    # -- ingest thread ------------------------------------------------------

    def _run(self) -> None:
        try:
            self._open_source()
        except StreamUnavailableError as exc:
            logger.error("stream_open_failed", error=str(exc))
            self._set_state(StreamState.ERROR, str(exc))
            return

        self._set_state(StreamState.RUNNING)

        try:
            self._ingest_loop()
        except StreamUnavailableError as exc:
            logger.error("stream_failed", error=str(exc))
            self._set_state(StreamState.ERROR, str(exc))
        except Exception as exc:  # noqa: BLE001 — the worker must not kill the app
            logger.exception("ingest_loop_crashed", error=str(exc))
            self._set_state(StreamState.ERROR, "Video processing stopped unexpectedly.")
        finally:
            if self._input is not None:
                self._input.stop()

    def _open_source(self) -> None:
        video_cfg = self._settings.video

        if video_cfg.input_type != "local_file":
            raise StreamUnavailableError(
                f"Input type '{video_cfg.input_type}' is not implemented in M1 "
                "(local_file only; other adapters land in a later milestone)"
            )

        self._input = LocalFileInput(
            path=video_cfg.local_file.path,
            loop=video_cfg.local_file.loop,
        )
        self._input.start()
        self._stream_info = self._input.get_stream_info()

        self._decoder = FrameDecoder(
            stream_info=self._stream_info,
            target_width=video_cfg.analysis_resolution.width,
            target_height=video_cfg.analysis_resolution.height,
        )

        with self._stats_lock:
            self._stats.source_fps = self._stream_info.fps
            self._stats.resolution = (
                video_cfg.analysis_resolution.width,
                video_cfg.analysis_resolution.height,
            )

        # Project what the configured buffer will actually cost in RAM, and
        # which of the three limits will bind first. Silent 800 MB buffers are
        # how "bounded memory" quietly becomes untrue on the client's machine.
        frame_bytes = estimate_frame_bytes(
            video_cfg.analysis_resolution.width, video_cfg.analysis_resolution.height
        )
        buffer_cfg = self._settings.buffer
        by_count = buffer_cfg.max_decoded_frames
        by_duration = int(buffer_cfg.decoded_buffer_seconds * max(self._stream_info.fps, 1.0))
        by_bytes = (buffer_cfg.max_decoded_megabytes * 1024 * 1024) // max(frame_bytes, 1)
        binding = min(
            (by_count, "max_decoded_frames"),
            (by_duration, "decoded_buffer_seconds"),
            (by_bytes, "max_decoded_megabytes"),
        )

        logger.info(
            "video_source_opened",
            source=self._stream_info.source_id,
            codec=self._stream_info.codec,
            source_resolution=f"{self._stream_info.width}x{self._stream_info.height}",
            analysis_resolution=f"{video_cfg.analysis_resolution.width}x"
            f"{video_cfg.analysis_resolution.height}",
            fps=self._stream_info.fps,
            buffer_limit_frames=binding[0],
            buffer_limit_reason=binding[1],
            buffer_projected_mb=round(binding[0] * frame_bytes / 1_048_576, 1),
            buffer_projected_seconds=round(binding[0] / max(self._stream_info.fps, 1.0), 1),
        )

    def _ingest_loop(self) -> None:
        assert self._input is not None and self._decoder is not None

        source_fps = self._stream_info.fps if self._stream_info else 25.0
        frame_interval = 1.0 / source_fps if source_fps > 0 else 0.04

        # Preview repaints are capped independently of decode rate so a
        # high-fps source cannot flood the UI thread.
        preview_interval = 1.0 / self._settings.video.preview_max_fps
        last_preview_at = 0.0

        next_frame_deadline = time.monotonic()
        fps_window_start = time.monotonic()
        fps_window_frames = 0

        while not self._stop_event.is_set():
            packet = self._input.read_packet()
            if packet is None:
                logger.info("stream_ended", source=self._stream_info.source_id)
                break

            self._encoded_buffer.append(packet)

            for frame in self._decoder.decode(packet):
                if self._stop_event.is_set():
                    return

                self._decoded_buffer.append(frame)

                with self._stats_lock:
                    self._stats.decoded_frames += 1
                fps_window_frames += 1

                # Pace to real time. A file source would otherwise decode as
                # fast as the CPU allows, which is neither watchable nor
                # representative of a live feed.
                next_frame_deadline += frame_interval
                sleep_for = next_frame_deadline - time.monotonic()
                if sleep_for > 0:
                    self._stop_event.wait(sleep_for)
                elif sleep_for < -1.0:
                    # We fell more than a second behind; resync rather than
                    # trying to catch up frame by frame.
                    next_frame_deadline = time.monotonic()

                now = time.monotonic()
                if self._on_frame is not None and now - last_preview_at >= preview_interval:
                    last_preview_at = now
                    self._emit_frame(frame)
                else:
                    with self._stats_lock:
                        self._stats.dropped_frames += 1

                elapsed = now - fps_window_start
                if elapsed >= 1.0:
                    with self._stats_lock:
                        self._stats.decoded_fps = fps_window_frames / elapsed
                    fps_window_start = now
                    fps_window_frames = 0

    def _emit_frame(self, frame: FramePacket) -> None:
        try:
            self._on_frame(frame)
        except Exception as exc:  # noqa: BLE001 — a bad subscriber must not stop ingest
            logger.warning("frame_callback_failed", error=str(exc))

    def _set_state(self, state: StreamState, message: str | None = None) -> None:
        with self._state_lock:
            if self._state == state:
                return
            self._state = state

        if self._on_state_change is not None:
            try:
                self._on_state_change(state, message)
            except Exception as exc:  # noqa: BLE001
                logger.warning("state_callback_failed", error=str(exc))
