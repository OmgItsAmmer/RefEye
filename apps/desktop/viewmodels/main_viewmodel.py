"""MainViewModel — the single bridge between worker threads and the UI.

Background services (VideoService, ModelRegistry, AnalysisRequestManager)
know nothing about Qt; they publish through plain callbacks. This class turns
those callbacks into Qt signals, which Qt delivers to the UI thread as queued
connections.

That gives us the event model architecture.md section 32 asks for: the UI
reacts to signals and never reaches into worker internals to poll mutable
state.

Model loading happens on its own thread so a cold CUDA context or a large
checkpoint never blocks the window from opening (sections 38, 49).
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal

from ai.inference_runtime.scheduler import InferenceScheduler
from ai.model_registry.registry import ModelRegistry, ModelStatus
from analysis.pipelines.analysis_pipeline import AnalysisPipeline
from analysis.request_manager.manager import AnalysisRequestManager, TrackedRequest
from analysis.request_manager.states import RequestState, is_active
from core.config.schema import AppSettings
from core.domain.models import FramePacket
from apps.desktop.viewmodels.offside_runner import OffsideRunner
from observability.logging.setup import get_logger
from video.frame_access.video_service import StreamState, VideoService, VideoStats
from vision.features.feature_cache import FeatureCache
from vision.live_pipeline import LiveCVPipeline

logger = get_logger(__name__)


class MainViewModel(QObject):
    # Stream — camera 1's raw feed (always decoded/displayed).
    frame_ready = Signal(object)               # FramePacket
    stream_state_changed = Signal(str, str)    # StreamState value, message

    # Cameras 2-4 on Live Grid: their raw feeds (video.preview_cameras).
    camera_frame_ready = Signal(int, object)          # camera_index (2..4), FramePacket
    camera_stream_state_changed = Signal(int, str, str)  # camera_index, StreamState value, message

    # All four cameras decode and display continuously, but only one is ever
    # "focused" at a time: detection/tracking and the triggered "Analyse
    # recent play" pipeline both run against the focused camera only (one
    # analysis pipeline, architecture.md section 5) — CPU/GPU cost stays
    # bounded to a single camera's worth of inference regardless of how many
    # tiles are visible. Selecting a camera on Live Grid moves the focus.
    focused_camera_changed = Signal(int)       # camera_index (1..4)

    # Models
    model_state_changed = Signal(str, str)     # ModelStatus value, message

    # Analysis
    analysis_state_changed = Signal(str, str)  # request_id, RequestState value
    analysis_busy_changed = Signal(bool)       # drives the card-swap loader
    analysis_completed = Signal(str, object)   # request_id, AnalysisResult
    analysis_failed = Signal(str, str)         # request_id, operator-facing message

    # Diagnostics
    stats_updated = Signal(object)             # VideoStats

    def __init__(self, settings: AppSettings, parent: QObject | None = None):
        super().__init__(parent)
        self._settings = settings
        self._busy = False

        self._feature_cache = FeatureCache(max_frames=settings.ai.feature_cache.max_frames)
        self._scheduler = InferenceScheduler(
            max_queue_size=settings.runtime.inference_scheduler.max_queue_size
        )
        self._registry = ModelRegistry(settings, self._feature_cache)
        self._live_pipeline: LiveCVPipeline | None = None
        self._focused_camera = 1

        self._video = VideoService(
            settings=settings,
            on_frame=self._handle_frame,
            on_state_change=self._handle_stream_state,
        )
        self._preview_services: dict[int, VideoService] = self._build_preview_services(settings)

        self._requests = AnalysisRequestManager(
            max_queue_size=settings.runtime.inference_scheduler.max_queue_size,
            default_window_ms=settings.buffer.recent_window_seconds * 1000,
            on_state_change=self._handle_request_state,
        )

        self._model_thread: threading.Thread | None = None

        # M2.7: one pipeline instance for the life of the app, shared across
        # every confirmed frame — see OffsideRunner's own docstring for why a
        # plain thread rather than the request manager's state machine.
        self._offside = OffsideRunner(settings, self._registry)

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self._scheduler.start()
        self._requests.start()
        self._video.start()
        for service in self._preview_services.values():
            service.start()

        # Loading weights can take seconds. Do it off the UI thread so the
        # window is live and playing video while the AI warms up.
        self._model_thread = threading.Thread(
            target=self._load_models, name="model-loader", daemon=True
        )
        self._model_thread.start()

    def shutdown(self) -> None:
        self._video.stop()
        for service in self._preview_services.values():
            service.stop()
        self._requests.stop()
        self._scheduler.stop()

    def _build_preview_services(self, settings: AppSettings) -> dict[int, VideoService]:
        """Cameras 2-4, each decoded from its own file. Buffer sizing matches
        camera 1's exactly (not a smaller "preview-only" buffer) — any of the
        four can become the focused/analyzed camera, and `get_recent_clip`
        must have a full window ready the instant that happens, not after a
        refill delay."""
        services: dict[int, VideoService] = {}
        for offset, camera_cfg in enumerate(settings.video.preview_cameras[:3]):
            index = offset + 2  # cameras 2, 3, 4
            camera_settings = settings.model_copy(deep=True)
            camera_settings.video.local_file = camera_cfg

            services[index] = VideoService(
                settings=camera_settings,
                on_frame=lambda frame, i=index: self._handle_preview_frame(i, frame),
                on_state_change=lambda state, message, i=index: self._handle_preview_state(
                    i, state, message
                ),
            )
        return services

    def _load_models(self) -> None:
        try:
            state = self._registry.load_all()
        except Exception as exc:  # noqa: BLE001 — degrade, never crash
            logger.exception("model_load_crashed", error=str(exc))
            self.model_state_changed.emit(
                ModelStatus.UNAVAILABLE.value,
                "AI analysis is unavailable. Live video is unaffected.",
            )
            return

        if self._registry.is_ready:
            self._live_pipeline = LiveCVPipeline(
                detector=self._registry.get_detector(),
                tracker=self._registry.get_tracker(),
                ball_tracker=self._registry.get_ball_tracker(),
                feature_cache=self._feature_cache,
                scheduler=self._scheduler,
                frame_stride=self._settings.ai.detector.live_frame_stride,
            )
            self._attach_analysis_pipeline(self._video_for_camera(self._focused_camera))

        self.model_state_changed.emit(state.status.value, state.message)

    def _attach_analysis_pipeline(self, video_service: VideoService) -> None:
        self._requests.set_pipeline(
            AnalysisPipeline(
                settings=self._settings,
                video_service=video_service,
                registry=self._registry,
                feature_cache=self._feature_cache,
                scheduler=self._scheduler,
                live_pipeline=self._live_pipeline,
            )
        )

    def _video_for_camera(self, index: int) -> VideoService:
        return self._video if index == 1 else self._preview_services[index]

    def video_for_camera(self, index: int) -> VideoService:
        """Public accessor so UI widgets (e.g. the recent-clip preview) can
        follow the focused camera instead of being pinned to camera 1's feed."""
        return self._video_for_camera(index)

    # -- commands (called from the UI thread) -------------------------------

    def set_focused_camera(self, index: int) -> None:
        """Move detection/tracking and the "Analyse recent play" pipeline to
        a different camera (Live Grid camera/Best selection).

        Only one camera is ever detected/tracked at a time — that's the
        whole point of "focus" here, keeping inference cost to one camera's
        worth regardless of how many tiles are on screen. The old camera's
        tracker state is meaningless for the new footage, same as a scene
        cut (architecture.md section 26), so it's reset rather than carried
        over; the feature cache is cleared so no stale boxes from the old
        camera linger in the UI until fresh ones arrive.
        """
        if index == self._focused_camera or index not in (1, 2, 3, 4):
            return
        if index != 1 and index not in self._preview_services:
            return  # no camera configured for this slot — nothing to focus

        self._focused_camera = index
        self._feature_cache.clear()

        if self._live_pipeline is not None:
            self._registry.get_tracker().reset()
            self._registry.get_ball_tracker().reset()
            self._attach_analysis_pipeline(self._video_for_camera(index))

        self.focused_camera_changed.emit(index)

    @property
    def focused_camera(self) -> int:
        return self._focused_camera

    def trigger_analysis(self, trigger_source: str = "shortcut") -> None:
        """Handle the analyze hotkey.

        The trigger timestamp comes from the normalized video timeline, not
        the wall clock, so the analysis window is reproducible
        (architecture.md section 11). Always the *focused* camera's
        timeline — that's the one the analysis pipeline is currently
        attached to (`set_focused_camera`).
        """
        latest = self._video_for_camera(self._focused_camera).latest_frame()
        if latest is None:
            self.analysis_failed.emit(
                "",
                "No video buffered yet. Wait for playback to start, then try again.",
            )
            return

        if not self._registry.is_ready:
            self.analysis_failed.emit(
                "",
                self._registry.state.message
                or "AI is still loading. Please try again in a moment.",
            )
            return

        self._requests.submit(
            trigger_source=trigger_source,
            triggered_at_ms=latest.timestamp_ms,
        )

    def refresh_stats(self) -> None:
        self.stats_updated.emit(self._video.stats())

    # -- accessors ----------------------------------------------------------

    @property
    def video(self) -> VideoService:
        return self._video

    @property
    def requests(self) -> AnalysisRequestManager:
        return self._requests

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    @property
    def offside(self) -> OffsideRunner:
        return self._offside

    def check_offside(
        self,
        frame_id: int,
        image,
        warm_up_frames: list[tuple[int, object]] | None = None,
    ) -> None:
        """Run the M2 pipeline on a confirmed frame — the operator's trigger
        for it. Automatic on confirm, not automatic on every frame: the
        pipeline reads model checkpoints and mutates team/identity state, and
        running it continuously would fight the live preview for the GPU for
        no benefit — an offside call is only ever asked about the one frame
        the operator confirmed.

        `warm_up_frames` (oldest first, strictly before `frame_id`) let
        identity tracking and kit colours build real continuity before the
        confirmed frame is judged — see `OffsideRunner.analyse` and
        `OffsidePipeline.warm_up`."""
        self._offside.analyse(frame_id, image, warm_up_frames=warm_up_frames)

    @property
    def feature_cache(self) -> FeatureCache:
        return self._feature_cache

    def stats(self) -> VideoStats:
        return self._video.stats()

    def pipeline_stats(self) -> dict:
        stats: dict = {"scheduler": self._scheduler.stats()}
        if self._live_pipeline is not None:
            stats["live_cv"] = self._live_pipeline.stats()
        return stats

    def get_frame(self, frame_id: int):
        return self._video.get_frame(frame_id)

    def take_review_session(self):
        """Hand the UI the frozen frames from the most recent analysis.

        Consumed rather than left in place, so a stale session from a previous
        request can never be shown against a newer result.
        """
        pipeline = self._requests.pipeline
        if pipeline is None:
            return None
        session = getattr(pipeline, "last_review_session", None)
        if session is not None:
            pipeline.last_review_session = None
        return session

    # -- worker-thread callbacks (must not block) ---------------------------

    def _handle_frame(self, frame: FramePacket) -> None:
        # Background CV first (it only submits a job), then the UI. Only
        # when camera 1 is the focused camera — detection runs on one
        # camera at a time (class docstring's `focused_camera_changed`).
        if self._live_pipeline is not None and self._focused_camera == 1:
            self._live_pipeline.on_frame(frame)
        self.frame_ready.emit(frame)

    def _handle_stream_state(self, state: StreamState, message: str | None) -> None:
        self.stream_state_changed.emit(state.value, message or "")

    def _handle_preview_frame(self, index: int, frame: FramePacket) -> None:
        # Same as _handle_frame: only the focused camera gets processed.
        if self._live_pipeline is not None and self._focused_camera == index:
            self._live_pipeline.on_frame(frame)
        self.camera_frame_ready.emit(index, frame)

    def _handle_preview_state(self, index: int, state: StreamState, message: str | None) -> None:
        self.camera_stream_state_changed.emit(index, state.value, message or "")

    def _handle_request_state(self, tracked: TrackedRequest) -> None:
        state = tracked.state
        self.analysis_state_changed.emit(tracked.request_id, state.value)

        busy = is_active(state)
        if busy != self._busy:
            self._busy = busy
            self.analysis_busy_changed.emit(busy)

        if state == RequestState.COMPLETED and tracked.result is not None:
            self.analysis_completed.emit(tracked.request_id, tracked.result)
        elif state == RequestState.FAILED:
            self.analysis_failed.emit(
                tracked.request_id,
                tracked.error_message
                or "AI analysis could not complete. The live video is still available.",
            )
