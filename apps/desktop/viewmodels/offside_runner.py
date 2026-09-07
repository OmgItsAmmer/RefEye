"""Runs the offside pipeline on the confirmed frame, off the UI thread (M2.7).

Until this file, `OffsidePipeline` was only ever driven by the standalone
Pipeline Inspector and the test suite — the shipped app built a whole review
panel for the verdict (`OffsideReviewPanel`) with nothing behind it to call.
This is what closes that gap: the operator confirms the contact frame in the
normal M1 review flow, and that confirmation now also triggers the M2
pipeline on exactly that frame, automatically, with no extra button.

**Why a plain thread, not the `AnalysisRequestManager`.** That manager's state
machine is `QUEUED → PREPARING → SPOTTING_ACTIONS → REFINING → RANKING` — the
five stages of *finding a candidate play*, which is a different pipeline with
a different shape (it does not know about calibration or team assignment, and
never will). Reusing its enum would mean stretching a state machine to cover
two unrelated processes; this instead follows the project's other pattern for
off-thread work — `MainViewModel._load_models` — a plain daemon `threading.Thread`
emitting Qt signals, which are safe to emit from a worker thread because Qt
queues them onto the receiver's own thread automatically.

**Why per-stage signals, not "done".** `OffsidePipeline.analyse()` takes an
`on_stage` callback that fires the instant each of the pipeline's seven stages
finishes. Forwarding that live is what lets the review screen show a
checklist filling in — Calibration, Teams, Identity, Line — rather than a
generic spinner that reveals nothing about where the five-to-ten seconds of
model inference actually went.
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QObject, Signal

from core.config.schema import AppSettings
from observability.logging.setup import get_logger
from offside.pipeline import OffsidePipeline, StageReport

logger = get_logger(__name__)


class OffsideRunner(QObject):
    """Owns one `OffsidePipeline` and runs it on demand, one frame at a time.

    One pipeline instance is kept for the life of the app rather than
    rebuilt per frame: its state (team colour model, identity tracks, the
    operator's pitch marks) is exactly the continuity M2.3/M2.4 depend on to
    get more confident the longer a clip runs, and throwing it away every
    confirm would reset that continuity for no reason.
    """

    #: A run has started, for the (frame_id).
    started = Signal(int)
    #: One stage of the pipeline just finished. Carries the `StageReport`.
    stage_progress = Signal(object)
    #: The pipeline finished this frame. Carries the `FrameAnalysis`.
    completed = Signal(object)
    #: The pipeline could not run at all (not a stage failure — those are
    #: reported through `stage_progress` and still produce a `completed`).
    failed = Signal(int, str)

    def __init__(
        self, settings: AppSettings, registry, parent: QObject | None = None
    ):
        super().__init__(parent)
        self._settings = settings
        self._registry = registry
        # Built lazily on first use: constructing it does not need the models
        # to be loaded (each stage looks its own model up from the registry
        # when it runs), but building it before `start()` would mean doing
        # file/model-registry work on the import path instead of on demand.
        self._pipeline: OffsidePipeline | None = None
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        #: Guards against a stale result landing after a newer frame was
        #: confirmed — the operator moved on, and an old verdict arriving late
        #: must not silently overwrite what they are now looking at.
        self._latest_requested_frame_id: int | None = None

    @property
    def pipeline(self) -> OffsidePipeline:
        if self._pipeline is None:
            self._pipeline = OffsidePipeline(self._settings, self._registry)
        return self._pipeline

    def analyse(self, frame_id: int, image: np.ndarray) -> None:
        """Analyse `image` (the confirmed frame) in the background.

        A run already in flight is not cancelled — the pipeline is not
        interruptible mid-stage, and an operator who confirms twice quickly
        should see the second, more relevant result once it lands, not a
        crash from two threads sharing the same tracker state. `analyse` is
        expected to be called at the pace of operator confirmations, not
        per-frame, so overlap in practice is rare.
        """
        self._latest_requested_frame_id = frame_id
        self.started.emit(frame_id)

        self._worker = threading.Thread(
            target=self._run,
            args=(frame_id, image),
            name="offside-analysis",
            daemon=True,
        )
        self._worker.start()

    def _run(self, frame_id: int, image: np.ndarray) -> None:
        try:
            frame = self._frame_packet(frame_id, image)
            with self._lock:
                # The pipeline is not thread-safe against itself (team colour
                # and identity state are mutated in place); serialising here
                # is cheap because confirms are rare compared to live frames.
                analysis = self.pipeline.analyse(
                    frame, frame_id, on_stage=self._forward_stage
                )
        except Exception as exc:  # noqa: BLE001 — reported to the operator, not raised
            logger.error("offside_analysis_failed", frame_id=frame_id, error=str(exc))
            if frame_id == self._latest_requested_frame_id:
                self.failed.emit(frame_id, "The offside pipeline could not run on this frame.")
            return

        if frame_id != self._latest_requested_frame_id:
            # Superseded by a newer confirm — drop it rather than show a
            # verdict about a frame the operator is no longer looking at.
            logger.debug("offside_analysis_superseded", frame_id=frame_id)
            return

        self.completed.emit(analysis)

    def _forward_stage(self, report: StageReport) -> None:
        self.stage_progress.emit(report)

    @staticmethod
    def _frame_packet(frame_id: int, image: np.ndarray):
        from core.domain.models import FramePacket

        height, width = image.shape[:2]
        return FramePacket(
            frame_id=frame_id,
            pts=frame_id,
            timestamp_ms=frame_id,
            capture_timestamp_ms=frame_id,
            width=width,
            height=height,
            source_id="confirmed_frame",
            image=image,
        )

