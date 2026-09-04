"""AnalysisRequestManager — queues, tracks, and reports analysis requests.

M1.1 scope: the full request lifecycle runs, but no AI executes behind it.
The pipeline hook (`set_pipeline`) is where M1.2 plugs real inference in;
until then requests walk the state machine and complete with an empty result.

Repeated hotkey presses must never corrupt shared state (architecture.md
section 17), so the manager is fully lock-guarded and each request is
independently tracked by id.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from core.domain.models import AnalysisRequest, AnalysisResult
from observability.logging.setup import bind_request, get_logger
from analysis.request_manager.states import (
    InvalidStateTransition,
    RequestState,
    can_transition,
    is_terminal,
)

logger = get_logger(__name__)

#: Signature of the pluggable analysis pipeline (wired in M1.2).
#: Receives the request and a progress reporter; returns the finished result.
PipelineFn = Callable[[AnalysisRequest, "ProgressReporter"], AnalysisResult]


@dataclass
class TrackedRequest:
    request: AnalysisRequest
    state: RequestState = RequestState.CREATED
    result: AnalysisResult | None = None
    error_message: str | None = None
    created_at_ms: int = 0
    completed_at_ms: int | None = None
    history: list[RequestState] = field(default_factory=list)

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def duration_ms(self) -> int | None:
        if self.completed_at_ms is None:
            return None
        return self.completed_at_ms - self.created_at_ms


class ProgressReporter:
    """Handed to the pipeline so it can advance state without owning the manager."""

    def __init__(self, manager: AnalysisRequestManager, request_id: str):
        self._manager = manager
        self._request_id = request_id

    def advance(self, state: RequestState) -> None:
        self._manager.transition(self._request_id, state)

    def is_cancelled(self) -> bool:
        tracked = self._manager.get(self._request_id)
        return tracked is not None and tracked.state == RequestState.CANCELLED


class AnalysisRequestManager:
    """Owns request lifecycle and a bounded worker queue.

    The queue is bounded (architecture.md section 35): if the operator spams
    the hotkey faster than analysis completes, the oldest queued-but-unstarted
    request is dropped rather than growing the queue without limit. An
    in-flight request is never dropped — operator intent is preserved.
    """

    def __init__(
        self,
        max_queue_size: int,
        default_window_ms: int,
        on_state_change: Callable[[TrackedRequest], None] | None = None,
        history_limit: int = 50,
    ):
        self._default_window_ms = default_window_ms
        self._on_state_change = on_state_change
        self._history_limit = history_limit

        self._queue: queue.Queue[str] = queue.Queue(maxsize=max_queue_size)
        self._requests: dict[str, TrackedRequest] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()

        self._pipeline: PipelineFn | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()

    # -- wiring -------------------------------------------------------------

    def set_pipeline(self, pipeline: PipelineFn) -> None:
        """Install the analysis pipeline. Called once at startup (M1.2)."""
        self._pipeline = pipeline

    @property
    def pipeline(self) -> PipelineFn | None:
        return self._pipeline

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run, name="analysis-worker", daemon=True)
        self._worker.start()

    def stop(self, timeout: float = 3.0) -> None:
        # The worker polls with a timeout rather than waiting on a poison pill,
        # because a full queue would make the pill itself un-enqueueable.
        self._stop_event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=timeout)
        self._worker = None

    # -- request lifecycle --------------------------------------------------

    def submit(
        self,
        trigger_source: str,
        triggered_at_ms: int,
        window_ms: int | None = None,
        request_type: str = "general",
    ) -> TrackedRequest:
        request = AnalysisRequest(
            request_id=uuid.uuid4().hex[:12],
            triggered_at_ms=triggered_at_ms,
            trigger_source=trigger_source,
            requested_window_ms=window_ms or self._default_window_ms,
            request_type=request_type,
        )

        tracked = TrackedRequest(
            request=request,
            created_at_ms=int(time.monotonic() * 1000),
            history=[RequestState.CREATED],
        )

        with self._lock:
            self._requests[request.request_id] = tracked
            self._order.append(request.request_id)
            self._prune_history_locked()

        with bind_request(request.request_id):
            logger.info(
                "analysis_requested",
                trigger_source=trigger_source,
                triggered_at_ms=triggered_at_ms,
                window_ms=request.requested_window_ms,
            )

        self._notify(tracked)
        self._enqueue(tracked)
        return tracked

    def _enqueue(self, tracked: TrackedRequest) -> None:
        self.transition(tracked.request_id, RequestState.QUEUED)

        while True:
            try:
                self._queue.put_nowait(tracked.request_id)
                return
            except queue.Full:
                dropped_id = self._drop_oldest_queued()
                if dropped_id is None:
                    # Nothing droppable; refuse the new request rather than block.
                    self.fail(tracked.request_id, "Analysis queue is busy. Please retry.")
                    return

    def _drop_oldest_queued(self) -> str | None:
        try:
            oldest_id = self._queue.get_nowait()
        except queue.Empty:
            return None

        with bind_request(oldest_id):
            logger.warning("analysis_request_dropped", reason="queue_full")
        self.cancel(oldest_id, reason="Superseded by a newer analysis request")
        return oldest_id

    def transition(self, request_id: str, target: RequestState) -> TrackedRequest:
        with self._lock:
            tracked = self._requests.get(request_id)
            if tracked is None:
                raise KeyError(f"Unknown request_id: {request_id}")

            if tracked.state == target:
                return tracked

            if not can_transition(tracked.state, target):
                raise InvalidStateTransition(tracked.state, target)

            tracked.state = target
            tracked.history.append(target)
            if is_terminal(target):
                tracked.completed_at_ms = int(time.monotonic() * 1000)

        with bind_request(request_id):
            logger.debug("analysis_state_changed", state=target.value)

        self._notify(tracked)
        return tracked

    def cancel(self, request_id: str, reason: str = "Cancelled") -> None:
        with self._lock:
            tracked = self._requests.get(request_id)
            if tracked is None or is_terminal(tracked.state):
                return
            tracked.error_message = reason

        try:
            self.transition(request_id, RequestState.CANCELLED)
        except InvalidStateTransition:
            pass

    def fail(self, request_id: str, message: str) -> None:
        with self._lock:
            tracked = self._requests.get(request_id)
            if tracked is None or is_terminal(tracked.state):
                return
            tracked.error_message = message

        with bind_request(request_id):
            logger.error("analysis_failed", reason=message)

        try:
            self.transition(request_id, RequestState.FAILED)
        except InvalidStateTransition:
            pass

    def get(self, request_id: str) -> TrackedRequest | None:
        with self._lock:
            return self._requests.get(request_id)

    def latest(self) -> TrackedRequest | None:
        with self._lock:
            if not self._order:
                return None
            return self._requests.get(self._order[-1])

    def pending_count(self) -> int:
        return self._queue.qsize()

    # -- worker -------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                request_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if self._stop_event.is_set() or not request_id:
                return

            tracked = self.get(request_id)
            if tracked is None or is_terminal(tracked.state):
                continue

            self._execute(tracked)

    def _execute(self, tracked: TrackedRequest) -> None:
        request_id = tracked.request_id

        with bind_request(request_id):
            logger.info("analysis_started", trigger_source=tracked.request.trigger_source)
            started_ms = time.monotonic() * 1000

            try:
                if self._pipeline is None:
                    result = self._stub_result(tracked)
                else:
                    result = self._pipeline(tracked.request, ProgressReporter(self, request_id))

                with self._lock:
                    tracked.result = result

                self.transition(request_id, RequestState.COMPLETED)
                logger.info(
                    "analysis_completed",
                    candidate_count=len(result.candidates),
                    duration_ms=int(time.monotonic() * 1000 - started_ms),
                )
            except Exception as exc:  # noqa: BLE001 — surface, never crash the worker
                logger.exception("analysis_pipeline_error", error=str(exc))
                self.fail(
                    request_id,
                    "AI analysis could not complete. The live video is still available.",
                )

    def _stub_result(self, tracked: TrackedRequest) -> AnalysisResult:
        """M1.1 placeholder: walk the real states, return no candidates.

        Replaced wholesale by the real pipeline in M1.2 — this exists so the
        UI event flow and state machine can be exercised end-to-end before any
        model is integrated.
        """
        for state in (
            RequestState.PREPARING,
            RequestState.SPOTTING_ACTIONS,
            RequestState.REFINING,
            RequestState.RANKING,
        ):
            if self._stop_event.is_set():
                break
            self.transition(tracked.request_id, state)
            time.sleep(0.35)  # visible progression for the loader; removed in M1.2

        return AnalysisResult(
            request_id=tracked.request_id,
            candidates=[],
            selected_candidate_index=0,
            status=RequestState.COMPLETED.value,
            warnings=["AI pipeline not yet integrated (Phase M1.1 scaffold)."],
            diagnostics={"pipeline": "stub"},
        )

    # -- internals ----------------------------------------------------------

    def _prune_history_locked(self) -> None:
        while len(self._order) > self._history_limit:
            oldest = self._order.pop(0)
            tracked = self._requests.get(oldest)
            if tracked is not None and not is_terminal(tracked.state):
                # Never evict a live request; put it back and stop pruning.
                self._order.insert(0, oldest)
                return
            self._requests.pop(oldest, None)

    def _notify(self, tracked: TrackedRequest) -> None:
        if self._on_state_change is None:
            return
        try:
            self._on_state_change(tracked)
        except Exception as exc:  # noqa: BLE001
            logger.warning("request_state_callback_failed", error=str(exc))
