"""Priority inference scheduler.

Architecture.md section 33 is explicit: "Do not create arbitrary parallel GPU
threads." All model work is therefore funnelled through a single worker
thread, so GPU access is serialised regardless of who submits it.

Section 34 sets the priorities:

    HIGH     triggered action analysis   — the operator is waiting
    MEDIUM   contact refinement
    LOW      continuous background feature work

Background work must never starve a triggered analysis, so LOW jobs are also
gated: while a HIGH job is queued or running, background submissions are
rejected immediately rather than queueing up behind it.
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from observability.logging.setup import get_logger

logger = get_logger(__name__)


class InferencePriority(IntEnum):
    # Lower value sorts first in a PriorityQueue.
    HIGH = 0
    MEDIUM = 1
    LOW = 2


class SchedulerBusy(Exception):
    """Raised when a low-priority job is refused because the GPU is needed."""


@dataclass(order=True)
class _Job:
    priority: int
    sequence: int
    fn: Callable[[], Any] = field(compare=False)
    future: "JobResult" = field(compare=False)
    label: str = field(compare=False, default="")


class JobResult:
    """A minimal future: the submitting thread blocks here for the outcome."""

    def __init__(self):
        self._event = threading.Event()
        self._value: Any = None
        self._error: BaseException | None = None

    def set_result(self, value: Any) -> None:
        self._value = value
        self._event.set()

    def set_error(self, error: BaseException) -> None:
        self._error = error
        self._event.set()

    def wait(self, timeout: float | None = None) -> Any:
        if not self._event.wait(timeout):
            raise TimeoutError("Inference job did not complete in time")
        if self._error is not None:
            raise self._error
        return self._value

    @property
    def done(self) -> bool:
        return self._event.is_set()


class InferenceScheduler:
    def __init__(self, max_queue_size: int = 16):
        self._queue: queue.PriorityQueue[_Job] = queue.PriorityQueue(maxsize=max_queue_size)
        self._sequence = itertools.count()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._lock = threading.Lock()
        self._high_outstanding = 0
        self._stats = {"submitted": 0, "completed": 0, "failed": 0, "rejected": 0}

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run, name="inference", daemon=True)
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=timeout)
        self._worker = None

    def submit(
        self,
        fn: Callable[[], Any],
        priority: InferencePriority = InferencePriority.MEDIUM,
        label: str = "",
    ) -> JobResult:
        """Queue work for the inference thread.

        Raises SchedulerBusy for LOW-priority work while a triggered analysis
        is in flight — background feature extraction is always droppable, and
        dropping it is better than delaying the operator.
        """
        # Without a running worker the job would sit in the queue until the
        # caller's timeout expires. Failing immediately turns a mysterious
        # 60-second stall into an actionable error.
        if self._worker is None or not self._worker.is_alive():
            raise SchedulerBusy("Inference scheduler is not running")

        with self._lock:
            if priority == InferencePriority.LOW and self._high_outstanding > 0:
                self._stats["rejected"] += 1
                raise SchedulerBusy("Deferring background work: triggered analysis in progress")
            if priority == InferencePriority.HIGH:
                self._high_outstanding += 1

        result = JobResult()
        job = _Job(
            priority=int(priority),
            sequence=next(self._sequence),
            fn=fn,
            future=result,
            label=label,
        )

        try:
            self._queue.put_nowait(job)
        except queue.Full:
            if priority == InferencePriority.HIGH and self._evict_one_low_priority_job():
                # The gate on new LOW submissions above only stops jobs that
                # haven't been queued yet — it does nothing about LOW jobs
                # already sitting in the queue from before the trigger fired.
                # On a slow (e.g. CPU) machine those 16 slots fill within a
                # couple of seconds of background CV alone, so without this
                # eviction every triggered analysis would fail with "queue is
                # full" the moment the background loop got ahead of itself.
                try:
                    self._queue.put_nowait(job)
                except queue.Full:
                    pass
                else:
                    with self._lock:
                        self._stats["submitted"] += 1
                    return result

            with self._lock:
                if priority == InferencePriority.HIGH:
                    self._high_outstanding -= 1
                self._stats["rejected"] += 1
            raise SchedulerBusy("Inference queue is full") from None

        with self._lock:
            self._stats["submitted"] += 1

        return result

    def _evict_one_low_priority_job(self) -> bool:
        """Drop the queue's single lowest-priority job to make room.

        `queue.PriorityQueue` has no peek/remove-arbitrary API, so this drains
        the whole queue and re-inserts everything except the worst LOW-priority
        job found. Only ever called when the queue is already full and a HIGH
        job needs a slot, so this runs at most once per trigger, not per frame.
        """
        drained: list[_Job] = []
        try:
            while True:
                drained.append(self._queue.get_nowait())
        except queue.Empty:
            pass

        evict_index = None
        for i, job in enumerate(drained):
            if job.priority == int(InferencePriority.LOW):
                evict_index = i
                break

        if evict_index is not None:
            evicted = drained.pop(evict_index)
            evicted.future.set_error(
                SchedulerBusy("Dropped: superseded by a triggered analysis")
            )

        for job in drained:
            self._queue.put_nowait(job)

        with self._lock:
            self._stats["rejected"] += 1

        return evict_index is not None

    def stats(self) -> dict[str, int]:
        with self._lock:
            snapshot = dict(self._stats)
        snapshot["queued"] = self._queue.qsize()
        return snapshot

    # -- worker -------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            started = time.perf_counter()
            try:
                job.future.set_result(job.fn())
                with self._lock:
                    self._stats["completed"] += 1
            except Exception as exc:  # noqa: BLE001 — never kill the worker
                logger.warning("inference_job_failed", label=job.label, error=str(exc))
                job.future.set_error(exc)
                with self._lock:
                    self._stats["failed"] += 1
            finally:
                if job.priority == int(InferencePriority.HIGH):
                    with self._lock:
                        self._high_outstanding = max(0, self._high_outstanding - 1)

                duration_ms = int((time.perf_counter() - started) * 1000)
                if duration_ms > 500:
                    logger.debug(
                        "inference_job_slow", label=job.label, duration_ms=duration_ms
                    )
