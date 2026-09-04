"""Regression: a HIGH-priority job must never be rejected because the queue
is full of LOW-priority background work.

This is the exact failure the client hit: on a slow machine, the background
CV loop submits LOW jobs faster than they drain, filling the bounded queue
within seconds. The existing gate only stopped NEW low-priority submissions
once a HIGH job was outstanding — it did nothing about LOW jobs already
sitting in the queue, so a triggered analysis still failed immediately with
SchedulerBusy("Inference queue is full").
"""

from __future__ import annotations

import threading
import time

import pytest

from ai.inference_runtime.scheduler import (
    InferencePriority,
    InferenceScheduler,
    SchedulerBusy,
)


@pytest.fixture
def blocked_scheduler():
    """A scheduler whose worker is stalled on a slow job, queue full of LOW work.

    Mirrors the real scenario: one job is "in flight" (slow CPU inference)
    while more LOW jobs pile up behind it faster than they can drain.
    """
    scheduler = InferenceScheduler(max_queue_size=4)
    release = threading.Event()

    def slow_job():
        release.wait(timeout=5.0)
        return "done"

    scheduler.start()
    # Occupy the worker so nothing drains while we fill the queue.
    scheduler.submit(slow_job, priority=InferencePriority.LOW, label="blocker")
    time.sleep(0.05)  # let the worker actually pick it up

    # Fill every remaining queue slot with LOW-priority work.
    for i in range(4):
        scheduler.submit(lambda: None, priority=InferencePriority.LOW, label=f"bg-{i}")

    yield scheduler, release
    release.set()
    scheduler.stop(timeout=2.0)


class TestQueueEviction:
    def test_high_priority_job_is_accepted_when_queue_is_full_of_low_priority(
        self, blocked_scheduler
    ):
        scheduler, release = blocked_scheduler

        # Before the fix, this raised SchedulerBusy("Inference queue is full").
        result = scheduler.submit(lambda: "analysis", priority=InferencePriority.HIGH)

        release.set()
        assert result.wait(timeout=5.0) == "analysis"

    def test_evicted_low_priority_job_reports_its_own_failure(self):
        """The dropped LOW job's caller must be told, not left hanging forever."""
        scheduler = InferenceScheduler(max_queue_size=1)
        release = threading.Event()
        scheduler.start()

        # Occupy the worker with a blocking job (leaves the queue immediately,
        # so it doesn't count against maxsize), then fill the single queue
        # slot with one LOW job.
        scheduler.submit(lambda: release.wait(5.0), priority=InferencePriority.LOW)
        time.sleep(0.05)
        low_result = scheduler.submit(lambda: None, priority=InferencePriority.LOW)

        # Queue is now full (maxsize=1, occupied by this LOW job). A HIGH
        # submission must evict it to make room for itself rather than fail.
        scheduler.submit(lambda: "analysis", priority=InferencePriority.HIGH)
        release.set()

        with pytest.raises(SchedulerBusy):
            low_result.wait(timeout=3.0)

        scheduler.stop(timeout=2.0)

    def test_high_priority_still_fails_clearly_when_truly_saturated_with_high(self):
        """Eviction only removes LOW-priority work — a queue genuinely full of
        HIGH-priority jobs must still raise, not silently drop operator-
        triggered analyses."""
        scheduler = InferenceScheduler(max_queue_size=2)
        release = threading.Event()
        scheduler.start()

        scheduler.submit(lambda: release.wait(5.0), priority=InferencePriority.HIGH)
        time.sleep(0.05)
        scheduler.submit(lambda: None, priority=InferencePriority.HIGH)
        scheduler.submit(lambda: None, priority=InferencePriority.HIGH)

        with pytest.raises(SchedulerBusy):
            scheduler.submit(lambda: None, priority=InferencePriority.HIGH)

        release.set()
        scheduler.stop(timeout=2.0)

    def test_low_priority_eviction_does_not_break_normal_operation(self):
        """The scheduler must still work correctly once busy load subsides."""
        scheduler = InferenceScheduler(max_queue_size=4)
        scheduler.start()

        results = [
            scheduler.submit(lambda i=i: i * 2, priority=InferencePriority.MEDIUM)
            for i in range(3)
        ]
        values = [r.wait(timeout=5.0) for r in results]

        scheduler.stop(timeout=2.0)
        assert values == [0, 2, 4]
