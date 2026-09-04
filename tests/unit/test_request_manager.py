import time

import pytest

from analysis.request_manager.manager import AnalysisRequestManager
from analysis.request_manager.states import (
    InvalidStateTransition,
    RequestState,
    allowed_transitions,
    can_transition,
    is_active,
    is_terminal,
)
from core.domain.models import AnalysisResult


class TestStateMachine:
    def test_happy_path_is_reachable(self):
        path = [
            RequestState.CREATED,
            RequestState.QUEUED,
            RequestState.PREPARING,
            RequestState.SPOTTING_ACTIONS,
            RequestState.REFINING,
            RequestState.RANKING,
            RequestState.COMPLETED,
        ]
        for current, target in zip(path, path[1:]):
            assert can_transition(current, target), f"{current} -> {target}"

    def test_terminal_states_have_no_exits(self):
        for state in (RequestState.COMPLETED, RequestState.FAILED, RequestState.CANCELLED):
            assert is_terminal(state)
            assert allowed_transitions(state) == frozenset()

    def test_cannot_skip_stages(self):
        assert not can_transition(RequestState.QUEUED, RequestState.RANKING)
        assert not can_transition(RequestState.CREATED, RequestState.COMPLETED)

    def test_cannot_move_backwards(self):
        assert not can_transition(RequestState.REFINING, RequestState.PREPARING)

    def test_any_active_stage_can_fail_or_cancel(self):
        for state in (
            RequestState.QUEUED,
            RequestState.PREPARING,
            RequestState.SPOTTING_ACTIONS,
            RequestState.REFINING,
            RequestState.RANKING,
        ):
            assert can_transition(state, RequestState.FAILED)
            assert can_transition(state, RequestState.CANCELLED)

    def test_active_states_drive_the_card_swap_loader(self):
        """theme.md ties the card loader to active request states, deterministically."""
        assert is_active(RequestState.SPOTTING_ACTIONS)
        assert not is_active(RequestState.COMPLETED)
        assert not is_active(RequestState.CREATED)


@pytest.fixture
def manager():
    mgr = AnalysisRequestManager(max_queue_size=2, default_window_ms=20_000)
    yield mgr
    mgr.stop(timeout=1.0)


class TestRequestManager:
    def test_submit_assigns_unique_ids(self, manager):
        a = manager.submit("shortcut", triggered_at_ms=1000)
        b = manager.submit("shortcut", triggered_at_ms=2000)
        assert a.request_id != b.request_id

    def test_submitted_request_is_queued(self, manager):
        tracked = manager.submit("shortcut", triggered_at_ms=1000)
        assert tracked.state == RequestState.QUEUED

    def test_uses_configured_default_window(self, manager):
        tracked = manager.submit("shortcut", triggered_at_ms=1000)
        assert tracked.request.requested_window_ms == 20_000

    def test_explicit_window_overrides_default(self, manager):
        tracked = manager.submit("shortcut", triggered_at_ms=1000, window_ms=5000)
        assert tracked.request.requested_window_ms == 5000

    def test_invalid_transition_raises(self, manager):
        tracked = manager.submit("shortcut", triggered_at_ms=1000)
        with pytest.raises(InvalidStateTransition):
            manager.transition(tracked.request_id, RequestState.COMPLETED)

    def test_unknown_request_id_raises(self, manager):
        with pytest.raises(KeyError):
            manager.transition("nope", RequestState.PREPARING)

    def test_repeated_transition_to_same_state_is_a_noop(self, manager):
        tracked = manager.submit("shortcut", triggered_at_ms=1000)
        manager.transition(tracked.request_id, RequestState.QUEUED)
        assert tracked.state == RequestState.QUEUED

    def test_failure_records_operator_facing_message(self, manager):
        tracked = manager.submit("shortcut", triggered_at_ms=1000)
        manager.fail(tracked.request_id, "AI analysis could not complete.")

        assert tracked.state == RequestState.FAILED
        assert "could not complete" in tracked.error_message

    def test_terminal_request_cannot_be_failed_again(self, manager):
        tracked = manager.submit("shortcut", triggered_at_ms=1000)
        manager.cancel(tracked.request_id, reason="first")
        manager.fail(tracked.request_id, "second")

        assert tracked.state == RequestState.CANCELLED
        assert tracked.error_message == "first"

    def test_state_history_is_recorded(self, manager):
        tracked = manager.submit("shortcut", triggered_at_ms=1000)
        manager.transition(tracked.request_id, RequestState.PREPARING)

        assert tracked.history == [
            RequestState.CREATED,
            RequestState.QUEUED,
            RequestState.PREPARING,
        ]

    def test_state_changes_are_published(self):
        seen = []
        mgr = AnalysisRequestManager(
            max_queue_size=2,
            default_window_ms=1000,
            on_state_change=lambda t: seen.append(t.state),
        )
        tracked = mgr.submit("shortcut", triggered_at_ms=0)
        mgr.transition(tracked.request_id, RequestState.PREPARING)
        mgr.stop(timeout=1.0)

        assert RequestState.QUEUED in seen
        assert RequestState.PREPARING in seen

    def test_a_failing_subscriber_does_not_break_the_manager(self):
        def explode(_tracked):
            raise RuntimeError("subscriber is broken")

        mgr = AnalysisRequestManager(
            max_queue_size=2, default_window_ms=1000, on_state_change=explode
        )
        tracked = mgr.submit("shortcut", triggered_at_ms=0)
        mgr.stop(timeout=1.0)

        assert tracked.state == RequestState.QUEUED

    def test_queue_overflow_cancels_oldest_not_newest(self):
        """Operator intent: the most recent trigger is the one that matters."""
        mgr = AnalysisRequestManager(max_queue_size=1, default_window_ms=1000)
        first = mgr.submit("shortcut", triggered_at_ms=0)
        second = mgr.submit("shortcut", triggered_at_ms=100)

        assert first.state == RequestState.CANCELLED
        assert second.state == RequestState.QUEUED
        mgr.stop(timeout=1.0)


class TestPipelineExecution:
    def test_pipeline_result_is_attached_to_the_request(self):
        mgr = AnalysisRequestManager(max_queue_size=4, default_window_ms=1000)

        def pipeline(request, progress):
            for state in (
                RequestState.PREPARING,
                RequestState.SPOTTING_ACTIONS,
                RequestState.REFINING,
                RequestState.RANKING,
            ):
                progress.advance(state)
            return AnalysisResult(
                request_id=request.request_id,
                candidates=[],
                selected_candidate_index=0,
                status="COMPLETED",
            )

        mgr.set_pipeline(pipeline)
        mgr.start()
        tracked = mgr.submit("shortcut", triggered_at_ms=0)

        _wait_until(lambda: tracked.state == RequestState.COMPLETED)
        mgr.stop(timeout=1.0)

        assert tracked.state == RequestState.COMPLETED
        assert tracked.result is not None
        assert tracked.duration_ms is not None

    def test_pipeline_exception_fails_the_request_without_killing_the_worker(self):
        mgr = AnalysisRequestManager(max_queue_size=4, default_window_ms=1000)

        def broken_pipeline(request, progress):
            raise RuntimeError("model exploded")

        mgr.set_pipeline(broken_pipeline)
        mgr.start()

        first = mgr.submit("shortcut", triggered_at_ms=0)
        _wait_until(lambda: first.state == RequestState.FAILED)

        # The worker must still be alive to serve the next request.
        second = mgr.submit("shortcut", triggered_at_ms=100)
        _wait_until(lambda: second.state == RequestState.FAILED)
        mgr.stop(timeout=1.0)

        assert first.state == RequestState.FAILED
        assert second.state == RequestState.FAILED
        assert "could not complete" in first.error_message


def _wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Condition not met within timeout")
