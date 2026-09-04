"""Analysis request state machine. See architecture.md section 18.

Transitions are explicit and validated — an illegal transition raises rather
than silently corrupting request state, which matters because repeated hotkey
presses can race against an in-flight request.
"""

from __future__ import annotations

from enum import Enum


class RequestState(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    PREPARING = "PREPARING"
    SPOTTING_ACTIONS = "SPOTTING_ACTIONS"
    REFINING = "REFINING"
    RANKING = "RANKING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES: frozenset[RequestState] = frozenset(
    {RequestState.COMPLETED, RequestState.FAILED, RequestState.CANCELLED}
)

#: States in which the theme's card-swap loader is shown (theme.md section 6):
#: the operator is waiting on an AI decision.
ACTIVE_STATES: frozenset[RequestState] = frozenset(
    {
        RequestState.QUEUED,
        RequestState.PREPARING,
        RequestState.SPOTTING_ACTIONS,
        RequestState.REFINING,
        RequestState.RANKING,
    }
)

_ALLOWED: dict[RequestState, frozenset[RequestState]] = {
    RequestState.CREATED: frozenset({RequestState.QUEUED, RequestState.CANCELLED, RequestState.FAILED}),
    RequestState.QUEUED: frozenset(
        {RequestState.PREPARING, RequestState.CANCELLED, RequestState.FAILED}
    ),
    RequestState.PREPARING: frozenset(
        {RequestState.SPOTTING_ACTIONS, RequestState.CANCELLED, RequestState.FAILED}
    ),
    RequestState.SPOTTING_ACTIONS: frozenset(
        {RequestState.REFINING, RequestState.CANCELLED, RequestState.FAILED}
    ),
    RequestState.REFINING: frozenset(
        {RequestState.RANKING, RequestState.CANCELLED, RequestState.FAILED}
    ),
    RequestState.RANKING: frozenset(
        {RequestState.COMPLETED, RequestState.CANCELLED, RequestState.FAILED}
    ),
    RequestState.COMPLETED: frozenset(),
    RequestState.FAILED: frozenset(),
    RequestState.CANCELLED: frozenset(),
}


class InvalidStateTransition(Exception):
    def __init__(self, current: RequestState, target: RequestState):
        super().__init__(f"Cannot transition request from {current.value} to {target.value}")
        self.current = current
        self.target = target


def can_transition(current: RequestState, target: RequestState) -> bool:
    return target in _ALLOWED[current]


def allowed_transitions(current: RequestState) -> frozenset[RequestState]:
    return _ALLOWED[current]


def is_terminal(state: RequestState) -> bool:
    return state in TERMINAL_STATES


def is_active(state: RequestState) -> bool:
    return state in ACTIVE_STATES
