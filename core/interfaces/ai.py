"""AI adapter interfaces. See architecture.md sections 19, 59.

No code outside ai/action_spotting/<provider>/ may depend on a specific
model's tensors or schema — only on ActionSpotter and ActionCandidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.domain.models import (
    ActionCandidate,
    AnalysisRequest,
    RefinedCandidate,
)


@dataclass
class AnalysisClip:
    request_id: str
    start_frame_id: int
    end_frame_id: int
    start_timestamp_ms: int
    end_timestamp_ms: int
    frames: list  # list[FramePacket]; kept loose to avoid heavy import cycles


class ActionSpotter(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def supported_actions(self) -> set[str]: ...

    def warmup(self) -> None: ...

    def infer(self, clip: AnalysisClip) -> list[ActionCandidate]: ...


@dataclass
class RefinementContext:
    frame_buffer: object
    feature_cache: object
    window_before_frames: int
    window_after_frames: int


class ContactRefiner(Protocol):
    def refine(
        self,
        candidate: ActionCandidate,
        context: RefinementContext,
    ) -> RefinedCandidate: ...


@dataclass
class AnalysisContext:
    request: AnalysisRequest
    clip: AnalysisClip


class CandidateRanker(Protocol):
    def rank(
        self,
        candidates: list[RefinedCandidate],
        context: AnalysisContext,
    ) -> list[RefinedCandidate]: ...
