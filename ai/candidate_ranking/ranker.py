"""Candidate ranking.

Architecture.md section 30. Returns an *ordered list*, never a single result:
the operator must always be able to reach the alternatives (section 4.6,
rule 7 in section 58).

The blend combines the refined contact score with signals about how much the
evidence can be trusted — ball visibility, tracking quality — and a mild
recency preference, because the operator triggers analysis just after the
moment of interest, so later events in the window are usually the ones they
mean. All weights come from config.
"""

from __future__ import annotations

from core.config.schema import CandidateRankingConfig
from core.domain.models import RefinedCandidate
from core.interfaces.ai import AnalysisContext
from observability.logging.setup import get_logger

logger = get_logger(__name__)


class ScoreCandidateRanker:
    def __init__(self, config: CandidateRankingConfig):
        self._config = config

    def rank(
        self,
        candidates: list[RefinedCandidate],
        context: AnalysisContext,
    ) -> list[RefinedCandidate]:
        if not candidates:
            return []

        clip = context.clip
        window_start = clip.start_timestamp_ms
        window_end = max(clip.end_timestamp_ms, window_start + 1)
        span = window_end - window_start

        scored: list[tuple[float, RefinedCandidate]] = []
        for candidate in candidates:
            visibility = 1.0 if candidate.evidence.get("ball_visible") else 0.5
            track_quality = candidate.evidence.get("track_consistency")
            if track_quality is None:
                track_quality = candidate.proximity_score

            anchor_ms = candidate.evidence.get("timestamp_ms")
            if anchor_ms is None:
                recency = 0.5
            else:
                recency = max(0.0, min(1.0, (anchor_ms - window_start) / span))

            rank_score = (
                self._config.weight_final_score * candidate.final_score
                + self._config.weight_ball_visibility * visibility
                + self._config.weight_track_quality * float(track_quality)
                + self._config.weight_temporal_recency * recency
            )

            candidate.evidence["rank_score"] = round(rank_score, 4)
            scored.append((rank_score, candidate))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        ranked = [candidate for _, candidate in scored]

        for position, candidate in enumerate(ranked):
            candidate.evidence["rank"] = position

        limited = ranked[: self._config.max_candidates_returned]

        logger.debug(
            "candidates_ranked",
            total=len(candidates),
            returned=len(limited),
            top_score=round(scored[0][0], 4) if scored else None,
            top_action=limited[0].action_type if limited else None,
        )
        return limited
