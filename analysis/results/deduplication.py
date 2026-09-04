"""Candidate de-duplication.

Architecture.md section 29: models emit several nearby detections for one
real action. Four consecutive "pass" frames are one pass, not four candidates
for the operator to page through.

Clustering key, per the architecture: temporal distance + action type +
player track. Two candidates merge when they are close in time AND describe
the same action AND (where both are known) involve the same player. The
survivor is the highest-scoring member, so merging never degrades the best
frame estimate — it only removes near-duplicates of it.
"""

from __future__ import annotations

from core.domain.models import RefinedCandidate
from observability.logging.setup import get_logger

logger = get_logger(__name__)


def deduplicate(
    candidates: list[RefinedCandidate],
    max_frame_distance: int,
) -> list[RefinedCandidate]:
    """Collapse near-duplicate candidates, keeping the strongest of each cluster."""
    if len(candidates) <= 1:
        return list(candidates)

    # Strongest first, so each cluster is seeded by its best member and
    # weaker neighbours attach to it rather than the reverse.
    ordered = sorted(candidates, key=lambda c: c.final_score, reverse=True)

    survivors: list[RefinedCandidate] = []
    absorbed = 0

    for candidate in ordered:
        duplicate_of = next(
            (
                survivor
                for survivor in survivors
                if _is_duplicate(candidate, survivor, max_frame_distance)
            ),
            None,
        )

        if duplicate_of is None:
            survivors.append(candidate)
            continue

        absorbed += 1
        merged = duplicate_of.evidence.setdefault("merged_candidate_ids", [])
        merged.append(candidate.candidate_id)
        # Record the spread so the operator can see how wide the cluster was.
        duplicate_of.evidence["cluster_frame_span"] = max(
            duplicate_of.evidence.get("cluster_frame_span", 0),
            abs(candidate.refined_frame_id - duplicate_of.refined_frame_id),
        )

    if absorbed:
        logger.debug(
            "candidates_deduplicated",
            before=len(candidates),
            after=len(survivors),
            absorbed=absorbed,
        )

    return survivors


def _is_duplicate(
    candidate: RefinedCandidate,
    survivor: RefinedCandidate,
    max_frame_distance: int,
) -> bool:
    # Two candidates that refined onto the exact same frame are one candidate
    # from the operator's point of view — they would page between them and see
    # an identical image. Merge regardless of action label, keeping the
    # stronger one's label and recording the alternative so the information
    # is not lost.
    if candidate.refined_frame_id == survivor.refined_frame_id:
        if candidate.action_type != survivor.action_type:
            alternatives = survivor.evidence.setdefault("alternative_actions", [])
            if candidate.action_type not in alternatives:
                alternatives.append(candidate.action_type)
        return True

    if candidate.action_type != survivor.action_type:
        return False

    if abs(candidate.refined_frame_id - survivor.refined_frame_id) > max_frame_distance:
        return False

    # Same moment, same action, but demonstrably different players: that is a
    # genuine second event (a block, a deflection), not a duplicate.
    candidate_track = candidate.evidence.get("player_track_id")
    survivor_track = survivor.evidence.get("player_track_id")
    if candidate_track and survivor_track and candidate_track != survivor_track:
        return False

    return True
