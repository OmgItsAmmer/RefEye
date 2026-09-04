"""Contact refinement, de-duplication, ranking, and event-chain ordering."""

from __future__ import annotations

import math

import pytest

from ai.candidate_ranking.ranker import ScoreCandidateRanker
from ai.contact_refinement.refiner import ContactRefiner
from analysis.event_chain.chain import build_event_chain
from analysis.results.deduplication import deduplicate
from core.config.schema import (
    CandidateRankingConfig,
    ContactRefinementConfig,
    RefinementWeights,
)
from core.domain.models import (
    ActionCandidate,
    AnalysisRequest,
    FrameFeatures,
    RefinedCandidate,
    TrackObservation,
)
from core.interfaces.ai import AnalysisClip, AnalysisContext
from vision.features.feature_cache import CachedFrame, FeatureCache


def refinement_config(**overrides) -> ContactRefinementConfig:
    base = dict(
        enabled=True,
        window_before_frames=6,
        window_after_frames=6,
        proximity_radius_px=90.0,
        weights=RefinementWeights(
            model=0.4, proximity=0.2, velocity=0.15,
            direction=0.15, motion=0.05, track_consistency=0.05,
        ),
    )
    base.update(overrides)
    return ContactRefinementConfig(**base)


def build_cache(
    contact_frame: int,
    total: int = 40,
    ball_visible: bool = True,
    segment: int = 0,
    cut_at: int | None = None,
) -> FeatureCache:
    """A straight-line ball that sharply changes direction at `contact_frame`."""
    cache = FeatureCache(max_frames=200)

    for frame_id in range(total):
        if frame_id < contact_frame:
            velocity = (6.0, 0.0)
            x, y = 100 + frame_id * 6.0, 200.0
        else:
            velocity = (0.0, 6.0)   # 90-degree turn
            x = 100 + contact_frame * 6.0
            y = 200.0 + (frame_id - contact_frame) * 6.0

        is_cut = cut_at is not None and frame_id == cut_at
        current_segment = segment + (1 if cut_at is not None and frame_id >= cut_at else 0)

        features = FrameFeatures(
            frame_id=frame_id,
            timestamp_ms=frame_id * 40,
            ball_center=(x, y) if ball_visible else None,
            ball_confidence=0.8 if ball_visible else None,
            nearest_player_track_id="p1",
            ball_velocity=velocity,
            ball_speed=math.hypot(*velocity),
            scene_cut=is_cut,
        )

        # A player sitting right at the contact point.
        player = TrackObservation(
            track_id="p1",
            frame_id=frame_id,
            timestamp_ms=frame_id * 40,
            object_type="player",
            bbox_xyxy=(100 + contact_frame * 6.0 - 15, 185, 100 + contact_frame * 6.0 + 15, 245),
            center_xy=(100 + contact_frame * 6.0, 215.0),
            confidence=0.9,
        )

        cache.put(
            CachedFrame(
                features=features,
                tracks=[player],
                scene_cut=is_cut,
                segment=current_segment,
            )
        )
    return cache


def candidate(frame_id: int, action="pass", score=0.8) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=f"c{frame_id}",
        action_type=action,
        anchor_frame_id=frame_id,
        anchor_timestamp_ms=frame_id * 40,
        model_score=score,
        source_model="test:1.0",
    )


def refined(frame_id, action="pass", score=0.8, **evidence) -> RefinedCandidate:
    base_evidence = {"ball_visible": True, "refined": True}
    base_evidence.update(evidence)
    return RefinedCandidate(
        candidate_id=f"r{frame_id}-{action}",
        action_type=action,
        original_frame_id=frame_id,
        refined_frame_id=frame_id,
        final_score=score,
        model_score=score,
        contact_score=score,
        trajectory_score=0.5,
        proximity_score=0.5,
        temporal_score=0.5,
        evidence=base_evidence,
    )


class TestContactRefiner:
    def test_moves_the_anchor_towards_the_real_contact(self):
        cache = build_cache(contact_frame=20)
        refiner = ContactRefiner(refinement_config(), cache)

        # Model was 4 frames early.
        result = refiner.refine(candidate(16))

        assert result.evidence["refined"] is True
        assert abs(result.refined_frame_id - 20) < abs(16 - 20), (
            "refinement should land closer to the true contact than the anchor"
        )

    def test_records_the_shift_it_applied(self):
        cache = build_cache(contact_frame=20)
        refiner = ContactRefiner(refinement_config(), cache)
        result = refiner.refine(candidate(17))

        assert result.evidence["frame_shift"] == result.refined_frame_id - 17

    def test_keeps_anchor_when_no_features_exist(self):
        """Missing data must not fail the analysis (architecture.md section 25)."""
        refiner = ContactRefiner(refinement_config(), FeatureCache(max_frames=10))
        result = refiner.refine(candidate(500))

        assert result.refined_frame_id == 500
        assert result.evidence["refined"] is False
        assert "fallback_reason" in result.evidence

    def test_missing_ball_still_produces_a_candidate(self):
        cache = build_cache(contact_frame=20, ball_visible=False)
        refiner = ContactRefiner(refinement_config(), cache)
        result = refiner.refine(candidate(20))

        assert result is not None
        assert result.refined_frame_id > 0

    def test_search_never_crosses_a_camera_cut(self):
        """Architecture.md section 26 — a frame past a cut is a different shot."""
        cache = build_cache(contact_frame=20, cut_at=18)
        refiner = ContactRefiner(refinement_config(), cache)

        result = refiner.refine(candidate(22))
        chosen = cache.get(result.refined_frame_id)

        if result.evidence.get("refined"):
            anchor_segment = cache.get(22).segment
            assert chosen.segment == anchor_segment

    def test_reports_ball_visibility_in_evidence(self):
        cache = build_cache(contact_frame=20)
        refiner = ContactRefiner(refinement_config(), cache)
        assert refiner.refine(candidate(20)).evidence["ball_visible"] is True

    def test_keep_anchor_scores_below_a_refined_result(self):
        cache = build_cache(contact_frame=20)
        refiner = ContactRefiner(refinement_config(), cache)

        kept = refiner.keep_anchor(candidate(20, score=0.9), "no data")
        assert kept.final_score < 0.9
        assert kept.evidence["refined"] is False


class TestDeduplication:
    def test_merges_adjacent_same_action_candidates(self):
        """1201/1202/1203/1204 'pass' is one pass (architecture.md section 29)."""
        candidates = [refined(f, "pass", 0.8 - i * 0.01) for i, f in enumerate((1201, 1202, 1203, 1204))]
        result = deduplicate(candidates, max_frame_distance=8)

        assert len(result) == 1
        assert len(result[0].evidence["merged_candidate_ids"]) == 3

    def test_keeps_the_highest_scoring_member(self):
        candidates = [refined(100, "pass", 0.5), refined(102, "pass", 0.9)]
        result = deduplicate(candidates, max_frame_distance=8)

        assert len(result) == 1
        assert result[0].final_score == 0.9

    def test_keeps_distant_candidates_separate(self):
        candidates = [refined(100, "pass", 0.8), refined(400, "pass", 0.7)]
        assert len(deduplicate(candidates, max_frame_distance=8)) == 2

    def test_keeps_different_actions_at_different_frames(self):
        candidates = [refined(100, "shot", 0.8), refined(103, "goalkeeper_contact", 0.7)]
        assert len(deduplicate(candidates, max_frame_distance=8)) == 2

    def test_merges_different_actions_on_the_exact_same_frame(self):
        """The operator would otherwise page between two identical images."""
        candidates = [refined(100, "cross", 0.9), refined(100, "pass", 0.6)]
        result = deduplicate(candidates, max_frame_distance=8)

        assert len(result) == 1
        assert result[0].action_type == "cross"
        assert "pass" in result[0].evidence["alternative_actions"]

    def test_different_players_are_not_duplicates(self):
        """A block or deflection is a real second event."""
        candidates = [
            refined(100, "pass", 0.8, player_track_id="p1"),
            refined(103, "pass", 0.7, player_track_id="p2"),
        ]
        assert len(deduplicate(candidates, max_frame_distance=8)) == 2

    def test_empty_and_single_inputs_are_safe(self):
        assert deduplicate([], 8) == []
        assert len(deduplicate([refined(1)], 8)) == 1


def ranking_config(**overrides) -> CandidateRankingConfig:
    base = dict(
        max_candidates_returned=5,
        dedup_temporal_distance_frames=8,
        weight_final_score=0.6,
        weight_ball_visibility=0.15,
        weight_track_quality=0.1,
        weight_temporal_recency=0.15,
        review_frames_each_side=15,
    )
    base.update(overrides)
    return CandidateRankingConfig(**base)


def context(start_ms=0, end_ms=10_000) -> AnalysisContext:
    return AnalysisContext(
        request=AnalysisRequest(
            request_id="req", triggered_at_ms=end_ms,
            trigger_source="test", requested_window_ms=end_ms - start_ms,
        ),
        clip=AnalysisClip(
            request_id="req", start_frame_id=0, end_frame_id=100,
            start_timestamp_ms=start_ms, end_timestamp_ms=end_ms, frames=[],
        ),
    )


class TestRanking:
    def test_returns_an_ordered_list_not_one_result(self):
        """Architecture.md section 58 rule 7: never collapse to a single result."""
        ranker = ScoreCandidateRanker(ranking_config())
        result = ranker.rank([refined(10, "pass", 0.5), refined(20, "shot", 0.9)], context())

        assert len(result) == 2

    def test_higher_scores_rank_first_all_else_equal(self):
        ranker = ScoreCandidateRanker(ranking_config(weight_temporal_recency=0.0))
        result = ranker.rank(
            [
                refined(10, "pass", 0.4, timestamp_ms=1000),
                refined(20, "shot", 0.95, timestamp_ms=1000),
            ],
            context(),
        )
        assert result[0].final_score == 0.95

    def test_invisible_ball_is_penalised(self):
        ranker = ScoreCandidateRanker(ranking_config(weight_temporal_recency=0.0))
        result = ranker.rank(
            [
                refined(10, "pass", 0.7, ball_visible=False, timestamp_ms=1000),
                refined(20, "pass", 0.7, ball_visible=True, timestamp_ms=1000),
            ],
            context(),
        )
        assert result[0].evidence["ball_visible"] is True

    def test_recency_breaks_ties(self):
        """Operators trigger just after the moment they care about."""
        ranker = ScoreCandidateRanker(ranking_config())
        result = ranker.rank(
            [
                refined(10, "pass", 0.7, timestamp_ms=1000),
                refined(90, "pass", 0.7, timestamp_ms=9000),
            ],
            context(),
        )
        assert result[0].evidence["timestamp_ms"] == 9000

    def test_respects_the_configured_limit(self):
        ranker = ScoreCandidateRanker(ranking_config(max_candidates_returned=2))
        result = ranker.rank([refined(i * 10, "pass", 0.5 + i * 0.01) for i in range(6)], context())
        assert len(result) == 2

    def test_annotates_rank_position(self):
        ranker = ScoreCandidateRanker(ranking_config())
        result = ranker.rank([refined(10, "pass", 0.5), refined(20, "shot", 0.9)], context())
        assert [c.evidence["rank"] for c in result] == [0, 1]

    def test_empty_input(self):
        assert ScoreCandidateRanker(ranking_config()).rank([], context()) == []


class TestEventChain:
    def test_orders_chronologically_not_by_score(self):
        """Ranked order and chronological order answer different questions."""
        chain = build_event_chain(
            [
                refined(300, "shot", 0.9),
                refined(100, "cross", 0.4),
                refined(200, "goalkeeper_contact", 0.6),
            ]
        )
        assert chain.describe() == ["cross", "goalkeeper_contact", "shot"]

    def test_empty_chain(self):
        assert build_event_chain([]).is_empty
