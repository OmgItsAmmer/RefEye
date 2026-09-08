"""Candidate navigation must step through the play in chronological order
(pass 1 -> pass 2 -> ... -> the shot), not AI-confidence rank order — that's
what lets the operator find the specific pass they need to review."""

from __future__ import annotations

import cv2
import numpy as np

from analysis.results.review_session import ReviewSession, ReviewStrip
from core.domain.models import AnalysisResult, RefinedCandidate

_ok, _ENCODED_FRAME = cv2.imencode(".jpg", np.zeros((4, 4, 3), dtype=np.uint8))
_ENCODED_FRAME = _ENCODED_FRAME.tobytes()


def _candidate(candidate_id: str, action_type: str, frame_id: int, score: float) -> RefinedCandidate:
    return RefinedCandidate(
        candidate_id=candidate_id,
        action_type=action_type,
        original_frame_id=frame_id,
        refined_frame_id=frame_id,
        final_score=score,
        model_score=score,
        contact_score=score,
        trajectory_score=score,
        proximity_score=score,
        temporal_score=score,
    )


def _session_with_ranked_out_of_order_candidates() -> ReviewSession:
    # Ranked best-first (as AnalysisResult.candidates arrives), but the
    # events happened in a different order during the play:
    #   index 0: shot at frame 400 (best score)
    #   index 1: pass 1 at frame 100 (earliest)
    #   index 2: pass 3 at frame 300
    #   index 3: pass 2 at frame 200
    candidates = [
        _candidate("c0", "shot", 400, 0.95),
        _candidate("c1", "pass", 100, 0.40),
        _candidate("c2", "pass", 300, 0.35),
        _candidate("c3", "pass", 200, 0.30),
    ]
    result = AnalysisResult(
        request_id="req-1",
        candidates=candidates,
        selected_candidate_index=0,
        status="completed",
    )
    strips = [
        ReviewStrip({c.refined_frame_id: _ENCODED_FRAME}, center_frame_id=c.refined_frame_id)
        for c in candidates
    ]
    return ReviewSession(request_id="req-1", result=result, strips=strips)


def test_next_candidate_walks_in_play_order_not_rank_order():
    session = _session_with_ranked_out_of_order_candidates()

    # Starts on index 0 (rank-best = the shot at frame 400), but chronological
    # navigation should start the walk from the earliest event first.
    session.select_candidate(1)  # pass at frame 100 (earliest)
    assert session.current_candidate.refined_frame_id == 100

    session.next_candidate()
    assert session.current_candidate.refined_frame_id == 200

    session.next_candidate()
    assert session.current_candidate.refined_frame_id == 300

    session.next_candidate()
    assert session.current_candidate.refined_frame_id == 400

    # Already at the last event chronologically — stays put.
    session.next_candidate()
    assert session.current_candidate.refined_frame_id == 400


def test_previous_candidate_walks_backward_in_play_order():
    session = _session_with_ranked_out_of_order_candidates()
    session.select_candidate(0)  # shot at frame 400 (last chronologically)

    session.previous_candidate()
    assert session.current_candidate.refined_frame_id == 300

    session.previous_candidate()
    assert session.current_candidate.refined_frame_id == 200

    session.previous_candidate()
    assert session.current_candidate.refined_frame_id == 100

    session.previous_candidate()
    assert session.current_candidate.refined_frame_id == 100


def test_chronological_order_exposes_play_order_for_the_alternatives_list():
    session = _session_with_ranked_out_of_order_candidates()
    # Rank order (as AnalysisResult.candidates arrives) is [shot@400, pass@100,
    # pass@300, pass@200]. The alternatives list must show them in play order.
    ordered_frame_ids = [
        session.candidates[i].refined_frame_id for i in session.chronological_order
    ]
    assert ordered_frame_ids == [100, 200, 300, 400]


def test_navigation_freezes_on_the_event_frame():
    session = _session_with_ranked_out_of_order_candidates()
    session.select_candidate(1)
    frame_id, _ = session.current_frame()
    assert frame_id == 100

    session.next_candidate()
    frame_id, _ = session.current_frame()
    assert frame_id == 200


# -- frames_before: the offside pipeline's identity warm-up window ----------


def test_frames_before_returns_only_earlier_frames_in_order():
    strip = ReviewStrip(
        {100: _ENCODED_FRAME, 105: _ENCODED_FRAME, 110: _ENCODED_FRAME, 115: _ENCODED_FRAME},
        center_frame_id=110,
    )

    frames = strip.frames_before(110)

    assert [frame_id for frame_id, _ in frames] == [100, 105]
    assert all(isinstance(image, np.ndarray) for _, image in frames)


def test_frames_before_the_earliest_frame_is_empty():
    strip = ReviewStrip({100: _ENCODED_FRAME, 105: _ENCODED_FRAME}, center_frame_id=100)

    assert strip.frames_before(100) == []


def test_frames_before_an_empty_strip_is_empty():
    strip = ReviewStrip({}, center_frame_id=100)

    assert strip.frames_before(100) == []
