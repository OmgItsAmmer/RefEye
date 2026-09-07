"""Choosing the real ball from the detector's raw output.

Measured on client footage: more than one "ball" detection appeared on 4 of
every 10 sampled frames, and detected sizes reached 31% of the median
player's height on screen when a real football is ~12-13%. Trusting whichever
detection scored highest was routinely wrong. These tests protect the fix —
a size gate against every player's box on the same frame — and, just as
importantly, protect the refusal: when nothing plausible survives the gate,
this must return None, not the least-bad guess.
"""

from __future__ import annotations

from core.domain.models import Detection
from offside.ball_selection import select_ball
from vision.detection.classes import BALL, PLAYER


def player(x1: float, y1: float, x2: float, y2: float, confidence: float = 0.9) -> Detection:
    return Detection(
        frame_id=1, class_name=PLAYER, confidence=confidence, bbox_xyxy=(x1, y1, x2, y2),
        source_model="test",
    )


def ball(x1: float, y1: float, x2: float, y2: float, confidence: float = 0.5) -> Detection:
    return Detection(
        frame_id=1, class_name=BALL, confidence=confidence, bbox_xyxy=(x1, y1, x2, y2),
        source_model="test",
    )


# A "typical" player box: 30px wide, 90px tall (height is what the size
# reference is built from). A plausible ball at the default 0.06-0.22 band is
# roughly 5-20px across.
PLAYERS = [player(100, 10, 130, 100), player(300, 20, 330, 110), player(500, 15, 530, 105)]


def test_no_ball_detections_at_all():
    assert select_ball([], PLAYERS) is None


def test_a_single_plausible_ball_is_returned():
    candidate = ball(200, 200, 212, 212)  # 12px, ~13% of ~90px reference
    result = select_ball([*PLAYERS, candidate], PLAYERS)
    assert result is candidate


def test_an_implausibly_large_detection_is_rejected_not_trusted():
    """The failure this module exists to fix: a stray large blob (an ad-board
    logo, a bright patch) must not be handed downstream just because nothing
    else was on screen."""
    huge = ball(200, 200, 250, 250, confidence=0.9)  # 50px, ~56% of reference
    assert select_ball([*PLAYERS, huge], PLAYERS) is None


def test_an_implausibly_tiny_detection_is_rejected():
    speck = ball(200, 200, 202, 202, confidence=0.9)  # 2px, ~2% of reference
    assert select_ball([*PLAYERS, speck], PLAYERS) is None


def test_two_ball_detections_the_plausible_one_wins():
    """Measured on client footage: multiple 'ball' detections on the same
    frame happen on 4 of every 10 sampled frames. Only one is real."""
    real = ball(200, 200, 212, 212, confidence=0.4)
    fake = ball(400, 400, 450, 450, confidence=0.9)  # bigger, more confident, wrong size
    result = select_ball([*PLAYERS, real, fake], PLAYERS)
    assert result is real


def test_two_plausible_detections_the_more_confident_wins():
    low = ball(200, 200, 212, 212, confidence=0.3)
    high = ball(400, 400, 412, 412, confidence=0.7)
    result = select_ball([*PLAYERS, low, high], PLAYERS)
    assert result is high


def test_with_no_players_on_the_frame_falls_back_to_highest_confidence():
    """No size reference exists without players — nothing to gate against,
    so this is not the failure the module targets and shouldn't discard
    every candidate for an unrelated reason."""
    low = ball(200, 200, 212, 212, confidence=0.3)
    high = ball(400, 400, 450, 450, confidence=0.7)
    result = select_ball([low, high], [])
    assert result is high


def test_a_zero_height_player_box_does_not_crash_the_reference():
    degenerate = player(100, 50, 130, 50)  # zero-height box
    candidate = ball(200, 200, 212, 212)
    result = select_ball([degenerate, candidate], [degenerate])
    assert result is candidate  # falls back to no-reference behaviour


def test_the_ratio_band_is_configurable():
    candidate = ball(200, 200, 212, 212)  # 12px
    # Reference ~90px -> 12px is ~13%; a band starting above that rejects it.
    assert select_ball([*PLAYERS, candidate], PLAYERS, min_size_ratio=0.5, max_size_ratio=0.9) is None
    assert select_ball([*PLAYERS, candidate], PLAYERS, min_size_ratio=0.05, max_size_ratio=0.5) is candidate
