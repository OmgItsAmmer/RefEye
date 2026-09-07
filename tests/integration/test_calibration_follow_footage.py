"""Following the camera on real broadcast footage.

The synthetic tests prove the maths moves points correctly under a known
transform. Only real footage answers the question that matters: does a real
broadcast camera move enough to break a fixed calibration, and does the
follower survive real motion blur, crowd, and players running through frame?
"""

from __future__ import annotations

from pathlib import Path

import pytest

from offside.pitch_calibration.homography import PointCorrespondence
from offside.pitch_calibration.tracking import CalibrationFollower

CLIP = Path("data/videos/client_m2_test_video.mp4")
RUN_START = 90
RUN_LENGTH = 60


@pytest.fixture(scope="module")
def frames():
    if not CLIP.exists():
        pytest.skip(f"{CLIP} not present — the client reference clip is not in git")

    import cv2

    capture = cv2.VideoCapture(str(CLIP))
    images = []
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, RUN_START)
        for _ in range(RUN_LENGTH):
            ok, image = capture.read()
            if not ok:
                break
            images.append(image)
    finally:
        capture.release()

    if len(images) < 20:
        pytest.skip("could not decode a run of frames from the reference clip")
    return images


def _marks(points):
    return [
        PointCorrespondence(image_xy=point, pitch_xy=(index * 10.0, 0.0),
                            landmark=f"m{index}")
        for index, point in enumerate(points)
    ]


ANCHOR_POINTS = [(300.0, 300.0), (950.0, 320.0), (400.0, 600.0), (1000.0, 620.0)]


@pytest.fixture(scope="module")
def follow_run(frames):
    follower = CalibrationFollower()
    follower.anchor(_marks(ANCHOR_POINTS), frames[0], 0)
    results = []
    for index, image in enumerate(frames[1:], start=1):
        result = follower.update(image, index)
        if result is None:
            break
        results.append(result)
    return results


def test_the_camera_really_does_move_enough_to_matter(follow_run):
    """The justification for this whole module. If a broadcast camera held
    still, fixed marks would be fine and none of this would be needed."""
    final = follow_run[-1]
    drift = abs(final.correspondences[0].image_xy[0] - ANCHOR_POINTS[0][0])

    # Measured on the reference clip: ~350px of pan in about two seconds.
    assert drift > 100, (
        f"the marks only moved {drift:.0f}px; either the camera is static here "
        "or the follower has stopped following"
    )


def test_the_marks_are_never_lost_during_ordinary_play(follow_run):
    assert follow_run, "the follower gave up immediately"
    assert not any(result.lost for result in follow_run)
    assert all(result.inlier_ratio > 0.7 for result in follow_run)


def test_trust_decays_as_the_marks_are_carried(follow_run):
    """Drift is cumulative and invisible in the marks themselves, so the only
    honest thing to do is trust them less the further they travel."""
    assert follow_run[-1].confidence < follow_run[0].confidence
    assert all(
        later.confidence <= earlier.confidence + 1e-9
        for earlier, later in zip(follow_run, follow_run[1:])
    )


def test_the_operator_is_told_when_to_re_mark(follow_run):
    """Below the floor the follower must say so in words, not just in a number
    the operator has to interpret."""
    low = [result for result in follow_run if result.confidence < 0.3]
    for result in low:
        assert any("re-mark" in warning for warning in result.warnings)


def test_a_jump_to_an_unrelated_shot_is_reported_as_lost(frames):
    """Standing in for a camera cut: the marks belong to a view that is gone,
    and carrying them into a new one would be a confident wrong answer."""
    follower = CalibrationFollower()
    follower.anchor(_marks(ANCHOR_POINTS), frames[0], 0)

    import numpy as np

    result = follower.update(np.zeros_like(frames[0]), frame_id=1)

    assert result.lost
    assert not follower.has_anchor
