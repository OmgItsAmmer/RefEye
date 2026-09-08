"""Automatic pitch-landmark detection on real broadcast footage.

The unit tests prove the translation table produces valid `PitchModel`
landmark names. Only real footage answers the question that actually
matters: do the detected image points, once solved through the same
homography an operator's own clicks would use, describe a mapping that is
*geometrically consistent* — not just "a landmark string that exists", but
"a point that lands within centimetres of where the other points say it
should be." A systematically wrong pt1/pt2 assumption (see the module
docstring) would show up here as a high mean reprojection error even though
every individual name resolved fine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from offside.field_geometry.pitch import PitchModel
from offside.pitch_calibration.auto_landmarks import AutoLandmarkDetector
from offside.pitch_calibration.homography import solve_homography

CHECKPOINT = Path("models/pitch_keypoints/soccana_keypoint.pt")
CLIPS = [
    ("data/videos/demo_video_offside_1.mp4", 122),
    ("data/videos/client_m2_test_video.mp4", 200),
]


@pytest.fixture(scope="module")
def detector():
    if not CHECKPOINT.exists():
        pytest.skip(f"{CHECKPOINT} not present — see models/pitch_keypoints/README.md")
    return AutoLandmarkDetector(str(CHECKPOINT), device="cpu", visibility_threshold=0.5)


@pytest.fixture(scope="module")
def pitch() -> PitchModel:
    return PitchModel()


def _read_frame(path: str, frame_id: int):
    import cv2

    if not Path(path).exists():
        pytest.skip(f"{path} not present")
    capture = cv2.VideoCapture(path)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok:
        pytest.skip(f"could not decode frame {frame_id} of {path}")
    return image


@pytest.mark.parametrize("clip_path,frame_id", CLIPS)
def test_a_frame_with_a_visible_penalty_area_reaches_metric(detector, pitch, clip_path, frame_id):
    """The concrete claim this feature makes: on real broadcast footage with
    the box in view, this reaches METRIC by itself — no operator, no click."""
    image = _read_frame(clip_path, frame_id)
    correspondences = detector.detect(image, pitch)

    assert len(correspondences) >= 6, (
        f"expected at least 6 confident landmarks on {clip_path}#{frame_id}, "
        f"got {len(correspondences)}"
    )

    fit = solve_homography(correspondences, max_error_m=2.0)
    assert fit is not None, "the detected points did not describe a valid mapping"
    assert fit.confidence >= 0.7, f"fit confidence too low: {fit.confidence:.2f}"
    assert fit.mean_error_m < 0.5, (
        f"mean reprojection error too high: {fit.mean_error_m:.2f}m — the "
        "pt1/pt2 mapping assumption (see auto_landmarks.py) may be wrong"
    )


def test_the_calibrated_pitch_lines_reproject_close_to_where_they_should(detector, pitch):
    """Stronger than the numeric check above: project the pitch model's own
    line drawing back through the fitted homography, at a handful of known
    landmark points, and confirm each one lands close to the image point it
    was fitted from — the same check as looking at the overlay, in
    assertion form."""
    image = _read_frame(*CLIPS[0])
    correspondences = detector.detect(image, pitch)
    fit = solve_homography(correspondences, max_error_m=2.0)
    assert fit is not None

    for correspondence in correspondences:
        reprojected = fit.to_image([correspondence.pitch_xy])[0]
        error_px = (
            (reprojected[0] - correspondence.image_xy[0]) ** 2
            + (reprojected[1] - correspondence.image_xy[1]) ** 2
        ) ** 0.5
        assert error_px < 40, (
            f"{correspondence.landmark} reprojects {error_px:.0f}px from where "
            "it was detected — the fit or the mapping is inconsistent"
        )


def test_a_close_up_shot_with_no_box_lines_honestly_finds_little(detector, pitch):
    """The other half of the claim: this must not hallucinate a confident
    metric calibration when the box genuinely isn't in the shot."""
    image = _read_frame("data/videos/demo_video_offside_2.mp4", 300)
    correspondences = detector.detect(image, pitch)
    # Not a hard zero — the exact frame content isn't guaranteed stable
    # across re-encodes of the demo clip — but this must not quietly produce
    # a full, confident 8-point read on a frame with nothing to read.
    assert len(correspondences) <= 4
