"""Pitch calibration against the real client clip.

What this pins down is *behaviour on real footage*, not accuracy: that the
line finder rejects the impostors it is built to reject, and that automatic
calibration reports what it genuinely knows instead of overclaiming. The
reference clip is a development fixture, not a design target (M2_Plan
section 4), so nothing here asserts a number tuned to this one video.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from offside.field_geometry.pitch import PitchModel
from offside.pitch_calibration.calibrator import CalibrationLevel, PitchCalibrator
from offside.pitch_calibration.line_detection import detect_pitch_lines, pitch_line_mask

CLIP = Path("data/videos/client_m2_test_video.mp4")
DETECTOR_CHECKPOINT = Path("models/detector/yolo11n.pt")
SAMPLE_FRAMES = (120, 200, 300)


@pytest.fixture(scope="module")
def frames() -> list[np.ndarray]:
    if not CLIP.exists():
        pytest.skip(f"{CLIP} not present — the client reference clip is not in git")

    import cv2

    capture = cv2.VideoCapture(str(CLIP))
    images = []
    try:
        for index in SAMPLE_FRAMES:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if ok:
                images.append(image)
    finally:
        capture.release()

    if not images:
        pytest.skip("could not decode the reference clip")
    return images


@pytest.fixture(scope="module")
def player_boxes(frames) -> list[list[tuple[float, float, float, float]]]:
    if not DETECTOR_CHECKPOINT.exists():
        pytest.skip(f"{DETECTOR_CHECKPOINT} not present")

    from ultralytics import YOLO

    model = YOLO(str(DETECTOR_CHECKPOINT))
    boxes = []
    for image in frames:
        result = model.predict(image, verbose=False, conf=0.3, imgsz=1280, max_det=60)[0]
        boxes.append(
            [
                tuple(float(v) for v in box)
                for box, cls in zip(
                    result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()
                )
                if int(cls) == 0
            ]
        )
    return boxes


def test_finds_markings_on_real_footage(frames, player_boxes):
    counts = [
        len(detect_pitch_lines(image, exclude_boxes=boxes))
        for image, boxes in zip(frames, player_boxes)
    ]
    assert max(counts) >= 3, f"no usable pitch markings found on any frame: {counts}"


def test_the_mask_stays_off_the_crowd_and_the_boards(frames, player_boxes):
    """The stands and the advertising hoardings are full of long white edges.
    Before the grass and top-hat filters they produced more "lines" than the
    pitch did, and dragged every vanishing point towards the stands."""
    image, boxes = frames[1], player_boxes[1]
    mask = pitch_line_mask(image, exclude_boxes=boxes)

    height = mask.shape[0]
    # The upper fifth of this framing is crowd, roof and hoardings.
    upper = mask[: height // 5]
    assert upper.sum() / max(mask.sum(), 1) < 0.15


def test_player_exclusion_removes_white_shirt_false_lines(frames, player_boxes):
    """A white shirt is a thin bright object on green — the same description
    a line detector works from. Excluding player boxes must measurably cut
    what is found."""
    image, boxes = frames[1], player_boxes[1]
    if not boxes:
        pytest.skip("no players detected in this frame")

    with_players = pitch_line_mask(image, exclude_boxes=None)
    without_players = pitch_line_mask(image, exclude_boxes=boxes)
    assert without_players.sum() < with_players.sum()


def test_automatic_calibration_never_claims_metric(frames, player_boxes):
    """Automatic calibration can find groups of parallel markings, but not
    which group runs along the goal line — so it must never report metric
    positions. If this ever fails, a learned landmark model was added and
    this expectation needs revisiting deliberately."""
    calibrator = PitchCalibrator(PitchModel())
    for image, boxes in zip(frames, player_boxes):
        calibration = calibrator.calibrate_auto(image, exclude_boxes=boxes)
        assert calibration.level is not CalibrationLevel.METRIC
        assert not calibration.is_metric
        assert calibration.to_pitch([(100.0, 100.0)]) is None


def test_a_directional_calibration_states_its_assumption(frames, player_boxes):
    calibrator = PitchCalibrator(PitchModel())
    directional = [
        calibrator.calibrate_auto(image, exclude_boxes=boxes)
        for image, boxes in zip(frames, player_boxes)
    ]
    usable = [c for c in directional if c.level is CalibrationLevel.DIRECTIONAL]
    if not usable:
        pytest.skip("no frame reached a directional calibration")

    for calibration in usable:
        assert calibration.warnings, "an assumption-based result must warn"
        assert any("goal line" in warning for warning in calibration.warnings)
        # Capped precisely because the semantics are unverified.
        assert calibration.confidence <= 0.4
        assert calibration.can_draw_offside_line


def test_operator_confirmation_lifts_the_assumption(frames, player_boxes):
    calibrator = PitchCalibrator(PitchModel())
    image, boxes = frames[-1], player_boxes[-1]

    unconfirmed = calibrator.calibrate_auto(image, exclude_boxes=boxes)
    if not unconfirmed.families:
        pytest.skip("no line families found in this frame")

    confirmed = calibrator.calibrate_auto(
        image, exclude_boxes=boxes, goal_line_family_index=0
    )
    assert confirmed.confidence >= unconfirmed.confidence
    assert any("operator" in warning for warning in confirmed.warnings)


def test_calibration_failure_is_reported_not_faked():
    """A frame with no pitch in it must produce nothing, with a reason."""
    blank = np.full((720, 1280, 3), 40, dtype=np.uint8)
    calibration = PitchCalibrator(PitchModel()).calibrate_auto(blank)

    assert calibration.level is CalibrationLevel.NONE
    assert not calibration.can_draw_offside_line
    assert calibration.confidence == 0.0
    assert calibration.reasons
    assert calibration.offside_line_through((100.0, 100.0)) is None
