"""Mapping the keypoint model's raw output onto this project's landmarks.

The real model isn't exercised here (that's the integration test, on real
footage) — this is about the translation layer: does the index-to-landmark
table actually produce correspondences a homography solve can use, does the
visibility gate actually gate, and does a missing model degrade instead of
crashing the offside path. A wrong entry in that table would silently
mislabel a point without any of these ever catching it directly, which is
exactly why the integration test then checks the *geometry* the mapped
points actually produce, not just that a mapping exists.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from core.errors.exceptions import ModelLoadError
from offside.field_geometry.pitch import PitchModel
from offside.pitch_calibration.auto_landmarks import (
    _MODEL_KEYPOINT_NAMES,
    _TO_PITCH_LANDMARK,
    AutoLandmarkDetector,
)


@pytest.fixture
def pitch() -> PitchModel:
    return PitchModel()


def _fake_result(keypoints_xy: np.ndarray, visibility: np.ndarray, num_boxes: int = 1):
    result = MagicMock()
    result.boxes = MagicMock()
    result.boxes.__len__.return_value = num_boxes
    result.boxes.conf = MagicMock()
    result.boxes.conf.argmax.return_value = 0
    result.keypoints = MagicMock()
    result.keypoints.xy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: keypoints_xy))]
    result.keypoints.conf = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: visibility))]
    return result


def make_detector(**overrides) -> AutoLandmarkDetector:
    detector = AutoLandmarkDetector("fake.pt", **overrides)
    detector._model = MagicMock()  # noqa: SLF001 — skip real loading in these tests
    return detector


def test_every_mapped_model_keypoint_resolves_to_a_real_pitch_landmark(pitch):
    """The one test that would catch a typo in the translation table before
    it ever reaches real footage: every landmark name this module claims to
    produce must actually exist on `PitchModel`."""
    for landmark in _TO_PITCH_LANDMARK.values():
        pitch.landmark(landmark)  # raises KeyError if the name is wrong


def test_21_of_29_keypoints_are_mapped_the_rest_deliberately_arent():
    mapped = sum(1 for name in _MODEL_KEYPOINT_NAMES.values() if name in _TO_PITCH_LANDMARK)
    assert mapped == 21
    assert len(_MODEL_KEYPOINT_NAMES) == 29


def test_a_confident_point_becomes_a_correspondence(pitch):
    detector = make_detector(visibility_threshold=0.5)
    keypoints = np.zeros((29, 2))
    visibility = np.zeros(29)
    keypoints[0] = (100.0, 200.0)  # sideline_top_left -> corner_left_top
    visibility[0] = 0.9
    detector._model.predict.return_value = [_fake_result(keypoints, visibility)]  # noqa: SLF001

    correspondences = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8), pitch)

    assert len(correspondences) == 1
    assert correspondences[0].landmark == "corner_left_top"
    assert correspondences[0].image_xy == (100.0, 200.0)
    assert correspondences[0].pitch_xy == pitch.landmark("corner_left_top")


def test_a_low_visibility_point_is_left_out_not_trusted(pitch):
    detector = make_detector(visibility_threshold=0.5)
    keypoints = np.zeros((29, 2))
    visibility = np.zeros(29)
    keypoints[0] = (100.0, 200.0)
    visibility[0] = 0.3  # below threshold
    detector._model.predict.return_value = [_fake_result(keypoints, visibility)]  # noqa: SLF001

    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8), pitch) == []


def test_an_unmapped_keypoint_is_never_returned_even_at_full_confidence(pitch):
    """Index 13 (center_circle_top) is deliberately unmapped — see the module
    docstring. High confidence must not smuggle it in anyway."""
    detector = make_detector(visibility_threshold=0.5)
    keypoints = np.zeros((29, 2))
    visibility = np.zeros(29)
    keypoints[13] = (500.0, 500.0)
    visibility[13] = 0.99
    detector._model.predict.return_value = [_fake_result(keypoints, visibility)]  # noqa: SLF001

    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8), pitch) == []


def test_a_zero_coordinate_is_excluded_even_past_the_visibility_gate(pitch):
    """Ultralytics reports an undetected keypoint as (0, 0) — a second, cheap
    guard so a low threshold can never turn "not found" into a real point."""
    detector = make_detector(visibility_threshold=0.1)
    keypoints = np.zeros((29, 2))
    visibility = np.zeros(29)
    visibility[0] = 0.5  # passes the gate, but coordinate is (0, 0)
    detector._model.predict.return_value = [_fake_result(keypoints, visibility)]  # noqa: SLF001

    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8), pitch) == []


def test_no_pitch_detected_at_all_returns_empty_not_an_error(pitch):
    detector = make_detector()
    result = MagicMock()
    result.keypoints = None
    result.boxes = MagicMock()
    result.boxes.__len__.return_value = 0
    detector._model.predict.return_value = [result]  # noqa: SLF001

    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8), pitch) == []


def test_a_missing_checkpoint_raises_model_load_error_not_a_bare_exception(tmp_path):
    detector = AutoLandmarkDetector(str(tmp_path / "does_not_exist.pt"))
    with pytest.raises(ModelLoadError):
        detector.load()


def test_from_config_reads_every_field():
    config = MagicMock(
        imgsz=320,
        visibility_threshold=0.7,
        detection_confidence=0.4,
    )
    detector = AutoLandmarkDetector.from_config(config, "some/path.pt", device="cuda")
    assert detector._imgsz == 320  # noqa: SLF001
    assert detector._visibility_threshold == 0.7  # noqa: SLF001
    assert detector._detection_confidence == 0.4  # noqa: SLF001
    assert detector._device == "cuda"  # noqa: SLF001
