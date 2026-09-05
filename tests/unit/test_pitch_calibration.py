"""Pitch geometry and calibration: the maths, and what it refuses to claim.

No video and no models here — a synthetic camera stands in, so the geometry
can be checked against an exactly known answer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from offside.field_geometry.pitch import PENALTY_AREA_DEPTH, PitchModel
from offside.pitch_calibration.calibrator import CalibrationLevel, PitchCalibrator
from offside.pitch_calibration.homography import (
    PointCorrespondence,
    line_through_vanishing_point,
    solve_homography,
)
from offside.pitch_calibration.line_detection import (
    DetectedLine,
    estimate_vanishing_point,
    merge_lines,
)

PITCH = PitchModel()


def synthetic_camera() -> np.ndarray:
    """A pitch->image homography resembling a raised, off-centre broadcast
    camera: the far touchline compresses towards the top of the frame."""
    import cv2

    pitch_corners = np.array(
        [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]], dtype=np.float32
    )
    image_corners = np.array(
        [[280.0, 210.0], [1180.0, 250.0], [1500.0, 690.0], [-140.0, 620.0]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(pitch_corners, image_corners)


def project(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    vector = matrix @ np.array([point[0], point[1], 1.0])
    return (float(vector[0] / vector[2]), float(vector[1] / vector[2]))


def correspondences_for(names: list[str], camera: np.ndarray) -> list[PointCorrespondence]:
    return [
        PointCorrespondence(
            image_xy=project(camera, PITCH.landmark(name)),
            pitch_xy=PITCH.landmark(name),
            landmark=name,
        )
        for name in names
    ]


class TestPitchModel:
    def test_penalty_area_matches_the_laws(self):
        corner = PITCH.landmark("left_penalty_area_top_corner")
        assert corner[0] == pytest.approx(PENALTY_AREA_DEPTH)
        # 40.32m wide, centred on a 68m pitch
        top = PITCH.landmark("left_penalty_area_top_goalline")[1]
        bottom = PITCH.landmark("left_penalty_area_bottom_goalline")[1]
        assert bottom - top == pytest.approx(40.32)
        assert (top + bottom) / 2 == pytest.approx(34.0)

    def test_landmarks_scale_with_a_non_standard_pitch(self):
        small = PitchModel(length=100.0, width=64.0)
        assert small.landmark("corner_right_bottom") == (100.0, 64.0)
        # Box depth is fixed by the Laws even when the pitch is not standard.
        assert small.landmark("left_penalty_area_top_corner")[0] == pytest.approx(16.5)

    def test_contains_rejects_the_crowd(self):
        assert PITCH.contains((52.5, 34.0))
        assert not PITCH.contains((-15.0, 34.0))
        assert PITCH.contains((-1.0, 34.0), margin=2.0)

    def test_unknown_landmark_fails_loudly(self):
        with pytest.raises(KeyError):
            PITCH.landmark("penalty_spot")  # ambiguous: which end?

    def test_attacking_direction_is_towards_the_named_goal(self):
        assert PITCH.attacking_direction_for_goal("right") == (1.0, 0.0)
        assert PITCH.attacking_direction_for_goal("left") == (-1.0, 0.0)
        with pytest.raises(ValueError):
            PITCH.attacking_direction_for_goal("north")


class TestHomography:
    def test_recovers_a_known_camera(self):
        camera = synthetic_camera()
        fit = solve_homography(
            correspondences_for(
                [
                    "corner_left_top",
                    "corner_right_top",
                    "corner_right_bottom",
                    "corner_left_bottom",
                    "halfway_top",
                    "left_penalty_area_top_corner",
                ],
                camera,
            )
        )
        assert fit is not None

        # A point never used in the fit must still land where it belongs.
        truth = PITCH.landmark("right_penalty_spot")
        recovered = fit.to_pitch([project(camera, truth)])[0]
        assert recovered[0] == pytest.approx(truth[0], abs=0.5)
        assert recovered[1] == pytest.approx(truth[1], abs=0.5)

    def test_round_trips_through_the_inverse(self):
        fit = solve_homography(
            correspondences_for(
                [
                    "corner_left_top",
                    "corner_right_top",
                    "corner_right_bottom",
                    "corner_left_bottom",
                    "centre_mark",
                ],
                synthetic_camera(),
            )
        )
        image_point = [(640.0, 400.0)]
        assert fit.to_image(fit.to_pitch(image_point))[0] == pytest.approx(
            image_point[0], abs=1e-6
        )

    def test_four_points_cannot_cross_check_themselves(self):
        """Zero residual from four points is arithmetic, not evidence — the
        confidence must not read as certainty."""
        camera = synthetic_camera()
        four = solve_homography(
            correspondences_for(
                ["corner_left_top", "corner_right_top", "corner_right_bottom", "corner_left_bottom"],
                camera,
            )
        )
        assert four.mean_error_m == pytest.approx(0.0, abs=1e-6)
        assert four.confidence <= 0.6
        assert any("four reference points" in reason for reason in four.reasons)

    def test_more_points_earn_more_confidence(self):
        camera = synthetic_camera()
        four = solve_homography(
            correspondences_for(
                ["corner_left_top", "corner_right_top", "corner_right_bottom", "corner_left_bottom"],
                camera,
            )
        )
        many = solve_homography(
            correspondences_for(
                [
                    "corner_left_top",
                    "corner_right_top",
                    "corner_right_bottom",
                    "corner_left_bottom",
                    "halfway_top",
                    "halfway_bottom",
                    "left_penalty_area_top_corner",
                ],
                camera,
            )
        )
        assert many.confidence > four.confidence

    def test_a_mismarked_landmark_is_reported(self):
        camera = synthetic_camera()
        points = correspondences_for(
            [
                "corner_left_top",
                "corner_right_top",
                "corner_right_bottom",
                "corner_left_bottom",
                "halfway_top",
                "left_penalty_area_top_corner",
            ],
            camera,
        )
        # The operator clicked the six-yard corner but labelled it the 16.5m one.
        bad = points[-1]
        points[-1] = PointCorrespondence(
            image_xy=project(camera, PITCH.landmark("left_goal_area_top_corner")),
            pitch_xy=bad.pitch_xy,
            landmark=bad.landmark,
        )

        fit = solve_homography(points)
        assert fit.max_error_m > 2.0
        assert any("mismatched" in reason for reason in fit.reasons)

    def test_too_few_points_returns_nothing(self):
        camera = synthetic_camera()
        assert solve_homography(correspondences_for(["corner_left_top"], camera)) is None

    def test_degenerate_points_return_nothing_rather_than_nonsense(self):
        """Four points on one line cannot define a plane mapping. OpenCV will
        return a matrix anyway; this must not pass it on."""
        collinear = [
            PointCorrespondence(image_xy=(float(x), 100.0), pitch_xy=(float(x) / 10.0, 0.0))
            for x in (100, 200, 300, 400)
        ]
        assert solve_homography(collinear) is None

    def test_goal_line_vanishing_point_is_where_goal_lines_meet(self):
        camera = synthetic_camera()
        fit = solve_homography(
            correspondences_for(
                ["corner_left_top", "corner_right_top", "corner_right_bottom", "corner_left_bottom"],
                camera,
            )
        )
        vanishing_point = fit.vanishing_point_of((0.0, 1.0))
        assert vanishing_point is not None

        # Both goal lines, extended, must pass through it.
        for end in ("left", "right"):
            top = project(camera, PITCH.landmark(f"corner_{end}_top"))
            bottom = project(camera, PITCH.landmark(f"corner_{end}_bottom"))
            line = DetectedLine(top, bottom)
            bearing = math.degrees(
                math.atan2(vanishing_point[1] - top[1], vanishing_point[0] - top[0])
            ) % 180.0
            gap = abs(line.angle_deg - bearing) % 180.0
            assert min(gap, 180.0 - gap) < 0.5


class TestLineGeometry:
    def test_vanishing_point_from_converging_lines(self):
        target = (2000.0, -500.0)
        lines = []
        for start in ((100.0, 400.0), (300.0, 500.0), (500.0, 600.0)):
            dx, dy = target[0] - start[0], target[1] - start[1]
            scale = 200.0 / math.hypot(dx, dy)
            lines.append(DetectedLine(start, (start[0] + dx * scale, start[1] + dy * scale)))

        estimate = estimate_vanishing_point(lines)
        assert estimate is not None
        assert estimate.support == 3
        assert estimate.xy[0] == pytest.approx(target[0], rel=0.02)
        assert estimate.xy[1] == pytest.approx(target[1], rel=0.02)

    def test_parallel_lines_have_no_finite_meeting_point(self):
        lines = [
            DetectedLine((0.0, 100.0), (500.0, 100.0)),
            DetectedLine((0.0, 300.0), (500.0, 300.0)),
        ]
        assert estimate_vanishing_point(lines) is None

    def test_merging_joins_fragments_of_one_line(self):
        """A line a player stands on arrives from Hough in pieces."""
        merged = merge_lines(
            [
                DetectedLine((100.0, 200.0), (300.0, 205.0)),
                DetectedLine((320.0, 205.0), (500.0, 210.0)),
            ],
            angle_tolerance_deg=4.0,
            distance_tolerance_px=14.0,
        )
        assert len(merged) == 1
        assert merged[0].length > 380

    def test_merging_leaves_distant_collinear_markings_alone(self):
        """Two stretches of paint at opposite ends of the pitch are not one
        line, however well they line up."""
        merged = merge_lines(
            [
                DetectedLine((100.0, 200.0), (250.0, 200.0)),
                DetectedLine((900.0, 200.0), (1100.0, 200.0)),
            ],
            angle_tolerance_deg=4.0,
            distance_tolerance_px=14.0,
            max_join_gap_px=90.0,
        )
        assert len(merged) == 2

    def test_line_through_vanishing_point_aims_at_it(self):
        origin, direction = line_through_vanishing_point((100.0, 100.0), (900.0, 500.0), (0.0, 1.0))
        assert origin == (100.0, 100.0)
        assert direction[0] == pytest.approx(0.894, abs=0.01)
        assert direction[1] == pytest.approx(0.447, abs=0.01)

    def test_without_a_vanishing_point_the_fallback_direction_is_used(self):
        _, direction = line_through_vanishing_point((100.0, 100.0), None, (0.0, 1.0))
        assert direction == (0.0, 1.0)


class TestCalibrator:
    def test_manual_marking_gives_a_metric_calibration(self):
        calibrator = PitchCalibrator(PITCH)
        calibration = calibrator.calibrate_manual(
            correspondences_for(
                [
                    "corner_left_top",
                    "corner_right_top",
                    "corner_right_bottom",
                    "corner_left_bottom",
                    "halfway_top",
                ],
                synthetic_camera(),
            )
        )
        assert calibration.level is CalibrationLevel.METRIC
        assert calibration.is_metric
        assert calibration.can_draw_offside_line
        assert calibration.confidence > 0.6

    def test_three_points_is_not_a_calibration(self):
        calibration = PitchCalibrator(PITCH).calibrate_manual(
            correspondences_for(
                ["corner_left_top", "corner_right_top", "corner_right_bottom"],
                synthetic_camera(),
            )
        )
        assert calibration.level is CalibrationLevel.NONE
        assert not calibration.can_draw_offside_line
        assert calibration.to_pitch([(0.0, 0.0)]) is None
        assert "four are needed" in calibration.reasons[0]

    def test_offside_line_runs_towards_the_goal_line_vanishing_point(self):
        calibration = PitchCalibrator(PITCH).calibrate_manual(
            correspondences_for(
                [
                    "corner_left_top",
                    "corner_right_top",
                    "corner_right_bottom",
                    "corner_left_bottom",
                    "centre_mark",
                ],
                synthetic_camera(),
            )
        )
        point = (700.0, 450.0)
        result = calibration.offside_line_through(point)
        assert result is not None

        origin, direction = result
        assert origin == point
        assert math.hypot(*direction) == pytest.approx(1.0)

        # Stepping along that direction must keep the same pitch x (distance
        # to the goal line) — that is what "parallel to the goal line" means.
        here = calibration.to_pitch([point])[0]
        there = calibration.to_pitch([(point[0] + direction[0] * 50, point[1] + direction[1] * 50)])[0]
        assert there[0] == pytest.approx(here[0], abs=0.35)

    def test_a_failed_calibration_still_answers_questions_safely(self):
        calibration = PitchCalibrator(PITCH).calibrate_manual([])
        assert calibration.offside_line_through((100.0, 100.0)) is None
        assert calibration.to_image([(0.0, 0.0)]) is None
        assert calibration.confidence == 0.0
