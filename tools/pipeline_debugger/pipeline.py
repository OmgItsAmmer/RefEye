"""Running the offside pipeline over one frame, stage by stage.

The debugger's job is to make each stage's *actual* output visible, so this
module keeps every stage's result separate and never silently substitutes one
for another. Stages that do not exist yet are declared here too, as
`PENDING` — the panel lists the whole pipeline from the start and fills in as
milestones land, rather than the UI being rebuilt each phase.

Nothing here is part of the shipped app. It reads the same config, loads the
same models through the same `ModelRegistry`, and calls the same modules the
product does — if the debugger shows it, the product does it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from core.config.schema import AppSettings
from core.domain.models import Detection, FramePacket
from observability.logging.setup import get_logger
from offside.body_keypoints.keypoints import PlayerPose
from offside.field_geometry.pitch import PitchModel
from offside.pitch_calibration.calibrator import (
    CalibrationLevel,
    PitchCalibration,
    PitchCalibrator,
)
from offside.pitch_calibration.homography import PointCorrespondence
from offside.pitch_calibration.line_detection import pitch_line_mask
from offside.team_assignment.assigner import TeamAssigner
from offside.team_assignment.teams import PlayerRole, TeamAssignment
from vision.detection.classes import BALL

logger = get_logger(__name__)


class StageState(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"       # ran, but on a fallback or with low confidence
    UNAVAILABLE = "unavailable"  # should have run, could not
    PENDING = "pending"          # not implemented yet (a future milestone)


@dataclass
class StageReport:
    """One row in the debugger's pipeline panel."""

    key: str
    phase: str
    title: str
    state: StageState
    summary: str
    details: list[str] = field(default_factory=list)


@dataclass
class FrameAnalysis:
    """Everything known about one frame, per stage."""

    frame_index: int
    frame: FramePacket
    detections: list[Detection] = field(default_factory=list)
    poses: list[PlayerPose] = field(default_factory=list)
    calibration: PitchCalibration | None = None
    line_mask: np.ndarray | None = None
    teams: TeamAssignment | None = None
    reports: list[StageReport] = field(default_factory=list)

    @property
    def players(self) -> list[Detection]:
        return [d for d in self.detections if d.class_name != BALL]

    @property
    def ball(self) -> Detection | None:
        balls = [d for d in self.detections if d.class_name == BALL]
        return max(balls, key=lambda d: d.confidence) if balls else None

    @property
    def ball_xy(self) -> tuple[float, float] | None:
        ball = self.ball
        if ball is None:
            return None
        x1, y1, x2, y2 = ball.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class OffsidePipeline:
    """Runs the implemented stages over a frame and reports on all of them."""

    def __init__(self, settings: AppSettings, registry):
        self._settings = settings
        self._registry = registry

        calibration_config = settings.offside.pitch_calibration
        line_config = calibration_config.line_detection
        self._pitch = PitchModel(
            length=calibration_config.pitch_length_m,
            width=calibration_config.pitch_width_m,
        )
        self._line_kwargs = {
            "grass_hue_range": (line_config.grass_hue_min, line_config.grass_hue_max),
            "grass_min_saturation": line_config.grass_min_saturation,
            "line_min_brightness": line_config.line_min_brightness,
            "max_line_width_px": line_config.max_line_width_px,
            "tophat_threshold": line_config.tophat_threshold,
            "exclude_box_margin": line_config.exclude_box_margin,
            "min_line_length_px": line_config.min_line_length_px,
            "max_line_gap_px": line_config.max_line_gap_px,
            "hough_threshold": line_config.hough_threshold,
            "merge_angle_deg": line_config.merge_angle_deg,
            "merge_distance_px": line_config.merge_distance_px,
            "max_lines": line_config.max_lines,
        }
        self._calibrator = PitchCalibrator(
            self._pitch,
            max_reprojection_error_m=calibration_config.max_reprojection_error_m,
            min_family_support=calibration_config.min_family_support,
            inlier_angle_deg=calibration_config.inlier_angle_deg,
            min_lines_for_auto=calibration_config.min_lines_for_auto,
            line_detection_kwargs=self._line_kwargs,
        )

        team_config = settings.offside.team_assignment
        self._team_assigner = TeamAssigner.from_config(
            team_config,
            # The grass window belongs to the pitch, not to team assignment;
            # sharing it keeps the kit sampler and the line detector agreeing
            # on what grass is instead of drifting apart in two config blocks.
            grass_hue_range=(line_config.grass_hue_min, line_config.grass_hue_max),
            grass_min_saturation=line_config.grass_min_saturation,
        )

        #: Operator-marked landmarks, kept across frames of the same shot.
        self.manual_correspondences: list[PointCorrespondence] = []
        #: Which auto-detected line family the operator says runs along the
        #: goal line. None means "nobody has confirmed it" — which the auto
        #: calibration reports as an explicit assumption rather than a fact.
        self.goal_line_family_index: int | None = None

        # Detection resolution matters more than usual here: at the M1 live
        # setting of 640 a wide broadcast shot yields almost no players (see
        # M2_Plan M2.2). The offside path is triggered, not continuous, so it
        # can afford the larger size.
        self._detector_imgsz = max(settings.ai.detector.imgsz, 1280)

    @property
    def pitch(self) -> PitchModel:
        return self._pitch

    def analyse(self, frame: FramePacket, frame_index: int) -> FrameAnalysis:
        analysis = FrameAnalysis(frame_index=frame_index, frame=frame)

        self._run_detection(analysis)
        self._run_body_keypoints(analysis)
        self._run_pitch_calibration(analysis)
        self._run_team_assignment(analysis)
        self._report_pending_stages(analysis)
        return analysis

    # -- stages -------------------------------------------------------------

    def _run_detection(self, analysis: FrameAnalysis) -> None:
        detector = self._registry.get_detector()
        if detector is None:
            analysis.reports.append(
                StageReport(
                    key="detection",
                    phase="M1",
                    title="Player & ball detection",
                    state=StageState.UNAVAILABLE,
                    summary="detector not loaded",
                )
            )
            return

        previous = getattr(detector, "_imgsz", None)
        try:
            if previous is not None:
                detector._imgsz = self._detector_imgsz
            analysis.detections = detector.detect(analysis.frame)
        finally:
            if previous is not None:
                detector._imgsz = previous

        ball = analysis.ball
        analysis.reports.append(
            StageReport(
                key="detection",
                phase="M1",
                title="Player & ball detection",
                state=StageState.OK if analysis.players else StageState.DEGRADED,
                summary=f"{len(analysis.players)} players, "
                + ("ball found" if ball else "no ball this frame"),
                details=[
                    f"model: {detector.model_name}",
                    (
                        f"detection size: {self._detector_imgsz}px (live preview "
                        f"uses {self._settings.ai.detector.imgsz}px)"
                    ),
                ]
                + ([] if ball else ["a missing ball is normal and not an error"]),
            )
        )

    def _run_body_keypoints(self, analysis: FrameAnalysis) -> None:
        estimator = self._registry.get_pose_estimator()
        if estimator is None:
            analysis.reports.append(
                StageReport(
                    key="body_keypoints",
                    phase="M2.2",
                    title="Body keypoints (feet)",
                    state=StageState.UNAVAILABLE,
                    summary="pose model not loaded",
                    details=["see models/pose/README.md"],
                )
            )
            return

        analysis.poses = estimator.estimate(analysis.frame, analysis.detections)
        measured = [p for p in analysis.poses if p.ground_point.is_measured]
        posed = [p for p in analysis.poses if p.has_pose]

        state = StageState.OK
        if analysis.poses and len(measured) < len(analysis.poses) * 0.5:
            state = StageState.DEGRADED

        analysis.reports.append(
            StageReport(
                key="body_keypoints",
                phase="M2.2",
                title="Body keypoints (feet)",
                state=state,
                summary=f"{len(measured)}/{len(analysis.poses)} players with a "
                "measured foot position",
                details=[
                    f"model: {estimator.model_name}",
                    (
                        f"{len(posed)} skeletons found; the rest fall back to "
                        "the bottom of the player box"
                    ),
                    (
                        "green = measured from an ankle, orange = inferred "
                        "from a knee, red = box-bottom guess"
                    ),
                ],
            )
        )

    def _run_pitch_calibration(self, analysis: FrameAnalysis) -> None:
        config = self._settings.offside.pitch_calibration
        if not config.enabled:
            analysis.reports.append(
                StageReport(
                    key="pitch_calibration",
                    phase="M2.1",
                    title="Pitch calibration",
                    state=StageState.UNAVAILABLE,
                    summary="disabled in config",
                )
            )
            return

        image = analysis.frame.image
        boxes = [d.bbox_xyxy for d in analysis.players]

        analysis.line_mask = pitch_line_mask(
            image,
            exclude_boxes=boxes,
            grass_hue_range=self._line_kwargs["grass_hue_range"],
            grass_min_saturation=self._line_kwargs["grass_min_saturation"],
            line_min_brightness=self._line_kwargs["line_min_brightness"],
            max_line_width_px=self._line_kwargs["max_line_width_px"],
            tophat_threshold=self._line_kwargs["tophat_threshold"],
            exclude_box_margin=self._line_kwargs["exclude_box_margin"],
        )

        if len(self.manual_correspondences) >= 4:
            # The operator has marked the pitch: that outranks any automatic
            # guess, and is the only route to metric positions today.
            calibration = self._calibrator.calibrate_manual(self.manual_correspondences)
            calibration.detected_lines = self._calibrator.calibrate_auto(
                image,
                exclude_boxes=boxes,
                goal_line_family_index=self.goal_line_family_index,
            ).detected_lines
        else:
            calibration = self._calibrator.calibrate_auto(
                image,
                exclude_boxes=boxes,
                goal_line_family_index=self.goal_line_family_index,
            )

        analysis.calibration = calibration

        state = {
            CalibrationLevel.METRIC: StageState.OK,
            CalibrationLevel.DIRECTIONAL: StageState.DEGRADED,
            CalibrationLevel.NONE: StageState.UNAVAILABLE,
        }[calibration.level]

        summary = {
            CalibrationLevel.METRIC: "metric — positions in metres available",
            CalibrationLevel.DIRECTIONAL: "directional — offside lines only, no metres",
            CalibrationLevel.NONE: "not calibrated",
        }[calibration.level]

        details = [
            f"source: {calibration.source}",
            f"confidence: {calibration.confidence:.2f}",
            f"{len(calibration.detected_lines)} markings detected",
            (
                f"{len(self.manual_correspondences)} landmark(s) marked by hand "
                "(4 needed for metric)"
            ),
        ]
        details += calibration.reasons
        details += [f"WARNING: {warning}" for warning in calibration.warnings]

        analysis.reports.append(
            StageReport(
                key="pitch_calibration",
                phase="M2.1",
                title="Pitch calibration",
                state=state,
                summary=summary,
                details=details,
            )
        )

    def _run_team_assignment(self, analysis: FrameAnalysis) -> None:
        config = self._settings.offside.team_assignment
        if not config.enabled:
            analysis.reports.append(
                StageReport(
                    key="team_assignment",
                    phase="M2.3",
                    title="Team & goalkeeper assignment",
                    state=StageState.UNAVAILABLE,
                    summary="disabled in config",
                )
            )
            return

        analysis.teams = self._team_assigner.assign(
            analysis.frame.image,
            analysis.poses,
            calibration=analysis.calibration,
            ball_xy=analysis.ball_xy,
        )
        teams = analysis.teams
        counts = teams.counts()

        if teams.color_model is None:
            state = StageState.UNAVAILABLE
            summary = "the two kits could not be told apart on this frame"
        elif teams.sides_are_known and teams.confidence >= 0.5:
            state = StageState.OK
            summary = (
                f"team A {counts['team_a']}, team B {counts['team_b']}, "
                f"{counts['unassigned']} unplaced — "
                f"{teams.attacking_team_id[-1].upper()} attacking"
            )
        else:
            state = StageState.DEGRADED
            summary = (
                f"team A {counts['team_a']}, team B {counts['team_b']}, "
                f"{counts['unassigned']} unplaced — "
                + (
                    "attacking side unknown"
                    if not teams.sides_are_known
                    else "low confidence"
                )
            )

        keepers = teams.goalkeepers()
        details = [f"confidence: {teams.confidence:.2f}"]
        if teams.color_model is not None:
            details.append(
                "kit colours (BGR): "
                + ", ".join(
                    f"{team[-1].upper()}={teams.color_model.swatches[team]}"
                    for team in teams.color_model.swatches
                )
            )
        details.append(
            f"{len(keepers)} goalkeeper(s) identified"
            + (
                ""
                if not keepers
                else ": " + "; ".join(k.reason for k in keepers)
            )
        )
        if self._team_assigner.overrides.is_active:
            corrected = [p for p in teams.players if p.is_operator_set]
            details.append(
                f"operator corrections active ({len(corrected)} player(s) pinned"
                + (", labels swapped" if self._team_assigner.overrides.teams_swapped else "")
                + ")"
            )
        details += teams.reasons
        details += [f"WARNING: {warning}" for warning in teams.warnings]

        analysis.reports.append(
            StageReport(
                key="team_assignment",
                phase="M2.3",
                title="Team & goalkeeper assignment",
                state=state,
                summary=summary,
                details=details,
            )
        )

    def _report_pending_stages(self, analysis: FrameAnalysis) -> None:
        """The rest of the plan, listed so the gap is visible rather than
        implied. Each becomes a real stage as its milestone lands."""
        pending = [
            ("tracking", "M2.4", "Player identity tracking",
             "keeping identities stable through the contact moment"),
            ("offside_line", "M2.5", "Second-last defender & offside line",
             "the geometry that produces a verdict"),
            ("decision_support", "M2.6", "Confidence & reasoning",
             "offside / not offside / too close to call, with reasons"),
        ]
        for key, phase, title, summary in pending:
            analysis.reports.append(
                StageReport(
                    key=key,
                    phase=phase,
                    title=title,
                    state=StageState.PENDING,
                    summary=f"not built yet — {summary}",
                )
            )

    # -- calibration helpers used by the UI ---------------------------------

    def mark_landmark(self, image_xy: tuple[float, float], landmark: str) -> None:
        self.manual_correspondences = [
            c for c in self.manual_correspondences if c.landmark != landmark
        ]
        self.manual_correspondences.append(
            PointCorrespondence(
                image_xy=image_xy,
                pitch_xy=self._pitch.landmark(landmark),
                landmark=landmark,
            )
        )

    def clear_landmarks(self) -> None:
        self.manual_correspondences = []

    # -- team corrections used by the UI ------------------------------------
    #
    # Thin pass-throughs on purpose: the correction logic lives in
    # `TeamOverrides`, so the M2.7 review panel drives exactly the same path
    # this debug UI does, rather than a parallel implementation of it.

    @property
    def team_overrides(self):
        return self._team_assigner.overrides

    def pin_player(
        self,
        image_xy: tuple[float, float],
        *,
        team_id: str | None = None,
        role: PlayerRole | None = None,
    ) -> None:
        self._team_assigner.overrides.pin_player(image_xy, team_id=team_id, role=role)

    def clear_player_pin(self, image_xy: tuple[float, float]) -> bool:
        return self._team_assigner.overrides.clear_player(image_xy)

    def swap_teams(self) -> bool:
        return self._team_assigner.overrides.swap_teams()

    def set_attacking_team(self, team_id: str | None) -> None:
        self._team_assigner.overrides.set_attacking_team(team_id)

    def clear_team_overrides(self) -> None:
        self._team_assigner.overrides.clear()

    def reset_team_colors(self) -> None:
        """Forget the fitted kit colours — the right thing after a camera cut."""
        self._team_assigner.reset()
