"""Running the offside pipeline over one frame, stage by stage.

This is the product's offside pipeline. It used to live under
`tools/pipeline_debugger/`, with a docstring promising that the debug tool
"calls the same modules the product does" — which was true of the individual
stages (M2.1-M2.6) but not of this file itself: nothing in the shipped app
imported it, so the promise had a hole in it. It now lives here, in
`offside/`, and the debugger imports it — one implementation, not two that
can drift apart.

Every stage's result is kept separate and never silently substituted for
another, and `analyse()` takes an optional `on_stage` callback so a caller
(the review screen's per-stage progress UI, M2.7's runner) can show each
stage completing as it happens rather than waiting on the whole frame.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from core.config.schema import AppSettings
from core.domain.models import Detection, FramePacket
from observability.logging.setup import get_logger
from offside.ball_selection import select_ball
from offside.body_keypoints.keypoints import PlayerPose
from offside.decision_support.explainer import DecisionExplainer
from offside.decision_support.explanation import ConfidenceBand, DecisionExplanation
from offside.field_geometry.pitch import PitchModel
from offside.offside_line.line import OffsideDecision, OffsideLineCalculator, Verdict
from offside.pitch_calibration.calibrator import (
    CalibrationLevel,
    PitchCalibration,
    PitchCalibrator,
)
from offside.pitch_calibration.homography import PointCorrespondence
from offside.pitch_calibration.line_detection import pitch_line_mask
from offside.pitch_calibration.tracking import CalibrationFollower
from offside.player_identity.identity import IdentityResult
from offside.player_identity.tracker import IdentityTracker
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
    identities: IdentityResult | None = None
    offside: OffsideDecision | None = None
    explanation: DecisionExplanation | None = None
    reports: list[StageReport] = field(default_factory=list)
    #: Set by OffsidePipeline from `ai.detector.ball_*_size_ratio` — kept on
    #: the analysis (not just read from a global) so a debugger or a test can
    #: override them per frame without touching config.
    ball_min_size_ratio: float = 0.06
    ball_max_size_ratio: float = 0.22

    @property
    def players(self) -> list[Detection]:
        return [d for d in self.detections if d.class_name != BALL]

    @property
    def ball(self) -> Detection | None:
        """The one detection most likely to be the real ball, or None.

        Not just the highest-confidence "ball" detection — see
        `offside.ball_selection`. Measured on client footage: more than one
        "ball" appeared on 4 of every 10 sampled frames, so trusting whichever
        one scored highest was routinely wrong.
        """
        return select_ball(
            self.detections,
            self.players,
            min_size_ratio=self.ball_min_size_ratio,
            max_size_ratio=self.ball_max_size_ratio,
        )

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

        self._follower = CalibrationFollower.from_config(calibration_config.follow)

        identity_config = settings.offside.player_identity
        self._identity_tracker = IdentityTracker.from_config(identity_config)

        team_config = settings.offside.team_assignment
        self._team_assigner = TeamAssigner.from_config(
            team_config,
            # The grass window belongs to the pitch, not to team assignment;
            # sharing it keeps the kit sampler and the line detector agreeing
            # on what grass is instead of drifting apart in two config blocks.
            grass_hue_range=(line_config.grass_hue_min, line_config.grass_hue_max),
            grass_min_saturation=line_config.grass_min_saturation,
        )

        self._offside = OffsideLineCalculator.from_config(
            settings.offside.offside_line,
            min_keypoint_confidence=(
                settings.offside.body_keypoints.keypoint_confidence_threshold
            ),
        )
        self._explainer = DecisionExplainer.from_config(
            settings.offside.decision_support
        )

        #: Which way the attack is going, when the operator has said. None
        #: means "work it out from the keeper and the shape of the teams".
        self.attack_sign: float | None = None

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

    def analyse(
        self,
        frame: FramePacket,
        frame_index: int,
        *,
        on_stage: Callable[[StageReport], None] | None = None,
    ) -> FrameAnalysis:
        """Run every stage over one frame.

        `on_stage` fires once per stage, immediately after that stage's report
        is appended — before the next stage starts, so a live progress UI can
        show "calibrating…" become "calibrated" while identity is still
        running, rather than everything appearing at once at the end.
        """
        detector_config = self._settings.ai.detector
        analysis = FrameAnalysis(
            frame_index=frame_index,
            frame=frame,
            ball_min_size_ratio=detector_config.ball_min_size_ratio,
            ball_max_size_ratio=detector_config.ball_max_size_ratio,
        )

        def run(stage: Callable[[FrameAnalysis], None]) -> None:
            before = len(analysis.reports)
            stage(analysis)
            if on_stage is not None:
                for report in analysis.reports[before:]:
                    on_stage(report)

        run(self._run_detection)
        run(self._run_body_keypoints)
        # Identity runs before both of the stages that consume it: it stamps a
        # track id onto every pose for the team stage to pool votes against,
        # and it is what notices a camera cut — which ends the validity of the
        # operator's pitch marks as surely as it ends a player's identity.
        run(self._run_player_identity)
        run(self._run_pitch_calibration)
        run(self._run_team_assignment)
        run(self._run_offside_line)
        run(self._run_decision_support)
        run(self._report_pending_stages)
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

        cut = analysis.identities.cut_detected if analysis.identities else False
        follow = self._follow_marks(analysis, image, boxes, cut)

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
            # guess, and is the only route to metric positions today. The marks
            # used are the *followed* ones — where those points are now, not
            # where they were on the frame the operator clicked.
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

        if follow is not None:
            # Drift cannot be seen in the marks themselves — four points always
            # agree with each other — so the follower's own confidence is what
            # discounts a calibration that has been carried a long way.
            calibration.confidence *= follow.confidence
            calibration.reasons += follow.reasons
            calibration.warnings += follow.warnings

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

    def _run_player_identity(self, analysis: FrameAnalysis) -> None:
        config = self._settings.offside.player_identity
        if not config.enabled:
            analysis.reports.append(
                StageReport(
                    key="tracking",
                    phase="M2.4",
                    title="Player identity tracking",
                    state=StageState.UNAVAILABLE,
                    summary="disabled in config",
                )
            )
            return

        result = self._identity_tracker.update(analysis.frame, analysis.poses)
        analysis.identities = result

        if result.cut_detected:
            # Kit colours survive a cut (same teams, same match); identities
            # do not. Resetting the colour model here would throw away good
            # evidence and let the team labels flip.
            self._team_assigner.reset_identities()

        counts = result.counts()
        contested = result.contested()

        if not result.identities:
            state = StageState.UNAVAILABLE
        elif contested or result.confidence < 0.5:
            state = StageState.DEGRADED
        else:
            state = StageState.OK

        summary = (
            f"{len(result.trusted())}/{len(result.identities)} players followed "
            "reliably"
            + (f" — {len(contested)} contested" if contested else "")
        )

        details = [
            f"shot segment {result.segment} (identities restart at every camera cut)",
            "  ".join(
                f"{count} {state_name}" for state_name, count in counts.items() if count
            ),
        ]
        if result.cut_detected:
            details.append("camera cut on this frame — every identity restarted here")
        details += result.reasons
        details += [f"WARNING: {warning}" for warning in result.warnings]

        analysis.reports.append(
            StageReport(
                key="tracking",
                phase="M2.4",
                title="Player identity tracking",
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
            frame_id=analysis.frame_index,
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
        unsure = teams.needs_confirmation()
        details = [
            f"confidence: {teams.confidence:.2f}",
            (
                f"{len(teams.settled())} settled, {len(unsure)} awaiting a "
                "confirmation (dashed outline)"
            ),
        ]
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

    def _follow_marks(self, analysis: FrameAnalysis, image, boxes, cut: bool):
        """Move the operator's marks with the camera, or drop them at a cut.

        Marks are image points. Left alone they stay pinned to the same pixels
        while the camera pans, and the calibration silently describes a view
        that no longer exists — with a reprojection error of zero throughout,
        because four points always agree with each other.
        """
        if cut and self.manual_correspondences:
            self.manual_correspondences = []
            self._follower.reset()
            return None

        if not self.manual_correspondences:
            self._follower.reset()
            return None

        if not self._follower.has_anchor:
            self._follower.anchor(
                self.manual_correspondences, image, analysis.frame_index, boxes
            )
            return None

        result = self._follower.update(image, analysis.frame_index, boxes)
        if result is None:
            return None
        if result.lost:
            self.manual_correspondences = []
            return result

        self.manual_correspondences = result.correspondences
        return result

    def _run_offside_line(self, analysis: FrameAnalysis) -> None:
        config = self._settings.offside.offside_line
        if not config.enabled:
            analysis.reports.append(
                StageReport(
                    key="offside_line",
                    phase="M2.5",
                    title="Second-last defender & offside line",
                    state=StageState.UNAVAILABLE,
                    summary="disabled in config",
                )
            )
            return

        decision = self._offside.decide(
            analysis.poses,
            analysis.teams,
            analysis.calibration,
            (analysis.frame.width, analysis.frame.height),
            identities=analysis.identities,
            ball_xy=analysis.ball_xy,
            attack_sign=self.attack_sign,
        )
        analysis.offside = decision

        state = {
            Verdict.OFFSIDE: StageState.OK,
            Verdict.ONSIDE: StageState.OK,
            Verdict.TOO_CLOSE: StageState.DEGRADED,
            Verdict.INCONCLUSIVE: StageState.UNAVAILABLE,
        }[decision.verdict]

        details = [f"confidence: {decision.confidence:.2f}"]
        if decision.second_last_defender is not None:
            defender = decision.second_last_defender
            details.append(
                f"line drawn through player {defender.index} "
                f"({defender.source}: {defender.reason})"
            )
        if decision.attacker is not None:
            details.append(
                f"most advanced attacker is player {decision.attacker.index}: "
                f"{decision.attacker.reason}"
            )
            beyond = [
                str(a.index) for a in decision.attackers if a.verdict is Verdict.OFFSIDE
            ]
            if beyond:
                details.append("attackers beyond the line: " + ", ".join(beyond))
        details += decision.reasons
        details += [f"WARNING: {warning}" for warning in decision.warnings]

        analysis.reports.append(
            StageReport(
                key="offside_line",
                phase="M2.5",
                title="Second-last defender & offside line",
                state=state,
                summary=decision.headline(),
                details=details,
            )
        )

    def _run_decision_support(self, analysis: FrameAnalysis) -> None:
        config = self._settings.offside.decision_support
        if not config.enabled:
            analysis.reports.append(
                StageReport(
                    key="decision_support",
                    phase="M2.6",
                    title="Confidence & reasoning",
                    state=StageState.UNAVAILABLE,
                    summary="disabled in config",
                )
            )
            return

        explanation = self._explainer.explain(
            analysis.offside,
            calibration=analysis.calibration,
            teams=analysis.teams,
            identities=analysis.identities,
            poses=analysis.poses,
        )
        analysis.explanation = explanation

        if explanation.band is ConfidenceBand.NONE:
            state = StageState.UNAVAILABLE
        elif explanation.band is ConfidenceBand.HIGH and explanation.is_published:
            state = StageState.OK
        else:
            state = StageState.DEGRADED

        details = [
            f"confidence: {explanation.confidence:.2f} ({explanation.band.value})",
        ]
        if explanation.weakest is not None:
            # The useful half of a low score is which stage to go and fix.
            details.append(
                f"weakest link: {explanation.weakest.label} "
                f"({explanation.weakest.phase}) at {explanation.weakest.score:.2f}"
            )
        if explanation.withheld:
            details.append(
                "the geometry reached a verdict that is NOT being published — "
                f"it reads {explanation.geometry_verdict.value}"
            )
        if explanation.detail:
            details.append(explanation.detail)
        details += [
            f"{signal.phase} {signal.label}: "
            + ("n/a" if not signal.available else f"{signal.score:.2f}")
            + f" — {signal.reason}"
            for signal in explanation.signals
        ]
        details += [f"limited by: {limit}" for limit in explanation.limits]
        details += [f"you can: {action}" for action in explanation.actions]

        analysis.reports.append(
            StageReport(
                key="decision_support",
                phase="M2.6",
                title="Confidence & reasoning",
                state=state,
                summary=explanation.headline,
                details=details,
            )
        )

    def _report_pending_stages(self, analysis: FrameAnalysis) -> None:
        """The rest of the plan, listed so the gap is visible rather than
        implied. Each becomes a real stage as its milestone lands.

        Empty as of M2.6: every stage of the offside pipeline now exists. The
        hook stays because the alternative — deleting it and rebuilding the
        listing for M3 — is how a pipeline panel quietly stops showing the
        stages nobody has written yet.
        """
        pending: list[tuple[str, str, str, str]] = []
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
        # A new mark re-anchors the follower: the operator has just told us
        # exactly where the pitch is on *this* frame, which is better evidence
        # than anything carried forward from an older one.
        self._follower.reset()
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
        self._follower.reset()

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

    def set_attack_direction(self, sign: float | None) -> None:
        """Operator states which way the attack is going, or hands it back."""
        self.attack_sign = sign

    def reset_identities(self) -> None:
        """Forget who is who — on a camera cut or a new clip."""
        self._identity_tracker.reset()
        self._team_assigner.reset_identities()

    def reset_team_colors(self) -> None:
        """Forget the kit colours and the accumulated per-player evidence —
        the right thing after a camera cut, when both describe another shot."""
        self._team_assigner.reset()
