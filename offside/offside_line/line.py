"""The offside line, the comparison, and the verdict (M2.5).

## The geometry is trivial. The honesty is not.

Once the pitch is calibrated, everyone is placed, and the sides are known,
the Law is a comparison of two numbers along one axis. All the difficulty in
this phase is inherited: a wrong calibration, a foot point off by half a
stride, a swapped identity, or the wrong end assumed as the defended goal all
produce a verdict that looks exactly as confident as a correct one.

So the central thing this module computes is not the margin — it is the
**uncertainty**, and whether the margin survives it:

    margin  =  how far the attacker is beyond the line
    error   =  what the inputs could plausibly be wrong by

    margin >  error   ->  offside position
    margin < -error   ->  onside
    otherwise         ->  too close to call, with the frame and the numbers

That last branch is not a failure mode, it is the product. A tool that says
"3cm offside" from a foot point measured to the nearest half-metre is lying;
one that says "too close to call, here is the frame" is doing the job the
client asked for.

## Both players are measured the same way

Law 11 compares body parts *nearer to the opponents' goal line*. That is the
same physical direction for both: the attacker's most advanced legal part,
and the defender's most advanced legal part towards the goal they defend. So
both come from M2.2's `leading_offside_point` along the same axis — arms
excluded, because a player cannot play the ball with them.

## What is deliberately not decided here

Which attacker is *involved* in play. That is football judgement — active
interference, deliberate play by a defender — and M2_Plan section 5 puts it
out of scope. This module reports every attacker in an offside *position*,
most advanced first, and leaves the judgement to the operator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from offside.body_keypoints.ground_point import leading_offside_point
from offside.body_keypoints.keypoints import PlayerPose
from offside.offside_line.axis import (
    AttackDirection,
    DepthAxis,
    infer_attack_direction,
)
from offside.second_last_defender.ranking import (
    DefenderRanking,
    RankedPlayer,
    image_direction_toward_goal,
    rank_opponents,
)

Point = tuple[float, float]


class Verdict(str, Enum):
    OFFSIDE = "offside"
    ONSIDE = "onside"
    #: The margin is inside what the measurement could be wrong by.
    TOO_CLOSE = "too_close_to_call"
    #: Something upstream was missing; there is no call to make.
    INCONCLUSIVE = "inconclusive"


@dataclass
class AttackerComparison:
    """One attacker measured against the line and the ball."""

    index: int
    track_id: str | None
    point: Point
    depth: float
    #: Beyond the second-last opponent by this much. Negative = behind them.
    margin: float
    #: Beyond the ball by this much. None when no ball position was supplied.
    margin_to_ball: float | None
    #: What this particular comparison could be wrong by.
    uncertainty: float
    verdict: Verdict
    confidence: float
    reason: str

    @property
    def beyond_line(self) -> bool:
        return self.margin > 0


@dataclass
class OffsideDecision:
    """The call, the geometry behind it, and everything that qualifies it."""

    verdict: Verdict = Verdict.INCONCLUSIVE
    confidence: float = 0.0
    #: The attacker the call is about — the most advanced one.
    attacker: AttackerComparison | None = None
    attackers: list[AttackerComparison] = field(default_factory=list)
    second_last_defender: RankedPlayer | None = None
    ranking: DefenderRanking | None = None
    direction: AttackDirection | None = None
    axis_unit: str = ""
    #: Image-space line through the second-last defender: a point and a unit
    #: direction, ready to draw. None when no line could be established.
    line: tuple[Point, Point] | None = None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_conclusive(self) -> bool:
        return self.verdict in (Verdict.OFFSIDE, Verdict.ONSIDE)

    def headline(self) -> str:
        """One line an operator can read without decoding anything."""
        if self.verdict is Verdict.INCONCLUSIVE:
            return "No call — " + (self.warnings[0] if self.warnings else "not enough to judge")
        if self.attacker is None:
            return "No attacker could be measured against the line"
        margin = abs(self.attacker.margin)
        if self.verdict is Verdict.TOO_CLOSE:
            return (
                f"Too close to call — {margin:.2f}{self.axis_unit} is inside what "
                f"this measurement can resolve (±{self.attacker.uncertainty:.2f}"
                f"{self.axis_unit})"
            )
        if self.verdict is Verdict.OFFSIDE:
            return f"Offside position — attacker is {margin:.2f}{self.axis_unit} beyond the line"
        return f"Onside — attacker is {margin:.2f}{self.axis_unit} behind the line"


class OffsideLineCalculator:
    """Draws the line, compares everyone to it, and refuses to over-claim."""

    def __init__(
        self,
        *,
        min_keypoint_confidence: float = 0.5,
        metric_base_uncertainty_m: float = 0.15,
        metric_foot_uncertainty_m: float = 0.6,
        directional_uncertainty_boxes: float = 0.35,
        min_team_separation_m: float = 3.0,
        min_team_separation_px: float = 40.0,
        require_beyond_ball: bool = True,
        min_confidence_to_call: float = 0.35,
    ):
        self._min_keypoint_confidence = min_keypoint_confidence
        self._metric_base = metric_base_uncertainty_m
        self._metric_foot = metric_foot_uncertainty_m
        self._directional_boxes = directional_uncertainty_boxes
        self._min_separation_m = min_team_separation_m
        self._min_separation_px = min_team_separation_px
        self._require_beyond_ball = require_beyond_ball
        self._min_confidence_to_call = min_confidence_to_call

    @classmethod
    def from_config(cls, config, *, min_keypoint_confidence: float = 0.5):
        return cls(
            min_keypoint_confidence=min_keypoint_confidence,
            metric_base_uncertainty_m=config.metric_base_uncertainty_m,
            metric_foot_uncertainty_m=config.metric_foot_uncertainty_m,
            directional_uncertainty_boxes=config.directional_uncertainty_boxes,
            min_team_separation_m=config.min_team_separation_m,
            min_team_separation_px=config.min_team_separation_px,
            require_beyond_ball=config.require_beyond_ball,
            min_confidence_to_call=config.min_confidence_to_call,
        )

    def decide(
        self,
        poses: list[PlayerPose],
        teams,
        calibration,
        image_size: tuple[int, int],
        *,
        identities=None,
        ball_xy: Point | None = None,
        attack_sign: float | None = None,
    ) -> OffsideDecision:
        decision = OffsideDecision()

        if teams is None or not teams.players:
            decision.warnings.append("no players were assigned to teams on this frame")
            return decision
        if not teams.sides_are_known:
            decision.warnings.append(
                "which side is attacking is unknown, so there is nothing to "
                "measure against the line"
            )
            return decision

        axis = DepthAxis.from_calibration(calibration, image_size)
        decision.axis_unit = axis.unit
        if not axis.available:
            decision.warnings.append(
                "the pitch is not calibrated on this frame — " + axis.description
            )
            return decision

        direction = self._direction(
            axis, teams, poses, attack_sign
        )
        decision.direction = direction
        if not direction.known:
            decision.warnings.append(direction.reason)
            return decision
        decision.reasons.append(direction.reason)

        ranking = rank_opponents(
            teams.opponents(),
            poses,
            axis,
            direction,
            identities=identities,
            min_keypoint_confidence=self._min_keypoint_confidence,
        )
        decision.ranking = ranking
        decision.reasons += ranking.reasons
        decision.warnings += ranking.warnings
        if not ranking.usable:
            return decision

        defender = ranking.second_last
        decision.second_last_defender = defender
        decision.line = self._line_through(calibration, defender.point, axis, direction)

        ball_depth = (
            direction.toward_goal(axis.depth(ball_xy)) if ball_xy is not None else None
        )
        decision.attackers = self._compare_attackers(
            teams, poses, axis, direction, defender, ball_depth, identities
        )
        if not decision.attackers:
            decision.warnings.append(
                "no attacking player could be measured against the line"
            )
            return decision

        decision.attackers.sort(key=lambda a: a.margin, reverse=True)
        decision.attacker = decision.attackers[0]
        decision.verdict = decision.attacker.verdict
        decision.confidence = decision.attacker.confidence
        self._apply_confidence_floor(decision, direction, defender)

        beyond = [a for a in decision.attackers if a.verdict is Verdict.OFFSIDE]
        if beyond:
            decision.reasons.append(
                f"{len(beyond)} attacker(s) are beyond the line; whether any of "
                "them is involved in the play is a judgement for the operator"
            )
        if ball_depth is None and self._require_beyond_ball:
            decision.warnings.append(
                "the ball was not located on this frame, so 'nearer the goal than "
                "the ball' could not be checked"
            )
        return decision

    def _apply_confidence_floor(self, decision, direction, defender) -> None:
        """Refuse to announce a verdict the inputs cannot support.

        The geometry can be perfect and the answer still worthless: measured
        on the reference clip, a frame with the two teams mixed together gave
        "offside by 14.6m" at a confidence of 0.08, because the *direction of
        attack* was barely more than a guess. A number that precise next to a
        confidence that low is exactly the confident-and-wrong output this
        milestone exists to avoid, so it becomes a refusal that names its own
        weakest link instead.
        """
        if decision.verdict is Verdict.INCONCLUSIVE:
            return
        if decision.confidence >= self._min_confidence_to_call:
            return

        decision.verdict = Verdict.INCONCLUSIVE
        decision.warnings.append(
            "the geometry produced an answer, but the inputs behind it are too "
            f"uncertain to report it ({self._weakest_link(decision, direction, defender)})"
        )

    def _weakest_link(self, decision, direction, defender) -> str:
        """Which input is holding the whole call back — the useful half of a
        low confidence score, and what the operator should go and fix."""
        candidates = [
            (direction.confidence, "which end is being defended is barely established"),
            (defender.confidence, "the second-last defender could not be measured well"),
        ]
        if decision.attacker is not None:
            candidates.append(
                (decision.attacker.confidence, "the attacker could not be measured well")
            )
        return min(candidates, key=lambda item: item[0])[1]

    # -- steps --------------------------------------------------------------

    def _direction(self, axis: DepthAxis, teams, poses, attack_sign):
        def depths(players) -> list[float]:
            points = [
                poses[p.index].ground_point.xy
                for p in players
                if p.index < len(poses)
            ]
            return [d for d in axis.depths(points) if d is not None]

        keepers = [p for p in teams.goalkeepers() if p.index < len(poses)]
        return infer_attack_direction(
            axis,
            depths(teams.attackers()),
            depths(teams.opponents()),
            depths(keepers),
            min_team_separation=(
                self._min_separation_m if axis.is_metric else self._min_separation_px
            ),
            operator_sign=attack_sign,
        )

    def _compare_attackers(
        self, teams, poses, axis, direction, defender, ball_depth, identities
    ) -> list[AttackerComparison]:
        image_direction = image_direction_toward_goal(axis, direction)
        comparisons: list[AttackerComparison] = []

        for attacker in teams.attackers():
            if attacker.index >= len(poses):
                continue
            pose = poses[attacker.index]
            leading = leading_offside_point(
                pose,
                image_direction,
                min_keypoint_confidence=self._min_keypoint_confidence,
            )
            depth = direction.toward_goal(axis.depth(leading.xy))
            if depth is None:
                continue

            margin = depth - defender.depth
            margin_to_ball = None if ball_depth is None else depth - ball_depth
            uncertainty = self._uncertainty(axis, pose, leading, defender)

            identity = identities.by_index(attacker.index) if identities else None
            confidence = min(
                attacker.confidence if attacker.confidence > 0 else 1.0,
                leading.confidence,
                defender.confidence,
                direction.confidence,
                identity.confidence if identity is not None else 1.0,
            )

            verdict, reason = self._judge(
                margin, margin_to_ball, uncertainty, axis.unit
            )
            comparisons.append(
                AttackerComparison(
                    index=attacker.index,
                    track_id=attacker.track_id,
                    point=leading.xy,
                    depth=depth,
                    margin=margin,
                    margin_to_ball=margin_to_ball,
                    uncertainty=uncertainty,
                    verdict=verdict,
                    confidence=float(confidence),
                    reason=reason,
                )
            )
        return comparisons

    def _judge(
        self, margin: float, margin_to_ball: float | None, uncertainty: float, unit: str
    ) -> tuple[Verdict, str]:
        if abs(margin) <= uncertainty:
            return (
                Verdict.TOO_CLOSE,
                f"{margin:+.2f}{unit} from the line, which is inside the "
                f"±{uncertainty:.2f}{unit} this measurement can resolve",
            )
        if margin < 0:
            return (
                Verdict.ONSIDE,
                f"{abs(margin):.2f}{unit} behind the second-last opponent",
            )
        if (
            self._require_beyond_ball
            and margin_to_ball is not None
            and margin_to_ball <= 0
        ):
            # Beyond the defender but level with or behind the ball is onside,
            # and it is a mistake a geometry-only tool makes constantly if the
            # ball is left out of the comparison.
            return (
                Verdict.ONSIDE,
                f"{margin:.2f}{unit} beyond the second-last opponent but not "
                "beyond the ball, which is onside",
            )
        return (
            Verdict.OFFSIDE,
            f"{margin:.2f}{unit} beyond the second-last opponent"
            + (
                f" and {margin_to_ball:.2f}{unit} beyond the ball"
                if margin_to_ball is not None
                else ""
            ),
        )

    def _uncertainty(self, axis: DepthAxis, pose: PlayerPose, leading, defender) -> float:
        """What this comparison could plausibly be wrong by.

        Two players are being measured, so both contribute; a confident body
        point costs little and a box-bottom guess costs a lot. Nothing here
        pretends to be a statistical bound — it is a deliberately generous
        estimate, because the failure that matters is claiming a verdict the
        inputs cannot support.
        """
        if axis.is_metric:
            attacker_doubt = (1.0 - leading.confidence) * self._metric_foot
            defender_doubt = (1.0 - defender.confidence) * self._metric_foot
            return self._metric_base + attacker_doubt + defender_doubt

        # Directional: the axis is in pixels, so a player's own height is the
        # only scale available, and it is a good one — it shrinks with distance
        # exactly as the measurement error does.
        box_height = max(1.0, pose.box_height)
        return self._directional_boxes * box_height

    def _line_through(self, calibration, point: Point, axis: DepthAxis, direction):
        """The offside line in image space, ready to draw."""
        if calibration is not None:
            drawn = calibration.offside_line_through(point)
            if drawn is not None:
                return drawn
        if axis.normal is not None:
            # Perpendicular to the goal-to-goal direction is the line itself.
            nx, ny = axis.normal
            return (point, (-ny, nx))
        return None
