"""Finding the player the offside line is drawn through (M2.5).

## What the Law actually says, and what that changes

"Nearer to the opponents' goal line than **the second-last opponent**." Not
the last defender — the second-last *opponent*, whoever that turns out to be.
Usually that is the last outfield defender because the goalkeeper is deeper,
but not always: a keeper who has come for a cross is not the last opponent,
and then the second-last is a defender who would otherwise have been third.

So this module ranks **every opponent** — goalkeeper included, and without
caring which is which — and takes the second one. That is why M2.3's
goalkeeper identification is not load-bearing for the verdict: it makes the
explanation readable, and nothing more.

## Why the ranking uses the leading point, not the feet

The line is drawn at the body part nearest the goal line, which is a leg, a
shoulder, or a head depending on how the player is standing. M2.2 already
knows which parts may legally be measured (arms may not) and can give the
most advanced legal one along a direction — so the direction is handed in
here and the answer comes back Law-correct rather than approximated by a
bounding box.

## Refusing to rank

Two opponents are the minimum: with one, there is no second-last and the only
honest answer is that the line cannot be drawn. Players whose depth could not
be established are excluded from the ranking and *reported*, because a
missing opponent is exactly the one who might have been second-last — the
hardest player to measure (distant, occluded) is the one whose absence moves
the line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from offside.body_keypoints.ground_point import leading_offside_point
from offside.body_keypoints.keypoints import PlayerPose
from offside.offside_line.axis import AttackDirection, DepthAxis

Point = tuple[float, float]


@dataclass
class RankedPlayer:
    """One opponent, placed along the goal-to-goal axis."""

    index: int
    track_id: str | None
    #: The most advanced legal body part, in image coordinates.
    point: Point
    #: Position along the axis, already signed so bigger = nearer the goal
    #: being defended.
    depth: float
    #: How much this player's own measurement can be trusted — the foot point
    #: from M2.2 and the identity from M2.4 both feed it.
    confidence: float
    source: str
    reason: str


@dataclass
class DefenderRanking:
    """The opponents in order, and which one the line goes through."""

    ranked: list[RankedPlayer] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def last(self) -> RankedPlayer | None:
        return self.ranked[0] if self.ranked else None

    @property
    def second_last(self) -> RankedPlayer | None:
        """The player the offside line is drawn through, per Law 11."""
        return self.ranked[1] if len(self.ranked) > 1 else None

    @property
    def usable(self) -> bool:
        return self.second_last is not None


def rank_opponents(
    opponents,
    poses: list[PlayerPose],
    axis: DepthAxis,
    direction: AttackDirection,
    *,
    identities=None,
    min_keypoint_confidence: float = 0.5,
) -> DefenderRanking:
    """Order the defending side from nearest their own goal line outwards."""
    ranking = DefenderRanking()

    if not axis.available:
        ranking.warnings.append(
            "without a calibrated pitch there is no goal-to-goal direction, so "
            "the defenders cannot be put in order"
        )
        return ranking
    if not direction.known:
        ranking.warnings.append(
            "which end is being defended is unknown, so 'nearest the goal line' "
            "has no meaning yet — " + direction.reason
        )
        return ranking

    image_direction = image_direction_toward_goal(axis, direction)

    for opponent in opponents:
        if opponent.index >= len(poses):
            continue
        pose = poses[opponent.index]

        leading = leading_offside_point(
            pose, image_direction, min_keypoint_confidence=min_keypoint_confidence
        )
        point = leading.xy
        depth = direction.toward_goal(axis.depth(point))
        if depth is None:
            ranking.excluded.append(
                f"player {opponent.index} could not be placed on the pitch"
            )
            continue

        identity = identities.by_index(opponent.index) if identities else None
        # The weakest link decides: a perfectly measured body point on a
        # player whose identity was swapped is still the wrong player.
        confidence = min(
            opponent.confidence if opponent.confidence > 0 else 1.0,
            leading.confidence,
            identity.confidence if identity is not None else 1.0,
        )

        ranking.ranked.append(
            RankedPlayer(
                index=opponent.index,
                track_id=opponent.track_id,
                point=point,
                depth=depth,
                confidence=float(confidence),
                source="player box" if leading.is_fallback else "leading body point",
                reason=leading.reason,
            )
        )

    ranking.ranked.sort(key=lambda player: player.depth, reverse=True)

    if ranking.excluded:
        # Never silent: the player who could not be measured is exactly the one
        # who might have been second-last.
        ranking.warnings.append(
            f"{len(ranking.excluded)} defending player(s) could not be placed on "
            "the pitch and were left out of the ranking; if one of them was "
            "deeper than the line, the line is in the wrong place"
        )

    if not ranking.usable:
        ranking.warnings.append(
            f"only {len(ranking.ranked)} defending player(s) could be measured — "
            "the Law needs the second-last opponent, so at least two are required"
        )
        return ranking

    second = ranking.second_last
    ranking.reasons.append(
        f"{len(ranking.ranked)} defending players ranked; the line is drawn "
        f"through the second-deepest of them (player {second.index})"
    )
    gap = ranking.last.depth - second.depth
    ranking.reasons.append(
        f"the deepest defender is {abs(gap):.1f}{axis.unit} beyond them"
    )
    return ranking


def image_direction_toward_goal(axis: DepthAxis, direction: AttackDirection) -> Point:
    """Which way "towards the defended goal" points, in image coordinates.

    M2.2's leading-point policy works in the image, so the pitch-space
    direction has to be expressed there. At DIRECTIONAL level the axis already
    carries that vector; at METRIC level the pitch x-axis is used, which the
    homography maps into the image for us at the point of measurement.
    """
    if axis.normal is not None:
        return (axis.normal[0] * direction.sign, axis.normal[1] * direction.sign)

    calibration = axis.calibration
    if calibration is not None and getattr(calibration, "is_metric", False):
        # Two points a metre apart along the pitch's length, projected into the
        # image, give the image-space direction of "up the pitch".
        try:
            here = calibration.to_image([(50.0, 34.0)])[0]
            ahead = calibration.to_image([(51.0, 34.0)])[0]
            dx, dy = float(ahead[0] - here[0]), float(ahead[1] - here[1])
            length = (dx * dx + dy * dy) ** 0.5
            if length > 1e-6:
                return (dx / length * direction.sign, dy / length * direction.sign)
        except Exception:  # noqa: BLE001 — a bad homography must not crash a verdict
            pass
    return (direction.sign, 0.0)
