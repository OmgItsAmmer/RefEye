"""How far up the pitch a point is, and which way "up the pitch" runs.

## Two questions, deliberately separated

**Where is everyone along the goal-to-goal axis?** That is geometry, and it
comes from the calibration: metres at METRIC, and at DIRECTIONAL an ordering
with no unit — enough to say who is in front of whom, which is most of
offside, but not how far apart they are.

**Which end are the defenders defending?** That is not geometry at all. The
same picture read from either end gives opposite verdicts, so this has to be
established from evidence and reported with its own confidence, never assumed.
Two sources, in order of trust:

1. **The goalkeeper, but only when they are actually the deepest player.** A
   keeper stands at their own goal — until they come for a cross, at which
   point they are level with their own defensive line and prove nothing about
   which end anybody is defending. Reading the direction off them there would
   invert the whole verdict, so the keeper counts only when they are the most
   extreme player on the frame; otherwise this falls through to the shape.
2. **The shape of the two teams.** A defending side sits nearer its own goal
   than the attacking side does. That is reliable in open play and wrong in
   exactly the situation offside matters least — everyone camped in one box.
   So its confidence comes from how far apart the two teams' averages are
   compared with how spread out they each are.

When neither is convincing the direction is reported as unknown, and the
verdict that would have been built on it becomes inconclusive rather than a
coin flip. The operator can also state it outright, which outranks both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Point = tuple[float, float]

#: Where the attack direction came from.
SOURCE_GOALKEEPER = "goalkeeper"
SOURCE_TEAM_SHAPE = "team_shape"
SOURCE_OPERATOR = "operator"


@dataclass(frozen=True)
class DepthAxis:
    """Position along the goal-to-goal axis, in whatever units are available."""

    kind: str          # "metric" | "directional" | "none"
    unit: str
    description: str
    #: Image-space direction that "further up the pitch" points along, used at
    #: DIRECTIONAL level where there is no pitch coordinate to read.
    normal: Point | None = None
    calibration: object | None = None

    @property
    def available(self) -> bool:
        return self.kind != "none"

    @property
    def is_metric(self) -> bool:
        return self.kind == "metric"

    @classmethod
    def from_calibration(cls, calibration, image_size: tuple[int, int]) -> "DepthAxis":
        """image_size is (width, height) — needed for the directional fallback."""
        if calibration is None:
            return cls(
                kind="none",
                unit="",
                description="no pitch calibration, so nobody's depth on the pitch is known",
            )

        if getattr(calibration, "is_metric", False):
            return cls(
                kind="metric",
                unit="m",
                description="distance up the pitch, in metres",
                calibration=calibration,
            )

        if getattr(calibration, "can_draw_offside_line", False):
            width, height = image_size
            line = calibration.offside_line_through((width / 2.0, height / 2.0))
            if line is not None:
                _, (dx, dy) = line
                # Perpendicular to the offside line is "towards the goal".
                # Taken once at frame centre and reused: the true direction
                # fans out slightly across the frame, so this orders players
                # correctly but does not measure the distance between them.
                return cls(
                    kind="directional",
                    unit="px",
                    description=(
                        "ordering along the goal-to-goal direction (no metres available)"
                    ),
                    normal=(-dy, dx),
                    calibration=calibration,
                )

        return cls(
            kind="none",
            unit="",
            description="the pitch markings do not support a goal-line direction",
        )

    def depth(self, point: Point) -> float | None:
        """How far up the pitch this image point is, or None if unknowable."""
        depths = self.depths([point])
        return depths[0] if depths else None

    def depths(self, points: list[Point]) -> list[float | None]:
        if not points:
            return []
        if self.kind == "metric":
            projected = self.calibration.to_pitch(points)
            return [
                float(position[0]) if np.all(np.isfinite(position)) else None
                for position in projected
            ]
        if self.kind == "directional" and self.normal is not None:
            nx, ny = self.normal
            return [float(x * nx + y * ny) for x, y in points]
        return [None] * len(points)

    def span(self, points: list[Point]) -> float:
        known = [d for d in self.depths(points) if d is not None]
        return (max(known) - min(known)) if len(known) > 1 else 0.0


@dataclass
class AttackDirection:
    """Which way the attacking team is playing, and how sure that is."""

    #: Multiply a depth by this and bigger always means "nearer the goal the
    #: defenders are defending".
    sign: float
    confidence: float
    source: str
    reason: str

    @property
    def known(self) -> bool:
        return self.confidence > 0.0

    def toward_goal(self, depth: float | None) -> float | None:
        return None if depth is None else self.sign * depth


def infer_attack_direction(
    axis: DepthAxis,
    attacker_depths: list[float],
    defender_depths: list[float],
    goalkeeper_depths: list[float] | None = None,
    *,
    min_team_separation: float = 1.0,
    operator_sign: float | None = None,
) -> AttackDirection:
    """Work out which end is being defended, from the keeper or the shape."""
    if operator_sign is not None:
        return AttackDirection(
            sign=operator_sign,
            confidence=1.0,
            source=SOURCE_OPERATOR,
            reason="the operator said which way the attack is going",
        )

    everyone = attacker_depths + defender_depths + list(goalkeeper_depths or [])
    if len(everyone) < 3:
        return _unknown("too few players on the frame to tell which end is which")

    middle = float(np.mean(everyone))

    if goalkeeper_depths:
        keeper = float(np.mean(goalkeeper_depths))
        # Only a keeper who is deeper than everybody else is evidence. One who
        # has come for a cross sits level with their own defensive line and
        # says nothing about which end is being defended.
        extreme = keeper >= max(everyone) - 1e-6 or keeper <= min(everyone) + 1e-6
        if extreme and abs(keeper - middle) > 1e-6:
            return AttackDirection(
                sign=1.0 if keeper > middle else -1.0,
                confidence=0.9,
                source=SOURCE_GOALKEEPER,
                reason=(
                    "the goalkeeper is the deepest player at that end, so that "
                    "is the goal being defended"
                ),
            )

    if not attacker_depths or not defender_depths:
        return _unknown(
            "only one side could be placed on the pitch, so which goal is being "
            "defended is unknown"
        )

    attack_mean = float(np.mean(attacker_depths))
    defend_mean = float(np.mean(defender_depths))
    separation = defend_mean - attack_mean

    spread = float(
        np.mean(
            [
                np.std(attacker_depths) if len(attacker_depths) > 1 else 0.0,
                np.std(defender_depths) if len(defender_depths) > 1 else 0.0,
            ]
        )
    )
    if abs(separation) < max(min_team_separation, 1e-6):
        return _unknown(
            "the two teams are mixed together on this frame, so which end each "
            "is attacking cannot be read from their positions"
        )

    # A confident reading needs the gap between the sides to be large next to
    # how spread out each side is; two interleaved lines prove nothing.
    confidence = float(min(0.75, abs(separation) / max(spread * 2.0, 1e-6) * 0.5))
    return AttackDirection(
        sign=1.0 if separation > 0 else -1.0,
        confidence=confidence,
        source=SOURCE_TEAM_SHAPE,
        reason=(
            "the defending side is sitting nearer that end of the pitch than the "
            f"attacking side ({abs(separation):.0f}{axis.unit} apart on average)"
        ),
    )


def _unknown(reason: str) -> AttackDirection:
    return AttackDirection(sign=1.0, confidence=0.0, source=SOURCE_TEAM_SHAPE, reason=reason)
