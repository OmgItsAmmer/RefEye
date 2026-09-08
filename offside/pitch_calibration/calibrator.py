"""Turning a frame into a usable pitch mapping — and saying how usable it is.

## The honest shape of this problem

Detecting the painted lines is easy. Knowing *which* line is the 16.5m line
rather than the six-yard line is the hard part, and it is not solved here.
Two things were tried against the reference clip and are recorded so nobody
repeats them expecting a different answer:

  * **"Lines parallel to the goal line look steep."** True for a camera on
    the halfway line; false on the reference clip, where the camera sits off
    to one side and the penalty-area front line reads as a shallow diagonal.
    Absolute image angle carries no reliable pitch semantics.
  * **Brightness alone finds pitch lines.** It finds advertising boards and
    white shirts far more strongly — see `line_detection.pitch_line_mask`.

So this module does not pretend. It reports one of three capability levels
and lets everything downstream ask what it is actually allowed to compute:

    NONE         nothing usable. Say so; do not guess.
    DIRECTIONAL  the goal-line vanishing point is known, so an offside line
                 can be drawn through any player — but nobody's position in
                 metres is known, and "is the attacker in their own half?"
                 cannot be answered.
    METRIC       a full homography. Positions in metres, top-down map, half
                 checks, distances.

DIRECTIONAL is worth the extra concept because it is *most* of offside:
the whole judgement is "is this attacker beyond that defender, along the
goal-line direction", and a vanishing point answers it without a metric map.
It also needs far less evidence — two parallel markings anywhere in frame,
versus four identified landmarks.

## Where each level comes from

METRIC comes from the operator: four or more clicked landmarks
(`calibrate_manual`). That path is reliable today and is the one the debug
UI drives.

DIRECTIONAL can come automatically (`calibrate_auto`), but with a caveat that
must reach the operator: the line families it finds are geometrically real,
yet which family is parallel to the goal line is an assumption. The debug UI
shows both families so a human can confirm in one click. A learned pitch-
keypoint model (SoccerNet-style) is the upgrade that would make this
automatic and metric — it drops in behind this same interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from observability.logging.setup import get_logger
from offside.field_geometry.pitch import PitchModel
from offside.pitch_calibration.homography import (
    HomographyFit,
    PointCorrespondence,
    line_through_vanishing_point,
    solve_homography,
)
from offside.pitch_calibration.line_detection import (
    DetectedLine,
    VanishingPoint,
    detect_pitch_lines,
    estimate_vanishing_point,
)

logger = get_logger(__name__)

Point = tuple[float, float]


class CalibrationLevel(str, Enum):
    NONE = "none"
    DIRECTIONAL = "directional"
    METRIC = "metric"


@dataclass
class LineFamily:
    """Lines that are parallel on the pitch, and where they meet in the image."""

    lines: list[DetectedLine]
    vanishing_point: VanishingPoint

    @property
    def mean_angle_deg(self) -> float:
        return float(np.mean([line.angle_deg for line in self.lines]))


@dataclass
class PitchCalibration:
    """What this frame's geometry supports, and what it does not."""

    level: CalibrationLevel
    confidence: float
    source: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    fit: HomographyFit | None = None
    goal_line_vanishing_point: Point | None = None
    #: Every family found, so the operator (or a later phase) can choose a
    #: different one rather than being stuck with this module's guess.
    families: list[LineFamily] = field(default_factory=list)
    detected_lines: list[DetectedLine] = field(default_factory=list)
    #: Which shot this belongs to. A calibration is only valid for its own
    #: camera; crossing a cut invalidates it (architecture.md section 26).
    segment: int = 0

    @property
    def is_metric(self) -> bool:
        return self.level is CalibrationLevel.METRIC and self.fit is not None

    @property
    def can_draw_offside_line(self) -> bool:
        return self.level in (CalibrationLevel.METRIC, CalibrationLevel.DIRECTIONAL)

    def to_pitch(self, points) -> np.ndarray | None:
        """Image points -> pitch metres, or None without a metric fit."""
        return self.fit.to_pitch(points) if self.is_metric else None

    def to_image(self, points) -> np.ndarray | None:
        return self.fit.to_image(points) if self.is_metric else None

    def offside_line_through(self, point: Point) -> tuple[Point, Point] | None:
        """Point and unit direction of the offside line through `point`.

        Parallel to the goal line on the pitch, which in the image means
        "aimed at the goal-line vanishing point".
        """
        if not self.can_draw_offside_line:
            return None

        vanishing_point = self.goal_line_vanishing_point
        if vanishing_point is None and self.is_metric:
            # Along the goal line the pitch direction is (0, 1): across the
            # width, at constant length.
            vanishing_point = self.fit.vanishing_point_of((0.0, 1.0))

        if vanishing_point is None:
            return None
        try:
            return line_through_vanishing_point(point, vanishing_point, (0.0, 1.0))
        except ValueError:
            return None


class PitchCalibrator:
    """Builds `PitchCalibration`s, automatically or from operator input."""

    def __init__(
        self,
        pitch: PitchModel | None = None,
        *,
        max_reprojection_error_m: float = 2.0,
        min_family_support: int = 2,
        inlier_angle_deg: float = 2.5,
        min_lines_for_auto: int = 4,
        line_detection_kwargs: dict | None = None,
    ):
        self._pitch = pitch or PitchModel()
        self._max_error_m = max_reprojection_error_m
        self._min_family_support = min_family_support
        self._inlier_angle_deg = inlier_angle_deg
        self._min_lines_for_auto = min_lines_for_auto
        self._line_kwargs = line_detection_kwargs or {}

    @property
    def pitch(self) -> PitchModel:
        return self._pitch

    # -- the reliable path --------------------------------------------------

    def calibrate_manual(
        self,
        correspondences: list[PointCorrespondence],
        *,
        segment: int = 0,
        source: str = "manual",
    ) -> PitchCalibration:
        """Full metric calibration from a set of identified landmarks.

        This is the path that actually works today on arbitrary broadcast
        footage, and the plan requires it to exist regardless (M2_Plan
        section 7: manual override at every classification stage).

        `source` defaults to "manual" (an operator's own clicks) but is
        overridable — the auto landmark detector (`auto_landmarks.py`) feeds
        its own points through this exact same solve, and must be able to
        say so honestly rather than claiming to be the operator's work.
        """
        if len(correspondences) < 4:
            return PitchCalibration(
                level=CalibrationLevel.NONE,
                confidence=0.0,
                source=source,
                segment=segment,
                reasons=[
                    (
                        f"{len(correspondences)} reference points marked — four "
                        "are needed before the pitch can be mapped"
                    )
                ],
            )

        fit = solve_homography(correspondences, max_error_m=self._max_error_m)
        if fit is None:
            return PitchCalibration(
                level=CalibrationLevel.NONE,
                confidence=0.0,
                source=source,
                segment=segment,
                reasons=[
                    (
                        "those reference points do not define a valid mapping — "
                        "three of them may be in a straight line, or two may be "
                        "the same landmark"
                    )
                ],
            )

        warnings: list[str] = []
        if fit.max_error_m > self._max_error_m:
            warnings.append(
                f"one marked point is {fit.max_error_m:.1f}m away from where the "
                "others say it should be — check it is on the landmark it claims"
            )

        logger.info(
            "pitch_calibrated",
            component="pitch_calibration",
            source=source,
            points=len(correspondences),
            mean_error_m=round(fit.mean_error_m, 3),
            confidence=round(fit.confidence, 3),
        )

        return PitchCalibration(
            level=CalibrationLevel.METRIC,
            confidence=fit.confidence,
            source=source,
            reasons=list(fit.reasons),
            warnings=warnings,
            fit=fit,
            goal_line_vanishing_point=fit.vanishing_point_of((0.0, 1.0)),
            segment=segment,
        )

    # -- the automatic attempt ----------------------------------------------

    def calibrate_auto(
        self,
        image: np.ndarray,
        *,
        exclude_boxes: list[tuple[float, float, float, float]] | None = None,
        goal_line_family_index: int | None = None,
        segment: int = 0,
    ) -> PitchCalibration:
        """Best effort from the image alone.

        Reaches DIRECTIONAL at most: it can find families of pitch-parallel
        lines and where they converge, but not which family runs parallel to
        the goal line. `goal_line_family_index` lets the operator (or a later
        phase) say which one — that single choice is what turns a geometric
        observation into an offside-capable calibration.
        """
        lines = detect_pitch_lines(image, exclude_boxes=exclude_boxes, **self._line_kwargs)
        families = self.find_line_families(lines)

        if len(lines) < self._min_lines_for_auto or not families:
            return PitchCalibration(
                level=CalibrationLevel.NONE,
                confidence=0.0,
                source="auto",
                segment=segment,
                detected_lines=lines,
                families=families,
                reasons=[
                    (
                        f"only {len(lines)} pitch markings could be made out — "
                        "not enough to work out how the camera is looking at "
                        "the pitch"
                    )
                ],
                warnings=["mark the pitch manually to continue"],
            )

        if goal_line_family_index is None:
            chosen_index = 0
            assumption = (
                "assumed the largest group of parallel markings runs parallel to "
                "the goal line — this has NOT been verified and is wrong on some "
                "camera angles; confirm it before trusting an offside line"
            )
        else:
            chosen_index = goal_line_family_index
            assumption = "goal-line direction was chosen by the operator"

        if not 0 <= chosen_index < len(families):
            return PitchCalibration(
                level=CalibrationLevel.NONE,
                confidence=0.0,
                source="auto",
                segment=segment,
                detected_lines=lines,
                families=families,
                reasons=[f"no line group {chosen_index} was found in this frame"],
            )

        family = families[chosen_index]
        # Deliberately capped: the geometry is sound, the *semantics* are an
        # assumption, and confidence here feeds M2.6's decision about whether
        # a call is safe to make.
        confidence = family.vanishing_point.confidence
        if goal_line_family_index is None:
            confidence = min(confidence, 0.4)

        return PitchCalibration(
            level=CalibrationLevel.DIRECTIONAL,
            confidence=confidence,
            source="auto",
            segment=segment,
            detected_lines=lines,
            families=families,
            goal_line_vanishing_point=family.vanishing_point.xy,
            reasons=[
                (
                    f"{len(lines)} pitch markings found, forming "
                    f"{len(families)} group(s) of parallel lines"
                ),
                f"the chosen group has {family.vanishing_point.support} agreeing lines",
            ],
            warnings=[
                assumption,
                (
                    "no distances in metres are available — positions on the "
                    "pitch, and which half a player is in, cannot be checked"
                ),
            ],
        )

    def find_line_families(self, lines: list[DetectedLine]) -> list[LineFamily]:
        """Group lines by what they converge on, not by how they look.

        Greedy: take the strongest consensus vanishing point, claim its
        supporters, repeat on what is left. Grouping by absolute angle was
        tried first and is unsound — see the module docstring.
        """
        remaining = list(lines)
        families: list[LineFamily] = []

        while len(remaining) >= self._min_family_support:
            vanishing_point = estimate_vanishing_point(
                remaining,
                min_support=self._min_family_support,
                inlier_angle_deg=self._inlier_angle_deg,
            )
            if vanishing_point is None:
                break

            members = [
                line
                for line in remaining
                if _points_at(line, vanishing_point.xy, self._inlier_angle_deg)
            ]
            if len(members) < self._min_family_support:
                break

            families.append(LineFamily(lines=members, vanishing_point=vanishing_point))
            remaining = [line for line in remaining if line not in members]

        families.sort(key=lambda f: (f.vanishing_point.support, -f.vanishing_point.residual_deg), reverse=True)
        return families


def _points_at(line: DetectedLine, target: Point, tolerance_deg: float) -> bool:
    import math

    mid = line.midpoint
    dx, dy = target[0] - mid[0], target[1] - mid[1]
    if math.hypot(dx, dy) < 1e-6:
        return False
    bearing = math.degrees(math.atan2(dy, dx)) % 180.0
    gap = abs(line.angle_deg - bearing) % 180.0
    return min(gap, 180.0 - gap) <= tolerance_deg
