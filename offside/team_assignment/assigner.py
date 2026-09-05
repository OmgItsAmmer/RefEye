"""Who is attacking, who is defending, who is in goal (M2.3).

## The order of operations, and why it is this order

    1. measure every player's kit colour            (jersey_color.py)
    2. fit the two kits, dropping who fits neither  (clustering.py)
    3. label the dropped players by *position*, not colour
    4. decide which of the two kits is attacking, from who has the ball
    5. apply operator corrections over the top of all of it

Steps 3 and 4 are the two that most often get collapsed into step 2 by
mistake, and both matter:

**A goalkeeper cannot be found by colour alone.** "Wears something different"
also describes the referee, a substitute on the touchline and any player
whose shirt failed to measure. What separates the keeper is *where they
stand*: alone, behind everybody, at one end. So the outliers from step 2 are
ranked along the goal-to-goal axis and only a positionally extreme one is
called a goalkeeper. When the pitch is not calibrated there is no such axis,
and then this stage says it cannot tell — which is the correct answer, not a
degradation to be papered over.

**Attacking is not a property of a kit.** Clustering yields two anonymous
groups; which one is attacking is a fact about this moment, and it comes from
the ball. The player nearest the ball at the contact frame donates their
group as "attacking". If there is no ball detection, or nobody is credibly
near it, the sides stay unknown and M2.5 is told so rather than being handed
a coin flip.

## What "manual override" means here

M2_Plan section 7 requires an override at every classification stage. It
lives in `TeamOverrides` — a data-layer object, not a UI concern — so the
pipeline debugger, the M2.7 review panel and a test can all drive the same
correction path. Three kinds of correction cover what actually goes wrong:

    pin_player   this specific player is on that team / is the keeper
    swap_teams   the two groups are right, the labels are the wrong way round
    set_attacking_team   the ball-possession guess picked the wrong side

An overridden player is marked `SOURCE_OPERATOR` with confidence 1.0, so
M2.6 can say "the operator confirmed this" instead of claiming the vision
system was sure.

## Identity, until M2.4 lands

Overrides are keyed by track id where one exists and fall back to matching
the nearest player within a pixel radius. That fallback is honest but weak
across long gaps — it is what M2.4 (stable identities through the contact
moment) exists to replace, and pins should move to track ids the moment it
does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.errors.exceptions import ConfigurationError
from observability.logging.setup import get_logger
from offside.body_keypoints.keypoints import PlayerPose
from offside.team_assignment.clustering import fit_team_colors
from offside.team_assignment.jersey_color import JerseyColorExtractor, TeamFeatureExtractor
from offside.team_assignment.teams import (
    SOURCE_KIT_COLOUR,
    SOURCE_OPERATOR,
    SOURCE_OUTLIER,
    SOURCE_UNMEASURED,
    TEAM_IDS,
    JerseyColor,
    PlayerRole,
    PlayerTeam,
    TeamAssignment,
    TeamColorModel,
)

logger = get_logger(__name__)

Point = tuple[float, float]


# -- operator corrections ---------------------------------------------------


@dataclass
class TeamOverride:
    """One operator correction, and how to find the player it applies to."""

    anchor_xy: Point
    team_id: str | None = None
    role: PlayerRole | None = None
    track_id: str | None = None

    def describe(self) -> str:
        parts = []
        if self.team_id:
            parts.append(f"team {self.team_id[-1].upper()}")
        if self.role:
            parts.append(self.role.value)
        return " / ".join(parts) if parts else "cleared"


class TeamOverrides:
    """The operator's corrections, applied over whatever the vision produced."""

    def __init__(self, match_distance_px: float = 40.0):
        self._match_distance_px = match_distance_px
        self.pins: list[TeamOverride] = []
        #: True when the operator says the two groups carry each other's names.
        self.teams_swapped: bool = False
        #: Set when the operator overrules the ball-possession guess.
        self.attacking_team_id: str | None = None

    @property
    def is_active(self) -> bool:
        return bool(self.pins) or self.teams_swapped or self.attacking_team_id is not None

    def pin_player(
        self,
        anchor_xy: Point,
        *,
        team_id: str | None = None,
        role: PlayerRole | None = None,
        track_id: str | None = None,
    ) -> TeamOverride:
        self.clear_player(anchor_xy)
        override = TeamOverride(
            anchor_xy=anchor_xy, team_id=team_id, role=role, track_id=track_id
        )
        self.pins.append(override)
        return override

    def clear_player(self, anchor_xy: Point) -> bool:
        before = len(self.pins)
        self.pins = [
            pin
            for pin in self.pins
            if _distance(pin.anchor_xy, anchor_xy) > self._match_distance_px
        ]
        return len(self.pins) != before

    def swap_teams(self) -> bool:
        self.teams_swapped = not self.teams_swapped
        return self.teams_swapped

    def set_attacking_team(self, team_id: str | None) -> None:
        self.attacking_team_id = team_id

    def clear(self) -> None:
        self.pins = []
        self.teams_swapped = False
        self.attacking_team_id = None

    def resolve(self, poses: list[PlayerPose]) -> dict[int, TeamOverride]:
        """Match each pin to at most one player in this frame."""
        resolved: dict[int, TeamOverride] = {}
        taken: set[int] = set()

        for pin in self.pins:
            index = self._match(pin, poses, taken)
            if index is None:
                continue
            resolved[index] = pin
            taken.add(index)
        return resolved

    def _match(
        self, pin: TeamOverride, poses: list[PlayerPose], taken: set[int]
    ) -> int | None:
        if pin.track_id is not None:
            for index, pose in enumerate(poses):
                if index not in taken and pose.track_id == pin.track_id:
                    return index

        best_index, best_distance = None, self._match_distance_px
        for index, pose in enumerate(poses):
            if index in taken:
                continue
            distance = _distance(_anchor_of(pose), pin.anchor_xy)
            if distance <= best_distance:
                best_index, best_distance = index, distance
        return best_index


# -- the stage itself -------------------------------------------------------


@dataclass
class _DepthAxis:
    """How "how far up the pitch" was measured for this frame."""

    values: list[float | None]
    unit: str
    description: str

    @property
    def available(self) -> bool:
        return any(value is not None for value in self.values)

    def span(self) -> float:
        known = [v for v in self.values if v is not None]
        return (max(known) - min(known)) if len(known) > 1 else 0.0


class TeamAssigner:
    """Groups players by kit, finds the goalkeepers, names the attacking side."""

    def __init__(
        self,
        *,
        extractor: TeamFeatureExtractor | None = None,
        min_samples: int = 6,
        confident_samples: int = 10,
        min_sample_confidence: float = 0.25,
        outlier_mad_scale: float = 3.0,
        min_outlier_distance: float = 18.0,
        min_cluster_fraction: float = 0.2,
        min_cluster_size: int = 3,
        max_trim_rounds: int = 3,
        good_separation: float = 25.0,
        min_separation_ratio: float = 2.0,
        max_iterations: int = 25,
        assignment_margin: float = 6.0,
        goalkeeper_min_gap_m: float = 5.0,
        goalkeeper_min_gap_fraction: float = 0.12,
        goalkeeper_search_depth: int = 2,
        ball_max_distance_boxes: float = 1.5,
        ball_ambiguous_ratio: float = 1.25,
        persist_colors_across_frames: bool = True,
    ):
        self._extractor = extractor or JerseyColorExtractor()
        self._fit_kwargs = {
            "min_samples": min_samples,
            "confident_samples": confident_samples,
            "min_sample_confidence": min_sample_confidence,
            "outlier_mad_scale": outlier_mad_scale,
            "min_outlier_distance": min_outlier_distance,
            "min_cluster_fraction": min_cluster_fraction,
            "min_cluster_size": min_cluster_size,
            "max_trim_rounds": max_trim_rounds,
            "good_separation": good_separation,
            "min_separation_ratio": min_separation_ratio,
            "max_iterations": max_iterations,
        }
        self._assignment_margin = assignment_margin
        self._goalkeeper_min_gap_m = goalkeeper_min_gap_m
        self._goalkeeper_min_gap_fraction = goalkeeper_min_gap_fraction
        self._goalkeeper_search_depth = goalkeeper_search_depth
        self._ball_max_distance_boxes = ball_max_distance_boxes
        self._ball_ambiguous_ratio = ball_ambiguous_ratio
        self._persist_colors = persist_colors_across_frames

        self.overrides = TeamOverrides()
        self._last_model: TeamColorModel | None = None

    @classmethod
    def from_config(cls, config, *, grass_hue_range=None, grass_min_saturation=None):
        """Build from `settings.offside.team_assignment`.

        The grass window is passed in rather than duplicated in this section:
        it describes the *pitch*, not team assignment, and two copies of it
        drifting apart would make the line detector and the kit sampler
        disagree about what grass is.
        """
        if config.provider != "jersey_color":
            # The seam for a future embedding extractor is `TeamFeatureExtractor`;
            # until one exists, an unknown provider is a config error rather
            # than something to fall back from silently.
            raise ConfigurationError(
                f"unknown team assignment provider '{config.provider}' — "
                "the only one built is 'jersey_color'"
            )

        extractor = JerseyColorExtractor(
            keypoint_confidence=config.torso_keypoint_confidence,
            torso_shrink=config.torso_shrink,
            fallback_top_fraction=config.fallback_top_fraction,
            fallback_bottom_fraction=config.fallback_bottom_fraction,
            fallback_width_fraction=config.fallback_width_fraction,
            grass_hue_range=grass_hue_range or (30, 95),
            grass_min_saturation=(
                grass_min_saturation if grass_min_saturation is not None else 40
            ),
            exclude_skin=config.exclude_skin,
            min_sample_pixels=config.min_sample_pixels,
            min_kept_fraction=config.min_kept_fraction,
        )
        return cls(
            extractor=extractor,
            min_samples=config.min_players_for_clustering,
            confident_samples=config.confident_player_count,
            min_sample_confidence=config.min_sample_confidence,
            outlier_mad_scale=config.outlier_mad_scale,
            min_outlier_distance=config.min_outlier_distance,
            min_cluster_fraction=config.min_cluster_fraction,
            min_cluster_size=config.min_cluster_size,
            max_trim_rounds=config.max_trim_rounds,
            good_separation=config.good_separation,
            min_separation_ratio=config.min_separation_ratio,
            max_iterations=config.max_iterations,
            assignment_margin=config.assignment_margin,
            goalkeeper_min_gap_m=config.goalkeeper_min_gap_m,
            goalkeeper_min_gap_fraction=config.goalkeeper_min_gap_fraction,
            goalkeeper_search_depth=config.goalkeeper_search_depth,
            ball_max_distance_boxes=config.ball_max_distance_boxes,
            ball_ambiguous_ratio=config.ball_ambiguous_ratio,
            persist_colors_across_frames=config.persist_colors_across_frames,
        )

    def reset(self) -> None:
        """Forget the fitted kit colours — call on a camera cut or a new clip."""
        self._last_model = None

    # -- main entry point ---------------------------------------------------

    def assign(
        self,
        image: np.ndarray,
        poses: list[PlayerPose],
        *,
        calibration=None,
        ball_xy: Point | None = None,
    ) -> TeamAssignment:
        if not poses:
            return TeamAssignment(reasons=["no players to assign"])

        colors = [self._extractor.extract(image, pose) for pose in poses]
        model, fit_reasons = fit_team_colors(colors, **self._fit_kwargs)

        assignment = TeamAssignment(color_model=model, reasons=list(fit_reasons))

        if model is None:
            assignment.warnings.append(
                "the two teams could not be told apart on this frame — no offside "
                "call can be made from it without operator input"
            )
            assignment.players = [
                _unassigned_player(index, pose, colour, "no team colours were measured")
                for index, (pose, colour) in enumerate(zip(poses, colors))
            ]
            self._apply_overrides(assignment, poses)
            return assignment

        if self._persist_colors:
            model = model.aligned_to(self._last_model)
            assignment.color_model = model
        self._last_model = model

        assignment.players = [
            self._classify(index, pose, colour, model)
            for index, (pose, colour) in enumerate(zip(poses, colors))
        ]

        axis = self._depth_axis(image, poses, calibration)
        for player in assignment.players:
            player.depth = axis.values[player.index]

        self._identify_goalkeepers(assignment, axis, calibration)
        self._name_attacking_team(assignment, poses, ball_xy)
        self._apply_overrides(assignment, poses)
        self._score(assignment, model)

        logger.debug(
            "team_assignment",
            component="team_assignment",
            counts=assignment.counts(),
            attacking=assignment.attacking_team_id,
            separation=round(model.separation, 1),
            confidence=round(assignment.confidence, 3),
        )
        return assignment

    # -- steps --------------------------------------------------------------

    def _classify(
        self, index: int, pose: PlayerPose, colour: JerseyColor, model: TeamColorModel
    ) -> PlayerTeam:
        if not colour.is_usable:
            return _unassigned_player(index, pose, colour, colour.reason)

        team_id, distance, margin = model.classify(colour)

        if distance > model.outlier_threshold:
            # Not a team member by colour. Whether this is a goalkeeper is a
            # question about position, answered in the next step.
            return PlayerTeam(
                index=index,
                team_id=None,
                role=PlayerRole.UNKNOWN,
                confidence=0.0,
                source=SOURCE_OUTLIER,
                reason=(
                    "kit matches neither team — may be a goalkeeper, a match "
                    "official, or a colour that could not be measured cleanly"
                ),
                anchor_xy=_anchor_of(pose),
                track_id=pose.track_id,
                color=colour,
                distance=distance,
                margin=margin,
            )

        # A player sitting between the two kits is reported as such rather
        # than as a confident member of the nearer one: on similar kits this
        # margin is the difference between a usable call and a coin flip.
        certainty = min(1.0, margin / max(1e-6, self._assignment_margin))
        confidence = float(colour.confidence * max(0.0, certainty))
        reason = (
            f"shirt colour is {distance:.0f} from this team's kit and "
            f"{distance + margin:.0f} from the other"
        )
        if certainty < 1.0:
            reason += " — close to the boundary between the two kits"

        return PlayerTeam(
            index=index,
            team_id=team_id,
            role=PlayerRole.OUTFIELD,
            confidence=confidence,
            source=SOURCE_KIT_COLOUR,
            reason=reason,
            anchor_xy=_anchor_of(pose),
            track_id=pose.track_id,
            color=colour,
            distance=distance,
            margin=margin,
        )

    def _depth_axis(self, image, poses: list[PlayerPose], calibration) -> _DepthAxis:
        """Rank players along the goal-to-goal axis, if the frame supports one."""
        points = [pose.ground_point.xy for pose in poses]

        if calibration is not None and calibration.is_metric:
            projected = calibration.to_pitch(points)
            values: list[float | None] = []
            for position in projected:
                values.append(
                    float(position[0]) if np.all(np.isfinite(position)) else None
                )
            return _DepthAxis(values, "m", "distance up the pitch, in metres")

        if calibration is not None and calibration.can_draw_offside_line:
            height, width = image.shape[:2]
            centre_line = calibration.offside_line_through((width / 2.0, height / 2.0))
            if centre_line is not None:
                _, (dx, dy) = centre_line
                # Perpendicular to the offside line is the direction "towards
                # the goal". Taken once at frame centre and reused: the true
                # direction fans out slightly across the frame, so this orders
                # players correctly but does not measure distance between them.
                normal = (-dy, dx)
                values = [float(x * normal[0] + y * normal[1]) for x, y in points]
                return _DepthAxis(
                    values,
                    "px",
                    "ordering along the goal-to-goal direction (no metres available)",
                )

        return _DepthAxis(
            [None] * len(poses),
            "",
            "no pitch calibration, so nobody's depth on the pitch is known",
        )

    def _identify_goalkeepers(
        self, assignment: TeamAssignment, axis: _DepthAxis, calibration
    ) -> None:
        outliers = [p for p in assignment.players if p.team_id is None and p.color and p.color.is_usable]
        if not outliers:
            assignment.reasons.append(
                "no player's kit stood apart from the two teams — the goalkeepers "
                "may be out of shot, or dressed like their own outfield team"
            )
            return

        if not axis.available:
            assignment.warnings.append(
                f"{len(outliers)} player(s) wear neither team's kit, but {axis.description}"
                " — they cannot be confirmed as goalkeepers and are excluded from "
                "the defender ranking"
            )
            return

        ranked = sorted(
            (p for p in assignment.players if p.depth is not None),
            key=lambda p: p.depth,
        )
        if len(ranked) < 3:
            return

        metric = calibration is not None and calibration.is_metric
        gap_needed = (
            self._goalkeeper_min_gap_m
            if metric
            else axis.span() * self._goalkeeper_min_gap_fraction
        )

        for end, ordered in (("near", ranked), ("far", list(reversed(ranked)))):
            keeper = self._goalkeeper_at_end(ordered, gap_needed)
            if keeper is None:
                continue
            player, rank, gap = keeper
            player.role = PlayerRole.GOALKEEPER
            player.source = SOURCE_OUTLIER
            player.confidence = (0.85 if rank == 0 else 0.6) * (1.0 if metric else 0.8)
            player.reason = (
                f"wears neither team's kit and stands {gap:.0f}{axis.unit} beyond "
                f"every other player at the {end} end of the pitch"
            )

    def _goalkeeper_at_end(
        self, ordered: list[PlayerTeam], gap_needed: float
    ) -> tuple[PlayerTeam, int, float] | None:
        """The outlier standing alone at this end, if there is one.

        Searched a couple of players deep, because a defender dropping goal-side
        of the keeper is ordinary football, not a failure.
        """
        for rank in range(min(self._goalkeeper_search_depth, len(ordered) - 1)):
            candidate = ordered[rank]
            if candidate.team_id is not None or candidate.role is PlayerRole.GOALKEEPER:
                continue
            following = ordered[rank + 1]
            gap = abs(following.depth - candidate.depth)
            if gap >= gap_needed > 0:
                return candidate, rank, gap
        return None

    def _name_attacking_team(
        self, assignment: TeamAssignment, poses: list[PlayerPose], ball_xy: Point | None
    ) -> None:
        if ball_xy is None:
            assignment.warnings.append(
                "no ball was detected on this frame, so which side is attacking is "
                "unknown — the two groups are reported without sides"
            )
            return

        distances = []
        for player in assignment.players:
            pose = poses[player.index]
            box_height = max(1.0, pose.box_height)
            distances.append(
                (_distance(pose.ground_point.xy, ball_xy) / box_height, player)
            )
        distances.sort(key=lambda item: item[0])

        nearest_distance, nearest = distances[0]
        if nearest_distance > self._ball_max_distance_boxes:
            assignment.warnings.append(
                "the ball is not close to any player on this frame, so which side "
                "is attacking cannot be established from it"
            )
            return

        if nearest.team_id is None:
            assignment.warnings.append(
                "the player nearest the ball could not be put on either team, so "
                "which side is attacking is unknown"
            )
            return

        assignment.attacking_team_id = nearest.team_id
        assignment.reasons.append(
            "attacking side taken from the player nearest the ball "
            f"({nearest_distance:.1f} player-widths away)"
        )

        contested = [
            player
            for distance, player in distances[1:]
            if player.team_id not in (None, nearest.team_id)
            and distance <= nearest_distance * self._ball_ambiguous_ratio
        ]
        if contested:
            assignment.warnings.append(
                "players from both sides are equally close to the ball — the "
                "attacking side may be the other one; confirm before trusting a call"
            )

    def _apply_overrides(self, assignment: TeamAssignment, poses: list[PlayerPose]) -> None:
        overrides = self.overrides
        if not overrides.is_active:
            return

        if overrides.teams_swapped:
            for player in assignment.players:
                if player.team_id in TEAM_IDS:
                    player.team_id = (
                        TEAM_IDS[1] if player.team_id == TEAM_IDS[0] else TEAM_IDS[0]
                    )
            if assignment.attacking_team_id in TEAM_IDS:
                assignment.attacking_team_id = (
                    TEAM_IDS[1]
                    if assignment.attacking_team_id == TEAM_IDS[0]
                    else TEAM_IDS[0]
                )
            if assignment.color_model is not None:
                assignment.color_model = assignment.color_model.swapped()
            assignment.reasons.append("operator swapped the two team labels")

        for index, pin in overrides.resolve(poses).items():
            player = assignment.by_index(index)
            if player is None:
                continue
            if pin.team_id is not None:
                player.team_id = pin.team_id
            if pin.role is not None:
                player.role = pin.role
                if pin.role is PlayerRole.UNKNOWN:
                    player.team_id = None
            player.source = SOURCE_OPERATOR
            player.confidence = 1.0
            player.reason = f"set by the operator: {pin.describe()}"

        if overrides.attacking_team_id is not None:
            assignment.attacking_team_id = overrides.attacking_team_id
            assignment.reasons.append("operator chose the attacking side")
            assignment.warnings = [
                warning
                for warning in assignment.warnings
                if "which side is attacking" not in warning
            ]

    def _score(self, assignment: TeamAssignment, model: TeamColorModel) -> None:
        """One honest number: no part of the chain may be better than its worst link."""
        assigned = [p for p in assignment.players if p.is_assigned]
        coverage = len(assigned) / max(1, len(assignment.players))
        mean_player_confidence = (
            float(np.mean([p.confidence for p in assigned])) if assigned else 0.0
        )

        confidence = min(model.confidence, mean_player_confidence)
        if not assignment.sides_are_known:
            # Two anonymous groups are not a team assignment M2.5 can use.
            confidence = min(confidence, 0.35)

        assignment.confidence = float(max(0.0, min(1.0, confidence)))
        assignment.reasons.append(
            f"{len(assigned)} of {len(assignment.players)} players placed on a team "
            f"({coverage * 100:.0f}%)"
        )
        if coverage < 0.6:
            assignment.warnings.append(
                "under two thirds of the players on screen could be placed on a "
                "team — the defender ranking may be missing someone"
            )


# -- helpers ----------------------------------------------------------------


def _unassigned_player(
    index: int, pose: PlayerPose, colour: JerseyColor, reason: str
) -> PlayerTeam:
    return PlayerTeam(
        index=index,
        team_id=None,
        role=PlayerRole.UNKNOWN,
        confidence=0.0,
        source=SOURCE_UNMEASURED,
        reason=reason,
        anchor_xy=_anchor_of(pose),
        track_id=pose.track_id,
        color=colour if colour.is_usable else None,
    )


def _anchor_of(pose: PlayerPose) -> Point:
    x1, y1, x2, y2 = pose.bbox_xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
