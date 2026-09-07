"""What the inspector shows, worked out separately from how Qt draws it.

## Why this module exists at all

The inspector's job is trust: an operator has to be able to look at the
screen and see *why* the app believes what it believes. That makes the
selection and wording of what appears on screen part of the product, not
incidental UI code — so it lives here, as plain data, and can be tested
without a display attached.

The widgets in `panels.py` do nothing but render what these functions
return. If a number is wrong on screen, it is wrong here, and there is a
test for it.

## The one rule that governs all of it

**Never show a conclusion without its evidence.** A team colour is shown as
the swatch that was actually measured, next to the crop of the player it was
measured from, so a wrong reading is visible as a mismatch rather than
inferred from a low number. Confidence is shown next to what produced it. A
stage that could not run says which input it was missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from offside.team_assignment.teams import (
    TEAM_A,
    TEAM_B,
    PlayerRole,
    PlayerTeam,
    TeamAssignment,
)

#: How much of the frame around a player box a gallery thumbnail includes.
CROP_PADDING = 0.12


@dataclass
class PlayerCard:
    """One player as the inspector shows them: the evidence, then the verdict."""

    index: int
    label: str
    track_id: str | None
    crop: np.ndarray | None
    primary_bgr: tuple[int, int, int]
    secondary_bgr: tuple[int, int, int]
    patterned: bool
    confidence: float
    needs_confirmation: bool
    is_operator_set: bool
    role: PlayerRole
    lines: list[str] = field(default_factory=list)
    anchor_xy: tuple[float, float] = (0.0, 0.0)

    @property
    def has_two_colours(self) -> bool:
        return self.patterned and self.primary_bgr != self.secondary_bgr


@dataclass
class TeamColumn:
    """One column of the team view — a kit, and everyone measured into it."""

    team_id: str | None
    title: str
    subtitle: str
    kit_bgr: tuple[int, int, int] | None
    kit_secondary_bgr: tuple[int, int, int] | None
    cards: list[PlayerCard] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.cards)


@dataclass
class FlowStep:
    """One row of the pipeline flow — what ran, on what, and what came out."""

    phase: str
    title: str
    state: str
    headline: str
    facts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def player_crop(image: np.ndarray, bbox, padding: float = CROP_PADDING) -> np.ndarray | None:
    """The pixels a measurement was taken from, for showing beside it."""
    if image is None:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    x1, y1 = int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))
    x2, y2 = int(min(width, x2 + pad_x)), int(min(height, y2 + pad_y))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return image[y1:y2, x1:x2].copy()


def player_card(analysis, player: PlayerTeam) -> PlayerCard:
    colour = player.color
    crop = None
    if player.index < len(analysis.poses):
        crop = player_crop(analysis.frame.image, analysis.poses[player.index].bbox_xyxy)

    lines = [player.reason]

    # How well this player is being *followed* belongs on the same card as
    # what team they are: a kit measured perfectly on a player whose identity
    # was swapped is still the wrong answer, and the operator should see both
    # facts together rather than on two screens.
    identities = getattr(analysis, "identities", None)
    identity = identities.by_index(player.index) if identities else None
    if identity is not None:
        lines.append(f"identity {identity.track_id}: {identity.reason}")
    if player.vote_share is not None and player.frames_pooled > 1:
        lines.append(
            f"seen in {player.frames_pooled} frames, {player.vote_share * 100:.0f}% agree"
        )
    if colour is not None:
        lines.append(colour.reason)
    if player.depth is not None:
        lines.append(f"position along the pitch: {player.depth:.1f}")

    return PlayerCard(
        index=player.index,
        label=_label(player),
        track_id=player.track_id,
        crop=crop,
        primary_bgr=colour.bgr if colour else (60, 60, 60),
        secondary_bgr=colour.secondary_bgr if colour else (60, 60, 60),
        patterned=bool(colour and colour.patterned),
        confidence=player.confidence,
        needs_confirmation=player.needs_confirmation
        or (identity is not None and identity.needs_confirmation),
        is_operator_set=player.is_operator_set,
        role=player.role,
        lines=lines,
        anchor_xy=player.anchor_xy,
    )


def team_columns(analysis) -> list[TeamColumn]:
    """The team view: one column per kit, plus one for everyone else.

    Two separate grids rather than one list, because the question an operator
    is answering here is "does everything in this column look like the same
    team?" — which the eye answers instantly on a grid of swatches and not at
    all on a mixed list.
    """
    teams: TeamAssignment | None = analysis.teams
    if teams is None:
        return []

    model = teams.color_model
    columns: list[TeamColumn] = []

    for team_id in (TEAM_A, TEAM_B):
        members = teams.members_of(team_id)
        role = ""
        if teams.attacking_team_id == team_id:
            role = " · attacking"
        elif teams.defending_team_id == team_id:
            role = " · defending"

        unsure = sum(1 for member in members if member.needs_confirmation)
        columns.append(
            TeamColumn(
                team_id=team_id,
                title=f"Team {team_id[-1].upper()}{role}",
                subtitle=(
                    f"{len(members)} player(s)"
                    + (f" · {unsure} to confirm" if unsure else " · all settled")
                ),
                kit_bgr=model.swatches.get(team_id) if model else None,
                kit_secondary_bgr=None,
                cards=[player_card(analysis, member) for member in members],
            )
        )

    others = [p for p in teams.players if p.team_id is None]
    keepers = sum(1 for p in others if p.role is PlayerRole.GOALKEEPER)
    columns.append(
        TeamColumn(
            team_id=None,
            title="Not on a team",
            subtitle=(
                f"{len(others)} player(s)"
                + (f" · {keepers} goalkeeper(s) identified" if keepers else "")
            ),
            kit_bgr=None,
            kit_secondary_bgr=None,
            cards=[player_card(analysis, other) for other in others],
        )
    )
    return columns


def kit_summary(analysis) -> list[str]:
    """The measured kits themselves, stated in numbers the operator can check."""
    teams: TeamAssignment | None = analysis.teams
    if teams is None or teams.color_model is None:
        return ["No kit colours could be measured on this frame."]

    model = teams.color_model
    spread = sum(model.spreads.values())
    return [
        f"measured from {model.sample_count} player(s)",
        f"the two kits are {model.separation:.0f} colour units apart",
        f"players of one kit vary by {spread:.0f} between themselves",
        (
            f"anything more than {model.outlier_threshold:.0f} from both kits is "
            "treated as neither team"
        ),
    ]


def flow_steps(analysis) -> list[FlowStep]:
    """The whole pipeline as an ordered list, including what has not run.

    A stage that is missing must *look* missing rather than be absent from the
    page — the same rule the stage panel has followed since M2.1, applied to
    the operator-facing view.
    """
    steps: list[FlowStep] = []
    for report in analysis.reports:
        facts = [detail for detail in report.details if not detail.startswith("WARNING:")]
        warnings = [
            detail[len("WARNING:"):].strip()
            for detail in report.details
            if detail.startswith("WARNING:")
        ]
        steps.append(
            FlowStep(
                phase=report.phase,
                title=report.title,
                state=report.state.value,
                headline=report.summary,
                facts=facts,
                warnings=warnings,
            )
        )
    return steps


def identity_summary(analysis) -> list[str]:
    """How the following is going, in the operator's terms."""
    identities = getattr(analysis, "identities", None)
    if identities is None or not identities.identities:
        return ["Nobody is being followed on this frame."]

    counts = identities.counts()
    lines = [
        f"shot segment {identities.segment} — identities restart at every camera cut",
        f"{len(identities.trusted())} of {len(identities.identities)} followed reliably",
    ]
    wording = {
        "confirmed": "followed cleanly",
        "recovered": "picked up again after being hidden",
        "tentative": "only just seen, not yet trusted",
        "contested": "could not be told apart from another player",
    }
    lines += [
        f"{count} {wording[state]}" for state, count in counts.items() if count
    ]
    if identities.cut_detected:
        lines.append("the camera cut on this frame; everyone was re-identified")
    return lines


def confirmation_summary(analysis) -> str:
    """The one line that says whether the operator has anything to do."""
    teams: TeamAssignment | None = analysis.teams
    if teams is None:
        return "Team assignment has not run on this frame."
    unsure = teams.needs_confirmation()
    if not teams.players:
        return "No players on this frame."
    if not unsure:
        return f"All {len(teams.players)} players settled — nothing to confirm."
    return (
        f"{len(unsure)} of {len(teams.players)} players need a look "
        f"({len(teams.settled())} settled)"
    )


def _label(player: PlayerTeam) -> str:
    if player.role is PlayerRole.GOALKEEPER:
        base = "GK"
    elif player.team_id is None:
        base = "?"
    else:
        base = player.team_id[-1].upper()
    return base + ("*" if player.is_operator_set else "")


# -- naming the pitch in words anybody can follow ----------------------------
#
# The technical landmark names (`left_goal_area_bottom_corner`) are correct and
# useless: they assume the reader knows a goal area from a penalty area, which
# an operator running this tool has no reason to. Each one therefore carries a
# plain label and a sentence describing where it physically is, phrased against
# the map the operator is looking at rather than against football vocabulary.
#
# Order matters too: the points that are easiest to pick out of a broadcast
# frame come first, because a calibration is only as good as the operator's
# ability to click the same point the app means.

#: (technical name, short label, where it is in plain words)
LANDMARK_GUIDE: list[tuple[str, str, str]] = [
    ("corner_left_top", "Corner flag - top left",
     "The very corner of the pitch, top-left of the map, where the corner flag stands."),
    ("corner_right_top", "Corner flag - top right",
     "The very corner of the pitch, top-right of the map, where the corner flag stands."),
    ("corner_left_bottom", "Corner flag - bottom left",
     "The very corner of the pitch, bottom-left of the map, where the corner flag stands."),
    ("corner_right_bottom", "Corner flag - bottom right",
     "The very corner of the pitch, bottom-right of the map, where the corner flag stands."),
    ("centre_mark", "Centre spot",
     "The painted spot in the exact middle of the pitch, inside the big circle."),
    ("halfway_top", "Halfway line - top edge",
     "Where the line across the middle of the pitch meets the top side line."),
    ("halfway_bottom", "Halfway line - bottom edge",
     "Where the line across the middle of the pitch meets the bottom side line."),
    ("left_penalty_spot", "Penalty spot - left goal",
     "The painted spot in front of the left goal, inside the big rectangle."),
    ("right_penalty_spot", "Penalty spot - right goal",
     "The painted spot in front of the right goal, inside the big rectangle."),
    ("left_penalty_area_top_corner", "Big box (left) - far top corner",
     "The big rectangle around the left goal: its top corner furthest from the goal."),
    ("left_penalty_area_bottom_corner", "Big box (left) - far bottom corner",
     "The big rectangle around the left goal: its bottom corner furthest from the goal."),
    ("right_penalty_area_top_corner", "Big box (right) - far top corner",
     "The big rectangle around the right goal: its top corner furthest from the goal."),
    ("right_penalty_area_bottom_corner", "Big box (right) - far bottom corner",
     "The big rectangle around the right goal: its bottom corner furthest from the goal."),
    ("left_penalty_area_top_goalline", "Big box (left) - top, on the goal line",
     "Where the top edge of the big left rectangle touches the end line of the pitch."),
    ("left_penalty_area_bottom_goalline", "Big box (left) - bottom, on the goal line",
     "Where the bottom edge of the big left rectangle touches the end line of the pitch."),
    ("right_penalty_area_top_goalline", "Big box (right) - top, on the goal line",
     "Where the top edge of the big right rectangle touches the end line of the pitch."),
    ("right_penalty_area_bottom_goalline", "Big box (right) - bottom, on the goal line",
     "Where the bottom edge of the big right rectangle touches the end line of the pitch."),
    ("left_goal_area_top_corner", "Small box (left) - far top corner",
     "The small rectangle right next to the left goal: its top corner furthest from the goal."),
    ("left_goal_area_bottom_corner", "Small box (left) - far bottom corner",
     "The small rectangle right next to the left goal: its bottom corner furthest from the goal."),
    ("right_goal_area_top_corner", "Small box (right) - far top corner",
     "The small rectangle right next to the right goal: its top corner furthest from the goal."),
    ("right_goal_area_bottom_corner", "Small box (right) - far bottom corner",
     "The small rectangle right next to the right goal: its bottom corner furthest from the goal."),
    ("left_goal_area_top_goalline", "Small box (left) - top, on the goal line",
     "Where the top edge of the small left rectangle touches the end line of the pitch."),
    ("left_goal_area_bottom_goalline", "Small box (left) - bottom, on the goal line",
     "Where the bottom edge of the small left rectangle touches the end line of the pitch."),
    ("right_goal_area_top_goalline", "Small box (right) - top, on the goal line",
     "Where the top edge of the small right rectangle touches the end line of the pitch."),
    ("right_goal_area_bottom_goalline", "Small box (right) - bottom, on the goal line",
     "Where the bottom edge of the small right rectangle touches the end line of the pitch."),
    ("left_goalpost_top", "Left goal - top post",
     "The foot of the top post of the left goal, where it meets the ground."),
    ("left_goalpost_bottom", "Left goal - bottom post",
     "The foot of the bottom post of the left goal, where it meets the ground."),
    ("right_goalpost_top", "Right goal - top post",
     "The foot of the top post of the right goal, where it meets the ground."),
    ("right_goalpost_bottom", "Right goal - bottom post",
     "The foot of the bottom post of the right goal, where it meets the ground."),
]

_LANDMARK_LABELS = {name: label for name, label, _ in LANDMARK_GUIDE}
_LANDMARK_DESCRIPTIONS = {name: text for name, _, text in LANDMARK_GUIDE}


def landmark_choices(pitch) -> list[tuple[str, str]]:
    """(label, technical name) pairs for the marking dropdown, easiest first.

    Driven by the guide rather than by the pitch model's own ordering, so the
    operator is offered the points a human can actually find in a broadcast
    frame before the fiddly ones. Anything the guide has not described falls
    to the end rather than disappearing — a landmark that exists but cannot be
    chosen would be a silent gap.
    """
    known = {name for name, _, _ in LANDMARK_GUIDE}
    available = set(pitch.landmarks())
    choices = [
        (label, name) for name, label, _ in LANDMARK_GUIDE if name in available
    ]
    choices += [
        (name.replace("_", " "), name) for name in sorted(available - known)
    ]
    return choices


def landmark_label(name: str) -> str:
    return _LANDMARK_LABELS.get(name, name.replace("_", " "))


def landmark_description(name: str) -> str:
    return _LANDMARK_DESCRIPTIONS.get(
        name, "Click this point on the pitch in the video."
    )
