"""Team, goalkeeper and role vocabulary — the data offside geometry consumes.

## Why the labels are anonymous

Clustering shirt colours can say "these eleven look alike and those eleven
look alike". It cannot say which group is *attacking*: that is not a property
of a kit, it is a property of a moment. So the two colour groups are named
`TEAM_A` / `TEAM_B` — deliberately meaningless — and `attacking_team_id` is
decided separately, at the contact frame, from who has the ball.

Keeping those two ideas apart is what stops the pipeline inventing football
knowledge it does not have. A wrong attacking-team call is a different
failure from a wrong colour cluster, has a different cause, and must be
reported separately so M2.6 can explain which one it was.

## Why roles are deliberately few

`OUTFIELD`, `GOALKEEPER`, `UNKNOWN`. There is no `REFEREE` role, and that is
on purpose: nothing here can tell a referee from a substitute warming up, a
ball boy, or a player whose shirt colour simply failed to measure. All of
them are "not usable as a team member", which is exactly what `UNKNOWN`
means, and all of them are excluded from the defender ranking either way.
Inventing a referee label would be a claim the evidence does not support.

## What M2.5 actually needs from this

Law 11 counts the second-last **opponent**, whoever that is — usually the
goalkeeper plus the last defender, but not necessarily. So `opponents()`
returns the defending side *including* its goalkeeper, and goalkeeper
identification is not load-bearing for the verdict: it keeps the colour
clusters clean and makes the explanation readable. A missing goalkeeper
label degrades the wording, not the geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

import numpy as np

#: The two colour groups. Anonymous by design — see the module docstring.
TEAM_A: Final = "team_a"
TEAM_B: Final = "team_b"
TEAM_IDS: Final = (TEAM_A, TEAM_B)


class PlayerRole(str, Enum):
    OUTFIELD = "outfield"
    GOALKEEPER = "goalkeeper"
    #: Kit matches neither group, or no kit colour could be measured. Covers
    #: referees, substitutes, crowd false-positives and measurement failures
    #: alike — all of them are excluded from the defender ranking.
    UNKNOWN = "unknown"


#: How a player's team came to be what it is. Travels with every assignment
#: so a decision built on an operator correction can be reported as such.
SOURCE_KIT_COLOUR: Final = "kit_colour"
SOURCE_OUTLIER: Final = "kit_outlier"
SOURCE_OPERATOR: Final = "operator"
SOURCE_UNMEASURED: Final = "unmeasured"


@dataclass(frozen=True)
class JerseyColor:
    """One player's kit appearance, as the feature clustering runs on.

    `vector` is what the clustering actually uses; `bgr` exists only so the
    debug view can paint a swatch a human recognises. They are separate
    because a future extractor (a crop embedding, say) would produce a
    high-dimensional `vector` with no colour meaning at all, while still
    wanting a swatch to show — see `jersey_color.py`.

    Two swatches, because a striped kit has two colours and averaging them
    produces a blend that matches neither and is unstable between frames.
    """

    vector: tuple[float, ...]
    bgr: tuple[int, int, int]
    pixel_count: int
    confidence: float
    source: str
    reason: str
    secondary_bgr: tuple[int, int, int] = (0, 0, 0)
    #: True when the two measured colours are genuinely different — a striped
    #: or hooped kit rather than a solid one.
    patterned: bool = False
    #: How many frames this colour was pooled from. One frame is a snapshot;
    #: a pooled measurement over a track is what makes the answer stable.
    sample_count: int = 1

    @property
    def is_usable(self) -> bool:
        return self.confidence > 0.0 and self.pixel_count > 0

    def as_array(self) -> np.ndarray:
        return np.asarray(self.vector, dtype=np.float64)


@dataclass
class TeamColorModel:
    """The two kit colours this footage actually contains.

    Fitted per clip at runtime. Nothing is stored between sessions and no kit
    colour is ever configured by hand: a model that had to be told "the home
    team is red" would be useless on the next match, which is the whole point
    of M2_Plan section 4.
    """

    centroids: dict[str, tuple[float, ...]]
    swatches: dict[str, tuple[int, int, int]]
    #: Mean distance from a member to its own centroid — how tight each kit is.
    spreads: dict[str, float]
    #: Distance between the two centroids. In CIELAB this is ~dE76, so ~2.3 is
    #: "a trained eye notices" and ~50 is "obviously different colours".
    separation: float
    #: Above this distance from both centroids, a player is not either team.
    #: Derived from the data (median + k*MAD), never a fixed colour.
    outlier_threshold: float
    sample_count: int
    confidence: float
    reasons: list[str] = field(default_factory=list)

    def centroid(self, team_id: str) -> np.ndarray:
        return np.asarray(self.centroids[team_id], dtype=np.float64)

    def classify(self, color: JerseyColor) -> tuple[str, float, float]:
        """Nearest team, its distance, and the margin over the other team.

        A small margin is the honest signal for "these two kits look alike" —
        it is what stops a similar-kit match being reported as a confident
        assignment.
        """
        vector = color.as_array()
        distances = {
            team_id: float(np.linalg.norm(vector - self.centroid(team_id)))
            for team_id in self.centroids
        }
        ordered = sorted(distances.items(), key=lambda item: item[1])
        nearest, best = ordered[0]
        runner_up = ordered[1][1] if len(ordered) > 1 else best
        return nearest, best, runner_up - best

    def aligned_to(self, previous: "TeamColorModel | None") -> "TeamColorModel":
        """Keep `team_a` meaning the same kit it meant on the last frame.

        Clustering has no memory, so a freshly fitted model may hand the same
        two kits the opposite names. Left alone that makes every team label in
        the debug view flicker between frames and would silently invert an
        offside call at the contact frame. Labels are therefore re-attached to
        whichever previous centroid they are nearest.
        """
        if previous is None or set(previous.centroids) != set(self.centroids):
            return self

        a, b = TEAM_IDS
        straight = float(
            np.linalg.norm(self.centroid(a) - previous.centroid(a))
            + np.linalg.norm(self.centroid(b) - previous.centroid(b))
        )
        swapped = float(
            np.linalg.norm(self.centroid(a) - previous.centroid(b))
            + np.linalg.norm(self.centroid(b) - previous.centroid(a))
        )
        return self if swapped >= straight else self.swapped()

    def swapped(self) -> "TeamColorModel":
        a, b = TEAM_IDS
        return TeamColorModel(
            centroids={a: self.centroids[b], b: self.centroids[a]},
            swatches={a: self.swatches[b], b: self.swatches[a]},
            spreads={a: self.spreads[b], b: self.spreads[a]},
            separation=self.separation,
            outlier_threshold=self.outlier_threshold,
            sample_count=self.sample_count,
            confidence=self.confidence,
            reasons=list(self.reasons),
        )


@dataclass
class PlayerTeam:
    """One player's team and role at one frame.

    `index` refers to the caller's list of `PlayerPose`s, so this never
    becomes a second source of truth about who is on the pitch — identity
    stays with the detector and tracker, exactly as in M2.2.
    """

    index: int
    team_id: str | None
    role: PlayerRole
    confidence: float
    source: str
    reason: str
    anchor_xy: tuple[float, float]
    track_id: str | None = None
    color: JerseyColor | None = None
    #: Distance to the assigned kit centroid, and how much closer that kit was
    #: than the other one. Both None when no colour could be measured.
    distance: float | None = None
    margin: float | None = None
    #: Position along the goal-to-goal axis, when calibration allowed one:
    #: metres at METRIC level, an arbitrary but monotonic image-space score at
    #: DIRECTIONAL level. None means depth could not be established.
    depth: float | None = None
    #: How much of the accumulated evidence across frames agrees with this
    #: team, and how many frames that evidence covers.
    vote_share: float | None = None
    frames_pooled: int = 1
    #: True when this player should be confirmed by the operator before a
    #: verdict is built on them. The design target is that everything *not*
    #: flagged is safe to act on — residual error should surface as a question
    #: rather than as a confident wrong answer.
    needs_confirmation: bool = False

    @property
    def is_assigned(self) -> bool:
        return self.team_id is not None

    @property
    def is_operator_set(self) -> bool:
        return self.source == SOURCE_OPERATOR


@dataclass
class TeamAssignment:
    """Everything M2.3 concluded about one frame, with its own confidence."""

    players: list[PlayerTeam] = field(default_factory=list)
    color_model: TeamColorModel | None = None
    attacking_team_id: str | None = None
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def defending_team_id(self) -> str | None:
        if self.attacking_team_id is None:
            return None
        return TEAM_B if self.attacking_team_id == TEAM_A else TEAM_A

    @property
    def sides_are_known(self) -> bool:
        return self.attacking_team_id is not None

    def by_index(self, index: int) -> PlayerTeam | None:
        for player in self.players:
            if player.index == index:
                return player
        return None

    def members_of(self, team_id: str | None) -> list[PlayerTeam]:
        if team_id is None:
            return []
        return [p for p in self.players if p.team_id == team_id]

    def opponents(self) -> list[PlayerTeam]:
        """The defending side, goalkeeper included — what M2.5 ranks.

        The Laws count the second-last *opponent*, not the second-last
        outfielder, so the goalkeeper must stay in this list.
        """
        return self.members_of(self.defending_team_id)

    def attackers(self) -> list[PlayerTeam]:
        return self.members_of(self.attacking_team_id)

    def goalkeepers(self) -> list[PlayerTeam]:
        return [p for p in self.players if p.role is PlayerRole.GOALKEEPER]

    def unassigned(self) -> list[PlayerTeam]:
        return [p for p in self.players if p.team_id is None]

    def needs_confirmation(self) -> list[PlayerTeam]:
        """The players the operator should glance at before a call is made.

        This is the safety valve for the whole stage: the thresholds behind it
        are set so that a player who is *not* on this list is one the pipeline
        is prepared to be judged on.
        """
        return [p for p in self.players if p.needs_confirmation]

    def settled(self) -> list[PlayerTeam]:
        return [p for p in self.players if not p.needs_confirmation and p.is_assigned]

    def counts(self) -> dict[str, int]:
        return {
            TEAM_A: len(self.members_of(TEAM_A)),
            TEAM_B: len(self.members_of(TEAM_B)),
            "unassigned": len(self.unassigned()),
        }
