"""One confidence signal per stage, in the stage's own words (M2.6).

Every phase of M2 already knows how much it trusts itself: the calibrator
reports whether it has metres or only a direction, M2.2 knows whether a foot
came from an ankle or from the bottom of a box, M2.3 knows which players it
wants confirmed, M2.4 knows who is contested, and M2.5 knows what its margin
could be wrong by. Nothing here recomputes any of that.

What this module does is make those numbers *comparable and sayable*. Each
stage becomes a `ConfidenceSignal` carrying three things the operator can act
on — how good it is, why it is that good, and what to do if it is not good
enough. A bare 0.31 tells an operator nothing; "the pitch is only marked at
three points, so distances are directions rather than metres — mark a fourth
landmark" tells them exactly where to spend the next ten seconds.

The signals are deliberately about *this* decision rather than the frame in
general. Twenty players with clean foot points do not help if the two the line
is drawn between are the two guesses, so where a stage can be narrowed to the
players actually being compared, it is.
"""

from __future__ import annotations

from dataclasses import dataclass

from offside.offside_line.line import OffsideDecision, Verdict

#: Stage keys, used by the UI to find a particular row.
PITCH = "pitch"
BODY_POINTS = "body_points"
TEAMS = "teams"
ATTACKING_SIDE = "attacking_side"
IDENTITY = "identity"
GEOMETRY = "geometry"


@dataclass(frozen=True)
class ConfidenceSignal:
    """How much one stage trusts what it handed on, and why."""

    key: str
    phase: str
    #: What this stage is called in front of an operator — no module names.
    label: str
    #: 0..1, or None when the stage has nothing to say about this frame and so
    #: should not cap anything.
    value: float | None
    reason: str
    #: How this stage reads when it is listed as a *caveat* rather than as a
    #: description. The two are not the same sentence: "the pitch is marked, so
    #: positions are in metres" is the right thing to say about a healthy
    #: stage, and nonsense under a heading that says "limited by". Defaults to
    #: `reason`, which is correct for every signal whose reason is already a
    #: complaint.
    concern: str = ""
    #: What the operator can do to raise it. Empty when nothing would help.
    action: str = ""
    #: True when this stage is missing outright rather than merely weak — the
    #: difference between "a poor measurement" and "no measurement".
    blocking: bool = False

    @property
    def available(self) -> bool:
        return self.value is not None

    @property
    def caveat(self) -> str:
        """How this stage reads in a list of what is holding the call back."""
        return self.concern or self.reason

    @property
    def score(self) -> float:
        return 0.0 if self.value is None else float(self.value)


def collect_signals(
    decision: OffsideDecision | None,
    *,
    calibration=None,
    teams=None,
    identities=None,
    poses=None,
) -> list[ConfidenceSignal]:
    """One signal per stage, in pipeline order."""
    return [
        pitch_signal(calibration),
        body_point_signal(poses, decision),
        team_signal(teams),
        attacking_side_signal(teams),
        identity_signal(identities, decision),
        geometry_signal(decision),
    ]


def pitch_signal(calibration) -> ConfidenceSignal:
    """M2.1 — is there a pitch to measure on, and in what units."""
    if calibration is None:
        return ConfidenceSignal(
            key=PITCH,
            phase="M2.1",
            label="Pitch calibration",
            value=0.0,
            reason="the pitch has not been calibrated on this frame",
            action="mark four pitch landmarks on this frame",
            blocking=True,
        )

    confidence = float(getattr(calibration, "confidence", 0.0))
    if getattr(calibration, "is_metric", False):
        return ConfidenceSignal(
            key=PITCH,
            phase="M2.1",
            label="Pitch calibration",
            value=confidence,
            reason="the pitch is marked, so positions are measured in metres",
            concern=(
                "the pitch marks have been carried a long way from the frame "
                "they were placed on, so the metres they give are less certain"
            ),
            action=(
                ""
                if confidence >= 0.6
                else "re-mark the landmarks on this frame — the marks have been "
                "carried a long way from where they were placed"
            ),
        )

    if getattr(calibration, "can_draw_offside_line", False):
        # A directional calibration can order players but cannot measure the
        # gap between them, so it is capped below the metric route however
        # confident the line detection itself was.
        return ConfidenceSignal(
            key=PITCH,
            phase="M2.1",
            label="Pitch calibration",
            value=min(confidence, 0.5),
            reason=(
                "the pitch markings give the direction of the goal line but not "
                "distances, so the tool can say who is in front, not by how much"
            ),
            action="mark four pitch landmarks to get distances in metres",
        )

    return ConfidenceSignal(
        key=PITCH,
        phase="M2.1",
        label="Pitch calibration",
        value=0.0,
        reason="no pitch markings could be established on this frame",
        action="mark four pitch landmarks on this frame",
        blocking=True,
    )


def body_point_signal(poses, decision: OffsideDecision | None) -> ConfidenceSignal:
    """M2.2 — how the two players being compared are standing on the pitch.

    Narrowed to those two wherever possible. A frame where eighteen players
    have clean ankles and the second-last defender is a box-bottom guess is a
    bad frame for this decision, and a frame-wide average would call it good.
    """
    measured = _measured_pair(poses, decision)
    if measured is not None:
        value, reason, action, concern = measured
        return ConfidenceSignal(
            key=BODY_POINTS,
            phase="M2.2",
            label="Foot positions",
            value=value,
            reason=reason,
            concern=concern,
            action=action,
        )

    if not poses:
        return ConfidenceSignal(
            key=BODY_POINTS,
            phase="M2.2",
            label="Foot positions",
            value=0.0,
            reason="no players were found on this frame",
            action="check that the frame is one where the players are visible",
            blocking=True,
        )

    good = [p for p in poses if p.ground_point.is_measured]
    share = len(good) / len(poses)
    return ConfidenceSignal(
        key=BODY_POINTS,
        phase="M2.2",
        label="Foot positions",
        value=share,
        reason=(
            f"{len(good)} of {len(poses)} players have a foot position measured "
            "from the body; the rest are estimated from the bottom of the box"
        ),
        action=(
            ""
            if share >= 0.8
            else "step to a frame where the players are less obscured"
        ),
    )


def team_signal(teams) -> ConfidenceSignal:
    """M2.3 — could the two kits be told apart, and every player placed.

    Deliberately independent of whether the *attacking side* is known — see
    `attacking_side_signal`. The two used to be one signal, and that was a
    real bug in how this got explained to an operator: a frame where the two
    kits were told apart perfectly (a clean red vs. white separation) but the
    ball just wasn't near anyone read as "Team colours: 0%", which sends
    someone looking at the wrong stage entirely. `TeamAssignment.confidence`
    itself is capped low whenever sides aren't known (`assigner.py._score`),
    for good reason at that layer — M2.5 genuinely cannot use two anonymous
    groups — but an *explanation* aimed at a person needs the two failure
    modes kept visibly apart, so this reads the clustering fit directly
    (`color_model.confidence`) rather than that already-capped number.
    """
    if teams is None or not teams.players:
        return ConfidenceSignal(
            key=TEAMS,
            phase="M2.3",
            label="Team colours",
            value=0.0,
            reason="no players were sorted into teams on this frame",
            action="check the kit colours in the team panel",
            blocking=True,
        )

    if teams.color_model is None:
        return ConfidenceSignal(
            key=TEAMS,
            phase="M2.3",
            label="Team colours",
            value=0.0,
            reason="the two kits could not be told apart on this frame",
            action="check the kit colours in the team panel",
            blocking=True,
        )

    unsure = teams.needs_confirmation()
    confidence = float(teams.color_model.confidence)
    if unsure:
        # Confirmations are not a warning to be dismissed: an unconfirmed
        # player may be the defender the line is drawn through.
        return ConfidenceSignal(
            key=TEAMS,
            phase="M2.3",
            label="Team colours",
            value=min(confidence, 0.6),
            reason=(
                f"{len(unsure)} player(s) could not be placed on a team from "
                "their kit with confidence"
            ),
            action="confirm the flagged players in the team panel",
        )

    return ConfidenceSignal(
        key=TEAMS,
        phase="M2.3",
        label="Team colours",
        value=confidence,
        reason="both kits were told apart and every player was placed on a side",
        concern="the two kits were told apart, but not cleanly",
        action="" if confidence >= 0.6 else "confirm the teams by hand",
    )


def attacking_side_signal(teams) -> ConfidenceSignal:
    """M2.3 — do we know *which* team is attacking, separately from whether
    their kits could be told apart (see `team_signal`).

    This is deliberately its own row rather than folded back into the team
    signal. The two questions have different answers surprisingly often: the
    exact moments offside calls matter most — a cross, a rebound, a loose
    ball — are precisely the moments nobody is standing next to the ball, so
    this fails constantly on frames where the kit clustering above is
    reading perfectly clean. Reporting one blended number for both would
    have an operator "fixing" a team-colour problem that was never broken.
    """
    if teams is None or not teams.players:
        # Already fully covered by team_signal's identical branch — nothing
        # new to say, and repeating it would just be noise.
        return ConfidenceSignal(
            key=ATTACKING_SIDE,
            phase="M2.3",
            label="Attacking side",
            value=None,
            reason="no players were sorted into teams on this frame",
        )

    if not teams.sides_are_known:
        reason = next(
            (w for w in teams.warnings if "attacking" in w),
            "which side is attacking has not been established",
        )
        return ConfidenceSignal(
            key=ATTACKING_SIDE,
            phase="M2.3",
            label="Attacking side",
            value=0.0,
            reason=reason,
            action="set the attacking team by hand",
            blocking=True,
        )

    if any("equally close to the ball" in warning for warning in teams.warnings):
        return ConfidenceSignal(
            key=ATTACKING_SIDE,
            phase="M2.3",
            label="Attacking side",
            value=0.6,
            reason=(
                "players from both sides were close enough to the ball that "
                "the attacking side is ambiguous"
            ),
            action="confirm the attacking side",
        )

    return ConfidenceSignal(
        key=ATTACKING_SIDE,
        phase="M2.3",
        label="Attacking side",
        value=1.0,
        reason="the attacking side was established from the ball or an operator override",
    )


def identity_signal(identities, decision: OffsideDecision | None) -> ConfidenceSignal:
    """M2.4 — is each player still the same player they were a frame ago.

    This is the signal most easily missed, because a swapped identity produces
    a decision that looks entirely normal: two real players, measured
    correctly, compared against each other — and the wrong two.
    """
    if identities is None or not identities.identities:
        return ConfidenceSignal(
            key=IDENTITY,
            phase="M2.4",
            label="Player tracking",
            value=None,
            reason="players are not being followed between frames on this clip",
        )

    contested = identities.contested()
    involved = _involved_indices(decision)
    contested_here = [i for i in contested if i.index in involved] if involved else contested

    if contested_here:
        return ConfidenceSignal(
            key=IDENTITY,
            phase="M2.4",
            label="Player tracking",
            value=min(float(identities.confidence), 0.4),
            reason=(
                f"{len(contested_here)} of the players this call depends on may "
                "have been mixed up with somebody nearby"
            ),
            action="step back a few frames to a moment where they are apart",
        )

    if contested:
        return ConfidenceSignal(
            key=IDENTITY,
            phase="M2.4",
            label="Player tracking",
            value=float(identities.confidence),
            reason=(
                f"{len(contested)} player(s) elsewhere on the frame are mixed up, "
                "but not the ones this call depends on"
            ),
        )

    trusted = identities.trusted()
    if not trusted and identities.confidence < 0.5:
        # A tracker that has just started has no evidence either way, and
        # measured on the reference clip that is the state of every frame
        # right after a cut. Scoring it as a failure would veto every call for
        # the first stretch of every shot, on the strength of nothing having
        # happened yet. The risk this signal exists to catch — two players
        # swapped — announces itself as *contested*, which is handled above;
        # absence of evidence is reported and left non-capping.
        return ConfidenceSignal(
            key=IDENTITY,
            phase="M2.4",
            label="Player tracking",
            value=None,
            reason=(
                "the shot has only just started, so nobody has a settled "
                "identity yet — no player has been mixed up with another"
            ),
            action="give the tracking a few frames if you can",
        )

    return ConfidenceSignal(
        key=IDENTITY,
        phase="M2.4",
        label="Player tracking",
        value=float(identities.confidence),
        reason=f"{len(trusted)} players are being followed reliably",
        concern=(
            "the shot has changed recently, so not every player is being "
            "followed with confidence yet"
        ),
        action=(
            ""
            if identities.confidence >= 0.6
            else "the shot has just changed — give the tracking a few frames"
        ),
    )


def geometry_signal(decision: OffsideDecision | None) -> ConfidenceSignal:
    """M2.5 — the line itself, and whether the margin survives its own error."""
    if decision is None:
        return ConfidenceSignal(
            key=GEOMETRY,
            phase="M2.5",
            label="Offside line",
            value=0.0,
            reason="the offside line was not computed on this frame",
            blocking=True,
        )

    if decision.verdict is Verdict.INCONCLUSIVE:
        return ConfidenceSignal(
            key=GEOMETRY,
            phase="M2.5",
            label="Offside line",
            value=0.0,
            reason=(
                decision.warnings[0]
                if decision.warnings
                else "no line could be drawn on this frame"
            ),
            blocking=True,
        )

    attacker = decision.attacker
    if attacker is None:
        return ConfidenceSignal(
            key=GEOMETRY,
            phase="M2.5",
            label="Offside line",
            value=0.0,
            reason="no attacker could be measured against the line",
            blocking=True,
        )

    unit = decision.axis_unit
    if decision.verdict is Verdict.TOO_CLOSE:
        return ConfidenceSignal(
            key=GEOMETRY,
            phase="M2.5",
            label="Offside line",
            value=float(decision.confidence),
            reason=(
                f"the gap of {abs(attacker.margin):.2f}{unit} is smaller than the "
                f"±{attacker.uncertainty:.2f}{unit} this measurement can resolve"
            ),
            action="judge this one from the frame",
        )

    return ConfidenceSignal(
        key=GEOMETRY,
        phase="M2.5",
        label="Offside line",
        value=float(decision.confidence),
        reason=(
            f"the gap of {abs(attacker.margin):.2f}{unit} is clear of the "
            f"±{attacker.uncertainty:.2f}{unit} this measurement can resolve"
        ),
        concern=(
            f"the gap of {abs(attacker.margin):.2f}{unit} is not much wider "
            f"than the ±{attacker.uncertainty:.2f}{unit} behind it"
        ),
    )


# -- helpers ---------------------------------------------------------------


def _involved_indices(decision: OffsideDecision | None) -> set[int]:
    """The players this particular call actually rests on."""
    if decision is None:
        return set()
    indices = set()
    if decision.attacker is not None:
        indices.add(decision.attacker.index)
    if decision.second_last_defender is not None:
        indices.add(decision.second_last_defender.index)
    return indices


def _measured_pair(poses, decision) -> tuple[float, str, str, str] | None:
    """The foot quality of the attacker and defender being compared."""
    if not poses or decision is None:
        return None
    if decision.attacker is None or decision.second_last_defender is None:
        return None

    pair = {
        "attacker": decision.attacker.index,
        "second-last defender": decision.second_last_defender.index,
    }
    guessed = []
    lowest = 1.0
    for role, index in pair.items():
        if index >= len(poses):
            return None
        ground = poses[index].ground_point
        lowest = min(lowest, float(ground.confidence))
        if not ground.is_measured:
            guessed.append(role)

    if guessed:
        who = " and the ".join(guessed)
        return (
            min(lowest, 0.4),
            (
                f"the {who} has no visible feet, so their position is estimated "
                "from the bottom of the player box"
            ),
            "step to a frame where their feet are visible",
            "",
        )
    return (
        lowest,
        "both players in this comparison have their feet measured from the body",
        "" if lowest >= 0.6 else "step to a frame where their feet are clearer",
        "the feet of the two players being compared are only partly visible",
    )
