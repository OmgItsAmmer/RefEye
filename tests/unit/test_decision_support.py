"""M2.6 — the confidence score and the sentence that goes with it.

These tests are almost entirely about *refusing*. The arithmetic in this stage
is one `min()`; what is worth protecting is the behaviour around it — that a
strong stage can never cover for a broken one, that a withheld verdict is
never quietly presented as a call, that an abstention is never withheld (it is
already the honest answer), and that every low score names the stage the
operator should go and fix.

The prose is asserted too, loosely. Wording that drifts into jargon defeats
the entire point of the phase, so a handful of tests check that the operator
is given a sentence rather than a number.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from offside.body_keypoints.keypoints import GroundPoint
from offside.decision_support.explainer import DecisionExplainer
from offside.decision_support.explanation import (
    SOURCE_OPERATOR,
    ConfidenceBand,
    operator_decision,
)
from offside.decision_support.signals import (
    ATTACKING_SIDE,
    BODY_POINTS,
    GEOMETRY,
    IDENTITY,
    PITCH,
    TEAMS,
    collect_signals,
)
from offside.offside_line.axis import AttackDirection
from offside.offside_line.line import AttackerComparison, OffsideDecision, Verdict
from offside.second_last_defender.ranking import RankedPlayer
from offside.team_assignment.teams import (
    TEAM_A,
    TEAM_B,
    PlayerRole,
    PlayerTeam,
    TeamAssignment,
    TeamColorModel,
)

# -- fakes ------------------------------------------------------------------
#
# Deliberately small stand-ins rather than the real stage outputs: this phase
# consumes only a confidence and a couple of flags from each stage, and
# building real calibrations here would test M2.1 all over again.


@dataclass
class FakeCalibration:
    confidence: float = 0.9
    is_metric: bool = True
    can_draw_offside_line: bool = True


@dataclass
class FakeIdentity:
    index: int
    contested: bool = False


class FakeIdentities:
    def __init__(self, identities, confidence=0.9):
        self.identities = identities
        self.confidence = confidence

    def contested(self):
        return [i for i in self.identities if i.contested]

    def trusted(self):
        return [i for i in self.identities if not i.contested]


@dataclass
class FakePose:
    ground_point: GroundPoint


def pose(confidence: float = 0.9, measured: bool = True) -> FakePose:
    return FakePose(
        ground_point=GroundPoint(
            xy=(0.0, 0.0),
            confidence=confidence,
            source="ankle" if measured else "bbox_bottom",
            reason="test",
        )
    )


def make_color_model(confidence: float = 0.9) -> TeamColorModel:
    return TeamColorModel(
        centroids={TEAM_A: (20.0, 10.0, 10.0), TEAM_B: (70.0, -5.0, -5.0)},
        swatches={TEAM_A: (40, 40, 200), TEAM_B: (220, 220, 220)},
        spreads={TEAM_A: 8.0, TEAM_B: 9.0},
        separation=50.0,
        outlier_threshold=30.0,
        sample_count=10,
        confidence=confidence,
    )


def make_teams(
    *, confidence=0.9, unsure=0, known=True, color_model=True, color_confidence=None
) -> TeamAssignment:
    """`confidence` and `color_confidence` are deliberately separate knobs —
    that split is the whole point of the M2.3 signal fix: whether the two
    kits were told apart is not the same question as whether the attacking
    side is known, and a test fixture that conflated them would hide exactly
    the bug it exists to catch."""
    players = [
        PlayerTeam(
            index=i,
            team_id=TEAM_A if i < 2 else TEAM_B,
            role=PlayerRole.OUTFIELD,
            confidence=0.9,
            source="kit_colour",
            reason="test",
            anchor_xy=(float(i), 0.0),
        )
        for i in range(4)
    ]
    for player in players[:unsure]:
        player.needs_confirmation = True
    return TeamAssignment(
        players=players,
        color_model=(
            make_color_model(color_confidence if color_confidence is not None else confidence)
            if color_model
            else None
        ),
        attacking_team_id=TEAM_A if known else None,
        confidence=confidence,
        warnings=[] if known else ["which side is attacking has not been established"],
    )


def make_decision(
    *,
    verdict=Verdict.OFFSIDE,
    confidence=0.8,
    margin=0.9,
    uncertainty=0.3,
    attacker_index=0,
    defender_index=2,
    margin_to_ball=1.0,
) -> OffsideDecision:
    attacker = AttackerComparison(
        index=attacker_index,
        track_id="s0t1",
        point=(10.0, 10.0),
        depth=50.0,
        margin=margin,
        margin_to_ball=margin_to_ball,
        uncertainty=uncertainty,
        verdict=verdict,
        confidence=confidence,
        reason="test",
    )
    defender = RankedPlayer(
        index=defender_index,
        track_id="s0t3",
        point=(20.0, 20.0),
        depth=49.0,
        confidence=0.9,
        source="ankle",
        reason="test",
    )
    return OffsideDecision(
        verdict=verdict,
        confidence=confidence,
        attacker=attacker,
        attackers=[attacker],
        second_last_defender=defender,
        direction=AttackDirection(
            sign=1.0, confidence=0.9, source="goalkeeper", reason="keeper is deepest"
        ),
        axis_unit="m",
        line=((0.0, 0.0), (1.0, 0.0)),
    )


def healthy(**overrides):
    """Every stage happy — the baseline the failure tests deviate from."""
    inputs = {
        "decision": make_decision(),
        "calibration": FakeCalibration(),
        "teams": make_teams(),
        "identities": FakeIdentities([FakeIdentity(i) for i in range(4)]),
        "poses": [pose() for _ in range(4)],
    }
    inputs.update(overrides)
    return inputs


def explain(explainer=None, **overrides):
    inputs = healthy(**overrides)
    decision = inputs.pop("decision")
    return (explainer or DecisionExplainer()).explain(decision, **inputs)


# -- the aggregation rule ---------------------------------------------------


def test_a_healthy_chain_publishes_the_verdict():
    result = explain()
    assert result.verdict is Verdict.OFFSIDE
    assert result.band is ConfidenceBand.HIGH
    assert result.is_published


def test_confidence_is_the_weakest_stage_not_the_average():
    """The whole design in one test. Four strong stages must not outvote the
    one that failed — which is invariably the one that decided the verdict."""
    result = explain(calibration=FakeCalibration(confidence=0.2))
    assert result.confidence == pytest.approx(0.2)
    assert result.weakest.key == PITCH


def test_the_weakest_stage_is_named_not_just_scored():
    result = explain(calibration=FakeCalibration(confidence=0.2))
    assert result.weakest is not None
    assert result.weakest.phase == "M2.1"
    assert result.weakest.action, "a weak stage that suggests nothing is not actionable"


def test_a_stage_with_nothing_to_say_does_not_cap_the_score():
    """Identity tracking is off on a still frame; that is not evidence of
    anything and must not be scored as though it were a failure."""
    result = explain(identities=None)
    assert result.confidence > 0.5
    assert result.signal(IDENTITY).available is False


def test_confidence_never_exceeds_the_geometry_it_describes():
    result = explain(decision=make_decision(confidence=0.42))
    assert result.confidence <= 0.42


# -- withholding ------------------------------------------------------------


def test_a_verdict_the_chain_cannot_carry_is_withheld():
    result = explain(calibration=FakeCalibration(confidence=0.15))
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.geometry_verdict is Verdict.OFFSIDE
    assert result.withheld


def test_a_withheld_verdict_is_still_visible_to_the_operator():
    """Withheld means "not in the tool's voice", not "hidden". The operator
    can still see what the geometry read and decide for themselves."""
    result = explain(calibration=FakeCalibration(confidence=0.15))
    assert result.geometry_verdict is Verdict.OFFSIDE
    assert "offside" in result.headline.lower()
    assert "please judge" in result.headline.lower()


def test_a_withheld_verdict_never_claims_to_be_published():
    result = explain(calibration=FakeCalibration(confidence=0.15))
    assert not result.is_published


def test_too_close_to_call_is_never_withheld():
    """An abstention is already the honest answer; a weak chain cannot make
    it wrong, and withholding it would leave the operator with nothing."""
    result = explain(
        decision=make_decision(verdict=Verdict.TOO_CLOSE, confidence=0.2, margin=0.05),
        calibration=FakeCalibration(confidence=0.2),
    )
    assert result.verdict is Verdict.TOO_CLOSE


def test_an_inconclusive_geometry_stays_inconclusive():
    result = explain(decision=OffsideDecision(warnings=["no line"]))
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.withheld is False


def test_the_publish_floor_is_configurable():
    strict = DecisionExplainer(min_confidence_to_publish=0.95)
    assert explain(strict).verdict is Verdict.INCONCLUSIVE


# -- bands ------------------------------------------------------------------


@pytest.mark.parametrize(
    "confidence,band",
    [
        (0.95, ConfidenceBand.HIGH),
        (0.55, ConfidenceBand.MEDIUM),
        (0.42, ConfidenceBand.LOW),
    ],
)
def test_bands_follow_the_thresholds(confidence, band):
    result = explain(decision=make_decision(confidence=confidence))
    assert result.band is band


def test_a_dead_chain_is_band_none_not_band_low():
    """"Low" is a weak answer; "none" is no answer. Collapsing them would let
    an empty frame read as a poor-quality call."""
    result = explain(decision=None, calibration=None, teams=None, poses=[])
    assert result.band is ConfidenceBand.NONE
    assert result.confidence == 0.0


# -- signals ----------------------------------------------------------------


def test_a_directional_pitch_is_capped_below_a_metric_one():
    """It can order players but not measure the gap, so however confident the
    line detection was, it cannot support a metric claim."""
    result = explain(
        calibration=FakeCalibration(confidence=0.95, is_metric=False)
    )
    assert result.signal(PITCH).score <= 0.5
    assert "distances" in result.signal(PITCH).reason


def test_a_missing_pitch_is_blocking():
    signals = collect_signals(make_decision(), calibration=None)
    pitch = next(s for s in signals if s.key == PITCH)
    assert pitch.blocking and pitch.score == 0.0


def test_foot_quality_is_judged_on_the_two_players_being_compared():
    """Eighteen clean foot points do not help when the defender the line runs
    through is the one guess on the frame."""
    poses = [pose() for _ in range(4)]
    poses[2] = pose(confidence=0.3, measured=False)
    result = explain(poses=poses)
    assert result.signal(BODY_POINTS).score <= 0.4
    assert "second-last defender" in result.signal(BODY_POINTS).reason


def test_clean_feet_elsewhere_do_not_rescue_a_guessed_defender():
    poses = [pose(confidence=0.99) for _ in range(4)]
    poses[2] = pose(confidence=0.2, measured=False)
    assert explain(poses=poses).confidence <= 0.4


def test_unconfirmed_teams_cap_the_team_signal():
    result = explain(teams=make_teams(confidence=0.95, unsure=1))
    assert result.signal(TEAMS).score <= 0.6
    assert "confirm" in result.signal(TEAMS).action


def test_unknown_sides_block_the_attacking_side_signal_not_team_colours():
    """The bug this split fixes, made concrete: a frame where the kits are
    read perfectly cleanly but nobody is near the ball must not report
    "team colours" as the thing that failed — an operator sent to recheck
    kit colours on a frame where they were already fine is being misled by
    the tool's own explanation."""
    result = explain(teams=make_teams(known=False))
    assert not result.signal(TEAMS).blocking
    assert result.signal(TEAMS).score >= 0.8, "kit colours were fine and must read that way"
    assert result.signal(ATTACKING_SIDE).blocking
    assert result.verdict is Verdict.INCONCLUSIVE


def test_a_kit_clustering_failure_still_blocks_team_colours():
    """The other half: when the kits genuinely couldn't be told apart, that
    is a real team-colour failure and must still be reported as one."""
    result = explain(teams=make_teams(color_model=False))
    assert result.signal(TEAMS).blocking
    assert "could not be told apart" in result.signal(TEAMS).reason


def test_a_contested_identity_matters_more_when_it_is_one_of_our_two():
    """A mix-up across the pitch is noise; a mix-up between the two players
    being compared produces a decision that looks entirely normal and is
    wrong."""
    ours = FakeIdentities(
        [FakeIdentity(0, contested=True)] + [FakeIdentity(i) for i in range(1, 4)]
    )
    theirs = FakeIdentities(
        [FakeIdentity(i) for i in range(3)] + [FakeIdentity(3, contested=True)]
    )
    involved = explain(identities=ours)
    elsewhere = explain(identities=theirs)
    assert involved.signal(IDENTITY).score < elsewhere.signal(IDENTITY).score


def test_a_contested_identity_elsewhere_is_reported_but_not_punished():
    theirs = FakeIdentities(
        [FakeIdentity(i) for i in range(3)] + [FakeIdentity(3, contested=True)]
    )
    signal = explain(identities=theirs).signal(IDENTITY)
    assert "not the ones this call depends on" in signal.reason


def test_a_tracker_that_has_only_just_started_does_not_veto_the_call():
    """Every frame right after a camera cut looks like this on real footage.
    Nothing has been mixed up — nothing has happened at all — and scoring
    that as a failure would refuse every call for the first stretch of every
    shot on the strength of no evidence."""
    fresh = FakeIdentities(
        [FakeIdentity(i, contested=True) for i in range(4)], confidence=0.1
    )
    fresh.contested = lambda: []  # tentative, not contested: no rival claims
    result = explain(identities=fresh)
    assert result.signal(IDENTITY).available is False
    assert result.verdict is Verdict.OFFSIDE


def test_a_mix_up_still_caps_the_call_however_young_the_tracker_is():
    """The distinction the previous test rests on: absence of evidence is
    non-capping, evidence of a swap is not."""
    mixed = FakeIdentities(
        [FakeIdentity(0, contested=True)] + [FakeIdentity(i) for i in range(1, 4)],
        confidence=0.2,
    )
    result = explain(identities=mixed)
    assert result.signal(IDENTITY).available
    assert result.verdict is Verdict.INCONCLUSIVE


def test_the_geometry_signal_reports_the_margin_against_its_own_error():
    signal = explain().signal(GEOMETRY)
    assert "0.90m" in signal.reason and "0.30m" in signal.reason


def test_every_stage_appears_even_when_it_is_fine():
    """A panel that only lists problems cannot be used to check that the good
    stages are actually good."""
    keys = {signal.key for signal in explain().signals}
    assert keys == {PITCH, BODY_POINTS, TEAMS, ATTACKING_SIDE, IDENTITY, GEOMETRY}


# -- the prose --------------------------------------------------------------


def test_the_headline_reads_as_a_sentence_not_a_score():
    headline = explain().headline
    assert headline.startswith("Offside position")
    assert "confidence" in headline
    assert "0." not in headline, "the band is the sentence; the number sits beside it"


def test_the_detail_states_the_measurement_with_its_error():
    detail = explain().detail
    assert "0.90m" in detail and "0.30m" in detail


def test_limits_are_listed_even_on_a_confident_call():
    """An operator told only the good half has been handed a sales pitch."""
    result = explain(teams=make_teams(confidence=0.62))
    assert result.limits
    assert any("team" in limit for limit in result.limits)


def test_a_caveat_never_quotes_a_healthy_stage_as_a_complaint():
    """A stage's description and its caveat are different sentences. "The
    pitch is marked, so positions are in metres" is the right thing to say
    about a working stage and nonsense under a heading that reads "limited
    by" — which is exactly what the first build of this panel printed."""
    result = explain(calibration=FakeCalibration(confidence=0.58))
    pitch_limit = next(limit for limit in result.limits if limit.startswith("pitch"))

    assert "so positions are measured in metres" not in pitch_limit
    assert "carried a long way" in pitch_limit


def test_limits_are_ordered_worst_first():
    result = explain(
        calibration=FakeCalibration(confidence=0.2),
        teams=make_teams(confidence=0.55),
    )
    assert result.limits[0].startswith("pitch calibration")


def test_actions_tell_the_operator_what_to_do_next():
    result = explain(calibration=FakeCalibration(confidence=0.2, is_metric=False))
    assert any("mark four" in action for action in result.actions)


def test_a_blocking_stage_explains_itself_in_the_headline():
    result = explain(teams=make_teams(known=False))
    assert result.headline.startswith("No call")
    assert "attacking" in result.headline


def test_as_lines_carries_the_whole_explanation():
    lines = explain(calibration=FakeCalibration(confidence=0.5)).as_lines()
    assert any(line.startswith("limited by:") for line in lines)
    assert any(line.startswith("you can:") for line in lines)


def test_the_summary_carries_the_first_caveat_with_the_headline():
    result = explain(teams=make_teams(confidence=0.5))
    assert result.headline in result.summary()
    assert result.limits[0] in result.summary()


# -- the operator's own call ------------------------------------------------


def test_an_operator_override_is_a_full_decision_not_a_flag():
    """The override travels the same path the pipeline's own answer does —
    one display, log and export route, not a second, less-tested one."""
    result = operator_decision(Verdict.ONSIDE, note="clear daylight")
    assert result.verdict is Verdict.ONSIDE
    assert result.confidence == 1.0
    assert result.is_published


def test_an_override_says_it_came_from_a_person():
    result = operator_decision(Verdict.OFFSIDE)
    assert result.source == SOURCE_OPERATOR
    assert result.is_operator_call
    assert "operator" in result.headline


def test_goalkeepers_do_not_change_the_signals():
    """A keeper is ranked like any other opponent (M2.5); nothing in the
    confidence chain should treat them specially."""
    teams = make_teams()
    teams.players[2].role = PlayerRole.GOALKEEPER
    assert explain(teams=teams).confidence == pytest.approx(explain().confidence)
