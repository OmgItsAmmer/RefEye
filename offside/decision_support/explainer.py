"""Turning every stage's confidence into one call the operator can act on.

## Why this stage exists at all

M2.5 already withholds a verdict its own inputs cannot carry — but "its own
inputs" there means the direction of attack, the defender and the attacker.
It never sees whether the pitch marks have drifted half a second from where
they were placed, whether the two players being compared were mixed up with
somebody else, or whether the teams were assigned from kit colours the stage
itself wanted confirmed. Each of those produces a decision that looks entirely
normal and is wrong, so this phase widens the check across the whole chain.

## The aggregation rule, and why it is `min`

Confidence here is the weakest signal, not the average of them. An average is
how a tool ends up publishing a confident wrong answer: four solid stages
outvote the one that failed, and the failure is exactly the thing that decided
the verdict. A chain is as strong as its worst link, so the worst link sets the
number *and gets named in the explanation* — the useful half of a low score is
which stage to go and fix.

## Withholding, versus abstaining

Two different outputs that must not be confused:

* **Too close to call** is a measurement result. The margin is genuinely inside
  what the geometry can resolve; that is an honest answer and it is published.
* **Withheld** means the geometry did reach a verdict, but the chain behind it
  is too weak to say it out loud. The verdict is kept and shown as *what was
  withheld*, never as the call — an operator can see the tool's working without
  the tool putting its name to it.

Neither is a failure. A tool that only ever answers when it is sure is worth
more to a match official than one that always answers.
"""

from __future__ import annotations

from offside.decision_support.explanation import (
    BAND_MEANING,
    VERDICT_WORDS,
    ConfidenceBand,
    DecisionExplanation,
)
from offside.decision_support.signals import ConfidenceSignal, collect_signals
from offside.offside_line.line import OffsideDecision, Verdict


class DecisionExplainer:
    """Aggregates the stage confidences and writes the operator's sentence."""

    def __init__(
        self,
        *,
        high_confidence: float = 0.7,
        medium_confidence: float = 0.45,
        min_confidence_to_publish: float = 0.4,
        max_limits: int = 3,
        max_actions: int = 3,
    ):
        self._high = high_confidence
        self._medium = medium_confidence
        self._publish_floor = min_confidence_to_publish
        self._max_limits = max_limits
        self._max_actions = max_actions

    @classmethod
    def from_config(cls, config) -> DecisionExplainer:
        return cls(
            high_confidence=config.high_confidence,
            medium_confidence=config.medium_confidence,
            min_confidence_to_publish=config.min_confidence_to_publish,
            max_limits=config.max_limits,
            max_actions=config.max_actions,
        )

    def explain(
        self,
        decision: OffsideDecision | None,
        *,
        calibration=None,
        teams=None,
        identities=None,
        poses=None,
    ) -> DecisionExplanation:
        signals = collect_signals(
            decision,
            calibration=calibration,
            teams=teams,
            identities=identities,
            poses=poses,
        )
        geometry_verdict = decision.verdict if decision else Verdict.INCONCLUSIVE

        confidence, weakest = self._aggregate(signals)
        explanation = DecisionExplanation(
            geometry_verdict=geometry_verdict,
            confidence=confidence,
            band=self._band(confidence),
            signals=signals,
            weakest=weakest,
        )

        explanation.verdict = self._publishable(geometry_verdict, confidence)
        explanation.withheld = (
            geometry_verdict in (Verdict.OFFSIDE, Verdict.ONSIDE)
            and explanation.verdict is not geometry_verdict
        )

        explanation.detail = self._detail(decision)
        explanation.headline = self._headline(explanation, decision)
        explanation.reasons = self._reasons(decision, signals)
        explanation.limits = self._limits(signals)
        explanation.actions = self._actions(signals)
        return explanation

    # -- aggregation --------------------------------------------------------

    def _aggregate(
        self, signals: list[ConfidenceSignal]
    ) -> tuple[float, ConfidenceSignal | None]:
        """The weakest available signal, and which one it was."""
        available = [s for s in signals if s.available]
        if not available:
            return 0.0, None
        weakest = min(available, key=lambda s: s.score)
        return float(max(0.0, min(1.0, weakest.score))), weakest

    def _band(self, confidence: float) -> ConfidenceBand:
        if confidence <= 0.0:
            return ConfidenceBand.NONE
        if confidence >= self._high:
            return ConfidenceBand.HIGH
        if confidence >= self._medium:
            return ConfidenceBand.MEDIUM
        return ConfidenceBand.LOW

    def _publishable(self, verdict: Verdict, confidence: float) -> Verdict:
        """A yes/no is only published when the whole chain can carry it.

        `TOO_CLOSE` and `INCONCLUSIVE` pass through untouched: they are already
        abstentions, and a weak chain cannot make an abstention wrong.
        """
        if verdict in (Verdict.TOO_CLOSE, Verdict.INCONCLUSIVE):
            return verdict
        if confidence < self._publish_floor:
            return Verdict.INCONCLUSIVE
        return verdict

    # -- prose --------------------------------------------------------------

    def _headline(
        self, explanation: DecisionExplanation, decision: OffsideDecision | None
    ) -> str:
        band = explanation.band
        if explanation.withheld:
            withheld_word = VERDICT_WORDS[explanation.geometry_verdict].lower()
            weak = explanation.weakest
            because = weak.reason if weak is not None else "the inputs are too uncertain"
            return (
                f"No call — the geometry reads {withheld_word}, but {because}. "
                "Here is the frame; please judge it."
            )

        if explanation.verdict is Verdict.INCONCLUSIVE:
            weak = explanation.weakest
            if weak is not None and weak.blocking:
                return f"No call — {weak.reason}"
            if decision is not None and decision.warnings:
                return f"No call — {decision.warnings[0]}"
            return "No call — not enough of the pipeline succeeded on this frame"

        word = VERDICT_WORDS[explanation.verdict]
        if explanation.verdict is Verdict.TOO_CLOSE:
            return f"{word} — {BAND_MEANING[ConfidenceBand.LOW]}"
        return f"{word} — {band.value} confidence, {BAND_MEANING[band]}"

    def _detail(self, decision: OffsideDecision | None) -> str:
        if decision is None or decision.attacker is None:
            return ""
        attacker = decision.attacker
        unit = decision.axis_unit
        margin = abs(attacker.margin)
        side = "beyond" if attacker.margin > 0 else "behind"
        detail = (
            f"The attacker is {margin:.2f}{unit} {side} the second-last "
            f"defender (±{attacker.uncertainty:.2f}{unit})."
        )
        if attacker.margin_to_ball is not None and attacker.margin_to_ball <= 0:
            detail += " They are not ahead of the ball, which makes them onside."
        return detail

    def _reasons(
        self, decision: OffsideDecision | None, signals: list[ConfidenceSignal]
    ) -> list[str]:
        """What is holding the call up — the stages that did their job."""
        reasons = [
            signal.reason
            for signal in signals
            if signal.available and not signal.blocking and signal.score >= self._medium
        ]
        if decision is not None and decision.direction is not None and decision.direction.known:
            reasons.append(decision.direction.reason)
        return reasons

    def _limits(self, signals: list[ConfidenceSignal]) -> list[str]:
        """What is holding it down, weakest first — never silently dropped.

        Even a high-confidence call lists whatever is dragging on it. An
        operator who is told only the good half has been handed a sales pitch,
        not a decision.
        """
        weak = [
            signal
            for signal in signals
            if signal.available and (signal.blocking or signal.score < self._high)
        ]
        weak.sort(key=lambda signal: signal.score)
        return [
            f"{signal.label.lower()}: {signal.caveat}"
            for signal in weak[: self._max_limits]
        ]

    def _actions(self, signals: list[ConfidenceSignal]) -> list[str]:
        actions: list[str] = []
        for signal in sorted(
            (s for s in signals if s.available and s.action),
            key=lambda s: s.score,
        ):
            if signal.action not in actions:
                actions.append(signal.action)
        return actions[: self._max_actions]
