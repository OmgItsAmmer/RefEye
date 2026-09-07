"""What the operator is actually shown: a band, a sentence, and the caveats.

The client's requirement for this phase is not another number. It is that the
tool says *"Offside — high confidence, defender and attacker both clearly
visible"* or *"Too close to call — the camera angle makes the attacker's foot
position uncertain, here is the frame, please judge."* So this module is the
vocabulary for that sentence, and nothing in it re-derives geometry.

Two rules hold everything here honest:

**Confidence is the weakest signal, never the average.** Averaging lets four
good stages hide one fatal one — a perfect calibration, clean body points and
settled teams will happily average away the fact that the two players being
compared were swapped a frame ago. The chain is exactly as strong as its worst
link, so that is what gets reported, and the link's name is reported with it.

**A band is a promise, so the bands are deliberately mean.** "High" has to mean
the operator can sign the decision without opening the frame; anything short of
that is "medium" at best. The cost of a band that is too cautious is an extra
look; the cost of one that is too generous is a wrong call published in the
tool's own voice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from offside.decision_support.signals import ConfidenceSignal
from offside.offside_line.line import Verdict

#: Where a verdict came from, which the UI shows verbatim — an operator must
#: always be able to see that the call on screen is their own, not the tool's.
SOURCE_PIPELINE = "pipeline"
SOURCE_OPERATOR = "operator"


class ConfidenceBand(str, Enum):
    """The four words the tool is allowed to use about its own certainty."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    #: Nothing usable at all — not a weak answer, an absent one.
    NONE = "none"


#: Plain-language gloss for each band. These are shown to the operator, so they
#: describe what to *do*, not what the number is.
BAND_MEANING = {
    ConfidenceBand.HIGH: "every stage behind this call is solid",
    ConfidenceBand.MEDIUM: "the call stands up, but check the frame before signing it",
    ConfidenceBand.LOW: "treat this as a suggestion and judge the frame yourself",
    ConfidenceBand.NONE: "there is not enough here to offer a call",
}

#: How each verdict reads in a sentence, without jargon.
VERDICT_WORDS = {
    Verdict.OFFSIDE: "Offside position",
    Verdict.ONSIDE: "Onside",
    Verdict.TOO_CLOSE: "Too close to call",
    Verdict.INCONCLUSIVE: "No call",
}


@dataclass
class DecisionExplanation:
    """One frame's decision, in the form the operator sees it."""

    #: What the tool is prepared to publish. Not always what the geometry
    #: concluded: a verdict whose inputs cannot carry it is withheld.
    verdict: Verdict = Verdict.INCONCLUSIVE
    #: What M2.5's geometry concluded, kept even when it is not published so
    #: the operator can see what was withheld and why.
    geometry_verdict: Verdict = Verdict.INCONCLUSIVE
    confidence: float = 0.0
    band: ConfidenceBand = ConfidenceBand.NONE
    source: str = SOURCE_PIPELINE

    headline: str = ""
    #: The measurement in one sentence, or "" when there is nothing measured.
    detail: str = ""
    #: What supports the call.
    reasons: list[str] = field(default_factory=list)
    #: What is holding the confidence down — the honest half.
    limits: list[str] = field(default_factory=list)
    #: What the operator could do about it, in order of usefulness.
    actions: list[str] = field(default_factory=list)

    signals: list[ConfidenceSignal] = field(default_factory=list)
    #: The single stage that is capping the confidence.
    weakest: ConfidenceSignal | None = None
    #: True when the geometry reached a verdict that was not published.
    withheld: bool = False
    #: Set when an operator has overruled the tool.
    operator_note: str = ""

    @property
    def is_published(self) -> bool:
        """Whether a yes/no call is being offered at all."""
        return self.verdict in (Verdict.OFFSIDE, Verdict.ONSIDE)

    @property
    def is_operator_call(self) -> bool:
        return self.source == SOURCE_OPERATOR

    def signal(self, key: str) -> ConfidenceSignal | None:
        for candidate in self.signals:
            if candidate.key == key:
                return candidate
        return None

    def summary(self) -> str:
        """Headline plus the one thing most worth knowing about it — what a
        log line or a narrow badge gets when there is no room for the panel."""
        if self.limits:
            return f"{self.headline} ({self.limits[0]})"
        return self.headline

    def as_lines(self) -> list[str]:
        """The whole explanation as text, for logs and the debug inspector."""
        lines = [self.headline]
        if self.detail:
            lines.append(self.detail)
        lines += self.reasons
        lines += [f"limited by: {limit}" for limit in self.limits]
        lines += [f"you can: {action}" for action in self.actions]
        return lines


def operator_decision(verdict: Verdict, note: str = "") -> DecisionExplanation:
    """The operator's own call, which outranks anything the tool concluded.

    Kept in the same type on purpose: an override is displayed, logged and
    exported through exactly the path a pipeline decision is, so there is no
    second, less-tested route for the answer that actually gets used.
    """
    word = VERDICT_WORDS.get(verdict, "No call")
    return DecisionExplanation(
        verdict=verdict,
        geometry_verdict=verdict,
        confidence=1.0,
        band=ConfidenceBand.HIGH,
        source=SOURCE_OPERATOR,
        headline=f"{word} — set by the operator",
        reasons=["the operator judged this frame themselves"],
        operator_note=note,
    )
