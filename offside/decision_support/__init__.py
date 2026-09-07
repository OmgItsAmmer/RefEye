"""Confidence scoring and plain-language reasoning (M2.6).

The stage that decides what the operator is *told*, as distinct from what the
geometry computed. It aggregates every earlier phase's own confidence into one
honest number — the weakest link, never the average — writes the sentence that
goes with it, and withholds a verdict the chain behind it cannot carry.
"""

from offside.decision_support.explainer import DecisionExplainer
from offside.decision_support.explanation import (
    BAND_MEANING,
    SOURCE_OPERATOR,
    SOURCE_PIPELINE,
    VERDICT_WORDS,
    ConfidenceBand,
    DecisionExplanation,
    operator_decision,
)
from offside.decision_support.signals import (
    ATTACKING_SIDE,
    BODY_POINTS,
    GEOMETRY,
    IDENTITY,
    PITCH,
    TEAMS,
    ConfidenceSignal,
    collect_signals,
)

__all__ = [
    "ATTACKING_SIDE",
    "BAND_MEANING",
    "BODY_POINTS",
    "GEOMETRY",
    "IDENTITY",
    "PITCH",
    "SOURCE_OPERATOR",
    "SOURCE_PIPELINE",
    "TEAMS",
    "VERDICT_WORDS",
    "ConfidenceBand",
    "ConfidenceSignal",
    "DecisionExplainer",
    "DecisionExplanation",
    "collect_signals",
    "operator_decision",
]
