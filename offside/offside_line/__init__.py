"""The offside line and the verdict (M2.5).

    DepthAxis / AttackDirection   where players are along the pitch, and which
                                  end is being defended
    OffsideLineCalculator         draws the line and compares everyone to it
    OffsideDecision / Verdict     the call, with the geometry and the caveats

Read `line.py`'s docstring first: the geometry is a comparison of two numbers,
and everything difficult in this phase is about whether that comparison
survives the uncertainty of the inputs it inherits.
"""

from offside.offside_line.axis import (
    AttackDirection,
    DepthAxis,
    infer_attack_direction,
)
from offside.offside_line.line import (
    AttackerComparison,
    OffsideDecision,
    OffsideLineCalculator,
    Verdict,
)

__all__ = [
    "AttackDirection",
    "AttackerComparison",
    "DepthAxis",
    "OffsideDecision",
    "OffsideLineCalculator",
    "Verdict",
    "infer_attack_direction",
]
