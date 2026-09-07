"""Who is who, and how sure we are — the vocabulary of player identity (M2.4).

## What this phase is actually protecting against

Offside is decided by *which* player is the second-last defender. If two
players swap identities in the frames leading up to the contact moment, every
later stage still runs perfectly and produces a confidently wrong answer:
the line gets drawn through the wrong person. Nothing downstream can detect
that, because nothing downstream knows the swap happened.

So this module's output is not just "here is a track id". It is a track id
**plus how much that id deserves to be believed**, in a form M2.5 and M2.6
can act on:

    CONFIRMED   seen for a while, appearance steady, association unambiguous
    TENTATIVE   too new to have earned trust yet
    RECOVERED   came back after being hidden — probably right, not certain
    CONTESTED   the association was ambiguous, or the appearance changed
                when it should not have. Do not build a verdict on this.

`CONTESTED` is the one that matters. A tracker that quietly picks the more
likely of two candidates is exactly how a swap happens silently; this one
says it could not tell, which is the plan's "no silent guessing" rule applied
to identity.

## Why identities are per-segment

A camera cut ends every identity in the frame. Nothing about the previous
shot constrains the next one — a different part of the pitch, a different
set of players, possibly a replay of something that already happened. Track
ids therefore carry their segment (`s3t12`), so an id from before a cut can
never be silently reused for somebody else afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IdentityState(str, Enum):
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    RECOVERED = "recovered"
    CONTESTED = "contested"


#: States a downstream verdict may be built on without asking a human.
TRUSTED_STATES = frozenset({IdentityState.CONFIRMED})


@dataclass
class PlayerIdentity:
    """One player's identity at one frame, with the evidence behind it."""

    index: int
    track_id: str
    state: IdentityState
    confidence: float
    frames_seen: int
    frames_missed: int
    segment: int
    reason: str
    #: How far this frame's appearance sits from the track's running
    #: appearance, in the same colour units M2.3 uses. None when appearance
    #: could not be measured (too small, too occluded).
    appearance_distance: float | None = None
    #: Recent centre positions, newest last — the visible proof of continuity.
    trail: list[tuple[float, float]] = field(default_factory=list)

    @property
    def is_trusted(self) -> bool:
        return self.state in TRUSTED_STATES

    @property
    def needs_confirmation(self) -> bool:
        """True when a human should look before this identity decides a call."""
        return not self.is_trusted


@dataclass
class IdentityResult:
    """What the tracker concluded about this frame."""

    identities: list[PlayerIdentity] = field(default_factory=list)
    segment: int = 0
    cut_detected: bool = False
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def by_index(self, index: int) -> PlayerIdentity | None:
        for identity in self.identities:
            if identity.index == index:
                return identity
        return None

    def counts(self) -> dict[str, int]:
        counts = {state.value: 0 for state in IdentityState}
        for identity in self.identities:
            counts[identity.state.value] += 1
        return counts

    def contested(self) -> list[PlayerIdentity]:
        return [i for i in self.identities if i.state is IdentityState.CONTESTED]

    def trusted(self) -> list[PlayerIdentity]:
        return [i for i in self.identities if i.is_trusted]
