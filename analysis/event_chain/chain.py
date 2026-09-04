"""Minimal event-chain ordering.

Architecture.md section 28 describes full event-chain reasoning (cross → shot
→ save → rebound → second shot). That is deliberately out of scope for M1;
what M1 provides is the foundation it needs — a temporally ordered view of
the candidates, kept alongside the ranked view.

Both orderings matter and neither replaces the other:
  * ranked order answers "which frame is most likely the one?"
  * chronological order answers "what happened, in what sequence?"

The architecture warns explicitly against assuming the last contact is the
one the operator wants, so nothing here reorders or discards candidates —
it only exposes the sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.domain.models import RefinedCandidate


@dataclass
class EventChain:
    events: list[RefinedCandidate] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.events

    def describe(self) -> list[str]:
        """Readable sequence, e.g. ['cross', 'shot', 'goalkeeper_contact']."""
        return [event.action_type for event in self.events]


def build_event_chain(candidates: list[RefinedCandidate]) -> EventChain:
    """Order candidates by their refined frame, earliest first."""
    return EventChain(events=sorted(candidates, key=lambda c: c.refined_frame_id))
