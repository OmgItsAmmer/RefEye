"""The action vocabulary every spotter must normalize into.

Model-native label sets differ (SoccerNet ball-action classes, T-DEED's own
head, a fine-tuned in-house model). Each adapter maps its own labels onto
these names so the review UI, ranking, and refinement never learn a specific
model's schema (architecture.md section 19).
"""

from __future__ import annotations

from typing import Final

PASS: Final = "pass"
SHOT: Final = "shot"
CROSS: Final = "cross"
HEADER: Final = "header"
REBOUND: Final = "rebound"
GOALKEEPER_CONTACT: Final = "goalkeeper_contact"
BALL_CONTACT: Final = "ball_contact"

SUPPORTED_ACTIONS: Final = frozenset(
    {PASS, SHOT, CROSS, HEADER, REBOUND, GOALKEEPER_CONTACT, BALL_CONTACT}
)

#: Human-readable labels for the review UI. Sentence case per theme.md.
DISPLAY_NAMES: Final = {
    PASS: "Pass",
    SHOT: "Shot",
    CROSS: "Cross",
    HEADER: "Header",
    REBOUND: "Rebound",
    GOALKEEPER_CONTACT: "Goalkeeper contact",
    BALL_CONTACT: "Ball contact",
}

#: SoccerNet ball-action-spotting labels -> domain vocabulary. Used by any
#: adapter whose checkpoint was trained on that label set.
SOCCERNET_TO_DOMAIN: Final = {
    "PASS": PASS,
    "DRIVE": PASS,
    "HIGH PASS": PASS,
    "CROSS": CROSS,
    "SHOT": SHOT,
    "HEADER": HEADER,
    "BALL PLAYER BLOCK": REBOUND,
    "PLAYER SUCCESSFUL TACKLE": BALL_CONTACT,
    "THROW IN": PASS,
    "FREE KICK": PASS,
    "GOAL": SHOT,
}


def display_name(action_type: str) -> str:
    return DISPLAY_NAMES.get(action_type, action_type.replace("_", " ").capitalize())
