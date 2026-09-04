"""Domain object classes and the mapping from detector-native labels.

The rest of the application speaks only these names. A detector trained on
COCO, on SoccerNet, or fine-tuned in-house all normalize into this vocabulary
inside their own adapter, so swapping detectors changes nothing downstream
(architecture.md section 14).
"""

from __future__ import annotations

from typing import Final

PLAYER: Final = "player"
GOALKEEPER: Final = "goalkeeper"
BALL: Final = "ball"
REFEREE: Final = "referee"

ALL_CLASSES: Final = frozenset({PLAYER, GOALKEEPER, BALL, REFEREE})

#: Classes that represent a person who can make contact with the ball.
PERSON_CLASSES: Final = frozenset({PLAYER, GOALKEEPER, REFEREE})

#: COCO -> domain. A generic COCO model cannot distinguish goalkeeper from
#: outfield player or referee, so everything person-shaped becomes `player`.
#: Role assignment (kit colour, position) is a later-milestone concern; until
#: then nothing downstream may assume goalkeeper labels exist.
COCO_TO_DOMAIN: Final = {
    "person": PLAYER,
    "sports ball": BALL,
}
