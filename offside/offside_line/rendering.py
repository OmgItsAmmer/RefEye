"""Drawing the offside decision on the frame (M2.5 geometry, M2.6 verdict).

This lives in the product, not in the debug tool, because two different
surfaces draw it — the operator's review screen and the pipeline inspector —
and an overlay that disagreed between them would be worse than no overlay at
all. The inspector imports this; it does not keep a second copy.

The overlay is the one part of the decision an operator can *argue with*. A
written "offside by 12cm" is a claim to be taken on trust; a line drawn through
the defender with the attacker's measured body point sitting in front of it is
evidence, and the operator can look at it and disagree.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Verdict colours. Note these are keyed on the *published* verdict, so a call
#: M2.6 withheld is drawn in the neutral grey of an inconclusive one rather
#: than in confident red — the frame must not say something the panel does not.
NEUTRAL = (170, 170, 170)

VERDICT_COLORS = {
    "offside": (60, 60, 235),
    "onside": (80, 220, 80),
    "too_close_to_call": (0, 200, 255),
    "inconclusive": (150, 150, 150),
}


def draw_offside_overlay(image: np.ndarray, decision, explanation=None) -> None:
    """The line, the two players it is measured between, and the verdict.

    This is the one overlay a verdict can actually be argued with. A written
    "offside by 12cm" is a claim; a line drawn through the defender with the
    attacker's measured body point sitting in front of it is something the
    operator can look at and disagree with, which is the whole point of the
    tool being decision *support*.

    When M2.6 has weighed in, its verdict is the one drawn — a call the chain
    cannot carry must not appear on the frame in confident colours just
    because the geometry reached it.
    """
    if decision is None or decision.line is None:
        return

    published = explanation.verdict if explanation is not None else decision.verdict
    headline = explanation.headline if explanation is not None else decision.headline()
    colour = VERDICT_COLORS.get(published.value, NEUTRAL)
    height, width = image.shape[:2]
    span = float(max(width, height)) * 2.0

    (x, y), (dx, dy) = decision.line
    cv2.line(
        image,
        (int(x - dx * span), int(y - dy * span)),
        (int(x + dx * span), int(y + dy * span)),
        colour,
        2,
        cv2.LINE_AA,
    )

    defender = decision.second_last_defender
    if defender is not None:
        point = (int(defender.point[0]), int(defender.point[1]))
        cv2.drawMarker(image, point, colour, cv2.MARKER_DIAMOND, 16, 2)
        cv2.putText(
            image,
            "2nd last",
            (point[0] + 10, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            colour,
            1,
            cv2.LINE_AA,
        )

    for attacker in decision.attackers:
        point = (int(attacker.point[0]), int(attacker.point[1]))
        marker_colour = VERDICT_COLORS.get(attacker.verdict.value, NEUTRAL)
        cv2.drawMarker(image, point, marker_colour, cv2.MARKER_TRIANGLE_UP, 14, 2)
        if attacker is decision.attacker:
            cv2.putText(
                image,
                f"{attacker.margin:+.2f}{decision.axis_unit}",
                (point[0] + 10, point[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                marker_colour,
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        image,
        ascii_safe(headline),
        (14, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        colour,
        2,
        cv2.LINE_AA,
    )



#: OpenCV's Hershey fonts are ASCII-only: anything outside it renders as a run
#: of "?" glyphs, which is how an em-dash in a headline turns into "Offside ???
#: high confidence" on the frame. Substitute rather than strip, so the
#: punctuation still reads.
_ASCII_SUBSTITUTIONS = {
    "—": "-",
    "–": "-",
    "±": "+/-",
    "→": "->",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "·": "-",
    "…": "...",
}


def ascii_safe(text: str) -> str:
    """Make a string safe for cv2.putText."""
    for source, replacement in _ASCII_SUBSTITUTIONS.items():
        text = text.replace(source, replacement)
    return text.encode("ascii", "replace").decode("ascii")
