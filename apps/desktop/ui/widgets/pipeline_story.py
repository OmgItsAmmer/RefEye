"""A-Z pipeline story — a premium, non-technical walkthrough of how offside
detection actually works, built for showing to a client.

Same visual language as everywhere else in this app (theme.md): monochrome,
flat, sharp radii, restrained motion. "Premium" here comes from precise
typography and a considered staggered reveal, not colour or gimmick — the
same restraint the rest of this system already reads by.

Every sentence in `_STAGES` describes something this build actually does,
worded for someone who has never heard of a homography or a Kalman filter —
nothing here is aspirational copy standing in for a feature that isn't real.
"""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer, QVariantAnimation
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from apps.desktop.ui.widgets.common import Panel, heading, meta

#: How long one stage takes to settle into place, and how far apart each
#: stage starts from the one before it — together these are what make the
#: reveal read as one continuous cascade down the page rather than a slide
#: deck advancing.
_REVEAL_MS = 420
_STAGGER_MS = 110

#: (letter, headline, plain-language description). The letters are a framing
#: device — "A to Z" — not a literal count; eight stages is the real
#: pipeline, not a padded-out alphabet.
_STAGES = [
    (
        "A",
        "Watching every frame",
        (
            "The system watches the match continuously as it happens — every "
            "camera frame is captured live, so nothing depends on reviewing "
            "footage after the fact."
        ),
    ),
    (
        "B",
        "Spotting the moment",
        (
            "AI automatically recognises the exact instant the ball is played — "
            "a pass, a shot — pinpointed to a single frame out of thousands, "
            "with no one scrubbing through replay to find it by hand."
        ),
    ),
    (
        "C",
        "Seeing every player",
        (
            "Every player and the ball are located automatically on the pitch, "
            "frame by frame — including inside a crowded penalty box, exactly "
            "where the human eye struggles most."
        ),
    ),
    (
        "D",
        "Mapping the real pitch",
        (
            "The camera's own angle is automatically translated into a true, "
            "real-world pitch map — so a distance is an actual measurement in "
            "metres, not a guess from pixels on a screen."
        ),
    ),
    (
        "E",
        "Telling the teams apart",
        (
            "Kit colours are read automatically to sort every player onto the "
            "correct side — including recognising the goalkeeper, who always "
            "stands apart from their own team."
        ),
    ),
    (
        "F",
        "Following every player",
        (
            "Each player is followed continuously through the passage of play — "
            "through a crowd, a camera pan, a moment out of sight — so the "
            "system always knows who is who when the moment matters."
        ),
    ),
    (
        "G",
        "Drawing the line",
        (
            "The offside line is calculated from real player positions on the "
            "true pitch map — a measurement, not an estimate by eye."
        ),
    ),
    (
        "H",
        "Knowing when to be sure",
        (
            "A verdict is only given once every step behind it is confident. If "
            "any part of the picture is uncertain, the system says exactly "
            "what — and asks for a quick confirmation — rather than guessing."
        ),
    ),
]


class _StageCard(QWidget):
    """One stage of the story: a lettered badge, a headline, plain-language
    copy, and the connecting line down to the next stage — all hidden until
    `reveal()` plays it in.
    """

    def __init__(self, letter: str, headline: str, body_text: str, *, is_last: bool, parent=None):
        super().__init__(parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(16)

        rail = QVBoxLayout()
        rail.setContentsMargins(0, 0, 0, 0)
        rail.setSpacing(0)
        rail.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        badge = QLabel(letter)
        badge.setObjectName("HelpStepBadge")
        badge.setFixedSize(28, 28)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rail.addWidget(badge)

        self._spine: QFrame | None = None
        self._spine_effect: QGraphicsOpacityEffect | None = None
        if not is_last:
            spine = QFrame()
            spine.setObjectName("StorySpine")
            spine.setFixedWidth(1)
            self._spine_effect = QGraphicsOpacityEffect(spine)
            self._spine_effect.setOpacity(0.0)
            spine.setGraphicsEffect(self._spine_effect)
            rail.addWidget(spine, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
            self._spine = spine
        else:
            rail.addStretch(1)

        outer.addLayout(rail)

        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 2, 0, 26)
        content_layout.setSpacing(4)

        # Collapses to 0 as the card settles into place — the "slide up
        # into position" half of the reveal, done by shrinking real layout
        # space rather than fighting the parent layout's own geometry.
        self._settle_spacer = QWidget()
        self._settle_spacer.setFixedHeight(16)
        content_layout.addWidget(self._settle_spacer)

        content_layout.addWidget(heading(headline))
        body = meta(body_text)
        body.setWordWrap(True)
        content_layout.addWidget(body)

        outer.addWidget(self._content, 1)

        self._content_effect = QGraphicsOpacityEffect(self._content)
        self._content_effect.setOpacity(0.0)
        self._content.setGraphicsEffect(self._content_effect)

        self._anims: list[QVariantAnimation] = []  # kept alive until finished

    def reveal(self, delay_ms: int) -> None:
        QTimer.singleShot(delay_ms, self._play)

    def _play(self) -> None:
        fade = QPropertyAnimation(self._content_effect, b"opacity", self)
        fade.setDuration(_REVEAL_MS)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        fade.finished.connect(lambda: self._content.setGraphicsEffect(None))

        settle = QVariantAnimation(self)
        settle.setDuration(_REVEAL_MS)
        settle.setStartValue(16)
        settle.setEndValue(0)
        settle.setEasingCurve(QEasingCurve.Type.OutCubic)
        settle.valueChanged.connect(self._settle_spacer.setFixedHeight)

        self._anims = [fade, settle]
        fade.start()
        settle.start()

        if self._spine is not None and self._spine_effect is not None:
            spine_fade = QPropertyAnimation(self._spine_effect, b"opacity", self)
            spine_fade.setDuration(_REVEAL_MS)
            spine_fade.setStartValue(0.0)
            spine_fade.setEndValue(1.0)
            spine_fade.setEasingCurve(QEasingCurve.Type.OutCubic)
            spine_fade.finished.connect(lambda: self._spine.setGraphicsEffect(None))
            self._anims.append(spine_fade)
            spine_fade.start()


class PipelineStoryPanel(Panel):
    """The full A-to-Z story, staggered in every time this panel is shown —
    built to be presented to a client, not just read once by an operator."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__("How offside detection works", parent=parent)

        subtitle = meta("From kickoff to verdict — every step, in plain terms.")
        subtitle.setWordWrap(True)
        subtitle.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.body().addWidget(subtitle)
        self.body().addSpacing(6)

        self._cards: list[_StageCard] = []
        for index, (letter, headline, body_text) in enumerate(_STAGES):
            card = _StageCard(letter, headline, body_text, is_last=(index == len(_STAGES) - 1))
            self._cards.append(card)
            self.body().addWidget(card)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Replays on every real show, not just the first — this panel exists
        # to be presented, and a client watching a demo more than once
        # should see the story again, not a page that only animates once.
        self.play()

    def play(self) -> None:
        for index, card in enumerate(self._cards):
            card.reveal(index * _STAGGER_MS)
