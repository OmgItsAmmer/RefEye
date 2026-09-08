"""The offside verdict in the review screen (M2.7, redesigned per operator feedback).

This is where the whole milestone is either trusted or ignored, so the panel is
built around one idea: **the operator must always be able to see what the tool
is unsure about, and always be able to overrule it.**

## Why this looks the way it does

The first build of this panel said everything in paragraphs — a headline
sentence, a detail sentence, a "limited by" sentence, a "you can" sentence.
Read individually each one was accurate; read together, on a real frame, nine
lines of prose held one clear-cut fact (a percentage) and it hid the one
actionable one (an off-target stage) inside a run-on sentence. Operator
feedback on that build was blunt: *"clearly show these so user gets clear cut
results, not necessarily a description."*

Rebuilt around four pieces, in the order the eye should hit them:

1. **The verdict, large, neutral colour.** What the tool concluded, stated
   once, not buried in a sentence. Deliberately *not* colour-coded by verdict
   — an offside call is not a failure and must not be painted red (theme.md's
   "never traffic light coding" rule already governs this everywhere else).
2. **A requirements checklist, not prose.** Every stage feeding the call, as
   one row each: a status dot, a percentage, a one-word read (GOOD / WEAK /
   POOR / N/A). *This* is "some metrics we're fulfilling and the percentage
   confidence we achieve in them" — the exact phrase the feedback used.
3. **A suggestions box.** Actionable next steps, visually set apart (a tinted
   card, not another paragraph) — literally requested as "suggestion box".
4. **The reason, last, in plain English.** The one sentence answering "why
   this result" — kept, because a checklist without a reason is just numbers;
   moved to the end because it's context for the verdict, not the headline.

Colour vocabulary is unchanged from the rest of the app: green/blue/amber/grey
means confidence (HIGH/MEDIUM/LOW/NONE), never the verdict itself.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from apps.desktop.ui.theme import tokens as t
from apps.desktop.ui.widgets.common import (
    AnimatedButton,
    BadgeVariant,
    StatusBadge,
    data_value,
    divider,
    label,
    meta,
    simple_dot,
)
from offside.decision_support.explanation import (
    BAND_MEANING,
    VERDICT_WORDS,
    ConfidenceBand,
    DecisionExplanation,
    operator_decision,
)
from offside.offside_line.line import Verdict
from offside.offside_line.rendering import draw_offside_overlay

#: The band a verdict is shown in. Deliberately not a traffic light: the colour
#: says how sure the tool is, never whether the answer is "good news" — an
#: offside call is not a failure and must not be painted like one.
_BAND_VARIANT = {
    ConfidenceBand.HIGH: BadgeVariant.ACCENT,
    ConfidenceBand.MEDIUM: BadgeVariant.INFO,
    ConfidenceBand.LOW: BadgeVariant.WARNING,
    ConfidenceBand.NONE: BadgeVariant.MUTED,
}

#: Requirement-row status, in the same three-colour vocabulary as the band
#: badge above — one dictionary, not a second colour scheme to learn. The
#: cutoffs mirror `DecisionExplainer`'s own defaults (high=0.7, medium=0.45);
#: this is a presentational read of the same numbers, not a second policy.
_ROW_HIGH = 0.7
_ROW_MEDIUM = 0.45
_ROW_COLOR = {
    "good": t.SUCCESS,
    "weak": t.PRIMARY,
    "poor": t.ALERT,
    "na": t.BORDER_STRONG,
}
_ROW_WORD = {"good": "GOOD", "weak": "WEAK", "poor": "POOR", "na": "N/A"}

#: The overrides an operator can set, in the order they appear.
OVERRIDE_CHOICES = (
    (Verdict.OFFSIDE, "Offside"),
    (Verdict.ONSIDE, "Onside"),
    (Verdict.TOO_CLOSE, "Too close"),
)


def _row_state(signal) -> str:
    if not signal.available:
        return "na"
    if signal.blocking:
        return "poor"
    if signal.score >= _ROW_HIGH:
        return "good"
    if signal.score >= _ROW_MEDIUM:
        return "weak"
    return "poor"


class ConfidenceMeter(QWidget):
    """A single bar for the confidence, with the publish floor marked on it.

    The floor tick is the point: a number alone does not tell an operator
    whether 0.38 was nearly enough, and the one thing they want to know when a
    call is withheld is how far short it fell.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("ConfidenceMeter")
        self.setFixedHeight(6)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._value = 0.0
        self._floor = 0.4
        self._band = ConfidenceBand.NONE

    def set_value(self, value: float, band: ConfidenceBand, floor: float = 0.4) -> None:
        self._value = max(0.0, min(1.0, float(value)))
        self._band = band
        self._floor = max(0.0, min(1.0, float(floor)))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        width = self.width()
        height = self.height()

        painter.fillRect(0, 0, width, height, QColor(t.BORDER))
        filled = int(width * self._value)
        colour = QColor(t.TEXT_MUTED if self._band is ConfidenceBand.NONE else t.PRIMARY)
        painter.fillRect(0, 0, filled, height, colour)

        # The floor, so "how far short" is visible rather than arithmetic.
        tick = int(width * self._floor)
        painter.fillRect(tick, 0, 1, height, QColor(t.BORDER_STRONG))
        painter.end()


class _MiniBar(QWidget):
    """A small, fixed-width fraction-filled bar for one requirement row.

    Deliberately tiny and wordless on its own — the percentage text next to
    it carries the exact number; this is the at-a-glance shape of it, the
    same division of labour the big `ConfidenceMeter` uses for the overall
    score.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(44, 6)
        self._fraction = 0.0
        self._colour = t.BORDER_STRONG

    def set_fraction(self, fraction: float, colour: str) -> None:
        self._fraction = max(0.0, min(1.0, float(fraction)))
        self._colour = colour
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor(t.BORDER))
        filled = int(self.width() * self._fraction)
        if filled > 0:
            painter.fillRect(0, 0, filled, self.height(), QColor(self._colour))
        painter.end()


class RequirementRow(QWidget):
    """One stage, read as a requirement: met, weak, or not met — and by how
    much. Replaces the earlier plain "label + number" row; the dot, the bar
    and the word are three views of the same score so a glance is enough and
    the tooltip carries the full sentence for whoever wants it."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("RequirementRow")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(t.SPACING_UNIT)

        self._dot = simple_dot(_ROW_COLOR["na"])
        layout.addWidget(self._dot)

        self._name = meta("")
        layout.addWidget(self._name)
        layout.addStretch(1)

        self._bar = _MiniBar()
        layout.addWidget(self._bar)

        # A score is a technical value — the data face (theme.md).
        self._score = data_value("")
        self._score.setMinimumWidth(64)
        self._score.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._score)

    def set_signal(self, signal) -> None:
        self._name.setText(signal.label)
        state = _row_state(signal)
        colour = _ROW_COLOR[state]
        self._dot.setStyleSheet(f"background-color: {colour}; border-radius: 4px;")

        if not signal.available:
            self._bar.set_fraction(0.0, colour)
            self._score.setText("N/A")
        else:
            self._bar.set_fraction(signal.score, colour)
            self._score.setText(f"{signal.score * 100:.0f}% {_ROW_WORD[state]}")
        self.setToolTip(f"{signal.phase} — {signal.reason}")


class SuggestionBox(QWidget):
    """Actionable next steps, set apart from the rest of the panel as a
    prominent actionable card."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("SuggestionBox")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.setSpacing(6)
        self._title = label("Suggestions")
        header_row.addWidget(self._title)
        header_row.addStretch(1)
        layout.addLayout(header_row)

        self._items_layout = QVBoxLayout()
        self._items_layout.setContentsMargins(0, 0, 0, 0)
        self._items_layout.setSpacing(4)
        layout.addLayout(self._items_layout)

        self._rows: list[QLabel] = []

    def set_actions(self, actions: list[str]) -> None:
        while len(self._rows) < len(actions):
            row = QLabel("")
            row.setObjectName("SuggestionItem")
            row.setWordWrap(True)
            self._items_layout.addWidget(row)
            self._rows.append(row)

        for index, row in enumerate(self._rows):
            if index < len(actions):
                row.setText(f"→  {actions[index]}")
                row.setVisible(True)
            else:
                row.setVisible(False)

        self.setVisible(bool(actions))


def _verdict_label() -> QLabel:
    """The one large, unmistakable word this panel exists to show.

    Not styled via the shared `title()` helper — that role is 15px, sized for
    chrome (panel headers), and this needs to read from across the room. A
    dedicated size, still driven by the same token scale, not a magic number.
    """
    lbl = QLabel("")
    lbl.setObjectName("VerdictWord")
    lbl.setWordWrap(True)
    font = lbl.font()
    font.setPointSize(t.FONT_SIZE_TITLE + 8)
    font.setWeight(QFont.Weight.DemiBold)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, t.TRACKING_TITLE)
    lbl.setFont(font)
    return lbl


class OffsideReviewPanel(QWidget):
    """The verdict, the requirements checklist, suggestions, and the reason."""

    #: The operator set (or cleared, with None) their own verdict.
    override_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("OffsideReviewPanel")

        self._explanation: DecisionExplanation | None = None
        self._override: Verdict | None = None
        self._publish_floor = 0.4

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(t.SPACING_UNIT)

        # -- 1. the verdict, large, neutral colour --------------------------
        self._verdict_word = _verdict_label()
        layout.addWidget(self._verdict_word)

        confidence_row = QHBoxLayout()
        confidence_row.setSpacing(t.SPACING_UNIT)
        self._confidence_badge = StatusBadge("No call", BadgeVariant.MUTED)
        confidence_row.addWidget(self._confidence_badge)
        confidence_row.addStretch(1)
        self._confidence_value = data_value("")
        confidence_row.addWidget(self._confidence_value)
        layout.addLayout(confidence_row)

        self._meter = ConfidenceMeter()
        layout.addWidget(self._meter)

        layout.addWidget(divider())

        # -- 2. the requirements checklist -----------------------------------
        layout.addWidget(label("Requirements"))
        self._rows: list[RequirementRow] = []
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        rows_host = QWidget()
        rows_host.setLayout(self._rows_layout)
        layout.addWidget(rows_host)

        # -- 3. suggestions ---------------------------------------------------
        self._suggestions = SuggestionBox()
        layout.addWidget(self._suggestions)

        # -- 4. the reason & why card ---------------------------------------
        layout.addWidget(divider())
        self._reason_card = QFrame()
        self._reason_card.setObjectName("ReasoningBox")
        reason_layout = QVBoxLayout(self._reason_card)
        reason_layout.setContentsMargins(12, 10, 12, 10)
        reason_layout.setSpacing(4)

        reason_header = label("Why")
        reason_layout.addWidget(reason_header)

        self._headline = QLabel("No frame has been analysed yet.")
        self._headline.setObjectName("ReasonHeadline")
        self._headline.setWordWrap(True)
        reason_layout.addWidget(self._headline)

        self._detail = meta("")
        self._detail.setObjectName("ReasonDetail")
        self._detail.setWordWrap(True)
        reason_layout.addWidget(self._detail)

        layout.addWidget(self._reason_card)

        layout.addStretch(1)
        layout.addWidget(divider())
        layout.addWidget(label("Operator override"))
        layout.addLayout(self._build_override_grid())

        self.clear()

    # -- construction -------------------------------------------------------

    def _build_override_grid(self) -> QGridLayout:
        """Two columns, not one row of four.

        Four `AnimatedButton`s side by side do not fit the review rail's own
        column width even at the app's stated minimum window size — the row
        forced the whole column wider than the space available, which is
        what a screen with too little width for it actually looks like:
        buttons and labels clipped at the edge rather than a readable panel.
        A 2-column grid caps the row at two buttons wide regardless of how
        many choices exist.
        """
        grid = QGridLayout()
        grid.setSpacing(t.SPACING_UNIT // 2)
        self._override_buttons: dict[Verdict, AnimatedButton] = {}

        choices = list(OVERRIDE_CHOICES) + [(None, "Clear")]
        for position, (verdict, text) in enumerate(choices):
            button = AnimatedButton(text)
            if verdict is None:
                button.setToolTip("Hand the call back to the tool")
                button.clicked.connect(lambda: self.set_override(None))
                self._clear_button = button
            else:
                button.setToolTip(
                    f"Record this frame as {text.lower()} — your call, not the tool's"
                )
                button.clicked.connect(lambda _=False, v=verdict: self.set_override(v))
                self._override_buttons[verdict] = button
            grid.addWidget(button, position // 2, position % 2)
        return grid

    # -- state --------------------------------------------------------------

    @property
    def explanation(self) -> DecisionExplanation | None:
        """What is currently on screen — the override when there is one."""
        if self._override is not None:
            return operator_decision(self._override, note="set in the review screen")
        return self._explanation

    @property
    def override(self) -> Verdict | None:
        return self._override

    def set_publish_floor(self, floor: float) -> None:
        """The threshold M2.6 publishes above, shown as a tick on the meter."""
        self._publish_floor = float(floor)

    def set_explanation(self, explanation: DecisionExplanation | None) -> None:
        self._explanation = explanation
        self._refresh()

    def set_override(self, verdict: Verdict | None) -> None:
        self._override = verdict
        self._refresh()
        self.override_changed.emit(verdict)

    def clear(self) -> None:
        self._explanation = None
        self._override = None
        self._refresh()

    # -- rendering ----------------------------------------------------------

    def _refresh(self) -> None:
        explanation = self.explanation
        for verdict, button in self._override_buttons.items():
            button.setProperty("selected", "true" if verdict == self._override else "false")
            button.style().unpolish(button)
            button.style().polish(button)
        self._clear_button.setEnabled(self._override is not None)

        if explanation is None:
            self._verdict_word.setText("NO ANALYSIS YET")
            self._confidence_badge.set_status("No call", BadgeVariant.MUTED)
            self._confidence_value.setText("")
            self._meter.set_value(0.0, ConfidenceBand.NONE, self._publish_floor)
            self._headline.setText("No frame has been analysed yet.")
            self._detail.setText("")
            self._suggestions.set_actions([])
            self._set_rows([])
            return

        self._verdict_word.setText(VERDICT_WORDS.get(explanation.verdict, "No call").upper())
        self._headline.setText(explanation.headline)
        self._detail.setText(explanation.detail)

        if explanation.is_operator_call:
            # An operator's own call has no pipeline confidence to report, and
            # showing one would suggest the tool endorsed it.
            self._confidence_badge.set_status("Your call", BadgeVariant.ACCENT)
            self._confidence_value.setText("")
            self._meter.set_value(1.0, ConfidenceBand.HIGH, self._publish_floor)
            # Overruling the tool must not erase what it said. The operator may
            # want to compare, and whoever reviews this decision later will
            # certainly want to see what was overruled.
            tool = self._explanation
            self._detail.setText(
                "" if tool is None else f"The tool read: {tool.headline}"
            )
            self._suggestions.set_actions([])
            self._set_rows(tool.signals if tool else [])
            return

        self._confidence_badge.set_status(
            explanation.band.value.upper(), _BAND_VARIANT[explanation.band]
        )
        self._confidence_value.setText(f"{explanation.confidence * 100:.0f}%")
        self._meter.set_value(
            explanation.confidence, explanation.band, self._publish_floor
        )
        self._suggestions.set_actions(explanation.actions)
        self._set_rows(explanation.signals)

    def _set_rows(self, signals) -> None:
        while len(self._rows) < len(signals):
            row = RequirementRow()
            self._rows_layout.addWidget(row)
            self._rows.append(row)

        for index, row in enumerate(self._rows):
            if index < len(signals):
                row.set_signal(signals[index])
                row.setVisible(True)
            else:
                row.setVisible(False)

    # -- the frame overlay --------------------------------------------------

    def render_overlay(self, image: np.ndarray, decision) -> np.ndarray:
        """Draw the line on a copy of `image` for the review video surface.

        A copy, because the caller's frame may be the cached one shared with
        every other view — drawing into it would leave the line burned onto the
        frame everywhere it is shown, including after the operator overrides.
        """
        if image is None or decision is None:
            return image
        painted = image.copy()
        draw_offside_overlay(painted, decision, self.explanation)
        return painted


def band_caption(explanation: DecisionExplanation | None) -> str:
    """One line for anywhere too narrow for the panel — a status bar, a log."""
    if explanation is None:
        return "No offside decision for this frame"
    return f"{explanation.headline} ({BAND_MEANING[explanation.band]})"


#: Kept so the review screen can align its own placeholder wording with the
#: panel's, rather than inventing a second vocabulary for "nothing here yet".
EMPTY_CAPTION = "No offside decision for this frame"
