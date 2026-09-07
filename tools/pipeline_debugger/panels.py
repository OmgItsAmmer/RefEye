"""The inspector's panels — the parts that make the pipeline visible.

Everything here renders what `presenter.py` computed and reports what the
operator clicked. No decisions are made in this file, which is what keeps the
decisions testable without a display.

## The team view, and why it is two grids

The operator's question at this stage is not "what does the app think?" but
"is it looking at the same thing I am?". A grid of measured kit swatches
answers that in one glance: every cell shows the player's own crop next to
the colour that was actually measured from it, so a wrong reading appears as
a swatch that does not match the shirt beside it. Two separate grids, one per
kit, make an odd player out obvious in a way a single mixed list never does.

Cells carry two colour bars, not one, because a player's signature *is* two
colours — that is how striped kits survive measurement. On a solid kit both
bars are the same, which is itself worth seeing.

A dashed border means the pipeline wants a human look before this player is
used in a verdict.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from offside.team_assignment.teams import PlayerRole
from tools.pipeline_debugger.presenter import PlayerCard, TeamColumn

BG = "#121212"
SURFACE = "#1A1A1A"
RAISED = "#202020"
BORDER = "#2A2A2A"
TEXT = "#E0E0E0"
MUTED = "#8A8A8A"
ACCENT = "#4C9AFF"

STATE_COLORS = {
    "ok": "#00FF66",
    "degraded": "#FFB020",
    "unavailable": "#FF3366",
    "pending": "#525252",
}
STATE_LABELS = {
    "ok": "OK",
    "degraded": "PARTIAL",
    "unavailable": "UNAVAILABLE",
    "pending": "NOT BUILT",
}

#: Actions the operator can apply to a selected player.
ACTION_CONFIRM = "confirm"
ACTION_TEAM_A = "team_a"
ACTION_TEAM_B = "team_b"
ACTION_GOALKEEPER = "goalkeeper"
ACTION_EXCLUDE = "exclude"
ACTION_CLEAR = "clear"


def bgr_to_pixmap(image: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    return QPixmap.fromImage(
        QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()
    )


def _hex(bgr: tuple[int, int, int] | None) -> str:
    if bgr is None:
        return "#3A3A3A"
    blue, green, red = (int(max(0, min(255, c))) for c in bgr)
    return f"#{red:02X}{green:02X}{blue:02X}"


class SwatchCell(QFrame):
    """One player: the crop it was measured from, and the colours measured."""

    clicked = Signal(int)

    CROP_WIDTH = 54
    CROP_HEIGHT = 92

    def __init__(self, card: PlayerCard, selected: bool = False):
        super().__init__()
        self.card = card
        border = ACCENT if selected else (TEXT if card.needs_confirmation else BORDER)
        style = "dashed" if card.needs_confirmation and not selected else "solid"
        self.setStyleSheet(
            f"QFrame {{ background:{RAISED}; border:1px {style} {border}; }}"
            "QLabel { border:none; }"
        )
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        crop = QLabel()
        crop.setFixedSize(self.CROP_WIDTH, self.CROP_HEIGHT)
        crop.setAlignment(Qt.AlignCenter)
        crop.setStyleSheet(f"background:#000; border:1px solid {BORDER};")
        if card.crop is not None and card.crop.size:
            crop.setPixmap(
                bgr_to_pixmap(card.crop).scaled(
                    self.CROP_WIDTH,
                    self.CROP_HEIGHT,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            crop.setText("no crop")
            crop.setStyleSheet(f"color:{MUTED}; background:#000; border:1px solid {BORDER};")
        layout.addWidget(crop, alignment=Qt.AlignHCenter)

        # The measurement itself, next to the pixels it came from. Two bars,
        # because two colours is what was measured — equal halves on a solid
        # kit, and the actual stripe colours on a patterned one.
        bars = QHBoxLayout()
        bars.setSpacing(2)
        for colour in (card.primary_bgr, card.secondary_bgr):
            bar = QLabel()
            bar.setFixedHeight(14)
            bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            bar.setStyleSheet(f"background:{_hex(colour)}; border:1px solid {BORDER};")
            bars.addWidget(bar)
        layout.addLayout(bars)

        header = QHBoxLayout()
        name = QLabel(card.label + (" ⚑" if card.needs_confirmation else ""))
        name.setStyleSheet(f"color:{TEXT}; font-weight:600; font-size:11px;")
        percent = QLabel(f"{card.confidence * 100:.0f}%")
        percent.setStyleSheet(f"color:{MUTED}; font-size:10px; font-family:monospace;")
        header.addWidget(name)
        header.addStretch()
        header.addWidget(percent)
        layout.addLayout(header)

        note = "two-colour kit" if card.has_two_colours else ""
        if card.is_operator_set:
            note = "you set this"
        elif card.role is PlayerRole.GOALKEEPER:
            note = "goalkeeper"
        if note:
            tag = QLabel(note)
            tag.setStyleSheet(f"color:{MUTED}; font-size:10px;")
            layout.addWidget(tag)

        self.setToolTip("\n".join(card.lines))

    def mousePressEvent(self, event):
        self.clicked.emit(self.card.index)


class TeamGrid(QWidget):
    """One kit: its measured colour, and every player measured into it."""

    card_clicked = Signal(int)
    confirm_all = Signal(object)  # list[int]

    COLUMNS = 3

    def __init__(self, column: TeamColumn, selected_index: int | None = None):
        super().__init__()
        self.column = column

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QFrame()
        header.setStyleSheet(f"QFrame {{ background:{SURFACE}; border:1px solid {BORDER}; }}")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 6, 8, 6)

        kit = QLabel()
        kit.setFixedSize(22, 22)
        kit.setStyleSheet(
            f"background:{_hex(column.kit_bgr)}; border:1px solid {BORDER};"
        )
        header_layout.addWidget(kit)

        titles = QVBoxLayout()
        titles.setSpacing(0)
        title = QLabel(column.title)
        title.setStyleSheet(f"color:{TEXT}; font-weight:600; border:none;")
        subtitle = QLabel(column.subtitle)
        subtitle.setStyleSheet(f"color:{MUTED}; font-size:11px; border:none;")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header_layout.addLayout(titles)
        header_layout.addStretch()

        flagged = [card.index for card in column.cards if card.needs_confirmation]
        if flagged:
            confirm = QPushButton(f"Confirm all {len(flagged)}")
            confirm.setStyleSheet(
                f"QPushButton {{ background:{RAISED}; color:{TEXT};"
                f" border:1px solid {BORDER}; padding:4px 8px; font-size:11px; }}"
                f"QPushButton:hover {{ background:#2C2C2C; }}"
            )
            confirm.clicked.connect(lambda: self.confirm_all.emit(flagged))
            header_layout.addWidget(confirm)
        layout.addWidget(header)

        grid = QGridLayout()
        grid.setSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)
        for position, card in enumerate(column.cards):
            cell = SwatchCell(card, selected=card.index == selected_index)
            cell.clicked.connect(self.card_clicked.emit)
            grid.addWidget(cell, position // self.COLUMNS, position % self.COLUMNS)

        if not column.cards:
            empty = QLabel("nobody")
            empty.setStyleSheet(f"color:{MUTED}; font-size:11px;")
            grid.addWidget(empty, 0, 0)

        holder = QWidget()
        holder.setLayout(grid)
        layout.addWidget(holder)
        layout.addStretch()


class TeamsPanel(QWidget):
    """The team view: two kit grids, the odd ones out, and the correction bar."""

    action_requested = Signal(str, object)  # action, list[int]
    selection_changed = Signal(int)

    def __init__(self):
        super().__init__()
        self._selected: int | None = None
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(8)

        self._banner = QLabel("waiting for a frame…")
        self._banner.setWordWrap(True)
        self._banner.setStyleSheet(
            f"background:{SURFACE}; color:{TEXT}; border:1px solid {BORDER}; padding:8px;"
        )
        self._root.addWidget(self._banner)

        self._kits = QLabel("")
        self._kits.setWordWrap(True)
        self._kits.setStyleSheet(f"color:{MUTED}; font-size:11px; padding:0 2px;")
        self._root.addWidget(self._kits)

        self._root.addWidget(self._build_action_bar())

        self._grids_holder = QWidget()
        self._grids = QHBoxLayout(self._grids_holder)
        self._grids.setContentsMargins(0, 0, 0, 0)
        self._grids.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._grids_holder)
        scroll.setStyleSheet(f"border:1px solid {BORDER};")
        self._root.addWidget(scroll, stretch=1)

    def _build_action_bar(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet(f"QFrame {{ background:{SURFACE}; border:1px solid {BORDER}; }}")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        self._selection_label = QLabel("select a player to correct")
        self._selection_label.setStyleSheet(f"color:{MUTED}; border:none;")
        layout.addWidget(self._selection_label)
        layout.addStretch()

        self._buttons: list[QPushButton] = []
        for label, action in (
            ("Confirm as shown", ACTION_CONFIRM),
            ("→ Team A", ACTION_TEAM_A),
            ("→ Team B", ACTION_TEAM_B),
            ("Goalkeeper", ACTION_GOALKEEPER),
            ("Not a player", ACTION_EXCLUDE),
            ("Undo my change", ACTION_CLEAR),
        ):
            button = QPushButton(label)
            button.setEnabled(False)
            button.setStyleSheet(
                f"QPushButton {{ background:{RAISED}; color:{TEXT};"
                f" border:1px solid {BORDER}; padding:5px 9px; font-size:11px; }}"
                f"QPushButton:hover:enabled {{ background:#2C2C2C; }}"
                f"QPushButton:disabled {{ color:#555; }}"
            )
            button.clicked.connect(
                lambda _=False, chosen=action: self._emit_action(chosen)
            )
            layout.addWidget(button)
            self._buttons.append(button)

        return bar

    def _emit_action(self, action: str) -> None:
        if self._selected is None:
            return
        self.action_requested.emit(action, [self._selected])

    def _on_card_clicked(self, index: int) -> None:
        self._selected = None if self._selected == index else index
        self.selection_changed.emit(index)

    @property
    def selected_index(self) -> int | None:
        return self._selected

    def update_from(self, analysis, columns, banner: str, kit_lines: list[str]) -> None:
        self._banner.setText(banner)
        self._kits.setText("  ·  ".join(kit_lines))

        while self._grids.count():
            item = self._grids.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        known = set()
        for column in columns:
            grid = TeamGrid(column, selected_index=self._selected)
            grid.card_clicked.connect(self._on_card_clicked)
            grid.confirm_all.connect(
                lambda indices: self.action_requested.emit(ACTION_CONFIRM, indices)
            )
            self._grids.addWidget(grid, stretch=1)
            known.update(card.index for card in column.cards)

        if self._selected is not None and self._selected not in known:
            self._selected = None

        enabled = self._selected is not None
        for button in self._buttons:
            button.setEnabled(enabled)
        self._selection_label.setText(
            f"player #{self._selected} selected" if enabled else "select a player to correct"
        )


class StageCard(QFrame):
    """One pipeline stage: state, headline, the facts behind it, its warnings."""

    def __init__(self, step):
        super().__init__()
        self.setStyleSheet(
            f"QFrame {{ background:{SURFACE}; border:1px solid {BORDER}; }}"
            "QLabel { border:none; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(3)

        header = QHBoxLayout()
        title = QLabel(f"{step.phase} · {step.title}")
        title.setStyleSheet(f"color:{TEXT}; font-weight:600;")
        badge = QLabel(STATE_LABELS.get(step.state, step.state.upper()))
        badge.setStyleSheet(
            f"color:{STATE_COLORS.get(step.state, MUTED)}; font-family:monospace;"
            " font-size:10px;"
        )
        header.addWidget(title)
        header.addStretch()
        header.addWidget(badge)
        layout.addLayout(header)

        summary = QLabel(step.headline)
        summary.setWordWrap(True)
        summary.setStyleSheet(f"color:{MUTED};")
        layout.addWidget(summary)

        for fact in step.facts:
            label = QLabel("· " + fact)
            label.setWordWrap(True)
            label.setStyleSheet("color:#6E6E6E; font-size:11px;")
            layout.addWidget(label)

        for warning in step.warnings:
            label = QLabel("⚠ " + warning)
            label.setWordWrap(True)
            label.setStyleSheet("color:#FFB020; font-size:11px;")
            layout.addWidget(label)


class FlowPanel(QWidget):
    """Every stage of the pipeline, in order, including the ones not built."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._container = QVBoxLayout()
        self._container.setSpacing(6)
        holder = QWidget()
        holder.setLayout(self._container)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        scroll.setStyleSheet(f"border:1px solid {BORDER};")
        layout.addWidget(scroll)

    def update_from(self, steps) -> None:
        while self._container.count():
            item = self._container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for step in steps:
            self._container.addWidget(StageCard(step))
        self._container.addStretch()
