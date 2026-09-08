"""Candidate review: video column + side column.

Architecture.md section 31. The operator sees the strongest candidate first,
can move between alternatives, can step frame by frame, and confirms the
final frame themselves — the human stays in the loop (section 4.6).

Frame navigation and candidate navigation are deliberately separate commands
with separate controls, because they answer different questions:
  * candidate nav — "which moment in the play is this?" Steps through the
    detected events (passes/shots/crosses) in the order they happened, not
    by AI confidence, so a play with four passes steps 1st -> 2nd -> 3rd ->
    4th pass and freezes the video on each one in turn.
  * frame nav     — "is this the exact frame of contact?" Fine-tunes within
    the currently selected event, one frame at a time.

This widget builds two independent containers — `video_widget` (frame view,
metadata, and a Spotify-style transport bar: skip-candidate / frame-step /
best / frame-step / skip-candidate) and `side_widget` (alternatives list,
retry/confirm) — rather than laying itself out, so AnalyzerScreen can place
the large video column on the left and the side column in the right rail.
Neither container is added to any layout here; whichever screen embeds them
determines position and size.

All styling comes from the stylesheet; this file sets object names and
dynamic properties only (theme.md).
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QVBoxLayout,
    QWidget,
)

from ai.action_spotting.common.actions import display_name
from analysis.results.review_session import ReviewSession
from apps.desktop.ui.theme import tokens as t
from apps.desktop.ui.theme.icons import skip_icon
from apps.desktop.ui.widgets.common import (
    AnimatedButton,
    BadgeVariant,
    StatusBadge,
    data_value,
    divider,
    meta,
)
from apps.desktop.ui.widgets.video_panel import VideoSurface
from offside.offside_line.rendering import draw_offside_overlay
from vision.features.feature_cache import FeatureCache


class CandidateRow(QWidget):
    """One selectable candidate in the alternatives pill list."""

    clicked = Signal(int)

    def __init__(self, index: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("CandidateRow")
        self._index = index
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # Rank pip: a single-hue ramp, never traffic-light coding (theme.md).
        self._pip = QLabel()
        self._pip.setObjectName("RankPip")
        self._pip.setFixedSize(3, 14)
        layout.addWidget(self._pip)

        self._action = QLabel()
        layout.addWidget(self._action)

        # A confidence score is a technical value, not prose — Space Mono
        self._score = data_value()
        layout.addWidget(self._score)

    def set_candidate(self, candidate, position: int, selected: bool) -> None:
        self._action.setText(f"{position}. {display_name(candidate.action_type)}")
        self._score.setText(f"{candidate.final_score:.2f}")

        self._pip.setStyleSheet(f"background-color: {t.rank_color(position - 1)};")
        self.setProperty("selected", "true" if selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt naming
        option = QStyleOption()
        option.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(
            QStyle.PrimitiveElement.PE_Widget, option, painter, self
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt naming
        self.clicked.emit(self._index)
        super().mousePressEvent(event)


class CandidateReviewPanel(QWidget):
    """Owns review state; exposes `video_widget` and `side_widget` for the
    caller to place. Never shown/laid out itself."""

    confirmed = Signal(int, int)      # candidate_index, frame_id
    retry_requested = Signal()
    candidate_selected = Signal(int)

    def __init__(self, feature_cache: FeatureCache | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._session: ReviewSession | None = None
        self._rows: list[CandidateRow] = []
        self._feature_cache = feature_cache
        # M2.7: the offside call for the frame currently shown, drawn onto it.
        # None means no offside analysis has been run — the M1 behaviour, which
        # this must not disturb.
        self._offside_decision = None
        self._offside_explanation = None

        self.video_widget = self._build_video_column()
        self.side_widget = self._build_side_column()

        self._prev_candidate.clicked.connect(self.previous_candidate)
        self._next_candidate.clicked.connect(self.next_candidate)
        self._prev_frame.clicked.connect(self.previous_frame)
        self._next_frame.clicked.connect(self.next_frame)
        self._best_button.clicked.connect(self.jump_to_best)
        self._retry_button.clicked.connect(self.retry_requested.emit)
        self._confirm_button.clicked.connect(self._on_confirm)

        self._set_enabled(False)

    # -- construction: video column & integrated control deck ----------

    def _build_video_column(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(t.SPACING_UNIT)

        self._surface = VideoSurface()
        self._surface.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._surface.setMinimumHeight(360)
        self._surface.set_placeholder("No candidate selected")
        layout.addWidget(self._surface, stretch=1)

        layout.addWidget(self._build_control_deck())
        return widget

    def _build_control_deck(self) -> QWidget:
        deck = QFrame()
        deck.setObjectName("ReviewControlDeck")
        deck_layout = QHBoxLayout(deck)
        deck_layout.setContentsMargins(14, 10, 14, 10)
        deck_layout.setSpacing(16)

        # -- Left section: metadata & evidence --
        meta_col = QVBoxLayout()
        meta_col.setContentsMargins(0, 0, 0, 0)
        meta_col.setSpacing(3)

        header = QHBoxLayout()
        header.setSpacing(8)
        self._action_badge = StatusBadge("—", BadgeVariant.ACCENT)
        header.addWidget(self._action_badge)
        self._frame_label = data_value("")
        header.addWidget(self._frame_label)
        header.addStretch(1)
        meta_col.addLayout(header)

        self._evidence = meta("")
        self._evidence.setWordWrap(True)
        meta_col.addWidget(self._evidence)
        deck_layout.addLayout(meta_col, stretch=3)

        # -- Center section: transport controls --
        transport_box = self._build_transport_bar()
        deck_layout.addWidget(transport_box, stretch=3)

        # -- Right section: candidate alternatives & confirmation actions --
        actions_col = QVBoxLayout()
        actions_col.setContentsMargins(0, 0, 0, 0)
        actions_col.setSpacing(6)

        alt_header = QHBoxLayout()
        alt_header.setSpacing(6)
        self._candidate_counter = meta("Candidates:")
        alt_header.addWidget(self._candidate_counter)

        self._rows_container = QHBoxLayout()
        self._rows_container.setSpacing(6)
        alt_header.addLayout(self._rows_container)
        alt_header.addStretch(1)
        actions_col.addLayout(alt_header)

        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(8)
        self._confirm_button = AnimatedButton("Confirm frame")
        self._confirm_button.setProperty("role", "primary")
        buttons_row.addWidget(self._confirm_button)

        self._retry_button = AnimatedButton("Retry")
        self._retry_button.setProperty("role", "danger")
        buttons_row.addWidget(self._retry_button)
        actions_col.addLayout(buttons_row)

        deck_layout.addLayout(actions_col, stretch=4)
        return deck

    def _build_transport_bar(self) -> QWidget:
        """Modern transport row: skip-candidate / step-frame / best /
        step-frame / skip-candidate."""
        bar = QFrame()
        bar.setObjectName("TransportBar")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(8, 4, 8, 4)
        bar_layout.setSpacing(6)

        self._prev_candidate = self._transport_button(
            "", "Previous event in the play", icon=skip_icon(forward=False)
        )
        self._prev_frame = self._transport_button("‹", "Previous frame")
        self._best_button = self._transport_button("BEST", "Jump to best candidate", best=True)
        self._next_frame = self._transport_button("›", "Next frame")
        self._next_candidate = self._transport_button(
            "", "Next event in the play", icon=skip_icon(forward=True)
        )

        for button in (
            self._prev_candidate,
            self._prev_frame,
            self._best_button,
            self._next_frame,
            self._next_candidate,
        ):
            bar_layout.addWidget(button)

        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(bar)
        return container

    @staticmethod
    def _transport_button(
        text: str, tooltip: str, best: bool = False, icon: QIcon | None = None
    ) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("TransportButton")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setToolTip(tooltip)
        if best:
            button.setProperty("best", "true")
        if icon is not None:
            button.setIcon(icon)
            button.setIconSize(QSize(18, 18))
        return button

    # -- construction: side column (kept for compatibility) -------------

    def _build_side_column(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        return widget

    # -- session ------------------------------------------------------------

    def set_session(self, session: ReviewSession | None) -> None:
        self._session = session
        self._rebuild_rows()
        if session is None or session.candidate_count == 0:
            self._surface.clear()
            self._surface.set_placeholder("No candidates found in the recent play")
            self._set_enabled(False)
            self._action_badge.set_status("None", BadgeVariant.MUTED)
            self._candidate_counter.setText("")
            self._frame_label.setText("")
            self._evidence.setText("")
            return

        self._set_enabled(True)
        self._refresh()

    # -- navigation commands (also bound to hotkeys) ------------------------

    def next_candidate(self) -> None:
        if self._session:
            self._session.next_candidate()
            self._refresh()
            self.candidate_selected.emit(self._session.candidate_index)

    def previous_candidate(self) -> None:
        if self._session:
            self._session.previous_candidate()
            self._refresh()
            self.candidate_selected.emit(self._session.candidate_index)

    def select_candidate(self, index: int) -> None:
        if self._session:
            self._session.select_candidate(index)
            self._refresh()
            self.candidate_selected.emit(self._session.candidate_index)

    def next_frame(self) -> None:
        if self._session:
            self._session.next_frame()
            self._refresh()

    def previous_frame(self) -> None:
        if self._session:
            self._session.previous_frame()
            self._refresh()

    def jump_to_best(self) -> None:
        if self._session:
            self._session.jump_to_best()
            self._refresh()
            self.candidate_selected.emit(0)

    def confirm(self) -> None:
        self._on_confirm()

    @property
    def has_candidates(self) -> bool:
        return self._session is not None and self._session.candidate_count > 0

    # -- rendering ----------------------------------------------------------

    def _refresh(self) -> None:
        session = self._session
        if session is None:
            return

        candidate = session.current_candidate
        if candidate is None:
            return

        current = session.current_frame()
        if current is not None:
            frame_id, image = current
            self._surface.set_frame(self._with_offside_line(image))
            self._apply_overlay(frame_id)
        else:
            frame_id = candidate.refined_frame_id
            self._surface.clear()
            self._surface.set_placeholder("Frame is no longer available")

        self._action_badge.set_status(
            display_name(candidate.action_type), BadgeVariant.ACCENT
        )
        self._candidate_counter.setText(
            f"Candidate {session.candidate_index + 1} of {session.candidate_count}"
        )

        offset = session.frame_offset_from_candidate
        position, total = session.frame_position
        offset_text = "AI frame" if offset == 0 else f"{offset:+d} from AI frame"
        self._frame_label.setText(f"Frame {frame_id} · {offset_text} · {position}/{total}")

        self._evidence.setText(self._describe(candidate))
        self._update_rows()

    # -- offside (M2.7) -----------------------------------------------------

    def set_offside(self, decision, explanation=None) -> None:
        """The offside call for the frame on screen, or None to clear it.

        The panel keeps both: the geometry supplies the line and the measured
        points, and M2.6's explanation decides what verdict — and therefore
        what colour — the line is drawn in. A call the chain could not carry
        must not appear on the frame in confident colours simply because the
        geometry reached it.
        """
        self._offside_decision = decision
        self._offside_explanation = explanation
        self._refresh()

    def _with_offside_line(self, image):
        """The frame with the offside overlay drawn on a copy of it.

        A copy because the frame comes from the shared review buffer: drawing
        into it would burn the line onto the frame everywhere else it is shown,
        and leave a stale line behind after an override changes the verdict.
        """
        if self._offside_decision is None:
            return image
        painted = image.copy()
        draw_offside_overlay(painted, self._offside_decision, self._offside_explanation)
        return painted

    def _apply_overlay(self, frame_id: int) -> None:
        """Unlike Live Grid (which shows the freshest cached entry), review
        frames are frozen and known exactly — an exact `frame_id` lookup is
        possible and more accurate than "latest"."""
        if self._feature_cache is None:
            return
        cached = self._feature_cache.get(frame_id)
        if cached is None:
            self._surface.clear_overlay()
            return
        self._surface.set_overlay(
            cached.tracks, cached.features.ball_center, cached.features.ball_confidence
        )

    @staticmethod
    def _describe(candidate) -> str:
        parts = [f"Confidence {candidate.final_score:.2f}"]

        shift = candidate.evidence.get("frame_shift")
        if candidate.evidence.get("refined") and shift:
            parts.append(f"refined {shift:+d} frames from the model's estimate")
        elif not candidate.evidence.get("refined"):
            parts.append("model frame kept (ball not visible)")

        if not candidate.evidence.get("ball_visible", True):
            parts.append("ball position estimated")

        alternatives = candidate.evidence.get("alternative_actions")
        if alternatives:
            names = ", ".join(display_name(a) for a in alternatives)
            parts.append(f"also read as {names}")

        merged = candidate.evidence.get("merged_candidate_ids")
        if merged:
            parts.append(f"{len(merged)} duplicate detection(s) merged")

        return " · ".join(parts)

    def _rebuild_rows(self) -> None:
        while self._rows_container.count():
            item = self._rows_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows = []

        if self._session is None:
            return

        # Rows list chronologically (pass 1, pass 2, pass 3, ...) rather than
        # by AI confidence — same order the transport bar steps through, so
        # what the operator sees in the side list matches what Up/Down does.
        for candidate_index in self._session.chronological_order:
            row = CandidateRow(candidate_index)
            row.clicked.connect(self.select_candidate)
            self._rows_container.addWidget(row)
            self._rows.append(row)

        self._update_rows()

    def _update_rows(self) -> None:
        if self._session is None:
            return
        for position, candidate_index in enumerate(self._session.chronological_order, start=1):
            row = self._rows[position - 1]
            row.set_candidate(
                self._session.candidates[candidate_index],
                position=position,
                selected=candidate_index == self._session.candidate_index,
            )

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (
            self._prev_candidate,
            self._next_candidate,
            self._prev_frame,
            self._next_frame,
            self._best_button,
            self._confirm_button,
        ):
            widget.setEnabled(enabled)

    def _on_confirm(self) -> None:
        if self._session is None:
            return
        current = self._session.current_frame()
        if current is None:
            return
        self.confirmed.emit(self._session.candidate_index, current[0])
