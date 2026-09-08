"""Per-stage progress while the offside pipeline runs on a confirmed frame.

Before this widget, confirming a frame gave the operator nothing to look at
for the several seconds the M2 pipeline takes to run — no busy state at all,
because nothing was calling the pipeline (`OffsideRunner` is what changed
that). A single spinner would technically fix "nothing happens", but the
whole design goal of this milestone is telling the operator what the tool
is actually doing, and a spinner tells them nothing: the seven stages take
noticeably different amounts of time (detection and pose are model inference;
team assignment reduces to arithmetic on already-known kit colours), and
whichever one is slow on a given frame is exactly the one an operator waiting
on it wants named.

So this is a checklist, not a bar: one row per pipeline stage, in the order
`offside/pipeline.py`'s `analyse()` actually runs them, filling in live as
each `StageReport` arrives from `OffsideRunner.stage_progress`. A stage that
finished DEGRADED or UNAVAILABLE is shown as such immediately — not silently
folded into "done" — because a slow-to-notice failed calibration is exactly
the kind of thing this whole project exists to surface rather than hide.
"""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from apps.desktop.ui.theme import tokens as t
from apps.desktop.ui.widgets.common import _PulsingDot, data_value, meta
from offside.pipeline import StageReport, StageState

#: Mirrors the stage order `OffsidePipeline.analyse()` actually runs in
#: (offside/pipeline.py). Pre-populating rows in this order — rather than
#: waiting for the first `stage_progress` signal to invent a row — is what
#: lets the panel show "detection done, pose next" instead of only ever
#: showing what has already finished.
PIPELINE_STAGES: tuple[tuple[str, str, str, str], ...] = (
    ("detection", "M1", "Player & ball detection", "Detection"),
    ("body_keypoints", "M2.2", "Body keypoints (feet)", "Body keypoints"),
    ("tracking", "M2.4", "Player identity tracking", "Identity"),
    ("pitch_calibration", "M2.1", "Pitch calibration", "Calibration"),
    ("team_assignment", "M2.3", "Team & goalkeeper assignment", "Teams"),
    ("offside_line", "M2.5", "Second-last defender & offside line", "Offside line"),
    ("decision_support", "M2.6", "Confidence & reasoning", "Confidence"),
)

_PENDING = "row_pending"
_RUNNING = "row_running"

_DOT_COLOR = {
    _PENDING: t.BORDER_STRONG,
    _RUNNING: t.PRIMARY,
    StageState.OK: t.SUCCESS,
    StageState.DEGRADED: t.TEXT_MUTED,
    StageState.UNAVAILABLE: t.ALERT,
    StageState.PENDING: t.BORDER_STRONG,
}

_STATE_WORD = {
    _PENDING: "waiting",
    _RUNNING: "running",
    StageState.OK: "done",
    StageState.DEGRADED: "done, degraded",
    StageState.UNAVAILABLE: "unavailable",
    StageState.PENDING: "not built",
}


class StageChecklistRow(QWidget):
    """One pipeline stage: a dot, its short label, and one line of state.
    Animates with a pulsing dot when active/running.
    """

    def __init__(self, short_label: str, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("StageChecklistRow")
        self._title = title

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(t.SPACING_UNIT)

        self._dot = _PulsingDot(self)
        layout.addWidget(self._dot)

        self._label = meta(short_label.upper())
        layout.addWidget(self._label)
        layout.addStretch(1)

        self._detail = data_value(_STATE_WORD[_PENDING])
        layout.addWidget(self._detail)
        self.setToolTip(title)
        self.set_pending()

    def set_pending(self) -> None:
        self._dot.set_color(_DOT_COLOR[_PENDING])
        self._dot.set_pulsing(False)
        self._detail.setText(_STATE_WORD[_PENDING])
        self.setProperty("running", "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.setToolTip(self._title)

    def set_running(self) -> None:
        self._dot.set_color(_DOT_COLOR[_RUNNING])
        self._dot.set_pulsing(True)
        self._detail.setText(_STATE_WORD[_RUNNING])
        self.setProperty("running", "true")
        self.style().unpolish(self)
        self.style().polish(self)

    def set_report(self, report: StageReport) -> None:
        self._dot.set_color(_DOT_COLOR[report.state])
        self._dot.set_pulsing(False)
        self._detail.setText(_STATE_WORD[report.state])
        self.setProperty("running", "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.setToolTip(f"{self._title}\n{report.summary}")


class OffsideProgressPanel(QWidget):
    """The checklist itself: one `StageChecklistRow` per pipeline stage plus live progress bar."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("OffsideProgressPanel")
        self.setVisible(False)
        self._completed_count = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._caption = meta("")
        layout.addWidget(self._caption)

        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("OffsideProgressBar")
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(3)
        self._progress_bar.setRange(0, len(PIPELINE_STAGES))
        layout.addWidget(self._progress_bar)

        self._rows: dict[str, StageChecklistRow] = {}
        for key, _phase, title, short_label in PIPELINE_STAGES:
            row = StageChecklistRow(short_label, title)
            layout.addWidget(row)
            self._rows[key] = row

        self._error = QLabel("")
        self._error.setWordWrap(True)
        self._error.setProperty("role", "meta")
        self._error.setVisible(False)
        layout.addWidget(self._error)

    # -- driven by OffsideRunner's signals -----------------------------

    def start(self, frame_id: int) -> None:
        """A run began on `frame_id`: reset every row and show the panel."""
        self.setVisible(True)
        self._error.setVisible(False)
        self._completed_count = 0
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._caption.setText(f"CHECKING FRAME {frame_id} FOR OFFSIDE")
        for row in self._rows.values():
            row.set_pending()
        first_key = PIPELINE_STAGES[0][0]
        self._rows[first_key].set_running()

    def report_stage(self, report: StageReport) -> None:
        row = self._rows.get(report.key)
        if row is None:
            return
        row.set_report(report)
        self._completed_count += 1
        self._progress_bar.setValue(self._completed_count)

        index = next(
            (i for i, (key, *_rest) in enumerate(PIPELINE_STAGES) if key == report.key),
            None,
        )
        if index is not None and index + 1 < len(PIPELINE_STAGES):
            next_key = PIPELINE_STAGES[index + 1][0]
            self._rows[next_key].set_running()

    def finish(self) -> None:
        self._caption.setText("OFFSIDE CHECK COMPLETE")
        self._progress_bar.setValue(len(PIPELINE_STAGES))

    def fail(self, message: str) -> None:
        self._caption.setText("OFFSIDE CHECK FAILED")
        self._error.setText(message)
        self._error.setVisible(True)
        for row in self._rows.values():
            if row._detail.text() in (_STATE_WORD[_PENDING], _STATE_WORD[_RUNNING]):
                row.set_pending()
