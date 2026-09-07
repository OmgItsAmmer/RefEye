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

from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from apps.desktop.ui.theme import tokens as t
from apps.desktop.ui.widgets.common import data_value, meta, simple_dot
from offside.pipeline import StageReport, StageState

#: Mirrors the stage order `OffsidePipeline.analyse()` actually runs in
#: (offside/pipeline.py). Pre-populating rows in this order — rather than
#: waiting for the first `stage_progress` signal to invent a row — is what
#: lets the panel show "detection done, pose next" instead of only ever
#: showing what has already finished.
#:
#: The third field is a short, at-a-glance label — not `StageReport.title`.
#: Uppercased and letter-tracked (theme.md), the full title of the longest
#: stage ("Second-last defender & offside line") sets a ~560px minimum width
#: on the row it's in, which is wider than this panel's own column at any
#: normal window size, not just a small one — this is what actually broke
#: the layout, caught only once the column could no longer silently squeeze
#: to fit. The full title survives as the row's tooltip.
PIPELINE_STAGES: tuple[tuple[str, str, str, str], ...] = (
    ("detection", "M1", "Player & ball detection", "Detection"),
    ("body_keypoints", "M2.2", "Body keypoints (feet)", "Body keypoints"),
    ("tracking", "M2.4", "Player identity tracking", "Identity"),
    ("pitch_calibration", "M2.1", "Pitch calibration", "Calibration"),
    ("team_assignment", "M2.3", "Team & goalkeeper assignment", "Teams"),
    ("offside_line", "M2.5", "Second-last defender & offside line", "Offside line"),
    ("decision_support", "M2.6", "Confidence & reasoning", "Confidence"),
)

#: Dot colour per stage state, plus the two states a row can be in before its
#: `StageReport` has arrived at all. Deliberately *not* the strings "pending"
#: or "running" — `StageState` is a `str` Enum, so `StageState.PENDING`
#: hashes and compares equal to the plain string "pending", and the two
#: sentinels below silently collided with `StageState.PENDING` as dict keys
#: until this was renamed (a real bug caught by the tests: a stage the
#: panel had never heard from read as "not built" instead of "waiting").
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

    `title` is the full `StageReport.title` and is never displayed — only
    used to seed the tooltip before a report has arrived, so hovering an
    untouched "waiting" row still says which stage it is.
    """

    def __init__(self, short_label: str, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("StageChecklistRow")
        self._title = title

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(t.SPACING_UNIT)

        self._dot = simple_dot(_DOT_COLOR[_PENDING])
        layout.addWidget(self._dot)

        self._label = meta(short_label.upper())
        layout.addWidget(self._label)
        layout.addStretch(1)

        self._detail = data_value(_STATE_WORD[_PENDING])
        layout.addWidget(self._detail)
        self.setToolTip(title)

    def set_pending(self) -> None:
        self._dot.setStyleSheet(f"background-color: {_DOT_COLOR[_PENDING]}; border-radius: 4px;")
        self._detail.setText(_STATE_WORD[_PENDING])
        self.setToolTip(self._title)

    def set_running(self) -> None:
        self._dot.setStyleSheet(f"background-color: {_DOT_COLOR[_RUNNING]}; border-radius: 4px;")
        self._detail.setText(_STATE_WORD[_RUNNING])

    def set_report(self, report: StageReport) -> None:
        self._dot.setStyleSheet(
            f"background-color: {_DOT_COLOR[report.state]}; border-radius: 4px;"
        )
        self._detail.setText(_STATE_WORD[report.state])
        # The full title plus the summary is the one line worth surfacing
        # without opening the inspector — everything else (details) stays
        # out of a checklist meant to be read at a glance.
        self.setToolTip(f"{self._title}\n{report.summary}")


class OffsideProgressPanel(QWidget):
    """The checklist itself: one `StageChecklistRow` per pipeline stage.

    Hidden by default (`setVisible(False)`) — a checklist with every row
    grey and pending, sitting under a review panel with nothing to review
    yet, would look like a tool stuck at start-up rather than a tool that
    simply hasn't been asked to do anything.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("OffsideProgressPanel")
        self.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._caption = meta("")
        layout.addWidget(self._caption)

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
        self._caption.setText(f"CHECKING FRAME {frame_id} FOR OFFSIDE")
        for row in self._rows.values():
            row.set_pending()
        # The first stage starts immediately — mark it running rather than
        # leaving every row identically grey the instant the panel appears.
        first_key = PIPELINE_STAGES[0][0]
        self._rows[first_key].set_running()

    def report_stage(self, report: StageReport) -> None:
        row = self._rows.get(report.key)
        if row is None:
            # A stage this panel doesn't know about (future milestone) — do
            # not crash the checklist over it, just skip the row it has no
            # slot for.
            return
        row.set_report(report)

        index = next(
            (i for i, (key, *_rest) in enumerate(PIPELINE_STAGES) if key == report.key),
            None,
        )
        if index is not None and index + 1 < len(PIPELINE_STAGES):
            next_key = PIPELINE_STAGES[index + 1][0]
            self._rows[next_key].set_running()

    def finish(self) -> None:
        self._caption.setText("OFFSIDE CHECK COMPLETE")

    def fail(self, message: str) -> None:
        self._caption.setText("OFFSIDE CHECK FAILED")
        self._error.setText(message)
        self._error.setVisible(True)
        for row in self._rows.values():
            if row._detail.text() in (_STATE_WORD[_PENDING], _STATE_WORD[_RUNNING]):
                row.set_pending()
