"""The top-down pitch map in the review screen (M2.7).

Everything this milestone has argued for — that the operator should see what
the tool is doing, not just be handed a verdict — was true of the Pipeline
Inspector's "Pitch" tab and untrue of the shipped app, which had no way to
show the flattened pitch at all. This widget is that view, moved into the
product: the same `render_top_down` the inspector draws with (from
`offside/pitch_calibration/rendering.py`), so the two can never disagree
about what the calibration actually produced.

Two things this widget exists to make honest, matching the map's job in the
inspector:

**A blank pitch is not a placeholder, it is the answer.** Before metric
calibration there is nothing to project players onto, and the map says so in
its own caption rather than silently drawing an empty diagram. The shipped
app has no manual-marking flow yet (that is presently inspector-only), so in
practice every map here is either "automatic, directional only" or "not
calibrated" — this widget does not pretend otherwise.

**The marks used to calibrate this frame are drawn on the map, not just
implied by a confidence number.** `render_top_down` already draws whichever
landmarks were supplied; this widget passes through whatever the pipeline
actually used, so "the marks used for calibration" is something the operator
can see, not a sentence they have to take on trust.

## Why this paints itself rather than using a QLabel + setPixmap

That was the first version, and it broke the review rail's width the same
way the checklist and the camera thumbnail already had: `QLabel.sizeHint()`
follows the *pixmap's* pixel size once one is set, so the very first render
— before layout had ever assigned this widget a real width — locked in the
map's native 760px render width as the label's preferred size, and the whole
column paid for it. `VideoSurface` (the main video panel) already solved
exactly this with a custom `paintEvent` that fits the pixmap into whatever
rect it is actually given, so this widget does the same rather than
reintroducing the bug a third time.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from apps.desktop.ui.theme import tokens as t
from apps.desktop.ui.widgets.common import AnimatedButton, meta
from offside.field_geometry.pitch import PitchModel
from offside.pitch_calibration.rendering import render_top_down

#: Rendered at a fixed pitch-aspect size, then fitted into whatever width the
#: review rail actually has — matching the aspect ratio keeps the pitch
#: looking like a pitch rather than a stretched rectangle at any column width.
_RENDER_SIZE = (760, 520)
_ASPECT = _RENDER_SIZE[1] / _RENDER_SIZE[0]

#: A rail-thumbnail minimum, not `VideoSurface`'s 320x180 (sized for the main
#: video) — reusing that one is what broke the "Last 20s" preview's column
#: width earlier in this same milestone.
_MIN_WIDTH = 140


class _PitchCanvas(QWidget):
    """Paints a BGR numpy canvas fitted to its own size — never wider than
    whatever the layout actually gives it, regardless of the source
    resolution."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(_MIN_WIDTH, int(_MIN_WIDTH * _ASPECT))
        self._pixmap: QPixmap | None = None

    def set_canvas(self, canvas: np.ndarray | None) -> None:
        if canvas is None:
            self._pixmap = None
            self.update()
            return
        height, width, _ = canvas.shape
        contiguous = np.ascontiguousarray(canvas)
        qimage = QImage(
            contiguous.data,
            width,
            height,
            contiguous.strides[0],
            QImage.Format.Format_BGR888,
        )
        self._pixmap = QPixmap.fromImage(qimage.copy())
        self.update()

    def heightForWidth(self, width: int) -> int:  # noqa: N802 — Qt naming
        return int(width * _ASPECT)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 — Qt naming
        return True

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(t.BG_BASE))

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor(t.TEXT_MUTED))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No frame analysed yet")
            return

        pw, ph = self._pixmap.width(), self._pixmap.height()
        scale = min(self.width() / pw, self.height() / ph)
        w, h = pw * scale, ph * scale
        target = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))


class PitchMapPanel(QWidget):
    """The flattened pitch: the calibration marks, and the players on it."""

    #: The operator wants to mark landmarks by hand — see
    #: `apps.desktop.ui.widgets.landmark_marking_dialog`. Emitted rather than
    #: opening the dialog directly: this widget has no access to the
    #: confirmed frame's raw image or the pipeline's `mark_landmark`/re-run
    #: hooks, both of which live at the screen/window level.
    mark_requested = Signal()

    def __init__(self, pitch: PitchModel | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("PitchMapPanel")
        self._pitch = pitch or PitchModel()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(t.SPACING_UNIT // 2)

        self._canvas = _PitchCanvas()
        layout.addWidget(self._canvas)

        self._caption = meta("")
        self._caption.setWordWrap(True)
        layout.addWidget(self._caption)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        # Always available, not just when calibration is weak — an operator
        # who disagrees with a confident-looking automatic read is entitled
        # to mark the pitch themselves regardless of what the number says.
        self._mark_button = AnimatedButton("Mark landmarks")
        self._mark_button.setToolTip(
            "Mark the pitch by hand — overrides automatic calibration for this frame"
        )
        self._mark_button.clicked.connect(self.mark_requested.emit)
        self._mark_button.setEnabled(False)
        action_row.addWidget(self._mark_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self._last_is_metric: bool | None = None
        self.clear()

    def set_analysis(self, analysis, pitch: PitchModel | None = None, marked_landmarks: dict | None = None) -> None:
        """Render the map for one frame's analysis.

        `pitch` is the `PitchModel` the pipeline that produced `analysis` was
        configured with (`OffsidePipeline.pitch`) — the map needs it to know
        the real-world dimensions and landmark positions to draw, and nothing
        on `FrameAnalysis` itself carries that.
        """
        if pitch is not None:
            self._pitch = pitch
        canvas = render_top_down(
            self._pitch,
            analysis.calibration,
            analysis.poses,
            analysis.teams,
            size=_RENDER_SIZE,
            marked_landmarks=marked_landmarks or {},
            # The click-to-mark *map* interaction stays inspector-only (that
            # UI lets you pick a landmark by pointing at this diagram); the
            # product's manual-marking flow is the separate dialog opened by
            # `mark_requested`, which clicks the video frame instead — see
            # `apps.desktop.ui.widgets.landmark_marking_dialog`.
            interactive=False,
        )
        self._canvas.set_canvas(canvas)
        calibration = analysis.calibration
        self._last_is_metric = bool(calibration is not None and calibration.is_metric)
        self._caption.setText(self._describe(analysis, marked_landmarks or {}))
        self._mark_button.setEnabled(True)

    def clear(self) -> None:
        canvas = render_top_down(
            self._pitch,
            None,
            [],
            None,
            size=_RENDER_SIZE,
            interactive=False,
        )
        self._canvas.set_canvas(canvas)
        self._caption.setText("No frame analysed yet.")
        self._mark_button.setEnabled(False)
        self._last_is_metric = None

    @property
    def needs_manual_marking(self) -> bool:
        """Whether the last frame shown fell short of metric calibration —
        the "Mark landmarks" action is always available; this is what a
        caller uses to decide whether to make the ask louder than a button
        sitting quietly in a rail (see `MainWindow._on_offside_completed`)."""
        return self._last_is_metric is False

    @staticmethod
    def _describe(analysis, marked_landmarks: dict) -> str:
        calibration = analysis.calibration
        if calibration is None or not calibration.can_draw_offside_line:
            return "Not calibrated on this frame — mark landmarks below for a precise line."

        source = "operator-marked" if marked_landmarks else "automatic"
        if calibration.is_metric:
            return f"Metric — {source}, positions in metres."
        return (
            f"Directional — {source}, no metric marks; ordering only, "
            "no distances."
        )
