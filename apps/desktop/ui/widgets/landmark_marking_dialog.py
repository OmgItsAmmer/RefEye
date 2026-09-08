"""Manual pitch-landmark marking, in the shipped app (M2.7, extends the plan).

Until this file, marking a pitch landmark by hand only existed in the
internal Pipeline Inspector — the automatic paths (directional line
detection, and now the auto-landmark detector in
`offside/pitch_calibration/auto_landmarks.py`) were the only routes to a
calibrated pitch an operator could reach from the shipped app. When both of
those come up short on a real frame, there was, until now, nowhere in the
product for a person to just point at the penalty spot and say so.

This is that place. The interaction is the same one the Inspector already
proved works: pick a landmark by name, click where it is in the frame,
repeat. Nothing about the underlying calibration math is new — this dialog
is a thin, product-facing front end onto the exact same
`PitchCalibrator.calibrate_manual` the automatic paths and the Inspector's
own marking flow both go through, so a hand-marked calibration here is not a
second, less-tested code path.

## Why a modal dialog, not inline in the review rail

The review rail is already a narrow column (see the width-budget tests in
`test_offside_review_screen.py`) — a clickable video frame at any usable
size does not fit next to the checklist and the verdict. A dialog gets the
frame real screen space to click on accurately, and the "confirm when done"
shape matches how marking actually works: several clicks building toward one
result, not something to react to click-by-click in the background.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from apps.desktop.ui.theme import tokens as t
from apps.desktop.ui.widgets.common import AnimatedButton, data_value, divider, label, meta
from offside.field_geometry.pitch import PitchModel
from offside.pitch_calibration.homography import PointCorrespondence
from tools.pipeline_debugger.presenter import (
    landmark_choices,
    landmark_description,
    landmark_label,
)

#: Below this many marks, calibration cannot reach METRIC — same constant
#: `PitchCalibrator.calibrate_manual` itself enforces; repeated here only so
#: the dialog can tell the operator how many more clicks are needed.
_MIN_MARKS = 4


class _ClickableFrame(QWidget):
    """The video frame, fitted to whatever size it's given, reporting clicks
    in the *original image's* pixel coordinates — not the widget's."""

    clicked = Signal(float, float)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 180)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._pixmap: QPixmap | None = None
        self._image_size = (1, 1)
        #: landmark name -> image xy, drawn as markers so already-placed
        #: points are visible while placing the rest.
        self._marks: dict[str, tuple[float, float]] = {}
        self._highlight: str | None = None

    def set_image(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        self._image_size = (width, height)
        contiguous = np.ascontiguousarray(image)
        qimage = QImage(
            contiguous.data, width, height, contiguous.strides[0], QImage.Format.Format_BGR888
        )
        self._pixmap = QPixmap.fromImage(qimage.copy())
        self.update()

    def set_marks(self, marks: dict[str, tuple[float, float]], highlight: str | None) -> None:
        self._marks = marks
        self._highlight = highlight
        self.update()

    def _fitted_rect(self) -> QRectF:
        pw, ph = self._pixmap.width(), self._pixmap.height()
        scale = min(self.width() / pw, self.height() / ph)
        w, h = pw * scale, ph * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _to_image_xy(self, pos) -> tuple[float, float] | None:
        if self._pixmap is None:
            return None
        target = self._fitted_rect()
        if not target.contains(pos):
            return None
        scale_x = self._image_size[0] / target.width()
        scale_y = self._image_size[1] / target.height()
        return (
            (pos.x() - target.x()) * scale_x,
            (pos.y() - target.y()) * scale_y,
        )

    def _to_widget_xy(self, image_xy: tuple[float, float]) -> tuple[float, float] | None:
        if self._pixmap is None:
            return None
        target = self._fitted_rect()
        scale_x = target.width() / self._image_size[0]
        scale_y = target.height() / self._image_size[1]
        return (target.x() + image_xy[0] * scale_x, target.y() + image_xy[1] * scale_y)

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt naming
        xy = self._to_image_xy(event.position())
        if xy is not None:
            self.clicked.emit(*xy)

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(t.BG_BASE))
        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor(t.TEXT_MUTED))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No frame")
            return

        target = self._fitted_rect()
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for name, xy in self._marks.items():
            point = self._to_widget_xy(xy)
            if point is None:
                continue
            colour = QColor(t.PRIMARY if name != self._highlight else t.SUCCESS)
            painter.setPen(colour)
            painter.drawLine(
                int(point[0] - 8), int(point[1]), int(point[0] + 8), int(point[1])
            )
            painter.drawLine(
                int(point[0]), int(point[1] - 8), int(point[0]), int(point[1] + 8)
            )
            painter.drawEllipse(point[0] - 5, point[1] - 5, 10, 10)


class LandmarkMarkingDialog(QDialog):
    """Pick a landmark, click where it is, repeat — the manual fallback.

    Returns via `exec()`: `QDialog.DialogCode.Accepted` when the operator
    marked at least 4 points and pressed Done, `Rejected` if they cancelled.
    Call `correspondences()` afterward either way — Accepted with fewer than
    4 cannot happen (the Done button is disabled until then), but a caller
    that only checks the dialog code rather than the count is still safe.
    """

    def __init__(
        self,
        image: np.ndarray,
        pitch: PitchModel,
        initial_marks: dict[str, tuple[float, float]] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Mark pitch landmarks")
        self.setModal(True)
        self.resize(1100, 680)

        self._pitch = pitch
        self._marks: dict[str, tuple[float, float]] = dict(initial_marks or {})

        root = QHBoxLayout(self)
        root.setSpacing(t.SPACING_UNIT * 2)

        self._frame = _ClickableFrame()
        self._frame.set_image(image)
        self._frame.clicked.connect(self._on_frame_clicked)
        root.addWidget(self._frame, stretch=3)

        side = QVBoxLayout()
        side.setSpacing(t.SPACING_UNIT)
        root.addLayout(side, stretch=2)

        side.addWidget(label("Landmark to mark"))
        self._picker = QComboBox()
        for choice_label, name in landmark_choices(pitch):
            self._picker.addItem(choice_label, name)
        self._picker.currentIndexChanged.connect(self._on_landmark_changed)
        side.addWidget(self._picker)

        self._hint = meta("")
        self._hint.setWordWrap(True)
        side.addWidget(self._hint)

        instruction = QLabel("Find this point in the frame on the left and click it.")
        instruction.setWordWrap(True)
        side.addWidget(instruction)

        side.addWidget(divider())
        side.addWidget(label("Marked so far"))
        self._marked_list = QListWidget()
        self._marked_list.setMaximumHeight(160)
        side.addWidget(self._marked_list)

        self._status = data_value("")
        side.addWidget(self._status)
        side.addStretch(1)

        buttons = QHBoxLayout()
        cancel_button = AnimatedButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(cancel_button)
        self._done_button = AnimatedButton("Done")
        self._done_button.setProperty("role", "primary")
        self._done_button.clicked.connect(self.accept)
        buttons.addWidget(self._done_button)
        side.addLayout(buttons)

        self._refresh()

    # -- interaction ----------------------------------------------------

    def _selected_landmark(self) -> str:
        return self._picker.currentData() or self._picker.currentText()

    def _on_landmark_changed(self, _index: int) -> None:
        name = self._selected_landmark()
        self._hint.setText(landmark_description(name))
        self._refresh()

    def _on_frame_clicked(self, x: float, y: float) -> None:
        landmark = self._selected_landmark()
        self._marks[landmark] = (x, y)
        self._advance_to_next_unmarked()
        self._refresh()

    def _advance_to_next_unmarked(self) -> None:
        for index in range(self._picker.count()):
            name = self._picker.itemData(index)
            if name not in self._marks:
                self._picker.setCurrentIndex(index)
                return

    # -- rendering --------------------------------------------------------

    def _refresh(self) -> None:
        self._frame.set_marks(self._marks, self._selected_landmark())

        self._marked_list.clear()
        for name, xy in self._marks.items():
            item = QListWidgetItem(f"{landmark_label(name)}  ({xy[0]:.0f}, {xy[1]:.0f})")
            self._marked_list.addItem(item)

        count = len(self._marks)
        if count < _MIN_MARKS:
            self._status.setText(f"{count} of {_MIN_MARKS} needed")
        else:
            self._status.setText(f"{count} marked — the pitch can be mapped")
        self._done_button.setEnabled(count >= _MIN_MARKS)

    # -- result -------------------------------------------------------------

    def correspondences(self) -> list[PointCorrespondence]:
        return [
            PointCorrespondence(
                image_xy=xy, pitch_xy=self._pitch.landmark(name), landmark=name
            )
            for name, xy in self._marks.items()
        ]
