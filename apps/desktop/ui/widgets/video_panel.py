"""Live video surface: renders decoded frames without blocking the UI thread.

The ingest thread never touches this widget directly. It emits a signal; Qt
delivers it as a queued connection on the UI thread, which stores the frame
and schedules a repaint. Painting is a single scaled blit — no decoding, no
resizing beyond the aspect-fit, no allocation of new frames.

Optionally also paints a detection overlay: outlines around what the app is
actually detecting (architecture.md section 14 — player/goalkeeper/ball/
referee), sourced from the live CV pipeline's `FeatureCache` rather than
recomputed here. `Detection.bbox_xyxy`/`TrackObservation.bbox_xyxy` already
share the displayed pixmap's coordinate space 1:1 (both derive from the same
analysis-resolution frame — video/decoding/decoder.py resizes once, before
either display or detection see the frame), so the only transform needed is
the existing pixmap-to-widget aspect-fit scale/offset.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from apps.desktop.ui.theme import tokens as t
from core.domain.models import TrackObservation

_BALL_MARKER_RADIUS = 9.0
_TAG_FONT_SIZE = 8


class VideoSurface(QWidget):
    """Aspect-preserving display for BGR frames arriving as numpy arrays."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setAutoFillBackground(False)
        self._pixmap: QPixmap | None = None
        self._placeholder = "No video"

        self._overlay_enabled = True
        self._tracks: list[TrackObservation] = []
        self._ball_center: tuple[float, float] | None = None
        self._ball_confidence: float | None = None

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        if self._pixmap is None:
            self.update()

    def clear(self) -> None:
        self._pixmap = None
        self.clear_overlay()

    # -- detection overlay ---------------------------------------------------

    def set_overlay_enabled(self, enabled: bool) -> None:
        self._overlay_enabled = enabled
        self.update()

    def set_overlay(
        self,
        tracks: list[TrackObservation],
        ball_center: tuple[float, float] | None,
        ball_confidence: float | None,
    ) -> None:
        """Latest detected/tracked objects for the currently displayed frame
        (or the closest one available — the background pipeline runs a few
        frames behind by design, architecture.md section 4.5)."""
        self._tracks = tracks
        self._ball_center = ball_center
        self._ball_confidence = ball_confidence
        self.update()

    def clear_overlay(self) -> None:
        self._tracks = []
        self._ball_center = None
        self._ball_confidence = None
        self.update()

    def set_frame(self, image: np.ndarray) -> None:
        """Accept a BGR uint8 frame and schedule a repaint.

        The incoming array is owned by the rolling buffer and may be evicted
        at any moment, so the QImage is copied into a QPixmap here rather than
        wrapping the buffer's memory and painting it later.
        """
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            return

        height, width, _ = image.shape
        contiguous = np.ascontiguousarray(image)

        qimage = QImage(
            contiguous.data,
            width,
            height,
            contiguous.strides[0],
            QImage.Format.Format_BGR888,
        )
        self._pixmap = QPixmap.fromImage(qimage.copy())
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Letterbox area stays base-dark so the frame reads as inset content.
        painter.fillRect(self.rect(), QColor(t.BG_BASE))

        if self._pixmap is None:
            painter.setPen(QColor(t.TEXT_MUTED))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                self._placeholder,
            )
            return

        target = self._fitted_rect()
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        if self._overlay_enabled:
            self._paint_overlay(painter, target)

    def _paint_overlay(self, painter: QPainter, target: QRectF) -> None:
        assert self._pixmap is not None
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw == 0 or ph == 0:
            return
        scale_x = target.width() / pw
        scale_y = target.height() / ph

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        box_pen = QPen(QColor(t.TEXT_MAIN))
        box_pen.setWidthF(1.6)
        for track in self._tracks:
            x1, y1, x2, y2 = track.bbox_xyxy
            rect = QRectF(
                target.x() + x1 * scale_x,
                target.y() + y1 * scale_y,
                (x2 - x1) * scale_x,
                (y2 - y1) * scale_y,
            )
            painter.setPen(box_pen)
            painter.drawRect(rect)
            self._draw_tag(painter, rect.topLeft(), track.object_type)

        if self._ball_center is not None:
            cx, cy = self._ball_center
            center = QPointF(target.x() + cx * scale_x, target.y() + cy * scale_y)

            ball_pen = QPen(QColor(t.PRIMARY))
            ball_pen.setWidthF(1.6)
            # Interpolated (no live detection this frame) reads as dashed —
            # it's a prediction, not an observation (architecture.md §25).
            if self._ball_confidence is None:
                ball_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(ball_pen)
            painter.drawEllipse(center, _BALL_MARKER_RADIUS, _BALL_MARKER_RADIUS)
            self._draw_tag(
                painter,
                QPointF(center.x() - _BALL_MARKER_RADIUS, center.y() - _BALL_MARKER_RADIUS),
                "ball",
            )

    @staticmethod
    def _draw_tag(painter: QPainter, anchor: QPointF, text: str) -> None:
        """A small glassy label chip, matching theme.md's on-video-overlay
        style (translucent surface + Space Mono technical text)."""
        painter.save()
        font = QFont(t.FONT_FAMILY_DATA, _TAG_FONT_SIZE)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        label = text.upper()

        pad_x, pad_y = 4.0, 2.0
        chip = QRectF(
            anchor.x(),
            anchor.y() - metrics.height() - pad_y * 2,
            metrics.horizontalAdvance(label) + pad_x * 2,
            metrics.height() + pad_y * 2,
        )
        background = QColor(t.BG_SURFACE)
        background.setAlpha(204)
        painter.setPen(QPen(QColor(t.BORDER)))
        painter.setBrush(background)
        painter.drawRect(chip)

        painter.setPen(QColor(t.TEXT_MAIN))
        painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, label)
        painter.restore()

    def _fitted_rect(self) -> QRectF:
        assert self._pixmap is not None
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw == 0 or ph == 0:
            return QRectF(self.rect())

        scale = min(self.width() / pw, self.height() / ph)
        width, height = pw * scale, ph * scale
        return QRectF(
            (self.width() - width) / 2,
            (self.height() - height) / 2,
            width,
            height,
        )
