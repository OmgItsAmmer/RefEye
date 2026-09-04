"""Small looping preview of the last `recent_window_seconds` of buffered
video — shown top-right on the Analyzer screen so the operator can see what
the AI is about to look at (or just looked at) without leaving the review.

Pulls its frames from VideoService.get_recent_clip (the same rolling
decoded-frame buffer a triggered analysis reads from), not from a live
frame_ready subscription — this widget replays a fixed recent window on its
own clock rather than tracking the live edge.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer

from apps.desktop.ui.widgets.video_panel import VideoSurface
from core.errors.exceptions import AnalysisWindowUnavailableError
from video.frame_access.video_service import VideoService

_REFRESH_MS = 42  # ~24fps playback of the buffered window


class RecentClipPreview(VideoSurface):
    def __init__(self, video_service: VideoService, window_seconds: int, parent=None):
        super().__init__(parent)
        self._video = video_service
        self._window_ms = window_seconds * 1000
        self._frames: list = []
        self._index = 0
        self.set_placeholder("No recent play buffered yet")

        self._timer = QTimer(self)
        self._timer.setInterval(_REFRESH_MS)
        self._timer.timeout.connect(self._tick)

    def showEvent(self, event) -> None:  # noqa: N802 — Qt naming
        super().showEvent(event)
        self._reload()
        self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 — Qt naming
        self._timer.stop()
        super().hideEvent(event)

    def refresh(self) -> None:
        """Re-pull the current recent-clip window — call when the operator
        navigates to the Analyzer screen so the loop reflects the play that
        was actually just live, not a stale one from a previous visit."""
        self._reload()

    def set_video_service(self, video_service: VideoService) -> None:
        """Switch which camera's buffer this preview follows.

        Without this the preview stays pinned to whichever VideoService it
        was constructed with (camera 1), so picking camera 2/3/4 on Live
        Grid would analyze the right footage but keep looping camera 1's
        feed here — a different video than the one just analyzed."""
        if video_service is self._video:
            return
        self._video = video_service
        self._reload()

    def _reload(self) -> None:
        latest = self._video.latest_frame()
        if latest is None:
            self._frames = []
            return
        try:
            clip = self._video.get_recent_clip(latest.timestamp_ms, self._window_ms)
        except AnalysisWindowUnavailableError:
            self._frames = []
            return
        self._frames = clip.frames
        self._index = 0

    def _tick(self) -> None:
        if not self._frames:
            return
        frame = self._frames[self._index]
        if frame.image is not None:
            self.set_frame(frame.image)
        self._index = (self._index + 1) % len(self._frames)
