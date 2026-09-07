"""Analyzer screen: 70% left = the candidate review workspace (frame view,
metadata, Spotify-style transport bar). 30% right = either a camera picker
or a looping preview of the last N seconds of buffered play, plus
alternatives/confirm/retry.

Idle/busy/review is a 3-page stack inside the left panel — busy shows the
card-swap loader (an AnalysisRequest is active), idle shows the "press <key>
to analyse" prompt, review shows CandidateReviewPanel's video_widget. The
side rail's alternatives/confirm/retry (CandidateReviewPanel's side_widget)
stays visible in every state — its buttons are simply disabled when there is
no session, same as before.

The top-right slot is a 2-page stack: a camera picker, and the recent-clip
preview. Arriving here straight from the sidebar means no camera was ever
picked — there is nothing to preview yet — so that's the picker's job: a
single "Watch camera" button that sends the operator back to Live Grid to
actually pick one. Arriving via a Live Grid camera/Best button (which
already selected one and kicked off analysis) shows the preview instead.
See `on_shown`.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from apps.desktop.ui.theme import tokens as t
from apps.desktop.ui.widgets.candidate_review import CandidateReviewPanel
from apps.desktop.ui.widgets.common import (
    AnimatedButton,
    BadgeVariant,
    Panel,
    StatusBadge,
    data_value,
    label,
)
from apps.desktop.ui.widgets.loaders import CardSwapLoader
from apps.desktop.ui.widgets.offside_progress import OffsideProgressPanel
from apps.desktop.ui.widgets.offside_review import OffsideReviewPanel
from apps.desktop.ui.widgets.pitch_map import PitchMapPanel
from apps.desktop.ui.widgets.recent_clip_preview import RecentClipPreview
from video.frame_access.video_service import VideoService
from vision.features.feature_cache import FeatureCache

PAGE_IDLE = 0
PAGE_BUSY = 1
PAGE_REVIEW = 2

_TOP_RIGHT_PICKER = 0
_TOP_RIGHT_PREVIEW = 1


class AnalyzerScreen(QWidget):
    # Emitted when the operator clicks a camera button on the "no camera
    # picked yet" panel — main_window.py routes this back to Live Grid,
    # since that's the only screen that can actually select one.
    camera_pick_requested = Signal()

    def __init__(
        self,
        video_service: VideoService,
        recent_window_seconds: int,
        analyze_shortcut_label: str,
        feature_cache: FeatureCache | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._feature_cache = feature_cache
        self._offside_decision = None

        root = QHBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        root.addWidget(self._build_analysis_panel(analyze_shortcut_label), stretch=7)
        root.addWidget(self._build_right_column(video_service, recent_window_seconds), stretch=3)

    # -- left: analysis workspace --------------------------------------

    def _build_right_column(
        self, video_service: VideoService, recent_window_seconds: int
    ) -> QScrollArea:
        """The camera/preview slot, alternatives, and the offside panel,
        inside a scroll area rather than three fixed-stretch panes.

        Three growing panels stacked with fixed VBox stretch factors and no
        escape hatch is what "small screen" actually breaks: on a window at
        or near the app's own minimum size, three panels compressed to fit
        whatever space is left is indistinguishable from labels silently
        losing their text and buttons losing their padding. A `QScrollArea`
        gives each panel its natural size and lets the *column* scroll
        instead of squeezing its contents — the operator loses nothing, they
        just occasionally scroll for it.
        """
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(self._build_top_right_stack(video_service, recent_window_seconds))
        layout.addWidget(self._build_alternatives_panel())
        # M2.7: the offside call sits beneath the alternatives, so the verdict
        # and its caveats are on screen at the same time as the frame it is
        # about — an operator should never have to change view to find out
        # what the tool was unsure about.
        layout.addWidget(self._build_offside_panel())
        # The flattened pitch, with whatever marks calibrated it — the same
        # view the Pipeline Inspector shows, moved into the product so the
        # operator does not need a separate debug tool to see what the
        # calibration actually produced.
        layout.addWidget(self._build_pitch_panel())
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("RightColumnScroll")
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # AsNeeded, not AlwaysOff: if some future row still doesn't fit, a
        # reachable scrollbar is a smaller failure than the content silently
        # being clipped with no way to see the rest of it.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        return scroll

    def _build_analysis_panel(self, analyze_shortcut_label: str) -> Panel:
        panel = Panel("Analyzer")
        self.analysis_badge = StatusBadge("Idle", BadgeVariant.MUTED)
        panel.add_header_widget(self.analysis_badge)

        self.analysis_stack = QStackedWidget()

        # --- idle ---
        idle_page = QWidget()
        idle_layout = QVBoxLayout(idle_page)
        idle_layout.setContentsMargins(0, 0, 0, 0)
        idle_layout.addStretch(1)
        self.analysis_message = QLabel(f"Press {analyze_shortcut_label} to analyse the recent play.")
        self.analysis_message.setWordWrap(True)
        self.analysis_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analysis_message.setProperty("role", "meta")
        idle_layout.addWidget(self.analysis_message)
        idle_layout.addSpacing(t.SPACING_UNIT * 2)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.analyze_button = AnimatedButton(f"Analyse recent play  ({analyze_shortcut_label})")
        self.analyze_button.setProperty("role", "primary")
        button_row.addWidget(self.analyze_button)
        button_row.addStretch(1)
        idle_layout.addLayout(button_row)
        idle_layout.addStretch(1)

        # --- busy (card swap) ---
        busy_page = QWidget()
        busy_layout = QVBoxLayout(busy_page)
        busy_layout.setContentsMargins(0, 0, 0, 0)
        busy_layout.addStretch(1)
        self.analysis_loader = CardSwapLoader()
        loader_row = QHBoxLayout()
        loader_row.addStretch(1)
        loader_row.addWidget(self.analysis_loader)
        loader_row.addStretch(1)
        busy_layout.addLayout(loader_row)
        self.analysis_stage = label("Analysing recent play…")
        self.analysis_stage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        busy_layout.addWidget(self.analysis_stage)
        busy_layout.addStretch(1)

        # --- review (frame view + transport bar) ---
        self.review = CandidateReviewPanel(feature_cache=self._feature_cache)

        self.analysis_stack.addWidget(idle_page)               # PAGE_IDLE
        self.analysis_stack.addWidget(busy_page)               # PAGE_BUSY
        self.analysis_stack.addWidget(self.review.video_widget)  # PAGE_REVIEW

        panel.body().addWidget(self.analysis_stack, stretch=1)

        self.request_label = data_value("No analysis run yet")
        panel.body().addWidget(self.request_label)
        return panel

    # -- right: camera picker / recent clip + alternatives/actions --------

    def _build_top_right_stack(self, video_service: VideoService, window_seconds: int) -> QStackedWidget:
        self._top_right_stack = QStackedWidget()
        self._top_right_stack.addWidget(self._build_camera_picker_panel())  # _TOP_RIGHT_PICKER
        self._top_right_stack.addWidget(
            self._build_recent_clip_panel(video_service, window_seconds)
        )  # _TOP_RIGHT_PREVIEW
        return self._top_right_stack

    def _build_camera_picker_panel(self) -> Panel:
        panel = Panel("Pick a camera")

        info = QLabel("No camera has been selected yet. Watch a camera to review its recent play.")
        info.setWordWrap(True)
        info.setProperty("role", "meta")
        panel.body().addWidget(info)

        panel.body().addStretch(1)
        watch_button = AnimatedButton("Watch camera")
        watch_button.setProperty("role", "primary")
        watch_button.clicked.connect(self.camera_pick_requested.emit)
        panel.body().addWidget(watch_button)
        panel.body().addStretch(1)
        return panel

    def _build_recent_clip_panel(self, video_service: VideoService, window_seconds: int) -> Panel:
        panel = Panel(f"Last {window_seconds}s")
        self.recent_clip = RecentClipPreview(video_service, window_seconds)
        # RecentClipPreview subclasses VideoSurface, which sets a 320x180
        # minimum sized for the *main* video panel — inherited here even
        # though this is a corner thumbnail. That inherited 320 (plus panel
        # padding) was the real width the review rail could never shrink
        # below, regardless of how narrow the window was: this call is what
        # actually fixes the "hidden text on small screens" complaint, more
        # than any scroll area does. 90px keeps a legible thumbnail without
        # setting the column's floor.
        self.recent_clip.setMinimumSize(90, 160)
        self.recent_clip.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        panel.body().addWidget(self.recent_clip, stretch=1)
        return panel

    def _build_alternatives_panel(self) -> Panel:
        panel = Panel("Alternatives")
        panel.body().addWidget(self.review.side_widget, stretch=1)
        return panel

    def _build_offside_panel(self) -> Panel:
        panel = Panel("Offside")
        self.offside_progress = OffsideProgressPanel()
        panel.body().addWidget(self.offside_progress)
        self.offside = OffsideReviewPanel()
        # An override changes the verdict, so it changes the colour of the line
        # drawn on the frame too — the panel and the picture must never
        # disagree about what the call currently is.
        self.offside.override_changed.connect(self._on_offside_override)
        panel.body().addWidget(self.offside, stretch=1)
        return panel

    def _on_offside_override(self, _verdict) -> None:
        self.review.set_offside(self._offside_decision, self.offside.explanation)

    def _build_pitch_panel(self) -> Panel:
        panel = Panel("Pitch map")
        self.pitch_map = PitchMapPanel()
        panel.body().addWidget(self.pitch_map)
        return panel

    # -- offside decisions (M2.7) ----------------------------------------

    def set_offside_decision(self, decision, explanation=None) -> None:
        """Show one frame's offside call: the panel, and the line on the frame.

        Passing None for both is how the screen goes back to plain M1 review —
        an offside line left over from a previous candidate would be worse than
        no line, since it would look like a call about the frame on screen.
        """
        self._offside_decision = decision
        self.offside_progress.finish()
        self.offside.set_explanation(explanation)
        self.review.set_offside(decision, self.offside.explanation)

    def set_pitch_analysis(self, analysis, pitch, marked_landmarks: dict | None = None) -> None:
        """The top-down map for the same frame `set_offside_decision` is
        showing. A separate call, not folded into `set_offside_decision`,
        because the map needs `pitch` (`OffsidePipeline.pitch`) and the raw
        `FrameAnalysis`, neither of which the offside panel itself needs."""
        self.pitch_map.set_analysis(analysis, pitch, marked_landmarks)

    def show_offside_started(self, frame_id: int) -> None:
        self.offside_progress.start(frame_id)
        # A fresh check is starting; the previous frame's verdict is now
        # stale and must not linger on screen while the new one is computed.
        self.offside.clear()
        self.pitch_map.clear()

    def show_offside_stage(self, report) -> None:
        self.offside_progress.report_stage(report)

    def show_offside_failed(self, frame_id: int, message: str) -> None:
        self.offside_progress.fail(message)

    def clear_offside_decision(self) -> None:
        self._offside_decision = None
        self.offside.clear()
        self.review.set_offside(None, None)
        self.pitch_map.clear()

    # -- called when the screen becomes visible --------------------------

    def on_shown(self, camera_context: bool = False) -> None:
        """`camera_context` is True only when navigation came from a Live
        Grid camera/Best button — the one case where there's an actual
        recent play to preview. A direct sidebar click gets the picker."""
        if camera_context:
            self._top_right_stack.setCurrentIndex(_TOP_RIGHT_PREVIEW)
            self.recent_clip.refresh()
        else:
            self._top_right_stack.setCurrentIndex(_TOP_RIGHT_PICKER)
