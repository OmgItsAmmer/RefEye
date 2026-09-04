"""Typed configuration schema, validated from YAML.

Nothing tuning-related (hotkeys, model choice, buffer sizes, refinement
weights) should ever be hard-coded elsewhere — it belongs here and in YAML.

Unknown keys are rejected so a typo fails loudly at startup rather than
silently falling back to a default nobody notices.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class StrictModel(BaseModel):
    model_config = {"extra": "forbid"}


#: Spotters that work from live CV features and need no model checkpoint.
CHECKPOINTLESS_PROVIDERS = frozenset({"kinematic"})


# --------------------------------------------------------------------------
# application
# --------------------------------------------------------------------------
class ApplicationConfig(StrictModel):
    name: str
    environment: Literal["development", "demo", "production"] = "development"


# --------------------------------------------------------------------------
# video
# --------------------------------------------------------------------------
class LocalFileVideoConfig(StrictModel):
    path: str
    loop: bool = True


class AnalysisResolutionConfig(StrictModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class VideoConfig(StrictModel):
    input_type: Literal["local_file", "capture_card", "rtsp", "srt", "screen", "ndi"]
    #: Camera 1 — the one real analysis source (architecture.md's single-
    #: stream analysis pipeline). Live CV/tracking/triggered analysis all run
    #: against this feed only.
    local_file: LocalFileVideoConfig
    #: Cameras 2..4 on Live Grid — decode-and-display only, each its own
    #: independent file/loop, never analyzed or buffered for `get_recent_clip`.
    #: A tile for a camera with no entry here just stays "No signal".
    preview_cameras: list[LocalFileVideoConfig] = Field(default_factory=list)
    analysis_resolution: AnalysisResolutionConfig
    # Cap on preview repaints per second. Decoding still runs at source rate;
    # this only bounds how often the UI thread is asked to repaint.
    preview_max_fps: int = Field(default=60, gt=0, le=240)


# --------------------------------------------------------------------------
# buffers
# --------------------------------------------------------------------------
class BufferConfig(StrictModel):
    recent_window_seconds: int = Field(gt=0)
    encoded_buffer_seconds: int = Field(gt=0)
    decoded_buffer_seconds: int = Field(gt=0)
    max_decoded_frames: int = Field(gt=0)
    max_encoded_packets: int = Field(default=3000, gt=0)
    # The bound that actually protects RAM. A frame count is meaningless
    # without the resolution behind it: 20s at 960x540 BGR is ~780 MB.
    # Whichever limit binds first wins.
    max_decoded_megabytes: int = Field(default=512, gt=0)

    @field_validator("encoded_buffer_seconds")
    @classmethod
    def _encoded_covers_analysis_window(cls, v: int, info) -> int:
        recent = info.data.get("recent_window_seconds")
        if recent is not None and v < recent:
            raise ValueError(
                "encoded_buffer_seconds must be >= recent_window_seconds, "
                "otherwise a triggered analysis window cannot be reconstructed"
            )
        return v


# --------------------------------------------------------------------------
# shortcuts
# --------------------------------------------------------------------------
class ShortcutsConfig(StrictModel):
    analyze: str
    previous_frame: str
    next_frame: str
    previous_candidate: str
    next_candidate: str
    jump_to_best: str
    confirm_frame: str
    # Live Grid screen's bottom nav bar — jump straight to a camera's (or the
    # best candidate's) analyzer view without touching the mouse.
    select_camera_1: str
    select_camera_2: str
    select_camera_3: str
    select_camera_4: str
    select_best: str


# --------------------------------------------------------------------------
# ai
# --------------------------------------------------------------------------
class ProviderCheckpointConfig(StrictModel):
    checkpoint: str


class ActionSpotterConfig(StrictModel):
    provider: str
    #: Used when `provider` cannot load (missing/unlicensed checkpoint), so the
    #: app degrades instead of refusing to analyse (architecture.md section 49).
    fallback_provider: str = "kinematic"

    # Kinematic baseline tuning. Thresholds are provisional and must be
    # validated on client footage before any accuracy claim is made.
    min_direction_change_deg: float = Field(default=25.0, ge=0.0, le=180.0)
    min_speed_ratio: float = Field(default=1.35, gt=1.0)
    proximity_radius_px: float = Field(default=90.0, gt=0.0)
    min_score: float = Field(default=0.15, ge=0.0, le=1.0)


class DetectorConfig(StrictModel):
    provider: str
    checkpoint: str
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    #: The ball is small, blurred and often occluded, so it gets a looser bar
    #: than people (architecture.md section 14). None = half the main threshold.
    ball_confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    imgsz: int = Field(default=640, gt=0)
    #: Run background detection on every Nth frame. Continuous full-rate
    #: detection buys little and starves the triggered path (section 13).
    live_frame_stride: int = Field(default=3, gt=0)


class TrackerConfig(StrictModel):
    provider: str
    high_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    low_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    match_iou: float = Field(default=0.25, ge=0.0, le=1.0)
    max_misses: int = Field(default=15, ge=0)
    ball_max_gap_frames: int = Field(default=6, ge=0)


class RefinementWeights(StrictModel):
    model: float = Field(ge=0.0)
    proximity: float = Field(ge=0.0)
    velocity: float = Field(ge=0.0)
    direction: float = Field(ge=0.0)
    motion: float = Field(ge=0.0)
    track_consistency: float = Field(ge=0.0)


class ContactRefinementConfig(StrictModel):
    enabled: bool
    window_before_frames: int = Field(ge=0)
    window_after_frames: int = Field(ge=0)
    weights: RefinementWeights
    #: Distance in analysis-resolution pixels within which a player counts as
    #: being in contact range of the ball.
    proximity_radius_px: float = Field(default=90.0, gt=0.0)


class CandidateRankingConfig(StrictModel):
    max_candidates_returned: int = Field(gt=0)
    dedup_temporal_distance_frames: int = Field(ge=0)
    #: Ranking blend. Kept in config so tuning never requires a code change
    #: (architecture.md section 58, rule 11).
    weight_final_score: float = Field(default=0.6, ge=0.0)
    weight_ball_visibility: float = Field(default=0.15, ge=0.0)
    weight_track_quality: float = Field(default=0.1, ge=0.0)
    weight_temporal_recency: float = Field(default=0.15, ge=0.0)
    #: Frames frozen either side of each candidate for review navigation.
    #: Bounds review memory: candidates x (2N+1) JPEG frames.
    review_frames_each_side: int = Field(default=15, ge=0, le=120)


class FeatureCacheConfig(StrictModel):
    max_frames: int = Field(default=900, gt=0)


class AIConfig(StrictModel):
    action_spotter: ActionSpotterConfig
    providers: dict[str, ProviderCheckpointConfig]
    detector: DetectorConfig
    tracker: TrackerConfig
    contact_refinement: ContactRefinementConfig
    candidate_ranking: CandidateRankingConfig
    feature_cache: FeatureCacheConfig = Field(default_factory=FeatureCacheConfig)

    @field_validator("providers")
    @classmethod
    def _selected_provider_is_configured(cls, v: dict, info) -> dict:
        spotter = info.data.get("action_spotter")
        if spotter is None:
            return v
        # The kinematic baseline derives everything from live CV features and
        # has no checkpoint, so it needs no ai.providers entry.
        if spotter.provider in CHECKPOINTLESS_PROVIDERS:
            return v
        if spotter.provider not in v:
            raise ValueError(
                f"action_spotter.provider '{spotter.provider}' has no entry under ai.providers"
            )
        return v


# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------
class InferenceSchedulerConfig(StrictModel):
    max_queue_size: int = Field(gt=0)


class RuntimeConfig(StrictModel):
    device: Literal["cuda", "cpu"]
    inference_scheduler: InferenceSchedulerConfig


# --------------------------------------------------------------------------
# logging / persistence
# --------------------------------------------------------------------------
class LoggingConfig(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    directory: str
    json_file: bool
    console: bool
    rotation_max_bytes: int = Field(default=10_485_760, gt=0)
    rotation_backup_count: int = Field(default=5, ge=0)


class PersistenceConfig(StrictModel):
    sqlite_path: str
    exports_directory: str


# --------------------------------------------------------------------------
# root
# --------------------------------------------------------------------------
class AppSettings(StrictModel):
    application: ApplicationConfig
    video: VideoConfig
    buffer: BufferConfig
    shortcuts: ShortcutsConfig
    ai: AIConfig
    runtime: RuntimeConfig
    logging: LoggingConfig
    persistence: PersistenceConfig
