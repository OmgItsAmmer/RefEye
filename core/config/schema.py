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
    local_file: LocalFileVideoConfig
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


# --------------------------------------------------------------------------
# ai
# --------------------------------------------------------------------------
class ProviderCheckpointConfig(StrictModel):
    checkpoint: str


class ActionSpotterConfig(StrictModel):
    provider: str


class DetectorConfig(StrictModel):
    provider: str
    checkpoint: str
    confidence_threshold: float = Field(ge=0.0, le=1.0)


class TrackerConfig(StrictModel):
    provider: str


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


class CandidateRankingConfig(StrictModel):
    max_candidates_returned: int = Field(gt=0)
    dedup_temporal_distance_frames: int = Field(ge=0)


class AIConfig(StrictModel):
    action_spotter: ActionSpotterConfig
    providers: dict[str, ProviderCheckpointConfig]
    detector: DetectorConfig
    tracker: TrackerConfig
    contact_refinement: ContactRefinementConfig
    candidate_ranking: CandidateRankingConfig

    @field_validator("providers")
    @classmethod
    def _selected_provider_is_configured(cls, v: dict, info) -> dict:
        spotter = info.data.get("action_spotter")
        if spotter is not None and spotter.provider not in v:
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
