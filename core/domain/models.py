"""Typed domain models shared across the application.

These are the only shapes that should cross module boundaries (video -> vision
-> ai -> analysis -> ui). Do not pass raw dicts between layers; convert into
one of these at the boundary instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class FramePacket:
    frame_id: int
    pts: int | None
    timestamp_ms: int
    capture_timestamp_ms: int
    width: int
    height: int
    source_id: str
    image: np.ndarray | None = None
    is_keyframe: bool = False


@dataclass
class Detection:
    frame_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    source_model: str


@dataclass
class TrackObservation:
    track_id: str
    frame_id: int
    timestamp_ms: int
    object_type: str
    bbox_xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float]
    confidence: float


@dataclass
class FrameFeatures:
    frame_id: int
    timestamp_ms: int
    ball_center: tuple[float, float] | None
    ball_confidence: float | None
    nearest_player_track_id: str | None
    ball_velocity: tuple[float, float] | None
    ball_speed: float | None
    scene_cut: bool
    model_features: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ActionCandidate:
    candidate_id: str
    action_type: str
    anchor_frame_id: int
    anchor_timestamp_ms: int
    model_score: float
    source_model: str
    player_track_id: str | None = None
    team_id: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RefinedCandidate:
    candidate_id: str
    action_type: str
    original_frame_id: int
    refined_frame_id: int
    final_score: float
    model_score: float
    contact_score: float
    trajectory_score: float
    proximity_score: float
    temporal_score: float
    evidence: dict = field(default_factory=dict)


@dataclass
class AnalysisRequest:
    request_id: str
    triggered_at_ms: int
    trigger_source: str
    requested_window_ms: int
    request_type: str = "general"


@dataclass
class AnalysisResult:
    request_id: str
    candidates: list[RefinedCandidate]
    selected_candidate_index: int
    status: str
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


@dataclass
class SceneState:
    is_gameplay: bool
    is_replay: bool
    is_closeup: bool
    has_major_graphic: bool
    confidence: float


@dataclass
class Incident:
    incident_id: str
    match_session_id: str
    trigger_timestamp_ms: int
    selected_frame_id: int
    selected_timestamp_ms: int
    action_type: str | None
    source_model: str
    confidence: float | None
    created_at: datetime
    metadata: dict = field(default_factory=dict)
