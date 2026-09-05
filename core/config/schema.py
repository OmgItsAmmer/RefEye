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
# offside (Milestone 2)
# --------------------------------------------------------------------------
class BodyKeypointsConfig(StrictModel):
    """Pose estimation for offside body points (M2.2).

    Defaults are chosen so an M1-era config file still loads; they are
    starting points measured on the reference clip, not validated accuracy
    settings, and M2.8 is where they get tuned against more footage.
    """

    #: When false the pose model is never loaded and offside body points are
    #: simply unavailable — M1's analysis loop is unaffected either way.
    enabled: bool = True
    provider: str = "yolo_pose"
    checkpoint: str = "./models/pose/yolo11n-pose.pt"

    #: Per-keypoint bar. Below this a body point is treated as not seen at
    #: all, rather than quietly steering a line it cannot support.
    keypoint_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Bar for the pose model's own person box inside a crop.
    person_confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    #: How much a pose's box must overlap the detection it was cropped for
    #: before its skeleton is accepted — the neighbouring-player guard.
    match_iou: float = Field(default=0.3, ge=0.0, le=1.0)

    #: Crops are resized to this before inference. Broadcast players are
    #: ~90px tall, so upscaling each crop is what makes keypoints possible.
    crop_imgsz: int = Field(default=256, gt=0)
    #: Padding around each player box, as a fraction of its size. Wider
    #: horizontally: what escapes a box is a stretched leg, not headroom.
    crop_padding_x: float = Field(default=0.25, ge=0.0, le=1.0)
    crop_padding_y: float = Field(default=0.15, ge=0.0, le=1.0)
    #: Players shorter than this in pixels are not posed at all; a skeleton
    #: on a smudge looks like data while being noise.
    min_box_height_px: float = Field(default=24.0, gt=0.0)

    #: Confidence multipliers for the two fallback rungs of the ground-point
    #: ladder (see offside/body_keypoints/ground_point.py). Both are < 1 so a
    #: guessed foot position can never outrank a measured one.
    knee_projection_penalty: float = Field(default=0.55, ge=0.0, le=1.0)
    bbox_fallback_penalty: float = Field(default=0.3, ge=0.0, le=1.0)

    max_players_per_frame: int = Field(default=30, gt=0)
    batch_size: int = Field(default=32, gt=0)


class PitchLineDetectionConfig(StrictModel):
    """Classical detection of the painted markings (M2.1).

    Every default here was measured against real broadcast footage, not
    guessed — see offside/pitch_calibration/line_detection.py for what each
    filter removes and what happens without it.
    """

    #: Grass hue window (OpenCV HSV, hue 0-180). Wide enough for floodlit
    #: night grass through to daylight, narrow enough to exclude the crowd.
    grass_hue_min: int = Field(default=30, ge=0, le=180)
    grass_hue_max: int = Field(default=95, ge=0, le=180)
    grass_min_saturation: int = Field(default=40, ge=0, le=255)

    #: Brightness floor for paint. A saturation-based "white" test was tried
    #: and discards most of the line: broadcast compression bleeds grass
    #: colour into the paint.
    line_min_brightness: int = Field(default=150, ge=0, le=255)
    #: Top-hat structuring size: keeps bright things thinner than this and
    #: rejects boards, sleeves and glare.
    max_line_width_px: int = Field(default=15, gt=2)
    tophat_threshold: int = Field(default=20, ge=0, le=255)
    #: Player boxes are blanked before line finding — a white shirt is a thin
    #: bright object on green, which is the definition a line detector uses.
    exclude_box_margin: float = Field(default=0.15, ge=0.0, le=1.0)

    min_line_length_px: int = Field(default=50, gt=0)
    max_line_gap_px: int = Field(default=25, ge=0)
    hough_threshold: int = Field(default=40, gt=0)
    merge_angle_deg: float = Field(default=4.0, gt=0.0)
    merge_distance_px: float = Field(default=14.0, gt=0.0)
    max_lines: int = Field(default=40, gt=0)


class PitchCalibrationConfig(StrictModel):
    """Mapping the camera view onto the pitch (M2.1)."""

    enabled: bool = True

    #: Pitch size in metres. The Laws permit a range and grounds differ;
    #: 105x68 is the FIFA/UEFA standard. Everything else (penalty area, goal
    #: area, centre circle) is fixed by the Laws and derived in code.
    pitch_length_m: float = Field(default=105.0, gt=0.0)
    pitch_width_m: float = Field(default=68.0, gt=0.0)

    #: A landmark further than this from where the other points imply it
    #: should be is treated as a mismarked click.
    max_reprojection_error_m: float = Field(default=2.0, gt=0.0)
    #: Lines must agree within this angle to count as converging on the same
    #: vanishing point. Judged by angle, not pixels — a vanishing point is
    #: often thousands of pixels off-screen.
    inlier_angle_deg: float = Field(default=2.5, gt=0.0)
    min_family_support: int = Field(default=2, ge=2)
    #: Below this many detected markings, automatic calibration reports
    #: nothing rather than fitting to noise.
    min_lines_for_auto: int = Field(default=4, ge=2)

    line_detection: PitchLineDetectionConfig = Field(
        default_factory=PitchLineDetectionConfig
    )


class TeamAssignmentConfig(StrictModel):
    """Grouping players by kit, and finding the goalkeepers (M2.3).

    Nothing here names a colour. Kit colours are discovered from the footage
    at runtime — a configured "home team is red" would be worthless on the
    next match (M2_Plan section 4). What is tunable is how much evidence is
    demanded before an assignment is claimed as confident.

    Colour distances are in CIELAB units (approximately dE76), where ~2.3 is
    "a trained eye can just tell them apart" and ~50 is "obviously different
    colours".
    """

    enabled: bool = True
    provider: str = "jersey_color"

    # -- where the shirt is sampled from ------------------------------------
    #: Bar for a shoulder/hip keypoint before the torso quad is trusted. Lower
    #: than the offside-measurement bar: a roughly-placed shoulder still
    #: samples shirt, whereas a roughly-placed ankle moves an offside line.
    torso_keypoint_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    #: Shrink the torso quad toward its centre; its border is where the shirt
    #: meets sleeve, neck, shorts and background.
    torso_shrink: float = Field(default=0.7, gt=0.0, le=1.0)
    #: Fallback band inside the player box, used when the torso keypoints are
    #: missing. A player who cannot be posed still needs a team, or the
    #: defender ranking loses them.
    fallback_top_fraction: float = Field(default=0.18, ge=0.0, le=1.0)
    fallback_bottom_fraction: float = Field(default=0.45, ge=0.0, le=1.0)
    fallback_width_fraction: float = Field(default=0.5, gt=0.0, le=1.0)

    #: Bare arms and necks look the same on both teams, so they are removed
    #: along with grass. Dark pixels are deliberately kept — excluding them
    #: would make black and navy kits unmeasurable.
    exclude_skin: bool = True
    min_sample_pixels: int = Field(default=20, gt=0)
    #: Below this surviving fraction of the sampled patch, the colour is
    #: reported with reduced confidence rather than as a clean measurement.
    min_kept_fraction: float = Field(default=0.35, gt=0.0, le=1.0)
    min_sample_confidence: float = Field(default=0.25, ge=0.0, le=1.0)

    # -- fitting the two kits ------------------------------------------------
    min_players_for_clustering: int = Field(default=6, ge=2)
    #: Player count at which the fit is considered well-evidenced; fewer
    #: caps the stage's confidence proportionally.
    confident_player_count: int = Field(default=10, gt=0)
    #: Outlier cut = median residual + this many MADs, floored at
    #: `min_outlier_distance`. Derived from the footage, because how tight a
    #: kit measures depends on the broadcast, not on anything knowable here.
    outlier_mad_scale: float = Field(default=3.0, gt=0.0)
    min_outlier_distance: float = Field(default=18.0, gt=0.0)
    #: A group this small is not a team — it is a goalkeeper, an official, or
    #: a badly measured shirt. Distance alone never catches them (a group of
    #: one sits exactly on its own centre), and while one survives it holds a
    #: team slot hostage and forces the two real kits to share the other.
    min_cluster_fraction: float = Field(default=0.2, gt=0.0, le=0.5)
    min_cluster_size: int = Field(default=3, ge=1)
    #: Removing outliers can expose more of them, so the fit is repeated.
    max_trim_rounds: int = Field(default=3, ge=0)
    #: Separation between the two kits at which colour alone is trustworthy.
    good_separation: float = Field(default=25.0, gt=0.0)
    #: ...and how many times the within-kit spread that separation must beat.
    min_separation_ratio: float = Field(default=2.0, gt=0.0)
    max_iterations: int = Field(default=25, gt=0)
    #: How much nearer a player must be to one kit than the other before the
    #: assignment counts as certain. Small margins are what "these two kits
    #: look alike" actually looks like in the data.
    assignment_margin: float = Field(default=6.0, gt=0.0)

    # -- goalkeepers ---------------------------------------------------------
    #: A goalkeeper is an odd kit that also stands alone behind everyone. With
    #: metric calibration the gap is in metres; without it, a fraction of the
    #: spread of all players along the goal-to-goal axis.
    goalkeeper_min_gap_m: float = Field(default=5.0, gt=0.0)
    goalkeeper_min_gap_fraction: float = Field(default=0.12, gt=0.0, le=1.0)
    #: How many players deep to look at each end — a defender dropping
    #: goal-side of the keeper is ordinary football, not a failure.
    goalkeeper_search_depth: int = Field(default=2, ge=1)

    # -- which side is attacking --------------------------------------------
    #: Distance from the ball to the nearest player, in multiples of that
    #: player's box height, before possession is claimed. Scale-free on
    #: purpose: players are far smaller in a wide shot than a tight one.
    ball_max_distance_boxes: float = Field(default=1.5, gt=0.0)
    #: If an opponent is within this multiple of the nearest player's
    #: distance, possession is reported as contested.
    ball_ambiguous_ratio: float = Field(default=1.25, ge=1.0)

    #: Keep `team_a` meaning the same kit between frames. Clustering has no
    #: memory, so without this the labels can flip from frame to frame.
    persist_colors_across_frames: bool = True


class OffsideConfig(StrictModel):
    """Milestone 2 settings. Later phases (decision support) add their
    sections alongside these."""

    body_keypoints: BodyKeypointsConfig = Field(default_factory=BodyKeypointsConfig)
    pitch_calibration: PitchCalibrationConfig = Field(
        default_factory=PitchCalibrationConfig
    )
    team_assignment: TeamAssignmentConfig = Field(default_factory=TeamAssignmentConfig)


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
    #: Defaulted so a config written before Milestone 2 still loads.
    offside: OffsideConfig = Field(default_factory=OffsideConfig)
    runtime: RuntimeConfig
    logging: LoggingConfig
    persistence: PersistenceConfig
