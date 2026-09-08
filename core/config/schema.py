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

    #: A ball's true size relative to a player barely changes with zoom,
    #: because both scale together — so a size band against the median
    #: player box height on the same frame is a solid, frame-invariant
    #: sanity check. A real football against a real player is ~12-13%;
    #: measured false positives on client footage reached 31%. Any "ball"
    #: detection outside this band is discarded rather than trusted, because
    #: a wrong ball position corrupts the attacking-side call (M2.3) and the
    #: "beyond the ball" offside check (M2.5) silently.
    ball_min_size_ratio: float = Field(default=0.06, ge=0.0, le=1.0)
    ball_max_size_ratio: float = Field(default=0.22, ge=0.0, le=1.0)


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


class CalibrationFollowConfig(StrictModel):
    """Carrying one calibration across a shot as the camera moves (M2.1).

    Marked landmarks are image points: without this, four marks stay pinned to
    the same pixels while the camera pans and the pitch moves out from under
    them. Nothing catches that on its own — the four marks still agree with
    each other perfectly, so the reprojection error stays at zero while the
    calibration describes a camera that stopped existing seconds ago.
    """

    enabled: bool = True

    #: Features are tracked anywhere in the frame, not only on the pitch: a
    #: camera that rotates and zooms without travelling moves every static
    #: point by one shared homography, however far away it is.
    max_features: int = Field(default=600, gt=0)
    quality_level: float = Field(default=0.01, gt=0.0, le=1.0)
    min_distance_px: int = Field(default=12, gt=0)
    #: Re-detect once tracking has worn the feature set down to this many.
    redetect_below: int = Field(default=120, gt=0)

    ransac_threshold_px: float = Field(default=3.0, gt=0.0)
    min_inliers: int = Field(default=25, gt=0)
    min_inlier_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    #: A fast pan moves the scene tens of pixels between frames. The library's
    #: default flow window is too small to find it again, so registration
    #: would fail exactly when the camera is moving most.
    flow_window_px: int = Field(default=31, gt=2)
    flow_pyramid_levels: int = Field(default=4, ge=0)

    #: Drift is cumulative and invisible in the marks themselves, so trust
    #: falls with every frame carried since a human last placed them.
    confidence_decay_per_frame: float = Field(default=0.004, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    #: Players are masked before features are chosen — a feature on a running
    #: player reports the player's motion, not the camera's.
    exclude_box_margin: float = Field(default=0.25, ge=0.0, le=1.0)


class AutoLandmarkConfig(StrictModel):
    """Automatic pitch-landmark detection — the METRIC path with nobody at
    the mouse (extends M2.1 past the plan).

    Runs a keypoint model (a football-pitch-specific one — see
    `models/pitch_keypoints/README.md`) that finds the same corner points an
    operator would otherwise click by hand, then feeds them through the
    identical homography solve manual marking uses. Validated against real
    footage before being wired in: three different broadcast clips, all
    reaching 0.90+ fit confidence with the projected pitch lines landing
    within centimetres of the real painted ones (see the module docstring in
    `offside/pitch_calibration/auto_landmarks.py` for the measurements).
    """

    enabled: bool = True
    checkpoint: str = "./models/pitch_keypoints/soccana_keypoint.pt"
    imgsz: int = Field(default=640, gt=0)
    #: Below this, a keypoint is "the model wasn't sure" and is left out
    #: rather than trusted — matches the model's own published threshold.
    visibility_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: The underlying pitch-instance detector's own confidence bar — loose,
    #: since there is only ever one pitch and this exists to gate a mostly
    #: pointless prediction on a frame with no field visible at all.
    detection_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    #: Fewer confident points than this and the result is discarded even
    #: though 4 is the mathematical minimum — extra points are what let
    #: `solve_homography`'s own RANSAC step catch a single mislabelled one
    #: rather than being forced to trust every point it's given.
    min_points: int = Field(default=6, ge=4)


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
    follow: CalibrationFollowConfig = Field(default_factory=CalibrationFollowConfig)
    auto_landmarks: AutoLandmarkConfig = Field(default_factory=AutoLandmarkConfig)


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

    # -- taking the lighting out of the measurement -------------------------
    #: The same shirt measures differently in sun and in the shadow of a
    #: stand, which splits one team into two colour groups. The grass around
    #: each player is used as a reference card to remove the lighting first.
    normalize_illumination: bool = True
    #: A correction beyond this factor means the reference was not grass; the
    #: correction is dropped rather than inventing a colour.
    illumination_max_gain: float = Field(default=2.0, gt=1.0)

    # -- striped and hooped kits --------------------------------------------
    #: A player is summarised by their *two* dominant torso colours: a single
    #: average is meaningless for stripes and hoops (a red/blue kit averages
    #: to a purple that matches neither, and shifts as the player turns).
    #: A second colour closer than this, or rarer than the share below, is
    #: treated as sponsor text or noise and collapsed into the first.
    #: A second colour counts as a real pattern only if it differs clearly in
    #: hue — a crease changes how bright fabric is, a stripe changes its
    #: colour. Measured on the reference clip, a brightness-based test called
    #: half the players striped in a match between two solid kits.
    pattern_collapse_distance: float = Field(default=25.0, gt=0.0)
    #: ...unless it differs this much in brightness, which is how black-and-
    #: white stripes are told apart from a crease in a solid shirt.
    pattern_lightness_distance: float = Field(default=35.0, gt=0.0)
    pattern_min_share: float = Field(default=0.3, gt=0.0, le=0.5)
    max_signature_pixels: int = Field(default=400, gt=0)
    #: Down-weighting lightness looks right and measured worse: on the
    #: reference clip it dropped kit separation from 58 to 43, because white
    #: and maroon differ mostly in brightness. Lighting is fixed at the source
    #: instead (normalize_illumination). Kept configurable for M2.8.
    lightness_weight: float = Field(default=1.0, gt=0.0, le=1.0)

    # -- fitting the two kits ------------------------------------------------
    #: The mathematical floor is 2 — a 2-means fit needs at least one point
    #: per cluster — and that floor is the default. This used to be a
    #: cliff-edge refusal set well above the floor (6): below it, the stage
    #: reported "the two kits could not be told apart" and produced nothing
    #: at all, even when 4 or 5 usable colours genuinely would have been
    #: enough to attempt a fit. That was inconsistent with how every other
    #: number in this module works — `confident_player_count` below already
    #: scales *confidence* down smoothly as evidence thins, rather than
    #: refusing outright, and a thin fit here gets exactly that treatment:
    #: `fit_team_colors`'s own confidence formula includes
    #: `len(usable) / confident_player_count`, so 2 or 3 samples already
    #: come out low-confidence on their own, and the M2.6 publish floor
    #: catches anything that slips through with too little evidence to
    #: trust. A hard gate above the mathematical minimum was refusing to
    #: even try what the rest of the pipeline is already built to handle
    #: honestly.
    min_players_for_clustering: int = Field(default=2, ge=2)
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
    #: A generic person detector fires on the crowd, stewards and photographers
    #: as readily as on a goalkeeper, and those detections are exactly the ones
    #: most likely to look like a goalkeeper to the ranking below: they wear a
    #: kit that matches neither team (a real "odd colour") and, being outside
    #: the pitch entirely, project to the single most extreme point along the
    #: goal-to-goal axis — more extreme than any real keeper standing near
    #: their own goal line. With metric calibration this margin is used with
    #: `PitchModel.contains()` to disqualify a candidate that projects this far
    #: (in metres) beyond the touchline/goal line before it is ever ranked.
    #: Generous on purpose — a keeper standing in the technical area for a
    #: throw-in, or just wide for a corner, must not be excluded by this.
    #: Without metric calibration there is no pitch-space position to check,
    #: so this has no effect (same "cannot claim what cannot be measured" rule
    #: as the rest of this stage).
    goalkeeper_pitch_margin_m: float = Field(default=8.0, gt=0.0)

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

    # -- deciding once per player, not once per frame -----------------------
    #: A player's team belongs to the person, not the frame. Measurements are
    #: pooled against a short-lived identity and every frame votes, so one
    #: blurred or half-occluded frame loses instead of deciding. Identities
    #: come from box overlap until M2.4 provides real ones.
    track_iou_threshold: float = Field(default=0.35, gt=0.0, le=1.0)
    track_max_age_frames: int = Field(default=12, ge=0)
    track_max_samples: int = Field(default=30, gt=0)

    # -- when to ask the operator -------------------------------------------
    #: The abstain thresholds. A player below any of these is flagged for
    #: confirmation rather than reported as settled — the design target is
    #: that residual error arrives as a question, never as a confident wrong
    #: answer. Raise these to ask more often and be wrong less often.
    vote_share_to_confirm: float = Field(default=0.75, ge=0.0, le=1.0)
    confident_player_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    min_frames_to_trust: int = Field(default=3, ge=1)


class PlayerIdentityConfig(StrictModel):
    """Following players through the contact moment (M2.4).

    Separate from `ai.tracker`, which stays tuned for M1's live preview. This
    is the triggered offside path: it can afford appearance checks and a
    longer memory, and it must be able to say "I could not tell" — something
    the live tracker's interface cannot express.
    """

    enabled: bool = True

    #: Box overlap needed to continue a track between frames.
    match_iou: float = Field(default=0.3, ge=0.0, le=1.0)
    #: How many frames a player may stay unmatched before their identity ends.
    max_misses: int = Field(default=12, ge=0)
    #: Frames of clean tracking before an identity is trusted on its own.
    min_frames_to_confirm: int = Field(default=5, ge=1)

    #: Kit colour is what stops a track being handed to the opponent who ran
    #: across it. Beyond this colour distance the pairing is refused and the
    #: player is flagged instead — a swap between kits is exactly the swap
    #: that breaks an offside call.
    appearance_max_distance: float = Field(default=30.0, gt=0.0)
    appearance_memory: int = Field(default=20, gt=0)
    #: When two tracks want the same player this closely and their kits match
    #: (teammates), the association is reported as contested rather than
    #: resolved by picking the likelier one.
    ambiguous_iou_margin: float = Field(default=0.1, ge=0.0, le=1.0)

    #: Re-identification after occlusion needs both a plausible position and
    #: a matching kit; position alone would hand the id to whoever is standing
    #: there now.
    reid_max_frames: int = Field(default=20, ge=0)
    reid_max_distance_boxes: float = Field(default=2.0, gt=0.0)

    #: A gap this short is ordinary tracking, not a re-identification: the
    #: detector drops a distant player for a frame constantly and the track
    #: coasts one step. Measured on the reference clip, treating every gap as
    #: a recovery left 11 of 18 players permanently flagged.
    recovery_gap_frames: int = Field(default=2, ge=0)
    #: How long doubt lingers after a recovery or a contested association, so
    #: it does not vanish on the next clean frame.
    doubt_frames: int = Field(default=5, ge=0)
    miss_penalty: float = Field(default=0.15, ge=0.0, le=1.0)

    #: A camera cut ends every identity: nothing about the previous shot
    #: constrains the next one, and carrying ids across invents continuity.
    detect_camera_cuts: bool = True
    cut_correlation_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    min_frames_between_cuts: int = Field(default=4, ge=0)

    #: Compensates a track's predicted box for the camera's own pan/tilt/zoom
    #: before overlap is measured, using the same homography-from-optical-flow
    #: technique `offside.pitch_calibration.tracking.CalibrationFollower`
    #: already uses to carry pitch marks across a moving shot
    #: (`vision/tracking/camera_motion.py`). Without it, a pan reads as every
    #: player having moved at once, which is the single biggest source of
    #: broken association at broadcast frame rates — far bigger than any
    #: player's own motion between two adjacent frames.
    camera_motion_compensation: bool = True
    #: Below this many surviving tracked features, re-detect from scratch
    #: rather than keep registering against a thinning set — mirrors
    #: `PitchCalibrationConfig.follow.redetect_below`.
    camera_motion_redetect_below: int = Field(default=120, ge=0)

    #: A general appearance signal (build, posture, boots — not colour; see
    #: `offside/player_identity/appearance_embedding.py`) alongside the
    #: kit-colour veto. Strictly one-directional, exactly like colour: it can
    #: refuse a pairing IoU and colour both accepted, but it never clears a
    #: doubt they raised. A crop this close to another player is exactly the
    #: crowded case where a neighbour's own pixels can bleed into the box
    #: edge — measured directly, on a synthetic two-teammate crossing, at
    #: 87% box overlap: a confident-looking embedding match there turned out
    #: to be reading the *other* player's leftover pixels, not this one's.
    #: Letting embeddings positively resolve an ambiguity the box already
    #: flagged would have made that kind of contamination look like proof.
    use_appearance_embedding: bool = True
    #: The `timm` backbone used as a general appearance feature extractor —
    #: see `appearance_embedding.py` for why an ImageNet classifier rather
    #: than a purpose-trained ReID checkpoint. Swappable without touching any
    #: caller: everything downstream only depends on an L2-normalised vector
    #: coming back.
    embedding_model: str = "mobilenetv3_small_100"
    embedding_crop_height: int = Field(default=128, gt=0)
    embedding_crop_width: int = Field(default=64, gt=0)
    #: How many recent embeddings a track keeps to form its own signature —
    #: shorter than kit-colour's memory on purpose: a general appearance
    #: embedding is more sensitive to pose and viewing angle than a colour
    #: swatch is, so an old measurement from a very different pose ages out
    #: sooner.
    embedding_memory: int = Field(default=10, gt=0)
    #: Distance beyond which a re-identification candidate is refused outright
    #: — an extra veto alongside kit colour in `_recover_lost`, where "this
    #: really is the same player" matters more than usual (occlusion is
    #: exactly when a teammate can end up standing where the hidden player
    #: was). In the same L2 units as two unit-normalised vectors (max 2.0).
    embedding_max_distance: float = Field(default=0.9, gt=0.0)


class OffsideLineConfig(StrictModel):
    """The line, the comparison, and when to refuse to call it (M2.5).

    The important numbers here are the uncertainties. The geometry is a
    comparison of two positions along one axis; what decides whether that
    comparison is worth reporting is how far out the inputs could be. Set
    these too low and the tool announces verdicts its own measurements cannot
    support — which is the one failure this milestone is built to avoid.
    """

    enabled: bool = True

    #: Fixed cost of the calibration itself, before any player is measured.
    metric_base_uncertainty_m: float = Field(default=0.15, ge=0.0)
    #: How wrong a *fully unconfident* body point may be. Scaled by how much
    #: M2.2 actually trusted the point, and counted once for the attacker and
    #: once for the defender, since both are being measured.
    metric_foot_uncertainty_m: float = Field(default=0.6, ge=0.0)
    #: Without metres the axis is in pixels, so a player's own height is the
    #: only available scale — and a good one, since it shrinks with distance
    #: exactly as the measurement error does.
    directional_uncertainty_boxes: float = Field(default=0.35, ge=0.0)

    #: How far apart the two sides' averages must be before their shape is
    #: accepted as evidence of which end is being defended.
    min_team_separation_m: float = Field(default=3.0, ge=0.0)
    min_team_separation_px: float = Field(default=40.0, ge=0.0)

    #: Law 11 requires the attacker to be nearer the goal line than the ball as
    #: well as the second-last opponent. Leaving the ball out is a mistake a
    #: geometry-only tool makes constantly.
    require_beyond_ball: bool = True

    #: Below this, the stage refuses to report the verdict at all. The geometry
    #: can be exact and the answer still worthless: measured on the reference
    #: clip, a frame with the teams mixed together produced "offside by 14.6m"
    #: at a confidence of 0.08, because which end was being defended was barely
    #: more than a guess. A precise number beside a confidence that low is the
    #: confident-and-wrong output this milestone exists to avoid.
    min_confidence_to_call: float = Field(default=0.35, ge=0.0, le=1.0)


class DecisionSupportConfig(StrictModel):
    """Confidence banding and when to publish a verdict at all (M2.6).

    These thresholds decide what the tool says in its own voice. They are set
    cautiously on purpose: the cost of a band that is too careful is that the
    operator opens the frame; the cost of one that is too generous is a wrong
    call published as the tool's own conclusion.
    """

    enabled: bool = True

    #: "High" has to mean the operator can sign the call without opening the
    #: frame, so it sits well above a coin flip.
    high_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    #: Below this the call is shown as a suggestion to be judged by eye.
    medium_confidence: float = Field(default=0.45, ge=0.0, le=1.0)

    #: Below this a yes/no verdict is withheld entirely and the weakest stage
    #: is named instead. Deliberately a little above M2.5's own floor, because
    #: this stage sees inputs M2.5 never does — drifted pitch marks, contested
    #: identities, unconfirmed kit colours — each of which produces a decision
    #: that looks entirely normal and is wrong. Abstentions ("too close to
    #: call") are never withheld: a weak chain cannot make an abstention wrong.
    min_confidence_to_publish: float = Field(default=0.4, ge=0.0, le=1.0)

    #: How many caveats and suggested fixes to show. A list nobody reads is
    #: the same as no list, so the weakest few are shown and the rest stay in
    #: the signal rows.
    max_limits: int = Field(default=3, ge=1, le=10)
    max_actions: int = Field(default=3, ge=1, le=10)


class OffsideConfig(StrictModel):
    """Milestone 2 settings. Later phases (decision support) add their
    sections alongside these."""

    body_keypoints: BodyKeypointsConfig = Field(default_factory=BodyKeypointsConfig)
    pitch_calibration: PitchCalibrationConfig = Field(
        default_factory=PitchCalibrationConfig
    )
    team_assignment: TeamAssignmentConfig = Field(default_factory=TeamAssignmentConfig)
    player_identity: PlayerIdentityConfig = Field(default_factory=PlayerIdentityConfig)
    offside_line: OffsideLineConfig = Field(default_factory=OffsideLineConfig)
    decision_support: DecisionSupportConfig = Field(
        default_factory=DecisionSupportConfig
    )


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
class PipelineRunLogConfig(StrictModel):
    """One detailed JSON file per offside pipeline run (extends M2.7).

    Separate from the firehose session log (`logs/session_<id>.jsonl`, every
    subsystem interleaved — video buffering, model loading, UI events, every
    triggered run, all in one growing file). That file answers "what did
    this session do"; this answers "what happened on *this* frame" — every
    stage's state, its full detail lines, how long it took, and the final
    verdict's confidence chain, as one self-contained file a developer or an
    agent can open in isolation rather than grep out of megabytes of
    unrelated log lines.
    """

    enabled: bool = True
    directory: str = "./logs/CLI"


class LoggingConfig(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    directory: str
    json_file: bool
    console: bool
    rotation_max_bytes: int = Field(default=10_485_760, gt=0)
    rotation_backup_count: int = Field(default=5, ge=0)
    pipeline_run_log: PipelineRunLogConfig = Field(default_factory=PipelineRunLogConfig)


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
