"""Contact-frame refinement.

Architecture.md section 24: the action model finds an approximate moment; a
dedicated local search around that anchor picks the frame where the ball is
actually struck.

Evidence used, each normalized to 0..1 and combined with configurable weights
(the weights live in YAML, never buried here — section 24, section 58 rule 11):

    model       the spotter's own score for the anchor
    proximity   how close the nearest player is to the ball
    velocity    magnitude of the ball's speed change
    direction   angle of the ball's direction change
    motion      how fast the nearest player is moving
    track       tracking continuity around the frame

Two hard rules from the architecture are enforced here:
  * Never compute motion across a camera cut (section 26) — frames in a
    different segment are excluded from the search window entirely.
  * Missing ball detections are normal (section 25) — a frame with no ball
    scores lower but does not abort refinement, and if the whole window lacks
    ball data the refiner falls back to the model's anchor frame rather than
    failing the analysis.
"""

from __future__ import annotations

import math

from core.config.schema import ContactRefinementConfig
from core.domain.models import ActionCandidate, RefinedCandidate
from observability.logging.setup import get_logger
from vision.features.feature_cache import CachedFrame, FeatureCache

logger = get_logger(__name__)


class ContactRefiner:
    def __init__(self, config: ContactRefinementConfig, feature_cache: FeatureCache):
        self._config = config
        self._cache = feature_cache

    def refine(self, candidate: ActionCandidate) -> RefinedCandidate:
        window = self._window_for(candidate)

        if not window:
            return self.keep_anchor(
                candidate, reason="no cached features around the anchor frame"
            )

        scored = [(self._score(frame, candidate), frame) for frame in window]
        scored = [(s, f) for s, f in scored if s is not None]

        if not scored:
            return self.keep_anchor(candidate, reason="no usable ball evidence in window")

        best_score, best_frame = max(scored, key=lambda pair: pair[0].total)

        return RefinedCandidate(
            candidate_id=candidate.candidate_id,
            action_type=candidate.action_type,
            original_frame_id=candidate.anchor_frame_id,
            refined_frame_id=best_frame.frame_id,
            final_score=round(best_score.total, 4),
            model_score=candidate.model_score,
            contact_score=round(best_score.contact, 4),
            trajectory_score=round(best_score.direction, 4),
            proximity_score=round(best_score.proximity, 4),
            temporal_score=round(best_score.velocity, 4),
            evidence={
                "frames_searched": len(window),
                "frame_shift": best_frame.frame_id - candidate.anchor_frame_id,
                "player_motion": round(best_score.motion, 4),
                "track_consistency": round(best_score.track, 4),
                "ball_visible": best_frame.features.ball_confidence is not None,
                "segment": best_frame.segment,
                "source_model": candidate.source_model,
                "refined": True,
            },
        )

    # -- window selection ---------------------------------------------------

    def _window_for(self, candidate: ActionCandidate) -> list[CachedFrame]:
        anchor_id = candidate.anchor_frame_id
        frames = self._cache.get_range(
            anchor_id - self._config.window_before_frames,
            anchor_id + self._config.window_after_frames,
        )
        if not frames:
            return []

        anchor = self._cache.get(anchor_id)
        if anchor is None:
            return frames

        # Confine the search to the anchor's own shot. A frame from the other
        # side of a cut is a different camera on a different moment.
        same_segment = [f for f in frames if f.segment == anchor.segment]
        return _contiguous_around(same_segment, anchor_id)

    # -- scoring ------------------------------------------------------------

    def _score(self, frame: CachedFrame, candidate: ActionCandidate) -> "_Score | None":
        features = frame.features
        weights = self._config.weights

        neighbours = self._neighbours(frame)
        if neighbours is None:
            return None
        before, after = neighbours

        velocity_score = _velocity_change_score(before, after)
        direction_score = _direction_change_score(before, after)
        proximity_score = self._proximity_score(frame)
        motion_score = self._player_motion_score(frame)
        track_score = self._track_consistency_score(frame)

        # Model score attaches to the anchor and decays with distance from it:
        # the spotter's opinion is about that moment, not the whole window.
        distance = abs(frame.frame_id - candidate.anchor_frame_id)
        span = max(self._config.window_before_frames, self._config.window_after_frames, 1)
        model_score = candidate.model_score * (1.0 - 0.5 * min(distance / span, 1.0))

        # A frame whose ball position is interpolated is weaker evidence than
        # one where the ball was actually seen.
        visibility = 1.0 if features.ball_confidence is not None else 0.7

        total = (
            weights.model * model_score
            + weights.proximity * proximity_score
            + weights.velocity * velocity_score
            + weights.direction * direction_score
            + weights.motion * motion_score
            + weights.track_consistency * track_score
        ) * visibility

        return _Score(
            total=total,
            contact=max(velocity_score, direction_score),
            velocity=velocity_score,
            direction=direction_score,
            proximity=proximity_score,
            motion=motion_score,
            track=track_score,
        )

    def _neighbours(self, frame: CachedFrame) -> tuple[CachedFrame, CachedFrame] | None:
        """Frames immediately before and after, within the same shot."""
        before = self._cache.get_range(frame.frame_id - 4, frame.frame_id - 1)
        after = self._cache.get_range(frame.frame_id + 1, frame.frame_id + 4)

        before = [f for f in before if f.segment == frame.segment and not f.scene_cut]
        after = [f for f in after if f.segment == frame.segment and not f.scene_cut]

        if not before or not after:
            return None
        return before[-1], after[0]

    def _proximity_score(self, frame: CachedFrame) -> float:
        ball = frame.features.ball_center
        if ball is None:
            return 0.0

        best = float("inf")
        for observation in frame.tracks:
            if observation.object_type == "ball":
                continue
            best = min(
                best,
                math.hypot(
                    observation.center_xy[0] - ball[0], observation.center_xy[1] - ball[1]
                ),
            )

        if best == float("inf"):
            return 0.0
        return max(0.0, 1.0 - best / self._config.proximity_radius_px)

    def _player_motion_score(self, frame: CachedFrame) -> float:
        """A striking player is usually moving; a bystander usually is not."""
        track_id = frame.features.nearest_player_track_id
        if track_id is None:
            return 0.0

        previous = self._cache.get_range(frame.frame_id - 3, frame.frame_id - 1)
        previous = [f for f in previous if f.segment == frame.segment]
        if not previous:
            return 0.0

        current_pos = _track_center(frame, track_id)
        earlier_pos = _track_center(previous[0], track_id)
        if current_pos is None or earlier_pos is None:
            return 0.0

        displacement = math.hypot(
            current_pos[0] - earlier_pos[0], current_pos[1] - earlier_pos[1]
        )
        return min(1.0, displacement / 25.0)

    def _track_consistency_score(self, frame: CachedFrame) -> float:
        """How stable tracking is here; unstable tracking means weak evidence."""
        neighbourhood = self._cache.get_range(frame.frame_id - 3, frame.frame_id + 3)
        neighbourhood = [f for f in neighbourhood if f.segment == frame.segment]
        if not neighbourhood:
            return 0.0

        with_ball = sum(1 for f in neighbourhood if f.features.ball_center is not None)
        return with_ball / len(neighbourhood)

    def keep_anchor(self, candidate: ActionCandidate, reason: str) -> RefinedCandidate:
        """Keep the model's anchor when refinement has nothing to work with.

        Architecture.md section 25: do not fail the whole analysis because
        ball detections are missing.
        """
        logger.debug(
            "refinement_fallback",
            candidate_id=candidate.candidate_id,
            anchor_frame=candidate.anchor_frame_id,
            reason=reason,
        )
        return RefinedCandidate(
            candidate_id=candidate.candidate_id,
            action_type=candidate.action_type,
            original_frame_id=candidate.anchor_frame_id,
            refined_frame_id=candidate.anchor_frame_id,
            final_score=round(candidate.model_score * 0.8, 4),
            model_score=candidate.model_score,
            contact_score=0.0,
            trajectory_score=0.0,
            proximity_score=0.0,
            temporal_score=0.0,
            evidence={
                "refined": False,
                "fallback_reason": reason,
                "frame_shift": 0,
                "source_model": candidate.source_model,
            },
        )


class _Score:
    __slots__ = ("total", "contact", "velocity", "direction", "proximity", "motion", "track")

    def __init__(self, total, contact, velocity, direction, proximity, motion, track):
        self.total = total
        self.contact = contact
        self.velocity = velocity
        self.direction = direction
        self.proximity = proximity
        self.motion = motion
        self.track = track


def _velocity_change_score(before: CachedFrame, after: CachedFrame) -> float:
    speed_in = before.features.ball_speed
    speed_out = after.features.ball_speed
    if speed_in is None or speed_out is None:
        return 0.0

    change = abs(speed_out - speed_in)
    return min(1.0, change / 12.0)


def _direction_change_score(before: CachedFrame, after: CachedFrame) -> float:
    v_in = before.features.ball_velocity
    v_out = after.features.ball_velocity
    if v_in is None or v_out is None:
        return 0.0

    mag_in, mag_out = math.hypot(*v_in), math.hypot(*v_out)
    if mag_in < 1e-6 or mag_out < 1e-6:
        return 0.0

    cosine = (v_in[0] * v_out[0] + v_in[1] * v_out[1]) / (mag_in * mag_out)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return min(1.0, angle / 90.0)


def _track_center(frame: CachedFrame, track_id: str) -> tuple[float, float] | None:
    for observation in frame.tracks:
        if observation.track_id == track_id:
            return observation.center_xy
    return None


def _contiguous_around(frames: list[CachedFrame], anchor_id: int) -> list[CachedFrame]:
    """Keep only frames reachable from the anchor without crossing a scene cut."""
    if not frames:
        return []

    ordered = sorted(frames, key=lambda f: f.frame_id)
    anchor_index = min(
        range(len(ordered)), key=lambda i: abs(ordered[i].frame_id - anchor_id)
    )

    start = anchor_index
    while start > 0 and not ordered[start].scene_cut:
        start -= 1
    if ordered[start].scene_cut and start != anchor_index:
        start += 1

    end = anchor_index
    while end < len(ordered) - 1 and not ordered[end + 1].scene_cut:
        end += 1

    return ordered[start : end + 1]
