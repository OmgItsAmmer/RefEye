"""Kinematic baseline action spotter.

**What this is.** A ball-trajectory analyser, not a learned action-spotting
model. It finds moments where the ball's motion changes abruptly while a
player is close by, which is the physical signature of a ball contact, and
classifies the contact from the resulting kinematics.

**Why it exists.** The T-DEED checkpoint is not redistributable and is not
present in this repository (see `ai/action_spotting/tdeed/adapter.py`). Rather
than ship an application that cannot demonstrate its own core loop, this
adapter provides a real, working spotter behind the same `ActionSpotter`
protocol. It reports itself honestly as `kinematic-baseline` in every
candidate, log line, and diagnostics field, so no result can be mistaken for
T-DEED output.

**What it is not.** It has no learned notion of football semantics. It cannot
distinguish a deliberate through-ball from a deflection, and its action-type
labels are heuristics over velocity and geometry. Accuracy on real broadcast
footage has not been measured, and the class labels it emits should be treated
as provisional until validated against client footage.

Swapping in T-DEED is a one-line config change (`ai.action_spotter.provider`)
once a licensed checkpoint is available — nothing outside this package needs
to change.
"""

from __future__ import annotations

import math
import uuid

from ai.action_spotting.common.actions import (
    BALL_CONTACT,
    CROSS,
    GOALKEEPER_CONTACT,
    HEADER,
    PASS,
    SHOT,
    SUPPORTED_ACTIONS,
)
from core.domain.models import ActionCandidate
from core.interfaces.ai import AnalysisClip
from observability.logging.setup import get_logger
from vision.features.feature_cache import CachedFrame, FeatureCache

logger = get_logger(__name__)

MODEL_NAME = "kinematic-baseline"
MODEL_VERSION = "1.0"


class KinematicActionSpotter:
    """Finds ball-contact moments from trajectory discontinuities."""

    def __init__(
        self,
        feature_cache: FeatureCache,
        frame_width: int,
        frame_height: int,
        min_direction_change_deg: float = 25.0,
        min_speed_ratio: float = 1.35,
        proximity_radius_px: float = 90.0,
        min_score: float = 0.15,
        max_candidates: int = 12,
    ):
        self._cache = feature_cache
        self._width = frame_width
        self._height = frame_height
        self._min_direction_change = min_direction_change_deg
        self._min_speed_ratio = min_speed_ratio
        self._proximity_radius = proximity_radius_px
        self._min_score = min_score
        self._max_candidates = max_candidates

    # -- ActionSpotter protocol --------------------------------------------

    @property
    def model_name(self) -> str:
        return MODEL_NAME

    @property
    def model_version(self) -> str:
        return MODEL_VERSION

    @property
    def supported_actions(self) -> set[str]:
        return set(SUPPORTED_ACTIONS)

    def warmup(self) -> None:
        """Nothing to warm: this spotter holds no weights and no GPU state."""

    def infer(self, clip: AnalysisClip) -> list[ActionCandidate]:
        frames = self._cache.get_range(clip.start_frame_id, clip.end_frame_id)
        if len(frames) < 5:
            logger.warning(
                "spotter_insufficient_features",
                model=self.model_name,
                frames_available=len(frames),
                requested_range=f"{clip.start_frame_id}-{clip.end_frame_id}",
            )
            return []

        candidates: list[ActionCandidate] = []

        for index in range(2, len(frames) - 2):
            evidence = self._evaluate(frames, index)
            if evidence is None:
                continue

            score, action_type, track_id = evidence
            if score < self._min_score:
                continue

            current = frames[index]
            candidates.append(
                ActionCandidate(
                    candidate_id=uuid.uuid4().hex[:10],
                    action_type=action_type,
                    anchor_frame_id=current.frame_id,
                    anchor_timestamp_ms=current.features.timestamp_ms,
                    model_score=round(score, 4),
                    source_model=f"{MODEL_NAME}:{MODEL_VERSION}",
                    player_track_id=track_id,
                    metadata={
                        "ball_speed": current.features.ball_speed,
                        "segment": current.segment,
                        "interpolated_ball": current.features.ball_confidence is None,
                    },
                )
            )

        candidates.sort(key=lambda c: c.model_score, reverse=True)
        return candidates[: self._max_candidates]

    # -- scoring ------------------------------------------------------------

    def _evaluate(
        self, frames: list[CachedFrame], index: int
    ) -> tuple[float, str, str | None] | None:
        current = frames[index]
        previous = frames[index - 1]
        following = frames[index + 1]

        # A contact cannot be inferred across a camera cut: the ball's
        # apparent motion either side belongs to different shots.
        if current.scene_cut or following.scene_cut:
            return None
        if not (previous.segment == current.segment == following.segment):
            return None

        incoming = previous.features.ball_velocity
        outgoing = following.features.ball_velocity
        if incoming is None or outgoing is None:
            return None

        speed_in = math.hypot(*incoming)
        speed_out = math.hypot(*outgoing)
        if speed_in < 1.0 and speed_out < 1.0:
            return None  # ball essentially stationary; nothing happened

        direction_change = _angle_between(incoming, outgoing)
        speed_ratio = speed_out / speed_in if speed_in > 0.5 else (2.0 if speed_out > 2 else 1.0)

        direction_score = _ramp(direction_change, self._min_direction_change, 140.0)
        speed_score = _ramp(abs(math.log(max(speed_ratio, 0.05))), math.log(self._min_speed_ratio), 1.6)

        ball_center = current.features.ball_center
        proximity_score, track_id, person_box = self._nearest_person(current, ball_center)

        # A trajectory change with nobody near it is a detection artefact,
        # not a contact — proximity gates the whole score rather than merely
        # contributing to it.
        if proximity_score <= 0.0:
            return None

        confidence_penalty = 0.75 if current.features.ball_confidence is None else 1.0

        score = (
            0.45 * max(direction_score, speed_score)
            + 0.25 * min(direction_score, speed_score)
            + 0.30 * proximity_score
        ) * confidence_penalty

        action_type = self._classify(
            ball_center=ball_center,
            outgoing=outgoing,
            speed_out=speed_out,
            speed_ratio=speed_ratio,
            person_box=person_box,
        )

        return score, action_type, track_id

    def _nearest_person(
        self, frame: CachedFrame, ball_center: tuple[float, float] | None
    ) -> tuple[float, str | None, tuple[float, float, float, float] | None]:
        if ball_center is None:
            return 0.0, None, None

        best_distance = float("inf")
        best_track: str | None = None
        best_box: tuple[float, float, float, float] | None = None

        for observation in frame.tracks:
            if observation.object_type == "ball":
                continue
            distance = math.hypot(
                observation.center_xy[0] - ball_center[0],
                observation.center_xy[1] - ball_center[1],
            )
            if distance < best_distance:
                best_distance = distance
                best_track = observation.track_id
                best_box = observation.bbox_xyxy

        if best_distance == float("inf") or best_distance > self._proximity_radius:
            return 0.0, best_track, best_box

        return 1.0 - (best_distance / self._proximity_radius), best_track, best_box

    def _classify(
        self,
        ball_center: tuple[float, float] | None,
        outgoing: tuple[float, float],
        speed_out: float,
        speed_ratio: float,
        person_box: tuple[float, float, float, float] | None,
    ) -> str:
        """Heuristic action typing. Provisional until validated on real footage."""
        if ball_center is None:
            return BALL_CONTACT

        # Ball meeting the upper third of a player's box reads as a header.
        if person_box is not None:
            box_top, box_bottom = person_box[1], person_box[3]
            box_height = max(box_bottom - box_top, 1.0)
            if (ball_center[1] - box_top) / box_height < 0.25:
                return HEADER

        in_wide_area = ball_center[0] < self._width * 0.2 or ball_center[0] > self._width * 0.8
        moving_fast = speed_out > 12.0 and speed_ratio > 1.4
        vertical_ratio = abs(outgoing[1]) / max(abs(outgoing[0]), 0.5)

        # Near a goal line and struck hard: most likely a shot.
        if moving_fast and not in_wide_area:
            return SHOT
        # From wide, travelling laterally into the middle: a cross.
        if in_wide_area and vertical_ratio < 1.2 and speed_out > 6.0:
            return CROSS
        # Deep in the penalty area, ball killed rather than redirected.
        if in_wide_area and speed_ratio < 0.6:
            return GOALKEEPER_CONTACT

        return PASS


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Angle in degrees between two velocity vectors."""
    mag_a, mag_b = math.hypot(*a), math.hypot(*b)
    if mag_a < 1e-6 or mag_b < 1e-6:
        return 0.0
    cosine = (a[0] * b[0] + a[1] * b[1]) / (mag_a * mag_b)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _ramp(value: float, low: float, high: float) -> float:
    """Linear 0..1 ramp between `low` and `high`, clamped at both ends."""
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))
