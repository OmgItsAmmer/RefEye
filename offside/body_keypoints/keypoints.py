"""Keypoint vocabulary and the pose types offside geometry consumes.

The keypoint names and their order are COCO-17, which is what every
mainstream pose model (YOLO-pose, RTMPose, ViTPose, OpenPose's COCO variant)
emits. Keeping the order here — rather than in the model adapter — means
swapping the pose model is an adapter change, not a rewrite of everything
downstream.

## The Law 11 detail that lives here, not in the geometry

Offside position is measured from the head, body and feet — **not the arms or
hands**, because a player cannot legally play the ball with them. That matters
mechanically, not just legally: a player reaching forward often has a wrist as
the furthest-forward point of their whole skeleton, so a naive "most advanced
keypoint" search would draw the offside line off an arm and call a legal
attacker offside.

`OFFSIDE_SURFACE_KEYPOINTS` encodes that exclusion once, here, so no later
phase has to remember it. `ARM_KEYPOINTS` is kept as a named set too, so the
exclusion is discoverable rather than looking like an oversight.

The shoulder is deliberately *included*: the boundary in the Laws is the
armpit, and the shoulder keypoint is the closest thing a COCO skeleton has to
it. That is an approximation, and an honest one to state — at broadcast
resolution the shoulder-vs-armpit difference is a few pixels, far below this
pipeline's real error bars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# -- COCO-17 vocabulary, in model output order ------------------------------

NOSE: Final = "nose"
LEFT_EYE: Final = "left_eye"
RIGHT_EYE: Final = "right_eye"
LEFT_EAR: Final = "left_ear"
RIGHT_EAR: Final = "right_ear"
LEFT_SHOULDER: Final = "left_shoulder"
RIGHT_SHOULDER: Final = "right_shoulder"
LEFT_ELBOW: Final = "left_elbow"
RIGHT_ELBOW: Final = "right_elbow"
LEFT_WRIST: Final = "left_wrist"
RIGHT_WRIST: Final = "right_wrist"
LEFT_HIP: Final = "left_hip"
RIGHT_HIP: Final = "right_hip"
LEFT_KNEE: Final = "left_knee"
RIGHT_KNEE: Final = "right_knee"
LEFT_ANKLE: Final = "left_ankle"
RIGHT_ANKLE: Final = "right_ankle"

#: Index -> name, in the exact order COCO-17 pose models emit keypoints.
COCO_KEYPOINT_NAMES: Final[tuple[str, ...]] = (
    NOSE,
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_EAR,
    RIGHT_EAR,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
)

#: The two points that actually touch the pitch. A homography maps the pitch
#: *plane*, so only a point on that plane projects correctly — which is why
#: feet, not box centres, are what pitch calibration (M2.1) must be handed.
ANKLE_KEYPOINTS: Final = (LEFT_ANKLE, RIGHT_ANKLE)

KNEE_KEYPOINTS: Final = (LEFT_KNEE, RIGHT_KNEE)

#: Arms and hands: excluded from offside measurement (see module docstring).
ARM_KEYPOINTS: Final = frozenset({LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST})

#: Body parts an offside line may legally be drawn against — everything the
#: skeleton offers except the arms.
OFFSIDE_SURFACE_KEYPOINTS: Final = frozenset(
    name for name in COCO_KEYPOINT_NAMES if name not in ARM_KEYPOINTS
)


# -- how a ground point was arrived at --------------------------------------

#: Measured from a visible ankle — the good case.
SOURCE_ANKLE: Final = "ankle"
#: No ankle survived the confidence bar; the knee gave a horizontal position
#: and the player box gave the vertical one.
SOURCE_KNEE_PROJECTED: Final = "knee_projected"
#: No usable body points at all — the bottom-centre of the detector's box.
#: This is a guess, and every consumer must treat it as one.
SOURCE_BBOX_BOTTOM: Final = "bbox_bottom"


@dataclass(frozen=True)
class Keypoint:
    name: str
    xy: tuple[float, float]
    confidence: float

    @property
    def x(self) -> float:
        return self.xy[0]

    @property
    def y(self) -> float:
        return self.xy[1]


@dataclass(frozen=True)
class GroundPoint:
    """Where a player meets the pitch, in full-frame image coordinates.

    This is the point M2.1's homography will project onto the top-down pitch
    map, so its accuracy sets the floor for every distance the offside
    geometry later computes. `source` and `reason` travel with it precisely
    so that a decision built on a `bbox_bottom` guess can be reported as
    uncertain rather than presented as a measurement (M2_Plan section 7).
    """

    xy: tuple[float, float]
    confidence: float
    source: str
    reason: str

    @property
    def is_measured(self) -> bool:
        """True when a real body keypoint produced this, not a box fallback."""
        return self.source in (SOURCE_ANKLE, SOURCE_KNEE_PROJECTED)


@dataclass(frozen=True)
class LeadingPoint:
    """The body part furthest along a given direction — the point an offside
    line is actually drawn against (consumed by M2.5).

    The direction is supplied by the caller because "towards the goal line"
    is only knowable once the pitch has been calibrated; this module knows
    which body parts are *legal* to measure, not which way the pitch faces.
    """

    keypoint_name: str
    xy: tuple[float, float]
    confidence: float
    reason: str
    #: True when no legal keypoint was confident enough and the player box had
    #: to stand in for the skeleton.
    is_fallback: bool = False


@dataclass
class PlayerPose:
    """One detected player at one frame, with whatever body points survived.

    Deliberately built per *detection*, not per pose: the M1 detector and
    tracker remain the authority on who is a player and which identity they
    carry, and the pose model only enriches those boxes. A player whose pose
    could not be estimated still appears here — carrying a fallback ground
    point and a warning — because silently dropping players would quietly
    corrupt the "second-last defender" ranking in M2.5.
    """

    frame_id: int
    bbox_xyxy: tuple[float, float, float, float]
    detection_confidence: float
    ground_point: GroundPoint
    source_model: str
    keypoints: dict[str, Keypoint] = field(default_factory=dict)
    track_id: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def has_pose(self) -> bool:
        return bool(self.keypoints)

    @property
    def box_height(self) -> float:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    def keypoint(self, name: str) -> Keypoint | None:
        return self.keypoints.get(name)

    def confident_keypoints(self, min_confidence: float) -> dict[str, Keypoint]:
        return {
            name: kp
            for name, kp in self.keypoints.items()
            if kp.confidence >= min_confidence
        }

    def offside_surface_points(self, min_confidence: float) -> list[Keypoint]:
        """Confident keypoints an offside line may legally be measured from.

        Arms and hands are excluded here rather than at the call site, so a
        future phase cannot accidentally measure off a wrist.
        """
        return [
            kp
            for name, kp in self.confident_keypoints(min_confidence).items()
            if name in OFFSIDE_SURFACE_KEYPOINTS
        ]


def keypoint_index(name: str) -> int:
    """Position of a keypoint in the COCO-17 model output order."""
    return COCO_KEYPOINT_NAMES.index(name)
