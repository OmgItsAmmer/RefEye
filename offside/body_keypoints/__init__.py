"""Body keypoints for detected players (M2.2).

Public surface, so later phases import from here rather than reaching into
the adapter:

    PlayerPose / Keypoint / GroundPoint / LeadingPoint  — the data
    estimate_ground_point / leading_offside_point       — the policy
    YoloPoseEstimator                                    — the model adapter

M2.1 consumes `PlayerPose.ground_point` (the only point on a player that sits
on the pitch plane, so the only one a homography projects correctly).
M2.5 consumes `leading_offside_point`, which knows that arms don't count.
"""

from offside.body_keypoints.ground_point import (
    estimate_ground_point,
    leading_offside_point,
)
from offside.body_keypoints.keypoints import (
    ANKLE_KEYPOINTS,
    ARM_KEYPOINTS,
    COCO_KEYPOINT_NAMES,
    OFFSIDE_SURFACE_KEYPOINTS,
    SOURCE_ANKLE,
    SOURCE_BBOX_BOTTOM,
    SOURCE_KNEE_PROJECTED,
    GroundPoint,
    Keypoint,
    LeadingPoint,
    PlayerPose,
)

__all__ = [
    "ANKLE_KEYPOINTS",
    "ARM_KEYPOINTS",
    "COCO_KEYPOINT_NAMES",
    "OFFSIDE_SURFACE_KEYPOINTS",
    "SOURCE_ANKLE",
    "SOURCE_BBOX_BOTTOM",
    "SOURCE_KNEE_PROJECTED",
    "GroundPoint",
    "Keypoint",
    "LeadingPoint",
    "PlayerPose",
    "estimate_ground_point",
    "leading_offside_point",
]
