"""The pitch as a fixed reference (M2.1).

`PitchModel` is the top-down truth an image gets calibrated against: named
landmarks in metres, and the painted lines for drawing a top-down map.
"""

from offside.field_geometry.pitch import (
    CENTRE_CIRCLE_RADIUS,
    GOAL_AREA_DEPTH,
    GOAL_AREA_WIDTH,
    PENALTY_AREA_DEPTH,
    PENALTY_AREA_WIDTH,
    PENALTY_SPOT_DISTANCE,
    PitchModel,
)

__all__ = [
    "CENTRE_CIRCLE_RADIUS",
    "GOAL_AREA_DEPTH",
    "GOAL_AREA_WIDTH",
    "PENALTY_AREA_DEPTH",
    "PENALTY_AREA_WIDTH",
    "PENALTY_SPOT_DISTANCE",
    "PitchModel",
]
