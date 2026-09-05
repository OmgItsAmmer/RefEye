"""The pitch itself: dimensions, named landmarks, and the lines painted on it.

This is the *reference* model — what a pitch looks like from directly above,
in metres — which is the thing an image is calibrated *against*. Nothing here
touches a camera, a frame or a model; it is the fixed half of the
correspondence that M2.1 solves.

## Coordinate system

    (0,0)                                        (105,0)
      +--------------------------------------------+   <- y = 0 touchline
      |                     |                      |
      |   [penalty area]    |     [penalty area]   |
      |                  centre                    |
      +--------------------------------------------+   <- y = 68 touchline
    (0,68)                                       (105,68)

x runs 0..length along the pitch (goal line to goal line), y runs 0..width
across it. y increases "downward" so the layout matches image conventions
when the top-down map is drawn, which removes a whole class of flipped-axis
bugs in the debug view.

Dimensions are configurable because the Laws permit a range (90-120m x
45-90m) and broadcast pitches genuinely differ. 105x68 is the FIFA/UEFA
standard and the default. Everything else — penalty area, goal area, centre
circle, penalty spot — is fixed by the Laws and derived, never guessed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Final

# Fixed by the Laws of the Game, in metres. These are not tunable and
# deliberately do not live in config: a penalty area is 16.5m deep at every
# ground in the world, and making it configurable would only let a typo
# silently distort every calibration built on it.
PENALTY_AREA_DEPTH: Final = 16.5
PENALTY_AREA_WIDTH: Final = 40.32
GOAL_AREA_DEPTH: Final = 5.5
GOAL_AREA_WIDTH: Final = 18.32
PENALTY_SPOT_DISTANCE: Final = 11.0
CENTRE_CIRCLE_RADIUS: Final = 9.15
GOAL_WIDTH: Final = 7.32

Point = tuple[float, float]
Segment = tuple[Point, Point]


@dataclass(frozen=True)
class PitchModel:
    """A pitch in metres, top-down. Landmark names are the vocabulary the
    operator's manual calibration clicks and the auto matcher both speak."""

    length: float = 105.0
    width: float = 68.0

    # -- derived geometry ---------------------------------------------------

    @property
    def centre(self) -> Point:
        return (self.length / 2.0, self.width / 2.0)

    def landmarks(self) -> dict[str, Point]:
        """Every point a human can unambiguously pick out of a broadcast frame.

        Corners of boxes are the useful ones: they are sharp, painted, and
        (unlike a touchline midpoint) identifiable without measuring.
        """
        length, width = self.length, self.width
        pa_top = (width - PENALTY_AREA_WIDTH) / 2.0
        pa_bottom = width - pa_top
        ga_top = (width - GOAL_AREA_WIDTH) / 2.0
        ga_bottom = width - ga_top
        goal_top = (width - GOAL_WIDTH) / 2.0
        goal_bottom = width - goal_top

        return {
            # Pitch corners
            "corner_left_top": (0.0, 0.0),
            "corner_left_bottom": (0.0, width),
            "corner_right_top": (length, 0.0),
            "corner_right_bottom": (length, width),
            # Halfway line meets the touchlines
            "halfway_top": (length / 2.0, 0.0),
            "halfway_bottom": (length / 2.0, width),
            "centre_mark": (length / 2.0, width / 2.0),
            # Left penalty area
            "left_penalty_area_top_goalline": (0.0, pa_top),
            "left_penalty_area_bottom_goalline": (0.0, pa_bottom),
            "left_penalty_area_top_corner": (PENALTY_AREA_DEPTH, pa_top),
            "left_penalty_area_bottom_corner": (PENALTY_AREA_DEPTH, pa_bottom),
            # Left goal area (six-yard box)
            "left_goal_area_top_goalline": (0.0, ga_top),
            "left_goal_area_bottom_goalline": (0.0, ga_bottom),
            "left_goal_area_top_corner": (GOAL_AREA_DEPTH, ga_top),
            "left_goal_area_bottom_corner": (GOAL_AREA_DEPTH, ga_bottom),
            "left_penalty_spot": (PENALTY_SPOT_DISTANCE, width / 2.0),
            "left_goalpost_top": (0.0, goal_top),
            "left_goalpost_bottom": (0.0, goal_bottom),
            # Right penalty area
            "right_penalty_area_top_goalline": (length, pa_top),
            "right_penalty_area_bottom_goalline": (length, pa_bottom),
            "right_penalty_area_top_corner": (length - PENALTY_AREA_DEPTH, pa_top),
            "right_penalty_area_bottom_corner": (length - PENALTY_AREA_DEPTH, pa_bottom),
            # Right goal area
            "right_goal_area_top_goalline": (length, ga_top),
            "right_goal_area_bottom_goalline": (length, ga_bottom),
            "right_goal_area_top_corner": (length - GOAL_AREA_DEPTH, ga_top),
            "right_goal_area_bottom_corner": (length - GOAL_AREA_DEPTH, ga_bottom),
            "right_penalty_spot": (length - PENALTY_SPOT_DISTANCE, width / 2.0),
            "right_goalpost_top": (length, goal_top),
            "right_goalpost_bottom": (length, goal_bottom),
        }

    def landmark(self, name: str) -> Point:
        try:
            return self.landmarks()[name]
        except KeyError as exc:
            raise KeyError(f"unknown pitch landmark: {name!r}") from exc

    def lines(self) -> list[Segment]:
        """The painted lines, for drawing the top-down map (M2.1's debug view
        and, later, M2.7's operator overlay)."""
        marks = self.landmarks()
        length = self.length

        return [
            # Touchlines and goal lines
            (marks["corner_left_top"], marks["corner_right_top"]),
            (marks["corner_left_bottom"], marks["corner_right_bottom"]),
            (marks["corner_left_top"], marks["corner_left_bottom"]),
            (marks["corner_right_top"], marks["corner_right_bottom"]),
            # Halfway
            (marks["halfway_top"], marks["halfway_bottom"]),
            # Left penalty area
            (marks["left_penalty_area_top_goalline"], marks["left_penalty_area_top_corner"]),
            (marks["left_penalty_area_top_corner"], marks["left_penalty_area_bottom_corner"]),
            (marks["left_penalty_area_bottom_corner"], marks["left_penalty_area_bottom_goalline"]),
            # Left goal area
            (marks["left_goal_area_top_goalline"], marks["left_goal_area_top_corner"]),
            (marks["left_goal_area_top_corner"], marks["left_goal_area_bottom_corner"]),
            (marks["left_goal_area_bottom_corner"], marks["left_goal_area_bottom_goalline"]),
            # Right penalty area
            (marks["right_penalty_area_top_goalline"], marks["right_penalty_area_top_corner"]),
            (marks["right_penalty_area_top_corner"], marks["right_penalty_area_bottom_corner"]),
            (
                marks["right_penalty_area_bottom_corner"],
                marks["right_penalty_area_bottom_goalline"],
            ),
            # Right goal area
            (marks["right_goal_area_top_goalline"], marks["right_goal_area_top_corner"]),
            (marks["right_goal_area_top_corner"], marks["right_goal_area_bottom_corner"]),
            (marks["right_goal_area_bottom_corner"], marks["right_goal_area_bottom_goalline"]),
            # Centre circle, as a polyline
            *_circle_segments(self.centre, CENTRE_CIRCLE_RADIUS),
            # Goals, drawn short of the goal line so they read as goals
            ((0.0, marks["left_goalpost_top"][1]), (-2.0, marks["left_goalpost_top"][1])),
            ((-2.0, marks["left_goalpost_top"][1]), (-2.0, marks["left_goalpost_bottom"][1])),
            ((-2.0, marks["left_goalpost_bottom"][1]), (0.0, marks["left_goalpost_bottom"][1])),
            ((length, marks["right_goalpost_top"][1]), (length + 2.0, marks["right_goalpost_top"][1])),
            (
                (length + 2.0, marks["right_goalpost_top"][1]),
                (length + 2.0, marks["right_goalpost_bottom"][1]),
            ),
            (
                (length + 2.0, marks["right_goalpost_bottom"][1]),
                (length, marks["right_goalpost_bottom"][1]),
            ),
        ]

    def contains(self, point: Point, margin: float = 0.0) -> bool:
        """Whether a pitch-space point is on the field of play.

        M2.5 uses this as a sanity check: a "player" who projects to 40m
        outside the touchline is a calibration failure or a crowd detection,
        not a defender to draw a line from.
        """
        x, y = point
        return (
            -margin <= x <= self.length + margin
            and -margin <= y <= self.width + margin
        )

    def attacking_direction_for_goal(self, goal: str) -> tuple[float, float]:
        """Unit vector, in pitch space, pointing at the goal being attacked.

        Offside is judged towards *one* goal line, so every geometric
        comparison in M2.5 needs to know which one.
        """
        if goal == "left":
            return (-1.0, 0.0)
        if goal == "right":
            return (1.0, 0.0)
        raise ValueError(f"goal must be 'left' or 'right', got {goal!r}")


def _circle_segments(centre: Point, radius: float, steps: int = 48) -> list[Segment]:
    import math

    cx, cy = centre
    points = [
        (cx + radius * math.cos(2 * math.pi * i / steps),
         cy + radius * math.sin(2 * math.pi * i / steps))
        for i in range(steps + 1)
    ]
    return list(itertools.pairwise(points))
