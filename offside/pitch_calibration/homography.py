"""Image <-> pitch mapping: the maths half of calibration.

Pure functions over point correspondences. No model, no video, no config
object — so the geometry can be tested exactly, and the messy business of
*finding* correspondences (line detection, operator clicks) stays in
`calibrator.py` where it belongs.

## Convention, fixed once here

`H` maps **image -> pitch** (pixels to metres). `H_inv` maps pitch -> image.
Everything downstream reads that from `HomographyFit` rather than deciding
per call site, because a silently inverted homography produces plausible-
looking numbers that are wrong by the whole geometry of the shot.

## Why four points is not enough, even though the maths only needs four

A homography has eight degrees of freedom, so four correspondences determine
it exactly — and therefore reproduce themselves with zero error. Zero error
from four points is not evidence of a good calibration; it is arithmetic. A
fifth point is the first one that can *disagree*, which is the first real
check on whether the operator clicked the corners they meant to.

`solve_homography` reflects that: with exactly four points it caps confidence
and says why, rather than reporting the perfect-looking residual it just
computed against itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]


@dataclass(frozen=True)
class PointCorrespondence:
    """One pairing of a place in the image with a place on the pitch."""

    image_xy: Point
    pitch_xy: Point
    #: Landmark name from `PitchModel.landmarks()` when known — carried for
    #: the operator UI and for diagnosing which click was misplaced.
    landmark: str | None = None


@dataclass
class HomographyFit:
    """A solved image<->pitch mapping, with an honest opinion of itself."""

    matrix: np.ndarray            # image -> pitch
    inverse: np.ndarray           # pitch -> image
    correspondences: list[PointCorrespondence]
    mean_error_m: float
    max_error_m: float
    confidence: float
    reasons: list[str] = field(default_factory=list)

    def to_pitch(self, points: np.ndarray | list[Point]) -> np.ndarray:
        return _transform(self.matrix, points)

    def to_image(self, points: np.ndarray | list[Point]) -> np.ndarray:
        return _transform(self.inverse, points)

    def vanishing_point_of(self, pitch_direction: Point) -> Point | None:
        """Where lines running along `pitch_direction` meet in the image.

        The offside line is parallel to the goal line, so every offside line
        in a frame passes through the vanishing point of the goal-line
        direction — which is what makes M2.5 able to draw one through any
        player without re-projecting the whole pitch.

        Returns None when that family of lines is parallel in the image too
        (a perfectly side-on camera): there is no finite meeting point, and
        the caller should use a direction vector instead.
        """
        dx, dy = pitch_direction
        at_infinity = np.array([dx, dy, 0.0], dtype=np.float64)
        image_point = self.inverse @ at_infinity

        if abs(image_point[2]) < 1e-9:
            return None
        return (float(image_point[0] / image_point[2]), float(image_point[1] / image_point[2]))


def solve_homography(
    correspondences: list[PointCorrespondence],
    *,
    max_error_m: float = 2.0,
    ransac_threshold_px: float = 8.0,
) -> HomographyFit | None:
    """Fit image -> pitch from >= 4 correspondences.

    Returns None (never a bad fit) when the points are too few or
    degenerate — three collinear corners and a fourth point cannot define a
    plane mapping, and OpenCV will happily return a matrix full of nonsense
    rather than refuse.
    """
    if len(correspondences) < 4:
        return None

    import cv2

    image_points = np.array([c.image_xy for c in correspondences], dtype=np.float64)
    pitch_points = np.array([c.pitch_xy for c in correspondences], dtype=np.float64)

    if len(correspondences) > 4:
        matrix, _ = cv2.findHomography(
            image_points, pitch_points, cv2.RANSAC, ransac_threshold_px
        )
    else:
        matrix, _ = cv2.findHomography(image_points, pitch_points, 0)

    if matrix is None or not np.all(np.isfinite(matrix)):
        return None

    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(inverse)):
        return None

    projected = _transform(matrix, image_points)
    errors = np.linalg.norm(projected - pitch_points, axis=1)
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))

    reasons: list[str] = []
    confidence = float(np.clip(1.0 - mean_error / max_error_m, 0.0, 1.0))

    if len(correspondences) == 4:
        # See module docstring: four points fit themselves perfectly by
        # construction, so the residual carries no information.
        confidence = min(confidence, 0.6)
        reasons.append(
            "only four reference points — the fit cannot be cross-checked, so "
            "a misplaced point would not show up as an error"
        )
    else:
        reasons.append(
            f"{len(correspondences)} reference points, average fit error "
            f"{mean_error:.2f}m"
        )

    if max_error > max_error_m:
        confidence *= 0.5
        reasons.append(
            f"one reference point is {max_error:.1f}m out — a landmark is "
            "likely mismatched"
        )

    return HomographyFit(
        matrix=matrix,
        inverse=inverse,
        correspondences=list(correspondences),
        mean_error_m=mean_error,
        max_error_m=max_error,
        confidence=confidence,
        reasons=reasons,
    )


def line_through_vanishing_point(
    point: Point, vanishing_point: Point | None, fallback_direction: Point
) -> tuple[Point, Point]:
    """A direction for the line through `point` that is parallel — on the
    pitch — to the family meeting at `vanishing_point`.

    Returns (point, unit direction). With no vanishing point the family is
    parallel in the image too, so the fallback direction is used unchanged.
    """
    if vanishing_point is None:
        dx, dy = fallback_direction
    else:
        dx = vanishing_point[0] - point[0]
        dy = vanishing_point[1] - point[1]

    magnitude = float(np.hypot(dx, dy))
    if magnitude < 1e-9:
        raise ValueError("cannot take a direction through the vanishing point itself")
    return point, (dx / magnitude, dy / magnitude)


def _transform(matrix: np.ndarray, points: np.ndarray | list[Point]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.hstack([array, np.ones((len(array), 1))])
    projected = homogeneous @ matrix.T

    w = projected[:, 2:3]
    # Points on the horizon divide by ~zero. Returning inf is correct and
    # loud; quietly clamping would place a player at a plausible-looking
    # position that is nowhere near where they are.
    with np.errstate(divide="ignore", invalid="ignore"):
        result = projected[:, :2] / w
    return result
