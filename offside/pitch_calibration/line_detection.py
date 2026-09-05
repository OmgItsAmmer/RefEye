"""Finding the painted lines on the grass, and where they meet.

Classical CV, deliberately: pitch markings are white paint on a green plane
under stadium light, which is close to the ideal case for colour masking plus
a Hough transform — and unlike a learned detector it needs no checkpoint, no
GPU, and no training data from a broadcaster we have never seen.

What it is *not* is a landmark identifier. Detecting that a line exists is
easy; knowing it is the 16.5m line rather than the six-yard line requires
either a learned pitch-keypoint model or an operator. That distinction is the
whole reason `calibrator.py` separates "lines found" from "pitch understood",
and why the manual path exists.

## Vanishing points, and why they matter more than they look

Lines that are parallel on the pitch meet at a single point in the image.
Two consequences the offside geometry leans on:

  * The offside line is parallel to the goal line, so **every** offside line
    in a frame passes through the vanishing point of the goal-line family.
    Find that point and M2.5 can draw a correct offside line through any
    player — without a full metric calibration.
  * It is recoverable from far less evidence than a homography needs: two
    goal-line-parallel lines anywhere in frame will do (the 16.5m line and
    the six-yard line, say), where a homography needs four identified
    landmarks.

So a frame can support offside geometry while still being unable to say where
anyone is in metres. `calibrator.py` reports those as different capability
levels rather than collapsing them into one pass/fail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Point = tuple[float, float]


@dataclass(frozen=True)
class DetectedLine:
    p1: Point
    p2: Point

    @property
    def length(self) -> float:
        return math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])

    @property
    def angle_deg(self) -> float:
        """Orientation in [0, 180) — lines have no direction, only a slope."""
        angle = math.degrees(
            math.atan2(self.p2[1] - self.p1[1], self.p2[0] - self.p1[0])
        )
        return angle % 180.0

    @property
    def midpoint(self) -> Point:
        return ((self.p1[0] + self.p2[0]) / 2.0, (self.p1[1] + self.p2[1]) / 2.0)

    def homogeneous(self) -> np.ndarray:
        """The line as ax+by+c=0, via the cross product of its endpoints."""
        a = np.array([self.p1[0], self.p1[1], 1.0])
        b = np.array([self.p2[0], self.p2[1], 1.0])
        return np.cross(a, b)


@dataclass(frozen=True)
class VanishingPoint:
    xy: Point
    #: Lines that agreed on it — more agreement, more trust.
    support: int
    #: Mean angular disagreement of supporting lines, in degrees.
    residual_deg: float
    confidence: float


def pitch_line_mask(
    image: np.ndarray,
    *,
    exclude_boxes: list[tuple[float, float, float, float]] | None = None,
    grass_hue_range: tuple[int, int] = (30, 95),
    grass_min_saturation: int = 40,
    line_min_brightness: int = 150,
    max_line_width_px: int = 15,
    tophat_threshold: int = 20,
    exclude_box_margin: float = 0.15,
) -> np.ndarray:
    """Binary mask of painted pitch markings. Exposed so the debug view can
    show exactly what the line finder is looking at — most calibration
    failures are obvious the moment you see this mask.

    Three filters, each removing a different impostor that measurably
    polluted the result on real broadcast footage:

    **Grass containment** removes the crowd, the roof and the scoreboard
    graphic. The mask is *eroded*, not dilated: the advertising boards sit
    directly above the touchline, and a dilated grass region reaches far
    enough to swallow them — on the reference clip those boards produced
    more "lines" than the pitch did.

    **A top-hat filter** keeps only bright structures thinner than
    `max_line_width_px`. Paint is a few pixels wide; a boarding, a sleeve or
    a patch of glare is not. Plain brightness thresholding cannot tell them
    apart, and on the reference clip it returned the boards as the longest,
    strongest lines in the frame.

    **Player exclusion** removes what remains: white kits. A player in a
    white shirt is a thin bright object on green, which is precisely the
    definition a line detector is built around — on the reference clip,
    every "steep line" found before this filter was a player, and the
    vanishing point they voted for was meaningless. Passing the detector's
    boxes in fixes it properly; the detector already ran for other reasons.
    """
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_low, hue_high = grass_hue_range

    grass = cv2.inRange(
        hsv,
        np.array([hue_low, grass_min_saturation, 40], dtype=np.uint8),
        np.array([hue_high, 255, 255], dtype=np.uint8),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # Close first so players and shadows do not punch holes in the field,
    # then pull the boundary inwards, away from the boards.
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, kernel)
    grass = cv2.erode(grass, kernel, iterations=1)

    # Thin-structure filter: a top-hat keeps what a wider opening removes.
    width = max(3, max_line_width_px | 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    )
    thin = cv2.threshold(tophat, tophat_threshold, 255, cv2.THRESH_BINARY)[1]

    # A plain brightness floor, not a saturation-based "white" test: paint
    # viewed through broadcast compression picks up so much green from the
    # grass around it that a strict white mask discards most of the line —
    # measured on the reference clip, it kept about a third of the pixels the
    # top-hat had already found.
    bright = cv2.threshold(gray, line_min_brightness, 255, cv2.THRESH_BINARY)[1]

    painted = cv2.bitwise_and(cv2.bitwise_and(bright, grass), thin)

    if exclude_boxes:
        height, frame_width = painted.shape[:2]
        for x1, y1, x2, y2 in exclude_boxes:
            pad_x = (x2 - x1) * exclude_box_margin
            pad_y = (y2 - y1) * exclude_box_margin
            cv2.rectangle(
                painted,
                (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))),
                (min(frame_width, int(x2 + pad_x)), min(height, int(y2 + pad_y))),
                0,
                thickness=-1,
            )

    # Bridge the gaps a shadow or a boot leaves across a line: Hough scores a
    # joined line far better than its fragments.
    #
    # Note there is deliberately no opening/erosion step afterwards. Paint is
    # only one to three pixels wide at this framing, so even a 3x3 erosion
    # deletes most of it — an earlier version of this function did exactly
    # that and reduced a frame from ~43 usable segments to 4. The top-hat
    # above is already the noise filter.
    return cv2.morphologyEx(
        painted, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )


def detect_pitch_lines(
    image: np.ndarray,
    *,
    exclude_boxes: list[tuple[float, float, float, float]] | None = None,
    grass_hue_range: tuple[int, int] = (30, 95),
    grass_min_saturation: int = 40,
    line_min_brightness: int = 150,
    max_line_width_px: int = 15,
    tophat_threshold: int = 20,
    exclude_box_margin: float = 0.15,
    min_line_length_px: int = 50,
    max_line_gap_px: int = 25,
    hough_threshold: int = 40,
    merge_angle_deg: float = 4.0,
    merge_distance_px: float = 14.0,
    max_lines: int = 40,
) -> list[DetectedLine]:
    """White pitch markings, as merged line segments.

    Pass `exclude_boxes` (the detector's player boxes) whenever they are
    available — see `pitch_line_mask` for why it matters so much.
    """
    import cv2

    painted = pitch_line_mask(
        image,
        exclude_boxes=exclude_boxes,
        grass_hue_range=grass_hue_range,
        grass_min_saturation=grass_min_saturation,
        line_min_brightness=line_min_brightness,
        max_line_width_px=max_line_width_px,
        tophat_threshold=tophat_threshold,
        exclude_box_margin=exclude_box_margin,
    )

    segments = cv2.HoughLinesP(
        painted,
        rho=1,
        theta=np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_line_length_px,
        maxLineGap=max_line_gap_px,
    )
    if segments is None:
        return []

    lines = [
        DetectedLine((float(x1), float(y1)), (float(x2), float(y2)))
        for x1, y1, x2, y2 in segments[:, 0, :]
    ]
    merged = merge_lines(lines, merge_angle_deg, merge_distance_px)
    merged.sort(key=lambda line: line.length, reverse=True)
    return merged[:max_lines]


def merge_lines(
    lines: list[DetectedLine],
    angle_tolerance_deg: float,
    distance_tolerance_px: float,
    max_join_gap_px: float = 90.0,
) -> list[DetectedLine]:
    """Collapse the several fragments Hough returns per painted line into one.

    A single 16.5m line typically arrives as five or six broken segments
    (players stand on it, shadows cross it). Left unmerged they each vote
    separately, and a heavily occluded line outvotes a cleanly visible one.

    Three conditions, all required — dropping any one of them makes merging
    snowball. Distance alone is the dangerous case: once a merged line grows,
    ever more fragments sit near its *infinite* extension, so a single line
    creeps across the frame absorbing unrelated markings until only a handful
    of frame-spanning lines remain.

    1. Similar orientation.
    2. *Both* endpoints close to the existing line — not just the midpoint,
       which a crossing segment also satisfies.
    3. Actually adjacent along the line: the fragments must overlap or sit
       within `max_join_gap_px` of each other, so two stretches of paint at
       opposite ends of the pitch stay separate even when collinear.
    """
    merged: list[DetectedLine] = []

    for line in sorted(lines, key=lambda item: item.length, reverse=True):
        absorbed = False
        for index, existing in enumerate(merged):
            if _angular_gap(line.angle_deg, existing.angle_deg) > angle_tolerance_deg:
                continue
            if max(
                _point_to_line_distance(line.p1, existing),
                _point_to_line_distance(line.p2, existing),
            ) > distance_tolerance_px:
                continue
            if _projection_gap(existing, line) > max_join_gap_px:
                continue
            merged[index] = _extend(existing, line)
            absorbed = True
            break
        if not absorbed:
            merged.append(line)

    return merged


def estimate_vanishing_point(
    lines: list[DetectedLine],
    *,
    min_support: int = 2,
    inlier_angle_deg: float = 2.5,
) -> VanishingPoint | None:
    """Where a family of pitch-parallel lines meets, by consensus.

    Every pair of lines proposes an intersection; the proposal supported by
    the most lines wins. Support is judged by *angle* — how far a line would
    have to rotate about its midpoint to pass through the candidate point —
    rather than by distance, because a vanishing point is often thousands of
    pixels off-screen where any pixel threshold becomes meaningless.
    """
    if len(lines) < 2:
        return None

    best: VanishingPoint | None = None

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            candidate = _intersect(lines[i], lines[j])
            if candidate is None:
                continue

            residuals = [
                gap
                for gap in (_angle_to_point(line, candidate) for line in lines)
                if gap is not None and gap <= inlier_angle_deg
            ]
            support = len(residuals)
            if support < min_support:
                continue

            mean_residual = float(np.mean(residuals)) if residuals else inlier_angle_deg
            # Agreement matters more than precision here: three lines that
            # roughly concur are better evidence than two that concur exactly,
            # since two lines always meet somewhere.
            confidence = float(
                np.clip(support / max(len(lines), 1), 0.0, 1.0)
                * np.clip(1.0 - mean_residual / max(inlier_angle_deg, 1e-6), 0.0, 1.0)
            )

            if best is None or (support, -mean_residual) > (best.support, -best.residual_deg):
                best = VanishingPoint(
                    xy=candidate,
                    support=support,
                    residual_deg=mean_residual,
                    confidence=confidence,
                )

    return best


def split_by_orientation(
    lines: list[DetectedLine], steep_threshold_deg: float = 35.0
) -> tuple[list[DetectedLine], list[DetectedLine]]:
    """Separate the two families a pitch shows: (steep, shallow).

    On a standard broadcast camera the lines parallel to the goal line
    (goal line, 16.5m line, six-yard line, halfway line) appear steep, while
    the touchlines and penalty-area sides appear shallow. That is a property
    of where the camera sits, not a law — `calibrator.py` treats the mapping
    from "steep" to "parallel to the goal line" as an assumption to state and
    let the operator check, never as a fact.
    """
    steep, shallow = [], []
    for line in lines:
        angle = line.angle_deg
        deviation = min(angle, 180.0 - angle)  # 0 = horizontal, 90 = vertical
        (steep if deviation >= steep_threshold_deg else shallow).append(line)
    return steep, shallow


# -- internals --------------------------------------------------------------


def _intersect(a: DetectedLine, b: DetectedLine) -> Point | None:
    point = np.cross(a.homogeneous(), b.homogeneous())
    if abs(point[2]) < 1e-9:
        return None  # parallel in the image: no finite meeting point
    return (float(point[0] / point[2]), float(point[1] / point[2]))


def _angle_to_point(line: DetectedLine, point: Point) -> float | None:
    """How far the line would rotate about its midpoint to hit `point`."""
    mid = line.midpoint
    dx, dy = point[0] - mid[0], point[1] - mid[1]
    if math.hypot(dx, dy) < 1e-6:
        return None
    return _angular_gap(line.angle_deg, math.degrees(math.atan2(dy, dx)) % 180.0)


def _projection_gap(a: DetectedLine, b: DetectedLine) -> float:
    """Distance between two segments measured *along* `a`'s direction.

    Zero when their spans overlap; otherwise the size of the hole between
    them. This is what keeps two distant stretches of collinear paint from
    being fused into one implausible line.
    """
    ax, ay = a.p1
    dx, dy = a.p2[0] - ax, a.p2[1] - ay
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return float("inf")
    ux, uy = dx / length, dy / length

    def project(point: Point) -> float:
        return (point[0] - ax) * ux + (point[1] - ay) * uy

    a_start, a_end = 0.0, length
    b_values = sorted((project(b.p1), project(b.p2)))
    return max(0.0, b_values[0] - a_end, a_start - b_values[1])


def _angular_gap(a: float, b: float) -> float:
    gap = abs(a - b) % 180.0
    return min(gap, 180.0 - gap)


def _point_to_line_distance(point: Point, line: DetectedLine) -> float:
    a, b, c = line.homogeneous()
    denominator = math.hypot(a, b)
    if denominator < 1e-9:
        return float("inf")
    return abs(a * point[0] + b * point[1] + c) / denominator


def _extend(a: DetectedLine, b: DetectedLine) -> DetectedLine:
    """The longest span across both segments, keeping them one line."""
    points = [a.p1, a.p2, b.p1, b.p2]
    best_pair, best_distance = (a.p1, a.p2), -1.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = math.hypot(
                points[j][0] - points[i][0], points[j][1] - points[i][1]
            )
            if distance > best_distance:
                best_pair, best_distance = (points[i], points[j]), distance
    return DetectedLine(best_pair[0], best_pair[1])
