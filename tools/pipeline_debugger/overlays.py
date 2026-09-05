"""Drawing pipeline output onto frames, and the top-down pitch map.

Colour is meaning here, not decoration, and one rule runs through all of it:
**measured things are green, inferred things are orange, guessed things are
red.** A debug view that renders a box-bottom guess identically to a measured
ankle would hide exactly the failure the operator is looking for.
"""

from __future__ import annotations

import cv2
import numpy as np

from offside.body_keypoints.keypoints import (
    COCO_KEYPOINT_NAMES,
    SOURCE_ANKLE,
    SOURCE_KNEE_PROJECTED,
    PlayerPose,
)
from offside.pitch_calibration.calibrator import PitchCalibration
from offside.team_assignment.teams import PlayerRole, PlayerTeam, TeamAssignment

# BGR, since these are drawn with OpenCV.
MEASURED = (80, 220, 80)
INFERRED = (0, 165, 255)
GUESSED = (60, 60, 235)
NEUTRAL = (170, 170, 170)
BALL_COLOR = (0, 230, 255)
LINE_FAMILY_COLORS = [(255, 160, 40), (200, 80, 255), (80, 230, 230), (160, 160, 160)]
MARKED = (255, 255, 255)

#: COCO skeleton, by keypoint index.
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_detections(image: np.ndarray, analysis) -> None:
    for detection in analysis.players:
        x1, y1, x2, y2 = (int(v) for v in detection.bbox_xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), NEUTRAL, 1)

    ball = analysis.ball
    if ball is not None:
        x1, y1, x2, y2 = (int(v) for v in ball.bbox_xyxy)
        centre = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.circle(image, centre, max(6, (x2 - x1) // 2 + 3), BALL_COLOR, 2)


def draw_poses(image: np.ndarray, poses: list[PlayerPose], min_confidence: float) -> None:
    for pose in poses:
        for start, end in SKELETON:
            a = pose.keypoint(COCO_KEYPOINT_NAMES[start])
            b = pose.keypoint(COCO_KEYPOINT_NAMES[end])
            if a is None or b is None:
                continue
            if a.confidence < min_confidence or b.confidence < min_confidence:
                continue
            cv2.line(
                image,
                (int(a.x), int(a.y)),
                (int(b.x), int(b.y)),
                (0, 210, 255),
                1,
                cv2.LINE_AA,
            )

        for keypoint in pose.confident_keypoints(min_confidence).values():
            cv2.circle(image, (int(keypoint.x), int(keypoint.y)), 2, (0, 235, 255), -1)


def draw_ground_points(image: np.ndarray, poses: list[PlayerPose]) -> None:
    for pose in poses:
        ground = pose.ground_point
        colour = {
            SOURCE_ANKLE: MEASURED,
            SOURCE_KNEE_PROJECTED: INFERRED,
        }.get(ground.source, GUESSED)
        cv2.drawMarker(
            image,
            (int(ground.xy[0]), int(ground.xy[1])),
            colour,
            cv2.MARKER_CROSS,
            11,
            2,
        )


def draw_team_assignment(
    image: np.ndarray, poses: list[PlayerPose], teams: TeamAssignment | None
) -> None:
    """Each player's box painted in their measured kit colour, plus a label.

    Painting the box in the colour the clustering actually measured — rather
    than in an arbitrary "team A is cyan" — makes a mis-measurement visible
    at a glance: a box whose colour does not match the shirt inside it is the
    failure, shown directly instead of inferred from a confidence number.
    """
    if teams is None:
        return

    for player in teams.players:
        if player.index >= len(poses):
            continue
        x1, y1, x2, y2 = (int(v) for v in poses[player.index].bbox_xyxy)
        swatch = _team_swatch(teams, player)
        cv2.rectangle(image, (x1, y1), (x2, y2), swatch, 2)

        label = _team_label(player)
        # The label follows the house rule (green measured, orange inferred,
        # red guessed) so the *certainty* is readable even where two kits
        # happen to be similar colours.
        if player.is_operator_set:
            colour = MARKED
        elif player.team_id is None:
            colour = GUESSED
        elif player.confidence >= 0.6:
            colour = MEASURED
        else:
            colour = INFERRED

        cv2.putText(
            image,
            label,
            (x1, max(10, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            colour,
            1,
            cv2.LINE_AA,
        )


def _team_label(player: PlayerTeam) -> str:
    if player.role is PlayerRole.GOALKEEPER:
        base = "GK"
    elif player.team_id is None:
        base = "?"
    else:
        base = player.team_id[-1].upper()
    return base + ("*" if player.is_operator_set else "")


def _team_swatch(teams: TeamAssignment, player: PlayerTeam) -> tuple[int, int, int]:
    if player.team_id is None or teams.color_model is None:
        return NEUTRAL
    swatch = teams.color_model.swatches.get(player.team_id, NEUTRAL)
    # Very dark kits would draw an invisible box on a dark frame; lift them
    # just enough to be seen without pretending the kit is a lighter colour.
    if max(swatch) < 70:
        return tuple(int(min(255, v + 70)) for v in swatch)
    return tuple(int(v) for v in swatch)


def draw_line_mask(image: np.ndarray, mask: np.ndarray) -> None:
    """Tint what the line finder actually sees — most calibration failures
    are obvious the moment this is visible."""
    if mask is None:
        return
    tint = np.zeros_like(image)
    tint[mask > 0] = (255, 255, 255)
    cv2.addWeighted(image, 1.0, tint, 0.55, 0.0, dst=image)


def draw_pitch_lines(image: np.ndarray, calibration: PitchCalibration | None) -> None:
    if calibration is None:
        return

    # Colour by family, so "these lines are parallel on the pitch" is visible
    # — that grouping is what a vanishing point is computed from.
    family_of: dict[int, int] = {}
    for index, family in enumerate(calibration.families):
        for line in family.lines:
            family_of[id(line)] = index

    for line in calibration.detected_lines:
        index = family_of.get(id(line))
        colour = (
            LINE_FAMILY_COLORS[index % len(LINE_FAMILY_COLORS)]
            if index is not None
            else (90, 90, 90)
        )
        cv2.line(
            image,
            (int(line.p1[0]), int(line.p1[1])),
            (int(line.p2[0]), int(line.p2[1])),
            colour,
            2,
            cv2.LINE_AA,
        )


def draw_offside_direction(
    image: np.ndarray, calibration: PitchCalibration | None, poses: list[PlayerPose]
) -> None:
    """A goal-line-parallel line through each player's feet.

    This is the single most useful thing to eyeball: if these lines do not
    look parallel to the real goal line, the calibration is wrong, and every
    verdict M2.5 would build on it is wrong too.
    """
    if calibration is None or not calibration.can_draw_offside_line:
        return

    height, width = image.shape[:2]
    span = float(max(width, height))

    for pose in poses:
        result = calibration.offside_line_through(pose.ground_point.xy)
        if result is None:
            continue
        (x, y), (dx, dy) = result
        colour = MEASURED if pose.ground_point.is_measured else GUESSED
        cv2.line(
            image,
            (int(x - dx * span), int(y - dy * span)),
            (int(x + dx * span), int(y + dy * span)),
            colour,
            1,
            cv2.LINE_AA,
        )


def draw_marked_landmarks(image: np.ndarray, calibration_points) -> None:
    for point in calibration_points:
        x, y = int(point.image_xy[0]), int(point.image_xy[1])
        cv2.drawMarker(image, (x, y), MARKED, cv2.MARKER_TILTED_CROSS, 16, 2)
        cv2.circle(image, (x, y), 9, MARKED, 1)
        if point.landmark:
            cv2.putText(
                image,
                point.landmark.replace("_", " "),
                (x + 12, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                MARKED,
                1,
                cv2.LINE_AA,
            )


def render_top_down(
    pitch,
    calibration: PitchCalibration | None,
    poses: list[PlayerPose],
    teams: TeamAssignment | None = None,
    size: tuple[int, int] = (760, 520),
) -> np.ndarray:
    """The pitch from above, with players projected onto it.

    Only possible with a metric calibration — with anything less this returns
    a panel saying so, rather than an empty pitch that looks like a pitch with
    nobody on it.
    """
    width, height = size
    canvas = np.full((height, width, 3), 26, dtype=np.uint8)

    margin = 40
    scale = min(
        (width - 2 * margin) / (pitch.length + 4),
        (height - 2 * margin) / (pitch.width + 4),
    )
    offset_x = (width - pitch.length * scale) / 2
    offset_y = (height - pitch.width * scale) / 2

    def to_canvas(point) -> tuple[int, int]:
        return (int(offset_x + point[0] * scale), int(offset_y + point[1] * scale))

    for start, end in pitch.lines():
        cv2.line(canvas, to_canvas(start), to_canvas(end), (70, 110, 70), 1, cv2.LINE_AA)

    if calibration is None or not calibration.is_metric:
        cv2.putText(
            canvas,
            "Metric calibration required",
            (margin, height // 2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Mark 4+ pitch landmarks to place players here",
            (margin, height // 2 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (110, 110, 110),
            1,
            cv2.LINE_AA,
        )
        return canvas

    ground_points = [pose.ground_point for pose in poses]
    if ground_points:
        projected = calibration.to_pitch([g.xy for g in ground_points])
        for index, (ground, position) in enumerate(zip(ground_points, projected)):
            if not np.all(np.isfinite(position)):
                continue
            # Off-pitch projections are shown, dimmed, rather than hidden:
            # they are the signature of a bad calibration or a crowd
            # detection, and hiding them hides the problem.
            on_pitch = pitch.contains(tuple(position), margin=5.0)
            certainty = (
                (MEASURED if ground.is_measured else GUESSED) if on_pitch else (70, 70, 70)
            )
            player = teams.by_index(index) if teams is not None else None

            if player is None:
                cv2.circle(canvas, to_canvas(position), 5, certainty, -1 if on_pitch else 1)
                continue

            # Two things at once, deliberately: the fill says which team, the
            # ring says how sure the *foot position* is. Collapsing them into
            # one colour would hide whichever failure was not being looked for.
            fill = _team_swatch(teams, player) if on_pitch else (70, 70, 70)
            centre = to_canvas(position)
            cv2.circle(canvas, centre, 5, fill, -1)
            cv2.circle(canvas, centre, 6, certainty, 1)
            if player.role is PlayerRole.GOALKEEPER:
                cv2.circle(canvas, centre, 9, MARKED, 1)

    if teams is not None and teams.attacking_team_id is not None:
        cv2.putText(
            canvas,
            f"attacking: team {teams.attacking_team_id[-1].upper()}",
            (margin, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

    return canvas
