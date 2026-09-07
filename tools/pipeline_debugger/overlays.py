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
from offside.offside_line.rendering import ascii_safe, draw_offside_overlay
from offside.pitch_calibration.calibrator import PitchCalibration

# Re-exported at module level for callers that reach these through
# `overlays.<name>` (app.py, and the tests that drive it) rather than
# importing them directly — hence the noqa: ruff can't see that usage.
from offside.pitch_calibration.rendering import (  # noqa: F401
    canvas_to_pitch,
    draw_marked_landmarks,
    render_top_down,
    team_swatch,
    top_down_transform,
)
from offside.team_assignment.teams import PlayerRole, PlayerTeam, TeamAssignment

# BGR, since these are drawn with OpenCV.
MEASURED = (80, 220, 80)
INFERRED = (0, 165, 255)
GUESSED = (60, 60, 235)
NEUTRAL = (170, 170, 170)
BALL_COLOR = (0, 230, 255)
LINE_FAMILY_COLORS = [(255, 160, 40), (200, 80, 255), (80, 230, 230), (160, 160, 160)]
MARKED = (255, 255, 255)
#: The player picked in the team grid, so the two halves of the window agree.
SELECTED = (255, 154, 76)

#: Identity states (M2.4), on the same measured/inferred/guessed scale: a
#: confirmed identity is a measurement, a recovered one is an inference, and a
#: contested one is a guess the operator is being warned not to rely on.
IDENTITY_COLORS = {
    "confirmed": MEASURED,
    "recovered": INFERRED,
    "tentative": (200, 200, 90),
    "contested": GUESSED,
}

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
        swatch = team_swatch(teams, player)
        cv2.rectangle(image, (x1, y1), (x2, y2), swatch, 2)

        if player.needs_confirmation:
            # The operator's whole job on this stage is "look at the flagged
            # ones". A dashed white surround says which without hiding the
            # measured kit colour underneath it.
            for offset in range(x1, x2, 8):
                cv2.line(image, (offset, y1 - 3), (min(offset + 4, x2), y1 - 3), MARKED, 1)
                cv2.line(image, (offset, y2 + 3), (min(offset + 4, x2), y2 + 3), MARKED, 1)

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
            ascii_safe(label),
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


def draw_identities(image: np.ndarray, poses: list[PlayerPose], identities) -> None:
    """Each player's id, how well they are being followed, and where they came from.

    The trail is the point of this overlay. A confidence number saying "this
    identity is solid" is an assertion; a path that visibly follows one player
    across the frame is evidence, and a path that jumps sideways onto somebody
    else is the swap this phase exists to prevent, made visible.
    """
    if identities is None:
        return

    for identity in identities.identities:
        if identity.index >= len(poses):
            continue
        colour = IDENTITY_COLORS.get(identity.state.value, NEUTRAL)

        points = identity.trail[-24:]
        for start, end in zip(points, points[1:]):
            cv2.line(
                image,
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                colour,
                1,
                cv2.LINE_AA,
            )

        x1, y1, x2, y2 = (int(v) for v in poses[identity.index].bbox_xyxy)
        cv2.putText(
            image,
            identity.track_id,
            (x1, min(image.shape[0] - 4, y2 + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            colour,
            1,
            cv2.LINE_AA,
        )


def highlight_player(image: np.ndarray, poses: list[PlayerPose], index: int | None) -> None:
    """Ring the player selected in the team grid.

    The link between the two halves of the window: picking a swatch on the
    right must point at somebody on the left, or the operator is being asked
    to judge a colour with no idea whose it is.
    """
    if index is None or not (0 <= index < len(poses)):
        return
    x1, y1, x2, y2 = (int(v) for v in poses[index].bbox_xyxy)
    cv2.rectangle(image, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), SELECTED, 2)
    cv2.putText(
        image,
        f"#{index}",
        (x1 - 4, max(12, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        SELECTED,
        1,
        cv2.LINE_AA,
    )


def draw_offside_line(image: np.ndarray, decision, explanation=None) -> None:
    """The line, the two players it is measured between, and the verdict.

    The drawing itself lives in `offside/offside_line/rendering.py` so that the
    inspector and the operator's review screen cannot drift apart: an overlay
    that disagreed between the debug view and the shipped one would be worse
    than no overlay at all.
    """
    draw_offside_overlay(image, decision, explanation)


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
