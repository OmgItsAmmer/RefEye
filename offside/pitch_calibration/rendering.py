"""The top-down pitch map: what to click, what has been clicked, and where
everybody actually is (M2.1, surfaced in the review screen as of M2.7).

This lives in the product, not in the debug tool, for the same reason
`offside/offside_line/rendering.py` does: two surfaces draw it — the
operator's review screen and the pipeline inspector — and a map that
disagreed between them would be worse than no map at all. The inspector
imports this; it does not keep a second copy.

## Why the map is not a placeholder before calibration

Asking an operator to mark "centre_mark" in a video, with no picture of
where that point is on a pitch, is not a question anybody can answer. So the
landmark being asked for — or, once auto-calibration exists, whatever the
pipeline actually used — is drawn on a diagram of the pitch at the position
it occupies, not hidden behind a "not calibrated yet" message.

## Colour is meaning, not decoration

The same measured/inferred/guessed vocabulary used for every other overlay
in this system holds here: a marked landmark is green (a measurement the
operator supplied), a projected player is ringed in green when their foot
position was measured and red when it was guessed, and off-pitch projections
are shown dimmed rather than hidden — they are the signature of a bad
calibration, and hiding them hides the problem.
"""

from __future__ import annotations

import cv2
import numpy as np

from offside.offside_line.rendering import ascii_safe
from offside.pitch_calibration.calibrator import PitchCalibration
from offside.team_assignment.teams import PlayerRole, PlayerTeam, TeamAssignment

# BGR, since these are drawn with OpenCV. Kept in sync with the same names in
# tools/pipeline_debugger/overlays.py, which imports them from here.
MEASURED = (80, 220, 80)
GUESSED = (60, 60, 235)
NEUTRAL = (170, 170, 170)
MARKED = (255, 255, 255)
#: The landmark being asked for, or the player picked in the team grid.
SELECTED = (255, 154, 76)


def team_swatch(teams: TeamAssignment, player: PlayerTeam) -> tuple[int, int, int]:
    if player.team_id is None or teams.color_model is None:
        return NEUTRAL
    swatch = teams.color_model.swatches.get(player.team_id, NEUTRAL)
    # Very dark kits would draw an invisible dot on a dark canvas; lift them
    # just enough to be seen without pretending the kit is a lighter colour.
    if max(swatch) < 70:
        return tuple(int(min(255, v + 70)) for v in swatch)
    return tuple(int(v) for v in swatch)


def wrap_text(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def top_down_transform(pitch, size: tuple[int, int] = (760, 520)):
    """Metres-to-canvas mapping, shared by drawing and by hit-testing clicks.

    One function so the dot an operator sees and the dot they click are the
    same dot — two copies of this arithmetic would drift and the map would
    quietly stop selecting what it displays.
    """
    width, height = size
    margin = 40
    scale = min(
        (width - 2 * margin) / (pitch.length + 4),
        (height - 2 * margin) / (pitch.width + 4),
    )
    offset_x = (width - pitch.length * scale) / 2
    offset_y = (height - pitch.width * scale) / 2
    return scale, offset_x, offset_y


def canvas_to_pitch(pitch, point, size: tuple[int, int] = (760, 520)):
    """A click on the top-down map, in metres."""
    scale, offset_x, offset_y = top_down_transform(pitch, size)
    return ((point[0] - offset_x) / scale, (point[1] - offset_y) / scale)


def draw_marked_landmarks(image: np.ndarray, calibration_points) -> None:
    """The operator's clicks, on the *video frame* (not the top-down map)."""
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


def _draw_landmarks(
    canvas,
    pitch,
    to_canvas,
    highlight: str | None,
    marked: dict,
    highlight_label: str | None = None,
) -> None:
    """Every landmark on the map: available, already marked, or being asked for."""
    for name, position in pitch.landmarks().items():
        point = to_canvas(position)
        if name in marked:
            # Green because it is a measurement the operator supplied, on the
            # same scale as everything else in this window.
            cv2.circle(canvas, point, 5, MEASURED, -1)
            cv2.putText(
                canvas,
                name.replace("_", " "),
                (point[0] + 8, point[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                MEASURED,
                1,
                cv2.LINE_AA,
            )
        elif name == highlight:
            # Deliberately loud. This is the one thing on the screen the
            # operator is being asked to act on.
            cv2.circle(canvas, point, 16, SELECTED, 1)
            cv2.circle(canvas, point, 9, SELECTED, 2)
            cv2.drawMarker(canvas, point, SELECTED, cv2.MARKER_CROSS, 22, 2)
            cv2.putText(
                canvas,
                ascii_safe(highlight_label or name.replace("_", " ")),
                (point[0] + 20, point[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                SELECTED,
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(canvas, point, 2, (95, 95, 95), -1)


def render_top_down(
    pitch,
    calibration: PitchCalibration | None,
    poses,
    teams: TeamAssignment | None = None,
    size: tuple[int, int] = (760, 520),
    *,
    highlight_landmark: str | None = None,
    highlight_label: str | None = None,
    highlight_description: str | None = None,
    marked_landmarks: dict | None = None,
    interactive: bool = True,
) -> np.ndarray:
    """The pitch from above: what to click, what has been clicked, and — once
    the calibration is metric — where everybody actually is.

    Before calibration this map is not a placeholder, it is the *instruction*.
    Asking an operator to click "centre_mark" in a video without showing them
    which point that is on a pitch is not a question anybody can answer, so
    the landmark being asked for is drawn here, on a diagram of the pitch, at
    the position it actually occupies.

    `interactive` is False for the review screen (M2.7), which has no
    click-to-mark flow — only the Pipeline Inspector does. Telling an
    operator to "click these 4 points" on a screen with no click handler for
    them would be actively misleading, not just unhelpful, so the read-only
    caller gets a status line instead of an instruction.
    """
    width, height = size
    canvas = np.full((height, width, 3), 26, dtype=np.uint8)

    scale, offset_x, offset_y = top_down_transform(pitch, size)

    def to_canvas(point) -> tuple[int, int]:
        return (int(offset_x + point[0] * scale), int(offset_y + point[1] * scale))

    for start, end in pitch.lines():
        cv2.line(canvas, to_canvas(start), to_canvas(end), (70, 110, 70), 1, cv2.LINE_AA)

    marked = marked_landmarks or {}
    _draw_landmarks(
        canvas, pitch, to_canvas, highlight_landmark, marked, highlight_label
    )

    if calibration is None or not calibration.is_metric:
        remaining = max(0, 4 - len(marked))
        if interactive:
            headline = (
                f"{len(marked)} of 4 marked - {remaining} more to go"
                if marked
                else "Click these 4 points in the video to map the pitch"
            )
        elif marked:
            headline = f"{len(marked)} of 4 landmarks marked"
        else:
            headline = (
                "Calibrated automatically - no landmarks marked"
                if calibration is not None
                else "Not calibrated on this frame"
            )
        cv2.putText(
            canvas,
            ascii_safe(headline),
            (16, height - 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        if highlight_landmark and interactive:
            cv2.putText(
                canvas,
                ascii_safe(
                    "click this one: "
                    + (highlight_label or highlight_landmark.replace("_", " "))
                ),
                (16, height - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                SELECTED,
                1,
                cv2.LINE_AA,
            )
        if highlight_description:
            # The description belongs on the map, not only in the panel: the
            # map is where the operator is looking while hunting for the point.
            for offset, line in enumerate(wrap_text(highlight_description, 62)):
                cv2.putText(
                    canvas,
                    ascii_safe(line),
                    (16, 22 + offset * 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (185, 185, 185),
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
            fill = team_swatch(teams, player) if on_pitch else (70, 70, 70)
            centre = to_canvas(position)
            cv2.circle(canvas, centre, 5, fill, -1)
            cv2.circle(canvas, centre, 6, certainty, 1)
            if player.role is PlayerRole.GOALKEEPER:
                cv2.circle(canvas, centre, 9, MARKED, 1)

    if teams is not None and teams.attacking_team_id is not None:
        cv2.putText(
            canvas,
            f"attacking: team {teams.attacking_team_id[-1].upper()}",
            (16, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

    return canvas
