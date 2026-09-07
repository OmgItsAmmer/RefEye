"""Choosing *the* ball from the detector's raw output (M2 hardening).

The detector's ball threshold is deliberately loose (0.15, versus 0.35 for
players) — missing the real ball is worse than a false positive, because a
missed ball silently breaks the attacking-side call (M2.3) and the
"beyond the ball" rule (M2.5), while a false positive is at least visible on
screen. That trade-off was always intentional. What was missing is this
module: something to clean up *after* it.

Measured on the client's own clips (`data/videos/`), sampling 10 frames each:
more than one "ball" detection appeared on **4 of every 10 frames** — and
there is only ever one real ball, so on those frames at least one detection
is definitely wrong. Sizes reached **31% of the median player's height** on
screen; a real football against a real player is physically **~12-13%**.
Boot studs, ball-shaped ad-board logos, and bright spots on the pitch lines
all pass the confidence threshold and had nothing filtering them out before
they reached the attacking-side and offside-line calculations.

## Why this is a size filter, not a trajectory filter

A natural second idea is "prefer whichever detection continues the ball's
recent path." That needs frames *before* this one, and the offside pipeline
here runs on a single confirmed frame at a time (Phase B is triggered once,
not a continuous loop) — there is no "recent path" available at this layer.
Trajectory smoothing already exists, just one layer up: `ai/contact_refinement/`
uses exactly that idea, across the buffered clip, to find the contact frame
in the first place (Phase A). By the time this module runs, that job is done;
this module's job is narrower — given the one detector output on the one
already-chosen frame, does this candidate even look like a ball.

## The rule, and why "reject" beats "trust anyway"

Every player detection on the same frame is already a size reference — no
extra model, no extra pass. A ball's true size (relative to a player) barely
changes with zoom, because both scale together, so a size band is a solid,
frame-invariant test.

If **no** detection passes the band, this returns `None` rather than falling
back to the least-implausible one. An implausible "ball" is worse than no
ball: everything downstream (M2.3's attacking side, M2.5's "beyond the ball"
check) already has an honest, tested path for "the ball was not found on this
frame" — reusing it here is safer than inventing a new, wrong answer.
"""

from __future__ import annotations

from statistics import median

from core.domain.models import Detection
from vision.detection.classes import BALL

Point = tuple[float, float]


def select_ball(
    detections: list[Detection],
    players: list[Detection],
    *,
    min_size_ratio: float = 0.06,
    max_size_ratio: float = 0.22,
) -> Detection | None:
    """The one detection most likely to be the real ball, or None.

    `players` supplies the size reference (median player box height on this
    same frame) — pass every player detection, not just the ones a later
    stage kept, so the reference doesn't inherit some other stage's filtering.
    """
    balls = [d for d in detections if d.class_name == BALL]
    if not balls:
        return None

    reference = _median_player_height(players)
    if reference is None:
        # No size reference on this frame (e.g. a detector-only smoke test
        # with no players) — can't apply the gate, so fall back to the old
        # behaviour rather than discarding every candidate for a reason that
        # has nothing to do with the candidates themselves.
        return max(balls, key=lambda d: d.confidence)

    plausible = [
        detection
        for detection in balls
        if min_size_ratio * reference <= _size(detection) <= max_size_ratio * reference
    ]
    if not plausible:
        return None
    return max(plausible, key=lambda d: d.confidence)


def _median_player_height(players: list[Detection]) -> float | None:
    heights = [d.bbox_xyxy[3] - d.bbox_xyxy[1] for d in players]
    heights = [h for h in heights if h > 0]
    return median(heights) if heights else None


def _size(detection: Detection) -> float:
    x1, y1, x2, y2 = detection.bbox_xyxy
    return max(x2 - x1, y2 - y1)
