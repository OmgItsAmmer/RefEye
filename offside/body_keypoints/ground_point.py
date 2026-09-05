"""Turning a skeleton into the two points offside geometry actually needs.

Pure functions, no model and no config object: everything they weigh arrives
as an argument, so the policy is unit-testable without a GPU and the tuning
values stay in `AppSettings.offside` where the cross-cutting rules put them.

Two points come out of a pose, and they are not the same point:

  **Ground point** — where the player meets the pitch. A homography maps the
  pitch *plane*, so this is the only point on a player that projects onto the
  top-down map correctly; a box centre floats somewhere above the grass and
  would land metres away after projection.

  **Leading point** — the most advanced legal body part along a direction.
  This is what an offside line is drawn against, and it is usually *not* the
  foot: a striker leaning goalwards is played onside or offside by their
  shoulder or knee.

Both carry a confidence and a reason. A ground point that had to fall back to
the bottom of a bounding box is still returned — dropping the player would
silently corrupt the second-last-defender ranking in M2.5 — but it says so,
loudly, so the decision built on it can be reported as uncertain.
"""

from __future__ import annotations

import math

from offside.body_keypoints.keypoints import (
    ANKLE_KEYPOINTS,
    KNEE_KEYPOINTS,
    SOURCE_ANKLE,
    SOURCE_BBOX_BOTTOM,
    SOURCE_KNEE_PROJECTED,
    GroundPoint,
    Keypoint,
    LeadingPoint,
    PlayerPose,
)

Box = tuple[float, float, float, float]


def estimate_ground_point(
    keypoints: dict[str, Keypoint],
    bbox: Box,
    box_confidence: float,
    *,
    min_keypoint_confidence: float,
    knee_projection_penalty: float,
    bbox_fallback_penalty: float,
) -> GroundPoint:
    """Best available estimate of where this player stands on the pitch.

    The ladder, most to least trustworthy:

    1. **A visible ankle.** With two, the *lower* one in image space wins —
       that is the planted foot. The raised foot of a running player is off
       the pitch plane, and projecting it would place them a stride further
       forward than they really are.
    2. **A visible knee.** The knee gives a horizontal position that survives
       an outstretched arm skewing the box; the bottom of the box gives the
       vertical one. Penalised, because the vertical half is a guess.
    3. **The bottom-centre of the detector box.** No skeleton evidence at all.
       Heavily penalised and clearly labelled.
    """
    x1, _, x2, y2 = bbox

    ankles = _confident(keypoints, ANKLE_KEYPOINTS, min_keypoint_confidence)
    if ankles:
        # Larger y is lower on screen: the foot in contact with the grass.
        planted = max(ankles, key=lambda kp: kp.y)
        side = "left" if planted.name.startswith("left") else "right"
        if len(ankles) == 2:
            reason = f"measured from the {side} ankle — the lower, planted foot"
        else:
            reason = (
                f"measured from the {side} ankle; the other foot was not "
                "visible, so a stride mid-air could shift this slightly"
            )
        return GroundPoint(
            xy=planted.xy,
            confidence=planted.confidence,
            source=SOURCE_ANKLE,
            reason=reason,
        )

    knees = _confident(keypoints, KNEE_KEYPOINTS, min_keypoint_confidence)
    if knees:
        lower_knee = max(knees, key=lambda kp: kp.y)
        return GroundPoint(
            xy=(lower_knee.x, y2),
            confidence=lower_knee.confidence * knee_projection_penalty,
            source=SOURCE_KNEE_PROJECTED,
            reason=(
                "no ankle was visible — foot position estimated from the knee "
                "and the bottom of the player box"
            ),
        )

    return GroundPoint(
        xy=((x1 + x2) / 2.0, y2),
        confidence=box_confidence * bbox_fallback_penalty,
        source=SOURCE_BBOX_BOTTOM,
        reason=(
            "no body keypoints were confident enough — using the bottom-centre "
            "of the player box, which can be off by a stride"
        ),
    )


def leading_offside_point(
    pose: PlayerPose,
    direction_xy: tuple[float, float],
    *,
    min_keypoint_confidence: float,
) -> LeadingPoint:
    """The most advanced body part along `direction_xy` that offside may be
    measured from (consumed by M2.5, which supplies the direction once the
    pitch is calibrated).

    Arms and hands are excluded by `PlayerPose.offside_surface_points`, not
    here — a reaching player's wrist is frequently their furthest-forward
    keypoint, and measuring off it would call a legal attacker offside.
    """
    dx, dy = direction_xy
    magnitude = math.hypot(dx, dy)
    if magnitude < 1e-9:
        raise ValueError("direction_xy must be a non-zero direction vector")
    unit = (dx / magnitude, dy / magnitude)

    candidates = pose.offside_surface_points(min_keypoint_confidence)
    if candidates:
        leading = max(candidates, key=lambda kp: kp.x * unit[0] + kp.y * unit[1])
        return LeadingPoint(
            keypoint_name=leading.name,
            xy=leading.xy,
            confidence=leading.confidence,
            reason=(
                f"most advanced legal body part is the "
                f"{leading.name.replace('_', ' ')}"
            ),
        )

    ground = pose.ground_point
    return LeadingPoint(
        keypoint_name=ground.source,
        xy=ground.xy,
        confidence=ground.confidence,
        reason=(
            "no body keypoint was confident enough to measure from — falling "
            "back to the estimated foot position, which ignores a leaning "
            "torso or stretched leg"
        ),
        is_fallback=True,
    )


def _confident(
    keypoints: dict[str, Keypoint],
    names: tuple[str, ...],
    min_confidence: float,
) -> list[Keypoint]:
    return [
        kp
        for kp in (keypoints.get(name) for name in names)
        if kp is not None and kp.confidence >= min_confidence
    ]
