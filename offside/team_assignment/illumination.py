"""Removing the lighting from a shirt colour, using the pitch as a grey card.

## The problem this solves

The same shirt measures as a different colour in floodlight and in daylight,
in sun and in the shadow of a stand, and under two broadcasters' colour
grading. That is not a small effect: on a half-shadowed pitch the *same team*
can split into two colour groups, which is precisely the failure that makes
clustering assign teammates to opposite sides.

## Why the grass, and not the usual white-balance tricks

The standard illuminant estimators assume the average of a scene is grey
(grey-world) or that its brightest pixel is white. Both are wrong here in the
same direction: a football frame is overwhelmingly green, so grey-world
"corrects" the grass away and drags every colour magenta.

But that same fact is an opportunity. **A football pitch is a giant reference
card of known colour.** Whatever the grass measures as in this frame *is* the
illuminant, near enough, and mapping it back to a canonical green removes the
lighting from everything else in the frame — including the shirts.

## Why the reference is taken locally, per player

A global correction fixes floodlights versus daylight, but not the case that
matters most: half the pitch in sun, half in the shadow of the stand, two
players in the same kit twenty metres apart. So the reference is sampled from
the grass immediately *around each player* where there is enough of it, and
falls back to the whole frame otherwise. A player standing in shade is then
measured against shaded grass, and comes out the same colour as their
teammate in the sun.

Gains are clamped: an extreme correction means the reference was not really
grass (a player on the touchline against advertising boards, say), and a wild
gain would invent a colour rather than reveal one.
"""

from __future__ import annotations

import cv2
import numpy as np

#: What grass is taken to look like under neutral light, in BGR. The exact
#: value does not matter — every player in the clip is mapped to the same
#: reference, so it cancels out of the comparisons that decide teams. It only
#: has to be a plausible green, so corrected colours stay recognisable to a
#: human looking at the debug view.
CANONICAL_GRASS: tuple[float, float, float] = (70.0, 130.0, 75.0)

#: Below this many grass pixels a reference is not trustworthy.
MIN_REFERENCE_PIXELS = 60


def grass_gains(
    image: np.ndarray,
    *,
    region: tuple[int, int, int, int] | None = None,
    hue_range: tuple[int, int] = (30, 95),
    min_saturation: int = 40,
    max_gain: float = 2.0,
) -> tuple[float, float, float] | None:
    """Per-channel gains that map this region's grass onto canonical grass.

    Returns None when the region does not contain enough grass to be a
    reference — the caller should fall back rather than invent a correction.
    """
    patch = image
    if region is not None:
        x1, y1, x2, y2 = region
        height, width = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        patch = image[y1:y2, x1:x2]

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    grass = (
        (hue >= hue_range[0])
        & (hue <= hue_range[1])
        & (saturation >= min_saturation)
        & (value > 20)  # deep shadow carries no reliable colour at all
    )
    if int(grass.sum()) < MIN_REFERENCE_PIXELS:
        return None

    reference = np.median(patch[grass], axis=0).astype(np.float64)
    if np.any(reference < 5.0):
        return None

    gains = np.array(CANONICAL_GRASS, dtype=np.float64) / reference
    if np.any(gains > max_gain) or np.any(gains < 1.0 / max_gain):
        # Not grass, most likely: an advertising hoarding, a shaded stand, or
        # a crop that is mostly player. A wild gain would invent a colour.
        return None
    return tuple(float(g) for g in gains)


def apply_gains(patch: np.ndarray, gains: tuple[float, float, float] | None) -> np.ndarray:
    """Apply per-channel gains, saturating rather than wrapping."""
    if gains is None:
        return patch
    corrected = patch.astype(np.float64) * np.array(gains, dtype=np.float64)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def local_grass_ring(
    bbox_xyxy: tuple[float, float, float, float], *, margin: float = 1.5
) -> tuple[int, int, int, int]:
    """A box around a player, wide enough to contain the pitch beside them.

    Deliberately generous horizontally: what surrounds a standing player at
    broadcast framing is pitch to the left and right, while directly above
    them is often the crowd or a hoarding.
    """
    x1, y1, x2, y2 = bbox_xyxy
    width, height = x2 - x1, y2 - y1
    return (
        int(x1 - width * margin),
        int(y1 + height * 0.35),
        int(x2 + width * margin),
        int(y2 + height * 0.35),
    )
