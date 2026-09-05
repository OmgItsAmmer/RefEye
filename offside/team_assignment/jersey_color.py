"""Measuring what colour a player is wearing — the feature teams cluster on.

## Nothing here knows any kit colour

There is no list of teams, no "home is red", no saved palette. Kit colours
are discovered from the footage every time, which is the only way this works
on a match nobody prepared for (M2_Plan section 4). What this module does is
narrower and mechanical: given a player and a frame, produce a number that is
close for two players in the same kit and far apart for two players in
different kits.

## Why the torso, taken from keypoints

The obvious approach — average the player's bounding box — measures mostly
grass. At broadcast framing a player box is maybe 40x100px of which the
shirt is a small central patch, and the rest is pitch, boots, shorts and
whoever is standing behind them. M2.2 already gives shoulders and hips, so
the shirt is sampled from the quadrilateral they define: the one region of a
player that is reliably kit and reliably not skin.

When those keypoints are missing the box fallback is used anyway (a player
who cannot be posed still needs a team, or the defender ranking loses them),
but it says so, and its confidence is lower.

## Why grass and skin are removed, and dark pixels are not

Grass leaks into every crop through gaps between arms and legs and around the
shoulders, and it leaks *the same green* into every player, which pulls all
players toward each other and destroys the very separation the clustering
needs. Skin does the same in a different direction: bare arms and necks are
the same colour on both teams.

Dark pixels are deliberately **kept**. Excluding them is a tempting
"shadow filter" that silently makes black and navy kits unmeasurable — a
whole class of real kits — so instead the summary statistic is the *median*,
which shrugs off a shaded minority without needing to know which pixels were
shaded.

## Why CIELAB and a median

LAB distance approximates how different two colours look to a person, so a
threshold expressed in it means something ("about as different as a trained
eye can just notice") rather than being an arbitrary RGB number. That matters
because the honest failure case here is "the two kits genuinely look alike",
and the measure needs to fail in the same way a human does.

## Swapping this out

`TeamFeatureExtractor` is the seam. If colour proves too weak on some
broadcast (similar kits, heavy floodlight cast), a crop-embedding extractor
drops in behind this protocol and everything downstream — clustering,
outlier trimming, goalkeeper logic — is unchanged, because all of it works on
`JerseyColor.vector` and never on colour as such.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from offside.body_keypoints.keypoints import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    PlayerPose,
)
from offside.team_assignment.teams import JerseyColor

#: Where the sampled patch came from, worst to best.
REGION_BOX: str = "box_estimate"
REGION_PARTIAL: str = "partial_keypoints"
REGION_TORSO: str = "torso_keypoints"

#: Confidence ceiling per region source. A colour taken from a guessed patch
#: can never outrank one taken from a measured torso, for the same reason the
#: ground-point ladder in M2.2 is ordered.
REGION_CONFIDENCE = {
    REGION_TORSO: 1.0,
    REGION_PARTIAL: 0.8,
    REGION_BOX: 0.55,
}


class TeamFeatureExtractor(Protocol):
    """Turns a player into a vector that clusters by kit."""

    @property
    def feature_name(self) -> str: ...

    def extract(self, image: np.ndarray, pose: PlayerPose) -> JerseyColor: ...


class JerseyColorExtractor:
    """Median torso colour in CIELAB, with grass and skin masked out."""

    def __init__(
        self,
        *,
        keypoint_confidence: float = 0.4,
        torso_shrink: float = 0.7,
        fallback_top_fraction: float = 0.18,
        fallback_bottom_fraction: float = 0.45,
        fallback_width_fraction: float = 0.5,
        grass_hue_range: tuple[int, int] = (30, 95),
        grass_min_saturation: int = 40,
        exclude_skin: bool = True,
        min_sample_pixels: int = 20,
        min_kept_fraction: float = 0.35,
    ):
        self._keypoint_confidence = keypoint_confidence
        self._torso_shrink = torso_shrink
        self._fallback_top = fallback_top_fraction
        self._fallback_bottom = fallback_bottom_fraction
        self._fallback_width = fallback_width_fraction
        self._grass_hue_range = grass_hue_range
        self._grass_min_saturation = grass_min_saturation
        self._exclude_skin = exclude_skin
        self._min_sample_pixels = min_sample_pixels
        self._min_kept_fraction = min_kept_fraction

    @property
    def feature_name(self) -> str:
        return "jersey_lab_median"

    def extract(self, image: np.ndarray, pose: PlayerPose) -> JerseyColor:
        polygon, region_source = self._torso_polygon(pose)
        patch, mask = _crop_polygon(image, polygon)
        if patch is None or mask is None or not mask.any():
            return _unmeasured("the shirt area falls outside the frame")

        keep = mask.copy()
        total = int(keep.sum())

        grass = _grass_mask(patch, self._grass_hue_range, self._grass_min_saturation)
        keep &= ~grass
        if self._exclude_skin:
            keep &= ~_skin_mask(patch)

        kept = int(keep.sum())
        if kept < self._min_sample_pixels:
            # Falling back to the unmasked patch would mean reporting grass as
            # a kit colour, which reads as a confident wrong answer rather
            # than a missing one.
            return _unmeasured(
                f"only {kept} shirt pixels survived after removing grass and skin "
                f"(need {self._min_sample_pixels}) — player too small, obscured, "
                "or the crop is mostly pitch"
            )

        lab = _to_lab(patch)
        median = np.median(lab[keep], axis=0)

        kept_fraction = kept / max(1, total)
        confidence = REGION_CONFIDENCE[region_source] * min(
            1.0, kept_fraction / max(1e-6, self._min_kept_fraction)
        )

        reason = (
            f"{kept} shirt pixels from the {region_source.replace('_', ' ')} "
            f"({kept_fraction * 100:.0f}% of the sampled area survived grass "
            "and skin removal)"
        )

        return JerseyColor(
            vector=tuple(float(v) for v in median),
            bgr=_lab_to_bgr(median),
            pixel_count=kept,
            confidence=float(min(1.0, max(0.0, confidence))),
            source=region_source,
            reason=reason,
        )

    # -- where to sample ----------------------------------------------------

    def _torso_polygon(self, pose: PlayerPose) -> tuple[np.ndarray, str]:
        """The quadrilateral between shoulders and hips, or a box estimate."""
        points = {
            name: pose.keypoint(name)
            for name in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
        }
        confident = {
            name: kp.xy
            for name, kp in points.items()
            if kp is not None and kp.confidence >= self._keypoint_confidence
        }

        shoulders = [confident[n] for n in (LEFT_SHOULDER, RIGHT_SHOULDER) if n in confident]
        hips = [confident[n] for n in (LEFT_HIP, RIGHT_HIP) if n in confident]

        if len(shoulders) == 2 and len(hips) == 2:
            quad = np.array(
                [confident[LEFT_SHOULDER], confident[RIGHT_SHOULDER],
                 confident[RIGHT_HIP], confident[LEFT_HIP]],
                dtype=np.float64,
            )
            return _shrink(quad, self._torso_shrink), REGION_TORSO

        if shoulders and hips:
            # One side of the torso is hidden (very common when players
            # overlap). A band between the visible shoulder and hip heights,
            # centred on the box, is still mostly shirt.
            x1, _, x2, _ = pose.bbox_xyxy
            top = min(p[1] for p in shoulders)
            bottom = max(p[1] for p in hips)
            quad = _band(x1, x2, top, bottom, self._fallback_width)
            return _shrink(quad, self._torso_shrink), REGION_PARTIAL

        x1, y1, x2, y2 = pose.bbox_xyxy
        height = y2 - y1
        quad = _band(
            x1,
            x2,
            y1 + height * self._fallback_top,
            y1 + height * self._fallback_bottom,
            self._fallback_width,
        )
        return quad, REGION_BOX


# -- pixel helpers ----------------------------------------------------------


def _band(x1: float, x2: float, top: float, bottom: float, width_fraction: float) -> np.ndarray:
    centre = (x1 + x2) / 2.0
    half = (x2 - x1) * width_fraction / 2.0
    return np.array(
        [
            (centre - half, top),
            (centre + half, top),
            (centre + half, bottom),
            (centre - half, bottom),
        ],
        dtype=np.float64,
    )


def _shrink(polygon: np.ndarray, factor: float) -> np.ndarray:
    """Pull a polygon toward its own centre, away from its edges.

    The border of a torso quad is where the shirt meets sleeve, neck, shorts
    and background; sampling it dilutes exactly the signal being measured.
    """
    centre = polygon.mean(axis=0)
    return centre + (polygon - centre) * factor


def _crop_polygon(
    image: np.ndarray, polygon: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    height, width = image.shape[:2]
    x1 = int(np.floor(polygon[:, 0].min()))
    y1 = int(np.floor(polygon[:, 1].min()))
    x2 = int(np.ceil(polygon[:, 0].max()))
    y2 = int(np.ceil(polygon[:, 1].max()))

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None, None

    patch = image[y1:y2, x1:x2]
    mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    shifted = np.round(polygon - np.array([x1, y1])).astype(np.int32)
    cv2.fillConvexPoly(mask, shifted, 1)
    return patch, mask.astype(bool)


def _grass_mask(patch: np.ndarray, hue_range: tuple[int, int], min_saturation: int) -> np.ndarray:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, saturation = hsv[:, :, 0], hsv[:, :, 1]
    return (hue >= hue_range[0]) & (hue <= hue_range[1]) & (saturation >= min_saturation)


def _skin_mask(patch: np.ndarray) -> np.ndarray:
    """The standard YCrCb skin window.

    Chrominance-only, so it holds across skin tones and lighting far better
    than an RGB rule: what varies between people is mostly luma.
    """
    ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)


def _to_lab(patch: np.ndarray) -> np.ndarray:
    """OpenCV's 8-bit LAB, rescaled to real L*a*b* so distances are dE76."""
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float64)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def _lab_to_bgr(lab: np.ndarray) -> tuple[int, int, int]:
    encoded = np.array(
        [[[lab[0] * 255.0 / 100.0, lab[1] + 128.0, lab[2] + 128.0]]], dtype=np.float64
    )
    bgr = cv2.cvtColor(np.clip(encoded, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return tuple(int(v) for v in bgr[0, 0])


def _unmeasured(reason: str) -> JerseyColor:
    return JerseyColor(
        vector=(),
        bgr=(0, 0, 0),
        pixel_count=0,
        confidence=0.0,
        source=REGION_BOX,
        reason=reason,
    )
