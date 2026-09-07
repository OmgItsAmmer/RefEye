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

## Why two colours per player, not one

**A single average colour is wrong for half the kits in football.** Stripes
and hoops — Inter, Barcelona, Juventus, QPR — average out to a muddy blend
that is unstable frame to frame, because it depends on how much of each
stripe happened to be visible. Worse, a red-and-blue striped kit can average
to something a solid purple kit would match exactly.

So a player's signature is the **two dominant colours of their torso**,
ordered by lightness so the pairing is stable. A solid kit produces two
near-identical colours and behaves exactly like the old single measurement; a
striped kit produces the two stripe colours. Distances stay in the same
units, because the two halves are scaled so that the combined distance is the
RMS of the per-colour distances.

A second cluster that is either tiny or barely different from the first is
collapsed away: sponsor text, a sleeve, and compression noise all produce a
spurious "second colour", and treating those as a pattern would make two
players in the same solid kit look different.

**That test is made on hue, not brightness**, and the difference matters:
measured on the reference clip (solid white and maroon kits), a brightness-
based test called 11 of 20 players "striped" — it was finding the shadow of a
fold in the shirt. A crease changes how bright the fabric is; a stripe changes
what colour it is.

## Why grass and skin are removed, and dark pixels are not

Grass leaks into every crop through gaps between arms and legs and around the
shoulders, and it leaks *the same green* into every player, which pulls all
players toward each other and destroys the very separation the clustering
needs. Skin does the same in a different direction: bare arms and necks are
the same colour on both teams.

Dark pixels are deliberately **kept**. Excluding them is a tempting
"shadow filter" that silently makes black and navy kits unmeasurable — a
whole class of real kits — so instead the summary statistic is a *median*,
which shrugs off a shaded minority without needing to know which pixels were
shaded.

## Lightness is kept at full weight, and that was measured

Down-weighting lightness looks obviously right — it is what a shadow moves
most — and on the reference clip it made things **worse**: separation between
the two kits fell from 58 to 43 and the separation-to-spread ratio from 2.9 to
2.3, because white and maroon differ mostly in *how bright they are*. The same
is true of any black-vs-white match, which is common.

The lesson is that lighting is a problem to fix at the source (see
`illumination.py`) rather than to blunt the feature against. The weight
remains configurable so a clip with a hard sun/shade split can be tested
against it in M2.8, but it defaults to 1.0 on the evidence.

## Why the lighting is removed first

See `illumination.py`. Briefly: the same shirt measures differently in sun
and in the shadow of a stand, which splits one team into two colour groups.
The grass around each player is used as a reference card to take the lighting
back out before the shirt is measured.

## Swapping this out

`TeamFeatureExtractor` is the seam. If colour proves too weak on some
broadcast (two white kits, say), a crop-embedding extractor drops in behind
this protocol and everything downstream — clustering, outlier trimming,
goalkeeper logic, per-track voting — is unchanged, because all of it works on
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
from offside.team_assignment.illumination import apply_gains, grass_gains, local_grass_ring
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

#: The two halves of a signature are scaled so that the distance between two
#: signatures is the RMS of their two colour distances — keeping every
#: threshold in this module family expressed in ordinary colour units.
_HALF_SCALE = 1.0 / np.sqrt(2.0)


class TeamFeatureExtractor(Protocol):
    """Turns a player into a vector that clusters by kit."""

    @property
    def feature_name(self) -> str: ...

    def extract(self, image: np.ndarray, pose: PlayerPose) -> JerseyColor: ...


class JerseyColorExtractor:
    """Two dominant torso colours in CIELAB, lighting-corrected against grass."""

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
        normalize_illumination: bool = True,
        illumination_max_gain: float = 2.0,
        pattern_collapse_distance: float = 25.0,
        pattern_lightness_distance: float = 35.0,
        pattern_min_share: float = 0.3,
        max_signature_pixels: int = 400,
        lightness_weight: float = 1.0,
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
        self._normalize = normalize_illumination
        self._max_gain = illumination_max_gain
        self._collapse_distance = pattern_collapse_distance
        self._collapse_lightness = pattern_lightness_distance
        self._pattern_min_share = pattern_min_share
        self._max_signature_pixels = max_signature_pixels
        self._lightness_weight = lightness_weight

        #: Frame-wide fallback reference, recomputed once per frame.
        self._frame_gains: tuple[float, float, float] | None = None
        self._frame_key: int | None = None

    @property
    def feature_name(self) -> str:
        return "jersey_lab_pair"

    def extract(self, image: np.ndarray, pose: PlayerPose) -> JerseyColor:
        polygon, region_source = self._torso_polygon(pose)
        patch, mask = _crop_polygon(image, polygon)
        if patch is None or mask is None or not mask.any():
            return unmeasured("the shirt area falls outside the frame")

        gains, lighting_note = self._lighting_for(image, pose)
        patch = apply_gains(patch, gains)

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
            return unmeasured(
                f"only {kept} shirt pixels survived after removing grass and skin "
                f"(need {self._min_sample_pixels}) — player too small, obscured, "
                "or the crop is mostly pitch"
            )

        lab = _to_lab(patch)[keep]
        primary, secondary, share, patterned = self._signature(lab)

        kept_fraction = kept / max(1, total)
        confidence = REGION_CONFIDENCE[region_source] * min(
            1.0, kept_fraction / max(1e-6, self._min_kept_fraction)
        )

        reason = (
            f"{kept} shirt pixels from the {region_source.replace('_', ' ')}"
            f"; {lighting_note}"
        )
        if patterned:
            reason += f"; two-colour kit ({share * 100:.0f}% / {(1 - share) * 100:.0f}%)"

        return JerseyColor(
            vector=_signature_vector(primary, secondary, self._lightness_weight),
            # The maths runs on the lighting-corrected colour; the swatch shown
            # to the operator is the colour *as filmed*, undoing the gain. A
            # correct measurement that displays as the wrong colour beside the
            # player's own crop reads as a bug and costs exactly the trust this
            # view exists to build.
            bgr=_display_bgr(primary, gains),
            secondary_bgr=_display_bgr(secondary, gains),
            patterned=patterned,
            pixel_count=kept,
            confidence=float(min(1.0, max(0.0, confidence))),
            source=region_source,
            reason=reason,
        )

    # -- lighting -----------------------------------------------------------

    def _lighting_for(
        self, image: np.ndarray, pose: PlayerPose
    ) -> tuple[tuple[float, float, float] | None, str]:
        if not self._normalize:
            return None, "lighting left as filmed"

        local = grass_gains(
            image,
            region=local_grass_ring(pose.bbox_xyxy),
            hue_range=self._grass_hue_range,
            min_saturation=self._grass_min_saturation,
            max_gain=self._max_gain,
        )
        if local is not None:
            return local, "lighting taken out using the grass beside this player"

        key = id(image)
        if self._frame_key != key:
            self._frame_key = key
            self._frame_gains = grass_gains(
                image,
                hue_range=self._grass_hue_range,
                min_saturation=self._grass_min_saturation,
                max_gain=self._max_gain,
            )
        if self._frame_gains is not None:
            return self._frame_gains, "lighting taken out using the whole pitch"
        return None, "no grass nearby to correct the lighting against"

    # -- the two-colour signature -------------------------------------------

    def _signature(self, lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
        """The two dominant torso colours, ordered by lightness.

        Returns (primary, secondary, share of the first, whether the kit is
        genuinely two-coloured).
        """
        median = np.median(lab, axis=0)
        if len(lab) < 2 * self._min_sample_pixels:
            return median, median, 1.0, False

        sample = lab
        if len(sample) > self._max_signature_pixels:
            # Deterministic thinning, not random sampling: the same frame must
            # produce the same answer on every run.
            step = int(np.ceil(len(sample) / self._max_signature_pixels))
            sample = sample[::step]

        first, second, share = _two_colours(sample)
        if share < self._pattern_min_share or (1 - share) < self._pattern_min_share:
            # One of them is a sliver — sponsor text, a sleeve, compression
            # noise. Treating that as a stripe would make two players in the
            # same solid kit look different from each other.
            return median, median, 1.0, False
        # Two ways to be a genuine pattern, and both are needed. Hue alone
        # misses black-and-white stripes, which differ in *nothing but*
        # brightness. Brightness alone finds a crease in a solid shirt: on the
        # reference clip's two solid kits it called half the players striped.
        # So a second colour must differ clearly in hue, or differ a lot more
        # in brightness than a fold ever does.
        chroma_gap = float(np.linalg.norm(first[1:] - second[1:]))
        lightness_gap = abs(float(first[0] - second[0]))
        if chroma_gap < self._collapse_distance and lightness_gap < self._collapse_lightness:
            return median, median, 1.0, False

        if first[0] > second[0]:
            first, second, share = second, first, 1.0 - share
        return first, second, share, True

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


def _signature_vector(
    primary: np.ndarray, secondary: np.ndarray, lightness_weight: float
) -> tuple[float, ...]:
    """Two colours as one comparable vector.

    Scaled so the distance between two signatures is the RMS of their two
    colour distances, keeping every threshold in ordinary colour units, and
    with lightness down-weighted so a shadow counts for less than a hue.
    """
    weights = np.array([lightness_weight, 1.0, 1.0])
    pair = np.concatenate([primary * weights, secondary * weights])
    return tuple(float(v) for v in pair * _HALF_SCALE)


def _two_colours(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Two-means over pixels, seeded deterministically by the extremes."""
    distances = np.linalg.norm(lab - lab.mean(axis=0), axis=1)
    far = lab[int(np.argmax(distances))]
    other = lab[int(np.argmax(np.linalg.norm(lab - far, axis=1)))]
    centres = np.array([far, other], dtype=np.float64)

    labels = np.zeros(len(lab), dtype=int)
    for iteration in range(12):
        to_centres = np.linalg.norm(lab[:, None, :] - centres[None, :, :], axis=2)
        new_labels = np.argmin(to_centres, axis=1)
        if iteration > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for slot in (0, 1):
            member = labels == slot
            if member.any():
                centres[slot] = lab[member].mean(axis=0)

    share = float((labels == 0).mean())
    return centres[0], centres[1], share


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


def _display_bgr(
    lab: np.ndarray, gains: tuple[float, float, float] | None
) -> tuple[int, int, int]:
    """A measured colour put back into the frame's own lighting, for display."""
    blue, green, red = _lab_to_bgr(lab)
    if gains is None:
        return (blue, green, red)
    undone = np.array([blue, green, red], dtype=np.float64) / np.array(gains)
    return tuple(int(v) for v in np.clip(undone, 0, 255))


def _lab_to_bgr(lab: np.ndarray) -> tuple[int, int, int]:
    encoded = np.array(
        [[[lab[0] * 255.0 / 100.0, lab[1] + 128.0, lab[2] + 128.0]]], dtype=np.float64
    )
    bgr = cv2.cvtColor(np.clip(encoded, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return tuple(int(v) for v in bgr[0, 0])


def unmeasured(reason: str) -> JerseyColor:
    return JerseyColor(
        vector=(),
        bgr=(0, 0, 0),
        secondary_bgr=(0, 0, 0),
        patterned=False,
        pixel_count=0,
        confidence=0.0,
        source=REGION_BOX,
        reason=reason,
    )
