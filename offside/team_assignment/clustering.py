"""Splitting the players on screen into two kits, and spotting who fits neither.

## Why the outliers are removed *before* the teams are finalised

This is the whole trick of the module, and getting the order wrong quietly
ruins the result. A goalkeeper wears a kit that matches neither team; a
referee likewise. Plain two-means has no concept of "neither" — it must put
every player in one of the two groups, so the keeper is forced into whichever
kit is marginally nearer and **drags that team's centroid toward themselves**.
A green keeper pulled into the red cluster shifts "red" toward green, and the
next borderline red-vs-blue player can then land on the wrong side. One
unclassifiable player thereby corrupts real outfielders.

So the fit runs twice:

    1. fit two kits over everyone
    2. measure how far each player is from their own kit and drop the ones
       the data itself says do not belong
    3. re-fit the two kits without them

Step 2's threshold is derived from the footage — the median residual plus a
multiple of its MAD — not from a fixed colour distance, because how tight a
kit measures depends on the broadcast, the floodlights and the compression,
none of which are knowable in advance. A floor is applied so that a very
tight fit on a handful of players cannot start calling teammates outliers.

Median/MAD rather than mean/standard deviation for the same reason the fit is
run twice: the outliers being looked for would otherwise inflate the very
statistic used to detect them.

## Why the k-means here is written out rather than imported

Two clusters over at most ~30 three-dimensional points is a handful of lines,
and this avoids adding scikit-learn to a PyInstaller desktop build for it.
The initialisation is deliberately **deterministic** — the two furthest-apart
players seed the clusters, no random restarts — because the same frame must
produce the same answer every run. An offside verdict that changes between
two runs of the same clip is not something the operator can be asked to trust.
"""

from __future__ import annotations

import numpy as np

from offside.team_assignment.teams import TEAM_IDS, JerseyColor, TeamColorModel


def fit_team_colors(
    colors: list[JerseyColor],
    *,
    #: 2 is the mathematical floor — a 2-means fit needs at least one point
    #: per cluster. Below that many usable colours, this refuses outright
    #: (see the guard below); at or above it, a thin sample is not refused,
    #: it just comes out low-confidence via `confident_samples` further down
    #: — the same "degrade honestly, don't refuse" rule the rest of this
    #: module already follows for outliers and pattern detection.
    min_samples: int = 2,
    confident_samples: int = 10,
    min_sample_confidence: float = 0.25,
    outlier_mad_scale: float = 3.0,
    min_outlier_distance: float = 18.0,
    min_cluster_fraction: float = 0.2,
    min_cluster_size: int = 3,
    max_trim_rounds: int = 3,
    good_separation: float = 25.0,
    min_separation_ratio: float = 2.0,
    max_iterations: int = 25,
) -> tuple[TeamColorModel | None, list[str]]:
    """Fit the two kits present in this frame, or explain why it could not."""
    usable = [c for c in colors if c.is_usable and c.confidence >= min_sample_confidence]
    if len(usable) < min_samples:
        return None, [
            f"only {len(usable)} of {len(colors)} players gave a usable shirt colour "
            f"({min_samples} needed) — too few to tell two kits apart"
        ]

    widths = {len(c.vector) for c in usable}
    if len(widths) != 1:
        return None, ["shirt colour features have inconsistent shapes"]

    vectors = np.array([c.as_array() for c in usable], dtype=np.float64)
    weights = np.array([c.confidence for c in usable], dtype=np.float64)

    reasons: list[str] = []
    keep = np.ones(len(vectors), dtype=bool)
    labels = np.zeros(len(vectors), dtype=int)
    centroids = np.zeros((2, vectors.shape[1]))
    threshold = min_outlier_distance
    dropped_far = 0
    dropped_lonely = 0
    stopped_early = False

    for _ in range(max_trim_rounds + 1):
        labels_kept, centroids = _kmeans2(vectors[keep], weights[keep], max_iterations)
        labels = np.full(len(vectors), -1)
        labels[keep] = labels_kept

        residuals = np.linalg.norm(vectors[keep] - centroids[labels_kept], axis=1)
        threshold = _outlier_threshold(residuals, outlier_mad_scale, min_outlier_distance)
        far = residuals > threshold

        # A cluster of one or two players out of a full frame is not a team;
        # it is a goalkeeper, a referee, or a badly measured shirt that
        # happened to sit far from everyone. Residual distance alone will
        # never catch it — a single-member cluster is its own centroid, so its
        # residual is zero — and while it survives it holds one of the two
        # slots hostage, forcing the *real* two kits to share the other. That
        # is exactly how a white-vs-maroon match collapses into "everyone" and
        # "the one player in red".
        floor = max(min_cluster_size, int(np.ceil(min_cluster_fraction * keep.sum())))
        lonely = np.zeros(len(residuals), dtype=bool)
        for slot in (0, 1):
            members = labels_kept == slot
            if 0 < members.sum() < floor:
                lonely |= members

        outliers = far | lonely
        if not outliers.any():
            break

        candidate = keep.copy()
        candidate[np.flatnonzero(keep)[outliers]] = False
        # Deliberately no "must still contain two clusters" check here: when a
        # lone odd kit is removed, the survivors *are* one cluster under the
        # old centroids, and the whole point of the next round is to let the
        # two real kits separate out of them.
        if candidate.sum() < min_samples:
            stopped_early = True
            break

        dropped_far += int((far & ~lonely).sum())
        dropped_lonely += int(lonely.sum())
        keep = candidate

    dropped = dropped_far + dropped_lonely
    if dropped:
        reasons.append(
            f"{dropped} player(s) matched neither kit and were excluded before the "
            "two team colours were measured, so they could not distort them"
            + (
                f" ({dropped_lonely} of them were alone in a group too small to be "
                "a team — typically a goalkeeper or a match official)"
                if dropped_lonely
                else ""
            )
        )
    if stopped_early:
        reasons.append(
            "some players still look unlike their own group, but removing them "
            "would leave too little to measure two kits from, so the colours "
            "below may be pulled by them"
        )

    assignment = np.clip(labels, 0, 1)
    order = _stable_order(centroids)
    ordered_centroids = centroids[order]
    remap = np.argsort(order)

    spreads: dict[str, float] = {}
    swatches: dict[str, tuple[int, int, int]] = {}
    centroid_map: dict[str, tuple[float, ...]] = {}
    for slot, team_id in enumerate(TEAM_IDS):
        members = [
            colour
            for colour, label, kept in zip(usable, assignment, keep)
            if kept and remap[label] == slot
        ]
        centroid_map[team_id] = tuple(float(v) for v in ordered_centroids[slot])
        spreads[team_id] = (
            float(
                np.mean(
                    [
                        np.linalg.norm(colour.as_array() - ordered_centroids[slot])
                        for colour in members
                    ]
                )
            )
            if members
            else 0.0
        )
        swatches[team_id] = _mean_swatch(members)

    separation = float(np.linalg.norm(ordered_centroids[0] - ordered_centroids[1]))
    spread_total = spreads[TEAM_IDS[0]] + spreads[TEAM_IDS[1]]
    ratio = separation / spread_total if spread_total > 1e-6 else float("inf")

    confidence = min(
        separation / good_separation,
        ratio / min_separation_ratio,
        len(usable) / max(1, confident_samples),
        1.0,
    )
    confidence = float(max(0.0, confidence))

    reasons.insert(
        0,
        f"two kits measured from {int(keep.sum())} players; they are "
        f"{separation:.0f} colour-units apart, against a within-kit spread of "
        f"{spread_total:.0f}",
    )
    if separation < good_separation:
        reasons.append(
            "the two kits are close in colour — individual players may be "
            "assigned to the wrong side"
        )

    return (
        TeamColorModel(
            centroids=centroid_map,
            swatches=swatches,
            spreads=spreads,
            separation=separation,
            outlier_threshold=float(threshold),
            sample_count=int(keep.sum()),
            confidence=confidence,
            reasons=reasons,
        ),
        reasons,
    )


# -- the maths --------------------------------------------------------------


def _kmeans2(
    vectors: np.ndarray, weights: np.ndarray, max_iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted two-means with a deterministic seed (the furthest-apart pair)."""
    distances = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    centroids = vectors[[first, second]].astype(np.float64).copy()

    labels = np.zeros(len(vectors), dtype=int)
    for iteration in range(max_iterations):
        to_centroids = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(to_centroids, axis=1)
        if iteration > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for slot in (0, 1):
            member = labels == slot
            if not member.any():
                # An empty cluster means the seeds collapsed. Hand it the point
                # furthest from the surviving centroid rather than returning a
                # one-team answer that would silently put everyone on one side.
                furthest = int(np.argmax(to_centroids[:, 1 - slot]))
                labels[furthest] = slot
                member = labels == slot
            centroids[slot] = np.average(vectors[member], axis=0, weights=weights[member])

    return labels, centroids


def _outlier_threshold(residuals: np.ndarray, mad_scale: float, floor: float) -> float:
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    return max(floor, median + mad_scale * mad)


def _stable_order(centroids: np.ndarray) -> np.ndarray:
    """Name the clusters by their colour, not by which seed happened to win.

    Without this, the same frame analysed twice could call the same kit
    `team_a` once and `team_b` the next time. Sorting the centroids
    lexicographically makes the naming a property of the footage.
    """
    return np.lexsort(tuple(centroids[:, i] for i in reversed(range(centroids.shape[1]))))


def _mean_swatch(colors: list[JerseyColor]) -> tuple[int, int, int]:
    if not colors:
        return (0, 0, 0)
    stacked = np.array([c.bgr for c in colors], dtype=np.float64)
    return tuple(int(v) for v in np.round(stacked.mean(axis=0)))
