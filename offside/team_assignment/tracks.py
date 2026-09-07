"""Deciding a player's team once, from many frames — not once per frame.

## The mistake this fixes

The first version of this stage classified every player independently on
every frame. That throws away the strongest signal available: **a player's
team is a property of the person, not of the frame.** A player who is
measured cleanly in eight frames out of ten and blurred in the other two does
not need to be guessed at in those two — the eight already answered it.

Deciding per frame also produces visible instability (a player flickering
between sides as they turn, get occluded, or run through shadow) which is
both wrong and destroys operator trust in everything else on screen.

So this module keeps a small identity for each player across frames and
accumulates two things against it:

    the pooled colour  — the median of every shirt measurement taken of them,
                         which is a far better estimate than any one frame's
    the votes          — how much confidence has accumulated for each team,
                         so a run of agreeing frames beats a single outlier

## Why there is a tracker in here at all

M2.4 is the phase that provides real identities. Until it lands, this uses
box overlap between consecutive frames, which is enough for the few seconds
around a contact frame and costs nothing. **The moment a pose carries a
`track_id`, that wins** — this tracker steps aside rather than competing with
it, so M2.4 upgrades the accuracy here without touching a line of team logic.

The overlap matching is deliberately conservative: an ambiguous match starts
a new identity instead of merging two players. A track that fragments loses
some pooling; a track that merges two players pools *the wrong player's
colours*, which is how a defender ends up labelled as an attacker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from offside.body_keypoints.keypoints import PlayerPose
from offside.team_assignment.teams import JerseyColor

Box = tuple[float, float, float, float]


@dataclass
class PlayerTrack:
    """One player followed across frames, with everything measured of them."""

    track_id: str
    box: Box
    last_frame: int
    frames_seen: int = 1
    colors: list[JerseyColor] = field(default_factory=list)
    votes: dict[str, float] = field(default_factory=dict)

    def record_color(self, color: JerseyColor, max_samples: int) -> None:
        if not color.is_usable:
            return
        self.colors.append(color)
        if len(self.colors) > max_samples:
            # Keep the most recent: a camera cut or a lighting change makes old
            # measurements actively misleading, not merely stale.
            self.colors = self.colors[-max_samples:]

    def record_vote(self, team_id: str, weight: float) -> None:
        self.votes[team_id] = self.votes.get(team_id, 0.0) + max(0.0, weight)

    @property
    def total_votes(self) -> float:
        return sum(self.votes.values())

    def leading_vote(self) -> tuple[str | None, float]:
        """The team with the most accumulated support, and its share."""
        if not self.votes or self.total_votes <= 0.0:
            return None, 0.0
        team_id = max(self.votes, key=lambda key: self.votes[key])
        return team_id, self.votes[team_id] / self.total_votes

    def pooled_color(self) -> JerseyColor | None:
        """The median of every measurement taken of this player.

        Median rather than mean: one frame where the player was half behind
        an opponent should not move the estimate, and with an even handful of
        samples that is exactly what a mean would let it do.
        """
        if not self.colors:
            return None
        if len(self.colors) == 1:
            return self.colors[0]

        widths = {len(c.vector) for c in self.colors}
        usable = (
            self.colors
            if len(widths) == 1
            else [c for c in self.colors if len(c.vector) == len(self.colors[-1].vector)]
        )
        vectors = np.array([c.as_array() for c in usable], dtype=np.float64)
        best = max(usable, key=lambda c: c.confidence)

        return JerseyColor(
            vector=tuple(float(v) for v in np.median(vectors, axis=0)),
            bgr=best.bgr,
            secondary_bgr=best.secondary_bgr,
            patterned=sum(c.patterned for c in usable) > len(usable) / 2,
            pixel_count=sum(c.pixel_count for c in usable),
            # Pooling raises confidence, but never above what a clean single
            # measurement would earn: more looks at a badly-lit player do not
            # make the lighting better.
            confidence=float(
                min(1.0, max(c.confidence for c in usable) * (1.0 + 0.05 * (len(usable) - 1)))
            ),
            source=best.source,
            reason=f"pooled from {len(usable)} frame(s); {best.reason}",
            sample_count=len(usable),
        )


class TrackRegistry:
    """Short-lived player identities, good enough to pool colours across frames."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.35,
        max_age_frames: int = 12,
        max_samples: int = 30,
    ):
        self._iou_threshold = iou_threshold
        self._max_age_frames = max_age_frames
        self._max_samples = max_samples
        self._tracks: dict[str, PlayerTrack] = {}
        self._next_id = 0

    def reset(self) -> None:
        """Forget every identity — the right thing on a camera cut."""
        self._tracks.clear()
        self._next_id = 0

    @property
    def tracks(self) -> dict[str, PlayerTrack]:
        return self._tracks

    def update(self, frame_id: int, poses: list[PlayerPose]) -> list[str]:
        """Assign each pose to an identity, returning one track id per pose."""
        self._expire(frame_id)

        assigned: list[str | None] = [None] * len(poses)
        claimed: set[str] = set()

        # A real track id from M2.4 outranks anything computed here.
        for index, pose in enumerate(poses):
            if pose.track_id:
                assigned[index] = pose.track_id
                claimed.add(pose.track_id)

        candidates = [
            (index, pose) for index, pose in enumerate(poses) if assigned[index] is None
        ]
        pairs = [
            (_iou(pose.bbox_xyxy, track.box), index, track_id)
            for index, pose in candidates
            for track_id, track in self._tracks.items()
            if track_id not in claimed
        ]
        # Greedy best-overlap-first: the clearest matches claim their tracks
        # before the ambiguous ones get a chance to steal them.
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

        matched_indices: set[int] = set()
        for overlap, index, track_id in pairs:
            if overlap < self._iou_threshold:
                break
            if index in matched_indices or track_id in claimed:
                continue
            assigned[index] = track_id
            claimed.add(track_id)
            matched_indices.add(index)

        result: list[str] = []
        for index, pose in enumerate(poses):
            track_id = assigned[index]
            if track_id is None:
                track_id = f"t{self._next_id}"
                self._next_id += 1
            track = self._tracks.get(track_id)
            if track is None:
                self._tracks[track_id] = PlayerTrack(
                    track_id=track_id, box=pose.bbox_xyxy, last_frame=frame_id
                )
            else:
                track.box = pose.bbox_xyxy
                if frame_id != track.last_frame:
                    track.frames_seen += 1
                track.last_frame = frame_id
            result.append(track_id)
        return result

    def record_color(self, track_id: str, color: JerseyColor) -> None:
        track = self._tracks.get(track_id)
        if track is not None:
            track.record_color(color, self._max_samples)

    def record_vote(self, track_id: str, team_id: str, weight: float) -> None:
        track = self._tracks.get(track_id)
        if track is not None:
            track.record_vote(team_id, weight)

    def get(self, track_id: str) -> PlayerTrack | None:
        return self._tracks.get(track_id)

    def _expire(self, frame_id: int) -> None:
        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_id - track.last_frame > self._max_age_frames
        ]
        for track_id in stale:
            del self._tracks[track_id]


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return float(intersection / union) if union > 0 else 0.0
