"""Following players through the contact moment, and saying when it failed.

## Why this is not just the M1 tracker

M1's `ByteTracker` associates by box overlap and returns boxes. That is the
right tool for the live preview, and it stays exactly as it is. But three
things offside needs cannot be expressed through it:

1. **An appearance gate.** Box overlap alone will happily hand a track to the
   opponent who just ran across it. Since M2.3 already measures every
   player's kit, a swap between *different kits* is detectable — and
   preventable — for free.
2. **Re-identification after occlusion.** A player who disappears behind an
   opponent for six frames and comes back is the same player. IoU cannot
   know that; position prediction plus appearance can.
3. **Saying "I could not tell".** The M1 protocol returns a track id or
   nothing. Offside needs a third answer — *this association was ambiguous,
   do not build a verdict on it* — because a tracker that silently picks the
   likelier of two candidates is precisely how an identity swap becomes an
   invisible wrong verdict.

So the geometry is reused (`iou`, `center_of` come straight from the M1
tracker) and the association loop is its own, with appearance and honesty
built in. M1's live path is untouched.

## What appearance can and cannot do here — and why that is enough

Kit colour cannot tell two *teammates* apart: they are dressed identically by
design. It separates only players in different kits.

That sounds like a severe limitation and is almost exactly the opposite,
because of what offside actually measures. The offside line is decided by
*positions* — the second-last opponent, whoever is standing there. Swapping
the identities of two players on the same team does not move a single
position, so the line is unchanged and the verdict is unchanged. The swap
that *does* break an offside call is attacker-for-defender, and that is a
swap between two different kits — the one case appearance catches.

## Why identities die at a camera cut

Nothing about the previous shot constrains the next one: different part of
the pitch, different players, possibly a replay of a passage of play that has
already happened. Carrying an id across a cut invents continuity that does
not exist, so a cut ends every identity and the new ones are minted with a
new segment number. They can never collide with the old ones.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.domain.models import FramePacket
from core.errors.exceptions import ModelLoadError
from observability.logging.setup import get_logger
from offside.body_keypoints.keypoints import PlayerPose
from offside.player_identity.appearance_embedding import AppearanceEmbedder
from offside.player_identity.identity import (
    IdentityResult,
    IdentityState,
    PlayerIdentity,
)
from offside.team_assignment.jersey_color import JerseyColorExtractor
from vision.scene_analysis.scene_cut import SceneCutDetector
from vision.tracking.camera_motion import (
    CameraMotionEstimator,
    to_gray,
    transform_box,
    transform_point,
)
from vision.tracking.iou_tracker import center_of, iou

logger = get_logger(__name__)

Box = tuple[float, float, float, float]


@dataclass(eq=False)
class _Track:
    """One followed player, and everything known about following them."""

    track_id: str
    bbox: Box
    segment: int
    last_frame: int
    velocity: tuple[float, float] = (0.0, 0.0)
    hits: int = 1
    misses: int = 0
    #: Recent appearance measurements; the median is the track's identity.
    appearances: list[np.ndarray] = field(default_factory=list)
    #: Recent general-appearance embeddings (build, posture, boots — not
    #: colour); the mean is the track's identity, matching how ReID galleries
    #: are ordinarily pooled, rather than the median used for kit colour —
    #: colour's median exists specifically to shrug off a lighting flash,
    #: which is not the noise these vectors carry.
    embeddings: list[np.ndarray] = field(default_factory=list)
    trail: list[tuple[float, float]] = field(default_factory=list)
    #: Doubt outlives the frame that caused it — a swap does not announce
    #: itself later — but the two kinds are counted separately so the operator
    #: is told which one actually happened. Reporting a crowded association as
    #: "came back after being hidden" is a wrong explanation, and a wrong
    #: explanation in a trust tool is worse than a vague one.
    contest_doubt: int = 0
    recovery_doubt: int = 0
    contested: bool = False
    recovered: bool = False
    last_appearance_distance: float | None = None
    last_embedding_distance: float | None = None

    @property
    def center(self) -> tuple[float, float]:
        return center_of(self.bbox)

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])

    def predict(self, motion_matrix: np.ndarray | None = None) -> Box:
        """Where this player should be next: the camera's motion applied to
        the last known box, then the player's own residual motion on top.

        `motion_matrix` carries everything a pan, tilt or zoom did to the
        whole scene between frames — without it, a camera pan reads as every
        player having moved, which is the single biggest source of broken
        association at broadcast frame rates.
        """
        box = self.bbox if motion_matrix is None else transform_box(motion_matrix, self.bbox)
        vx, vy = self.velocity
        x1, y1, x2, y2 = box
        return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)

    def appearance(self) -> np.ndarray | None:
        if not self.appearances:
            return None
        return np.median(np.stack(self.appearances), axis=0)

    def appearance_distance(self, vector: np.ndarray | None) -> float | None:
        mine = self.appearance()
        if mine is None or vector is None or mine.shape != vector.shape:
            return None
        return float(np.linalg.norm(mine - vector))

    def embedding(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        pooled = np.mean(np.stack(self.embeddings), axis=0)
        norm = np.linalg.norm(pooled)
        return pooled / norm if norm > 1e-9 else pooled

    def embedding_distance(self, vector: np.ndarray | None) -> float | None:
        mine = self.embedding()
        if mine is None or vector is None or mine.shape != vector.shape:
            return None
        return float(np.linalg.norm(mine - vector))

    def observe(
        self,
        bbox: Box,
        frame_id: int,
        vector: np.ndarray | None,
        memory: int,
        motion_matrix: np.ndarray | None = None,
        embedding: np.ndarray | None = None,
        embedding_memory: int | None = None,
    ) -> None:
        old_cx, old_cy = self.center
        if motion_matrix is not None:
            # Where the old centre would be now if only the camera had moved
            # — so the delta below is the player's own motion, not a mix of
            # the two. Skipping this would keep baking camera pan into the
            # velocity estimate, which is exactly the conflation `predict()`
            # now exists to undo.
            old_cx, old_cy = transform_point(motion_matrix, (old_cx, old_cy))
        self.bbox = bbox
        new_cx, new_cy = self.center

        # Smoothed velocity: raw frame-to-frame deltas jitter with the box.
        vx, vy = self.velocity
        self.velocity = (
            0.6 * vx + 0.4 * (new_cx - old_cx),
            0.6 * vy + 0.4 * (new_cy - old_cy),
        )
        self.hits += 1
        self.misses = 0
        self.last_frame = frame_id
        self.trail.append((new_cx, new_cy))
        if len(self.trail) > 60:
            self.trail.pop(0)

        if vector is not None:
            self.appearances.append(vector)
            if len(self.appearances) > memory:
                self.appearances.pop(0)

        if embedding is not None:
            self.embeddings.append(embedding)
            limit = embedding_memory if embedding_memory is not None else memory
            if len(self.embeddings) > limit:
                self.embeddings.pop(0)

    def coast(self, motion_matrix: np.ndarray | None = None) -> None:
        self.misses += 1
        self.bbox = self.predict(motion_matrix)


class IdentityTracker:
    """Keeps player identities through occlusion, and flags what it cannot."""

    def __init__(
        self,
        *,
        appearance_extractor=None,
        embedding_extractor=None,
        match_iou: float = 0.3,
        max_misses: int = 12,
        min_frames_to_confirm: int = 5,
        appearance_memory: int = 20,
        appearance_max_distance: float = 30.0,
        ambiguous_iou_margin: float = 0.1,
        reid_max_frames: int = 20,
        reid_max_distance_boxes: float = 2.0,
        recovery_gap_frames: int = 2,
        doubt_frames: int = 5,
        miss_penalty: float = 0.15,
        detect_camera_cuts: bool = True,
        cut_correlation_threshold: float = 0.6,
        min_frames_between_cuts: int = 4,
        camera_motion_compensation: bool = True,
        camera_motion_redetect_below: int = 120,
        use_appearance_embedding: bool = True,
        embedding_memory: int = 10,
        embedding_max_distance: float = 0.9,
    ):
        self._extractor = appearance_extractor or JerseyColorExtractor()
        self._match_iou = match_iou
        self._max_misses = max_misses
        self._min_frames_to_confirm = min_frames_to_confirm
        self._appearance_memory = appearance_memory
        self._appearance_max_distance = appearance_max_distance
        self._ambiguous_iou_margin = ambiguous_iou_margin
        self._reid_max_frames = reid_max_frames
        self._reid_max_distance_boxes = reid_max_distance_boxes
        self._recovery_gap_frames = recovery_gap_frames
        self._doubt_frames = doubt_frames
        self._miss_penalty = miss_penalty

        # A general appearance signal alongside kit colour (which cannot
        # separate two players on the same team by design — see
        # `appearance_embedding.py`). Strictly one-directional, like the kit
        # veto: it may refuse a pairing IoU and colour both accepted, but it
        # never clears a doubt they raised — a crop this close to another
        # player can be contaminated by that neighbour's own pixels, so a
        # confident-looking embedding match here is not proof of anything.
        self._use_embedding = use_appearance_embedding
        self._embedder = embedding_extractor or AppearanceEmbedder()
        self._embedding_memory = embedding_memory
        self._embedding_max_distance = embedding_max_distance

        self._detect_cuts = detect_camera_cuts
        self._cut_detector = SceneCutDetector(
            correlation_threshold=cut_correlation_threshold,
            min_frames_between_cuts=min_frames_between_cuts,
        )

        # Same primitive `CalibrationFollower` uses to carry pitch marks
        # across a pan (`vision/tracking/camera_motion.py`) — a broadcast
        # camera's motion is one shared homography for every static point in
        # the scene, and without compensating for it a pan reads as every
        # player having moved, which collapses IoU association exactly when
        # the camera is moving most.
        self._camera_motion_enabled = camera_motion_compensation
        self._motion_redetect_below = camera_motion_redetect_below
        self._motion = CameraMotionEstimator()
        self._previous_gray: np.ndarray | None = None
        self._motion_features: np.ndarray | None = None

        self._tracks: list[_Track] = []
        self._counter = itertools.count(1)
        self._segment = 0

    @classmethod
    def from_config(
        cls, config, *, appearance_extractor=None, embedding_extractor=None, device: str = "cpu"
    ):
        return cls(
            appearance_extractor=appearance_extractor,
            # Construction is cheap — the real model is loaded lazily on
            # first use — so this is built unconditionally and simply never
            # used when `use_appearance_embedding` is off.
            embedding_extractor=embedding_extractor
            or AppearanceEmbedder.from_config(config, device=device),
            match_iou=config.match_iou,
            max_misses=config.max_misses,
            min_frames_to_confirm=config.min_frames_to_confirm,
            appearance_memory=config.appearance_memory,
            appearance_max_distance=config.appearance_max_distance,
            ambiguous_iou_margin=config.ambiguous_iou_margin,
            reid_max_frames=config.reid_max_frames,
            reid_max_distance_boxes=config.reid_max_distance_boxes,
            recovery_gap_frames=config.recovery_gap_frames,
            doubt_frames=config.doubt_frames,
            miss_penalty=config.miss_penalty,
            detect_camera_cuts=config.detect_camera_cuts,
            cut_correlation_threshold=config.cut_correlation_threshold,
            min_frames_between_cuts=config.min_frames_between_cuts,
            camera_motion_compensation=config.camera_motion_compensation,
            camera_motion_redetect_below=config.camera_motion_redetect_below,
            use_appearance_embedding=config.use_appearance_embedding,
            embedding_memory=config.embedding_memory,
            embedding_max_distance=config.embedding_max_distance,
        )

    @property
    def segment(self) -> int:
        return self._segment

    def reset(self) -> None:
        self._tracks.clear()
        self._cut_detector.reset()
        self._previous_gray = None
        self._motion_features = None

    def start_new_segment(self) -> None:
        """A cut: every identity ends here and new ids carry a new segment.

        The camera motion state resets too — a feature tracked across a cut
        would report the difference between two unrelated shots as motion.
        """
        self._segment += 1
        self._tracks.clear()
        self._previous_gray = None
        self._motion_features = None

    # -- the stage ----------------------------------------------------------

    def update(self, frame: FramePacket, poses: list[PlayerPose]) -> IdentityResult:
        """Assign identities to this frame's players, and say how sure it is.

        Sets `track_id` on each pose — identity belongs to the tracker, which
        is the rule M2.2 established when it refused to let the pose model
        create or rename players.
        """
        cut = False
        if self._detect_cuts and frame.image is not None:
            cut = self._cut_detector.update(frame.image)
            if cut:
                self.start_new_segment()

        result = IdentityResult(segment=self._segment, cut_detected=cut)
        if cut:
            result.warnings.append(
                "the camera cut to a different shot — nobody can be followed "
                "across it, so every player is being identified from scratch"
            )
        motion_matrix = self._estimate_camera_motion(frame, poses)

        if not poses:
            # Still age the tracks: an early return here let a player vanish
            # for any number of frames and come back looking continuously
            # followed, which is precisely the false continuity this phase is
            # supposed to refuse.
            self._age_unmatched(frame.frame_id, motion_matrix)
            result.reasons.append("no players to follow on this frame")
            return result

        vectors = [self._appearance_of(frame, pose) for pose in poses]
        embeddings = self._embeddings_of(frame, poses)

        for track in self._tracks:
            track.contested = False

        matched = self._associate(poses, vectors, embeddings, frame.frame_id, motion_matrix)
        matched = self._recover_lost(
            poses, vectors, embeddings, frame.frame_id, matched, motion_matrix
        )
        self._start_new_tracks(poses, vectors, embeddings, frame.frame_id, matched)
        # `matched` is the association itself, carried to reporting rather than
        # re-derived from box coordinates: two players can share a box after
        # rounding, and re-deriving it would silently swap exactly the pair
        # this phase exists to keep apart.

        self._age_unmatched(frame.frame_id, motion_matrix)

        self._report(result, poses, matched)
        logger.debug(
            "player_identity",
            component="player_identity",
            segment=self._segment,
            tracked=len(result.identities),
            contested=len(result.contested()),
            cut=cut,
        )
        return result

    def _estimate_camera_motion(
        self, frame: FramePacket, poses: list[PlayerPose]
    ) -> np.ndarray | None:
        """The homography carrying the previous frame onto this one, or None
        when compensation is off, there is no image, or nothing to compare
        against yet — `predict()` and `observe()` both already treat that as
        "no compensation this frame" rather than a reason to fail.
        """
        if not self._camera_motion_enabled or frame.image is None:
            return None

        gray = to_gray(frame.image)
        exclude_boxes = [pose.bbox_xyxy for pose in poses]

        if self._previous_gray is None or self._motion_features is None:
            self._previous_gray = gray
            self._motion_features = self._motion.detect(gray, exclude_boxes)
            return None

        estimate = self._motion.estimate(self._previous_gray, self._motion_features, gray)
        self._previous_gray = gray

        if estimate is None:
            # Lost the camera. Re-detect from scratch rather than keep
            # tracking features that may already belong to a different view,
            # and predict uncompensated this frame rather than apply a
            # motion nobody believes.
            self._motion_features = self._motion.detect(gray, exclude_boxes)
            return None

        if len(estimate.features) < self._motion_redetect_below:
            self._motion_features = self._motion.detect(gray, exclude_boxes)
        else:
            self._motion_features = estimate.features

        return estimate.matrix

    def _age_unmatched(self, frame_id: int, motion_matrix: np.ndarray | None) -> None:
        for track in self._tracks:
            if track.last_frame != frame_id:
                track.coast(motion_matrix)
        self._tracks = [t for t in self._tracks if t.misses <= self._max_misses]

    # -- association --------------------------------------------------------

    def _associate(
        self,
        poses: list[PlayerPose],
        vectors: list,
        embeddings: list,
        frame_id: int,
        motion_matrix: np.ndarray | None,
    ) -> dict[int, _Track]:
        """Optimal overlap matching, refused when appearance disagrees.

        Hungarian assignment over the whole track/detection matrix, not the
        old track-by-track greedy pick — a locally-best pairing can force two
        *other* tracks into a mutual swap, which is exactly the failure this
        phase exists to prevent. Camera motion is compensated in each track's
        predicted box first (`_Track.predict`), or a pan reads as every
        player having moved and collapses every overlap at once.
        """
        live = [t for t in self._tracks if t.segment == self._segment]
        if not live or not poses:
            return {}

        n_tracks, n_poses = len(live), len(poses)
        overlap = np.zeros((n_tracks, n_poses), dtype=np.float64)
        appearance = np.full((n_tracks, n_poses), np.nan, dtype=np.float64)
        embed_dist = np.full((n_tracks, n_poses), np.nan, dtype=np.float64)
        for i, track in enumerate(live):
            predicted = track.predict(motion_matrix)
            for j, pose in enumerate(poses):
                overlap[i, j] = iou(predicted, pose.bbox_xyxy)
                distance = track.appearance_distance(vectors[j])
                if distance is not None:
                    appearance[i, j] = distance
                embed_distance = track.embedding_distance(embeddings[j])
                if embed_distance is not None:
                    embed_dist[i, j] = embed_distance

        # Kit colour is the veto here, same as before this phase gained an
        # embedding at all — proven, calibrated, and stable between adjacent
        # frames. The embedding is deliberately *not* folded in as a second
        # veto on ordinary frame-to-frame matching: a generic backbone's
        # features move measurably between two adjacent frames of real
        # footage (motion blur, a stride's change of pose, a half-turn) even
        # for the same player, and gating every single match on that would
        # be asking a noisy signal to out-vote an IoU+colour agreement that
        # was already solid. Measured directly on the reference clip: doing
        # so dropped confirmed players from the expected majority to well
        # under half, all newly forced through `_recover_lost`'s slower,
        # lower-confidence path for no reason — a real regression, not a
        # theoretical one. The embedding still gets a real job, just a
        # narrower one: an extra check in `_recover_lost`, where a multi-frame
        # gap has already weakened position's own claim and "is this really
        # the same player" is worth asking again.
        vetoed = appearance > self._appearance_max_distance
        # A pairing the box likes but appearance refuses is routed around,
        # not just later rejected — a large but finite penalty lets Hungarian
        # find a genuinely better home for both sides first, the same thing
        # the old greedy loop did by retrying, done as one global optimum
        # instead of a sequence of local ones.
        cost = np.where(vetoed, 1.0 + self._appearance_max_distance, 1.0 - overlap)

        row_idx, col_idx = linear_sum_assignment(cost)

        matched: dict[int, _Track] = {}
        for i, j in zip(row_idx, col_idx):
            track = live[i]
            best = overlap[i, j]
            if best < self._match_iou:
                # No acceptable box match for this track this frame — forced
                # only because the assignment must be square, not because
                # this pairing means anything.
                continue

            if vetoed[i, j]:
                # The box says yes, the kit says no. This is the cross-team
                # swap the gate exists for: refuse the pairing rather than
                # take the likelier of two wrong answers.
                track.contested = True
                track.contest_doubt = self._doubt_frames
                continue

            # Ambiguity, tested in *both* directions, off the original
            # overlap matrix — another track wanting this player is not
            # enough on its own (in a crowded box that's true of half the
            # frame); it is only genuinely ambiguous when this detection is
            # nearly as good for someone else AND this track has no clearly
            # better alternative of its own.
            rival_for_player = (
                float(np.max(np.delete(overlap[:, j], i))) if n_tracks > 1 else 0.0
            )
            alternative_for_track = (
                float(np.max(np.delete(overlap[i, :], j))) if n_poses > 1 else 0.0
            )
            contested = (
                rival_for_player >= self._match_iou
                and (best - rival_for_player) < self._ambiguous_iou_margin
                and (best - alternative_for_track) < self._ambiguous_iou_margin
            )
            if contested:
                # Neither kit colour nor the appearance embedding may clear
                # this — both are one-directional vetoes (see above), and a
                # position-only ambiguity between two candidates this close
                # is exactly the case a crop can be contaminated by its
                # neighbour, so a small embedding distance here is not proof
                # of anything.
                track.contested = True
                track.contest_doubt = self._doubt_frames

            # Read this before `observe`, which resets the miss counter.
            #
            # A *short* gap is not a recovery. Broadcast detection drops a
            # distant or half-occluded player for a frame constantly, and the
            # track coasts one step and re-matches with high overlap and the
            # same kit — that is ordinary tracking, not a re-identification.
            # Measured on the reference clip: treating every gap as a recovery
            # left 11 of 18 players permanently flagged, which would have made
            # M2.5 refuse to use most of the pitch.
            gap = track.misses
            was_missing = gap > self._recovery_gap_frames
            track.observe(
                poses[j].bbox_xyxy,
                frame_id,
                vectors[j],
                self._appearance_memory,
                motion_matrix,
                embeddings[j],
                self._embedding_memory,
            )
            distance = appearance[i, j]
            track.last_appearance_distance = None if np.isnan(distance) else float(distance)
            embed_distance = embed_dist[i, j]
            track.last_embedding_distance = None if np.isnan(embed_distance) else float(embed_distance)
            track.recovered = was_missing
            if was_missing:
                track.recovery_doubt = self._doubt_frames
            matched[j] = track

        return matched

    def _recover_lost(
        self,
        poses: list[PlayerPose],
        vectors: list,
        embeddings: list,
        frame_id: int,
        matched: dict[int, _Track],
        motion_matrix: np.ndarray | None,
    ) -> dict[int, _Track]:
        """Give a player who was hidden their own identity back.

        Position alone would hand the id to whoever happens to be standing
        there now, so a recovery needs a plausible position, a kit colour
        that still matches, and — when either track has one — an appearance
        embedding that still matches too. A track coasting through several
        frames of occlusion is exactly where a general appearance check earns
        its keep: kit colour survives occlusion fine on its own, but it
        cannot tell "this is genuinely the same player" from "this is a
        teammate who happened to end up nearby when they reappeared".
        """
        lost = [
            track
            for track in self._tracks
            if track.segment == self._segment
            and track.misses > 0
            and track not in matched.values()
            and frame_id - track.last_frame <= self._reid_max_frames
        ]
        if not lost:
            return matched

        for index, pose in enumerate(poses):
            if index in matched:
                continue

            best_track, best_score = None, None
            for track in lost:
                if track in matched.values():
                    continue
                gap = _distance(
                    center_of(pose.bbox_xyxy), center_of(track.predict(motion_matrix))
                )
                if gap > self._reid_max_distance_boxes * track.height:
                    continue
                distance = track.appearance_distance(vectors[index])
                if distance is not None and distance > self._appearance_max_distance:
                    continue
                embed_distance = track.embedding_distance(embeddings[index])
                if embed_distance is not None and embed_distance > self._embedding_max_distance:
                    continue
                score = gap + (distance or 0.0) + (embed_distance or 0.0)
                if best_score is None or score < best_score:
                    best_track, best_score = track, score

            if best_track is None:
                continue

            hidden_for = frame_id - best_track.last_frame
            best_track.observe(
                pose.bbox_xyxy,
                frame_id,
                vectors[index],
                self._appearance_memory,
                motion_matrix,
                embeddings[index],
                self._embedding_memory,
            )
            best_track.recovered = True
            best_track.recovery_doubt = self._doubt_frames
            best_track.last_appearance_distance = best_track.appearance_distance(
                vectors[index]
            )
            best_track.last_embedding_distance = best_track.embedding_distance(
                embeddings[index]
            )
            best_track.misses = 0
            matched[index] = best_track
            logger.debug(
                "identity_recovered",
                component="player_identity",
                track=best_track.track_id,
                hidden_frames=hidden_for,
            )

        return matched

    def _start_new_tracks(
        self,
        poses: list[PlayerPose],
        vectors: list,
        embeddings: list,
        frame_id: int,
        matched: dict[int, _Track],
    ) -> None:
        for index, pose in enumerate(poses):
            if index in matched:
                continue
            track = _Track(
                track_id=f"s{self._segment}t{next(self._counter)}",
                bbox=pose.bbox_xyxy,
                segment=self._segment,
                last_frame=frame_id,
                trail=[center_of(pose.bbox_xyxy)],
            )
            if vectors[index] is not None:
                track.appearances.append(vectors[index])
            if embeddings[index] is not None:
                track.embeddings.append(embeddings[index])
            self._tracks.append(track)
            matched[index] = track

    # -- reporting ----------------------------------------------------------

    def _report(
        self,
        result: IdentityResult,
        poses: list[PlayerPose],
        matched: dict[int, _Track],
    ) -> None:
        for index, pose in enumerate(poses):
            track = matched.get(index)
            if track is None:
                continue

            # Identity belongs to the tracker: this is where it is attached,
            # and the only place any stage should be setting it.
            pose.track_id = track.track_id

            state, confidence, reason = self._judge(track)
            track.contest_doubt = max(0, track.contest_doubt - 1)
            track.recovery_doubt = max(0, track.recovery_doubt - 1)

            result.identities.append(
                PlayerIdentity(
                    index=index,
                    track_id=track.track_id,
                    state=state,
                    confidence=confidence,
                    frames_seen=track.hits,
                    frames_missed=track.misses,
                    segment=track.segment,
                    reason=reason,
                    appearance_distance=track.last_appearance_distance,
                    embedding_distance=track.last_embedding_distance,
                    trail=list(track.trail),
                )
            )

        counts = result.counts()
        result.reasons.append(
            f"{len(result.identities)} player(s) followed: "
            + ", ".join(f"{count} {state}" for state, count in counts.items() if count)
        )

        trusted = result.trusted()
        result.confidence = (
            len(trusted) / len(result.identities) if result.identities else 0.0
        )

        contested = result.contested()
        if contested:
            result.warnings.append(
                f"{len(contested)} player(s) could not be followed unambiguously — "
                "another player overlapped them and their kit could not separate "
                "the two; do not build a call on these without checking"
            )

    def _judge(self, track: _Track) -> tuple[IdentityState, float, str]:
        """State, confidence and the sentence explaining both."""
        if track.contested or track.contest_doubt > 0:
            return (
                IdentityState.CONTESTED,
                0.2,
                "another player overlapped this one and the two could not be told "
                "apart — this identity may have been swapped"
                + ("" if track.contested else " a moment ago"),
            )

        if track.recovered or track.recovery_doubt > 0:
            return (
                IdentityState.RECOVERED,
                0.6,
                "went out of sight and was picked up again by position and kit "
                "colour — very likely the same player, not certain",
            )

        if track.hits < self._min_frames_to_confirm:
            return (
                IdentityState.TENTATIVE,
                round(0.5 * track.hits / self._min_frames_to_confirm, 3),
                f"only seen for {track.hits} frame(s); "
                f"{self._min_frames_to_confirm} are needed before an identity "
                "is trusted on its own",
            )

        confidence = 1.0 - self._miss_penalty * track.misses
        appearance_note = ""
        if track.last_appearance_distance is not None:
            steadiness = 1.0 - min(
                1.0, track.last_appearance_distance / self._appearance_max_distance
            )
            confidence *= 0.7 + 0.3 * steadiness
            appearance_note = " and their kit has stayed the same colour throughout"
        else:
            # Less evidence must never read as more: an unmeasurable kit means
            # the appearance gate never protected this track at all.
            confidence *= 0.85
            appearance_note = " (kit colour could not be measured to double-check)"

        return (
            IdentityState.CONFIRMED,
            round(max(0.0, min(1.0, confidence)), 3),
            f"followed for {track.hits} frames without interruption{appearance_note}",
        )

    def _appearance_of(self, frame: FramePacket, pose: PlayerPose):
        if frame.image is None:
            return None
        colour = self._extractor.extract(frame.image, pose)
        return colour.as_array() if colour.is_usable else None

    def _embeddings_of(
        self, frame: FramePacket, poses: list[PlayerPose]
    ) -> list[np.ndarray | None]:
        if not self._use_embedding or frame.image is None or not poses:
            return [None] * len(poses)
        try:
            return self._embedder.extract_batch(frame.image, poses)
        except ModelLoadError as exc:
            # Degrade to position-and-colour only, the same as before this
            # signal existed — not a reason to fail a frame that IoU and kit
            # colour could otherwise handle fine.
            logger.warning(
                "appearance_embedding_unavailable",
                component="player_identity",
                error=str(exc),
            )
            self._use_embedding = False
            return [None] * len(poses)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
