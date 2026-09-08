"""Following players, and admitting when following failed.

The failure this phase exists to prevent is silent: if two players swap
identities before the pass, every later stage still runs perfectly and draws
the offside line through the wrong person. So the tests here are mostly about
the *hard* cases — players crossing, players hidden, the camera cutting — and
about the tracker saying "I could not tell" instead of picking a winner.

Synthetic scenes, because the point is the decision logic: real footage
cannot be made to produce a crossing at a chosen frame on demand.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.domain.models import FramePacket
from offside.body_keypoints.keypoints import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    SOURCE_ANKLE,
    GroundPoint,
    Keypoint,
    PlayerPose,
)
from offside.player_identity import IdentityState, IdentityTracker

GRASS = (50, 130, 60)
RED_KIT = (40, 40, 200)
BLUE_KIT = (200, 60, 40)
PLAYER_W, PLAYER_H = 30, 90
FRAME_SIZE = (720, 1280)


def make_scene() -> np.ndarray:
    image = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
    image[:, :] = GRASS
    return image


def add_player(image: np.ndarray, x: float, y: float, kit) -> PlayerPose:
    x1, y1 = x - PLAYER_W / 2, y - PLAYER_H
    x2, y2 = x + PLAYER_W / 2, y
    shoulder_y, hip_y = y1 + PLAYER_H * 0.25, y1 + PLAYER_H * 0.55
    left_x, right_x = x1 + PLAYER_W * 0.15, x2 - PLAYER_W * 0.15

    cv2.rectangle(
        image, (int(left_x), int(shoulder_y)), (int(right_x), int(hip_y)), kit, -1
    )
    return PlayerPose(
        frame_id=0,
        bbox_xyxy=(x1, y1, x2, y2),
        detection_confidence=0.9,
        ground_point=GroundPoint(xy=(x, y), confidence=0.9, source=SOURCE_ANKLE, reason="t"),
        source_model="test",
        keypoints={
            LEFT_SHOULDER: Keypoint(LEFT_SHOULDER, (left_x, shoulder_y), 0.9),
            RIGHT_SHOULDER: Keypoint(RIGHT_SHOULDER, (right_x, shoulder_y), 0.9),
            LEFT_HIP: Keypoint(LEFT_HIP, (left_x, hip_y), 0.9),
            RIGHT_HIP: Keypoint(RIGHT_HIP, (right_x, hip_y), 0.9),
        },
    )


def frame_of(image: np.ndarray, frame_id: int) -> FramePacket:
    height, width = image.shape[:2]
    return FramePacket(
        frame_id=frame_id,
        pts=frame_id,
        timestamp_ms=frame_id * 33,
        capture_timestamp_ms=frame_id * 33,
        width=width,
        height=height,
        source_id="test",
        image=image,
    )


def run(tracker: IdentityTracker, positions, frame_id: int):
    """Draw a frame from (x, y, kit) triples and track it."""
    image = make_scene()
    poses = [add_player(image, x, y, kit) for x, y, kit in positions]
    return tracker.update(frame_of(image, frame_id), poses), poses


@pytest.fixture
def tracker() -> IdentityTracker:
    # Cuts off by default: these scenes are synthetic and change abruptly by
    # design, which the histogram cut detector would (correctly) call a cut.
    return IdentityTracker(detect_camera_cuts=False)


class TestFollowingPlayers:
    def test_a_player_keeps_their_id_while_they_walk(self, tracker):
        ids = []
        for step in range(8):
            result, _ = run(tracker, [(300 + step * 12, 400, RED_KIT)], step)
            ids.append(result.identities[0].track_id)

        assert len(set(ids)) == 1

    def test_identity_is_stamped_onto_the_pose(self, tracker):
        _, poses = run(tracker, [(300, 400, RED_KIT)], 0)

        # M2.2's rule: identity belongs to the tracker, and this is where it
        # gets attached. M2.3 pools its colour votes against exactly this id.
        assert poses[0].track_id is not None

    def test_a_new_player_is_tentative_before_it_is_confirmed(self, tracker):
        result, _ = run(tracker, [(300, 400, RED_KIT)], 0)
        assert result.identities[0].state is IdentityState.TENTATIVE
        assert result.identities[0].needs_confirmation

        for step in range(1, 8):
            result, _ = run(tracker, [(300 + step * 10, 400, RED_KIT)], step)

        assert result.identities[0].state is IdentityState.CONFIRMED
        assert not result.identities[0].needs_confirmation

    def test_two_players_keep_separate_ids(self, tracker):
        for step in range(6):
            result, _ = run(
                tracker,
                [(300 + step * 10, 400, RED_KIT), (700 - step * 10, 420, BLUE_KIT)],
                step,
            )

        assert len({i.track_id for i in result.identities}) == 2


class TestTheSwapThisPhaseExistsToPrevent:
    def test_players_of_different_kits_crossing_do_not_swap_ids(self, tracker):
        """The failure that breaks an offside call: an attacker and a defender
        cross, and the line is then drawn through the wrong person."""
        first_id = second_id = None
        for step in range(12):
            red_x = 400 + step * 20
            blue_x = 640 - step * 20
            result, _ = run(tracker, [(red_x, 400, RED_KIT), (blue_x, 400, BLUE_KIT)], step)
            if step == 0:
                first_id = result.identities[0].track_id
                second_id = result.identities[1].track_id

        # After crossing, the red player must still be the red player.
        assert result.identities[0].track_id == first_id
        assert result.identities[1].track_id == second_id

    def test_a_kit_that_changes_colour_is_refused_not_absorbed(self, tracker):
        """If the shirt inside a box stops matching the track's own kit, the
        box is not this player — taking it anyway is how a swap happens."""
        for step in range(6):
            run(tracker, [(400, 400, RED_KIT)], step)

        # Same position, opposite kit: the box says yes, the shirt says no.
        result, _ = run(tracker, [(400, 400, BLUE_KIT)], 6)

        identity = result.identities[0]
        assert identity.state in (IdentityState.TENTATIVE, IdentityState.CONTESTED)
        assert identity.needs_confirmation

    def test_ambiguous_overlap_between_teammates_is_reported_not_resolved(self):
        """Teammates are dressed identically, so appearance cannot separate
        them. The honest answer is 'contested', not a coin flip."""
        tracker = IdentityTracker(detect_camera_cuts=False, ambiguous_iou_margin=0.9)
        for step in range(6):
            run(tracker, [(400, 400, RED_KIT), (412, 400, RED_KIT)], step)

        result, _ = run(tracker, [(404, 400, RED_KIT), (408, 400, RED_KIT)], 6)

        assert result.contested()
        assert any("could not be followed unambiguously" in w for w in result.warnings)
        assert all(i.needs_confirmation for i in result.contested())


class TestOcclusion:
    def test_a_one_frame_detector_drop_does_not_downgrade_a_player(self, tracker):
        """Found on real footage: broadcast detection loses a distant player
        for a frame constantly. Calling every such gap a re-identification
        left 11 of 18 players permanently flagged, which would have made M2.5
        refuse to use most of the pitch."""
        for step in range(8):
            run(tracker, [(300 + step * 10, 400, RED_KIT)], step)
        tracker.update(frame_of(make_scene(), 8), [])

        result, _ = run(tracker, [(390, 400, RED_KIT)], 9)

        assert result.identities[0].state is IdentityState.CONFIRMED
        assert not result.identities[0].needs_confirmation

    def test_a_player_hidden_for_a_while_gets_their_own_id_back(self, tracker):
        for step in range(8):
            run(tracker, [(300 + step * 10, 400, RED_KIT), (800, 400, BLUE_KIT)], step)
        first_id = run(tracker, [(380, 400, RED_KIT)], 8)[0].identities[0].track_id

        for hidden in range(9, 14):  # properly out of sight, not a dropped frame
            assert tracker.update(frame_of(make_scene(), hidden), []).identities == []

        result, _ = run(tracker, [(385, 400, RED_KIT)], 14)

        identity = result.identities[0]
        assert identity.track_id == first_id
        assert identity.state is IdentityState.RECOVERED
        assert "picked up again" in identity.reason
        # Recovered is *likely* right, not certain — so it still gets flagged.
        assert identity.needs_confirmation

    def test_a_recovery_needs_the_kit_to_match_too(self, tracker):
        """Position alone would hand the identity to whoever is standing
        there now — which, after an occlusion, is often an opponent."""
        for step in range(6):
            run(tracker, [(300 + step * 10, 400, RED_KIT)], step)
        for hidden in range(6, 11):
            tracker.update(frame_of(make_scene(), hidden), [])

        result, _ = run(tracker, [(365, 400, BLUE_KIT)], 11)

        assert result.identities[0].state is IdentityState.TENTATIVE

    def test_a_player_gone_too_long_is_a_new_player(self):
        tracker = IdentityTracker(detect_camera_cuts=False, reid_max_frames=2, max_misses=2)
        for step in range(6):
            run(tracker, [(300, 400, RED_KIT)], step)
        original = tracker.update(frame_of(make_scene(), 6), []).identities

        for gap in range(7, 14):
            tracker.update(frame_of(make_scene(), gap), [])
        result, _ = run(tracker, [(300, 400, RED_KIT)], 14)

        assert original == []
        assert result.identities[0].state is IdentityState.TENTATIVE


class ScriptedEmbedder:
    """Returns embeddings by explicit script instead of reading real pixels.

    Full control over what "the same visual identity" means, so these tests
    are not confounded by the real risk `appearance_embedding.py` documents:
    a crowded box's crop can be contaminated by a neighbour's own pixels,
    which is exactly the kind of thing a synthetic scene of solid-colour
    rectangles can accidentally reproduce (found the hard way — see the
    veto-not-resolve design this module settled on).
    """

    def __init__(self):
        self.script: list[np.ndarray] | None = None

    def extract_batch(self, image, poses):
        assert self.script is not None and len(self.script) == len(poses)
        vectors, self.script = self.script, None
        return vectors


VECTOR_A = np.array([1.0, 0.0])
VECTOR_B = np.array([0.0, 1.0])  # orthogonal: as different as two unit vectors get


class TestAppearanceEmbeddingVeto:
    """Kit colour cannot tell two same-team players apart by design. The
    embedding is the extra signal that sometimes still can — but only ever
    as a veto (see `PlayerIdentityConfig.use_appearance_embedding`), never
    as a way to positively resolve a doubt colour or position already
    raised."""

    def test_a_recovery_needs_the_embedding_to_match_too(self):
        """Position and kit colour alone would hand a reappearing identity
        to whoever is standing nearby — even a teammate. This is the extra
        check for "this really is the same player", not just "the right
        colour, roughly where expected"."""
        embedder = ScriptedEmbedder()
        tracker = IdentityTracker(detect_camera_cuts=False, embedding_extractor=embedder)

        for step in range(6):
            embedder.script = [VECTOR_A]
            run(tracker, [(300 + step * 10, 400, RED_KIT)], step)
        for hidden in range(6, 9):
            tracker.update(frame_of(make_scene(), hidden), [])

        # Same kit, a plausible position for the hidden player to have
        # reached — but a different build entirely.
        embedder.script = [VECTOR_B]
        result, _ = run(tracker, [(365, 400, RED_KIT)], 9)

        assert result.identities[0].state is not IdentityState.RECOVERED

    def test_a_matching_embedding_still_recovers_normally(self):
        """The veto must not become a second, stricter gate that blocks
        ordinary recoveries — a genuine match still goes through."""
        embedder = ScriptedEmbedder()
        tracker = IdentityTracker(detect_camera_cuts=False, embedding_extractor=embedder)

        for step in range(6):
            embedder.script = [VECTOR_A]
            run(tracker, [(300 + step * 10, 400, RED_KIT)], step)
        for hidden in range(6, 9):
            tracker.update(frame_of(make_scene(), hidden), [])

        embedder.script = [VECTOR_A]
        result, _ = run(tracker, [(365, 400, RED_KIT)], 9)

        assert result.identities[0].state is IdentityState.RECOVERED

    def test_disabling_embeddings_falls_back_to_colour_and_position_only(self):
        """The feature flag must actually do something: with it off, the
        same wrong-build reappearance recovers exactly as it always did
        before this signal existed."""
        embedder = ScriptedEmbedder()
        tracker = IdentityTracker(
            detect_camera_cuts=False,
            embedding_extractor=embedder,
            use_appearance_embedding=False,
        )

        for step in range(6):
            run(tracker, [(300 + step * 10, 400, RED_KIT)], step)
        for hidden in range(6, 9):
            tracker.update(frame_of(make_scene(), hidden), [])

        result, _ = run(tracker, [(365, 400, RED_KIT)], 9)

        # The scripted embedder is never even called with embeddings off.
        assert embedder.script is None
        assert result.identities[0].state is IdentityState.RECOVERED


class TestCameraCuts:
    def test_a_cut_ends_every_identity(self):
        tracker = IdentityTracker(detect_camera_cuts=True, min_frames_between_cuts=0)
        for step in range(6):
            run(tracker, [(300, 400, RED_KIT)], step)
        before = tracker.update(
            frame_of(make_scene(), 6), [add_player(make_scene(), 300, 400, RED_KIT)]
        )

        # A completely different shot: white frame instead of a green pitch.
        white = np.full((*FRAME_SIZE, 3), 240, dtype=np.uint8)
        poses = [add_player(white, 300, 400, RED_KIT)]
        after = tracker.update(frame_of(white, 7), poses)

        assert after.cut_detected
        assert after.segment > before.segment
        assert any("camera cut" in warning for warning in after.warnings)

    def test_ids_from_before_a_cut_can_never_be_reused_after_it(self):
        tracker = IdentityTracker(detect_camera_cuts=True, min_frames_between_cuts=0)
        for step in range(4):
            result, _ = run(tracker, [(300, 400, RED_KIT)], step)
        before = {i.track_id for i in result.identities}

        white = np.full((*FRAME_SIZE, 3), 240, dtype=np.uint8)
        after_result = tracker.update(
            frame_of(white, 5), [add_player(white, 300, 400, RED_KIT)]
        )

        # The segment number is part of the id precisely so that a stale id
        # cannot silently come to mean somebody else after a cut.
        assert not before & {i.track_id for i in after_result.identities}


class TestReporting:
    def test_the_result_counts_and_explains_itself(self, tracker):
        for step in range(8):
            result, _ = run(
                tracker,
                [(300 + step * 10, 400, RED_KIT), (700, 420, BLUE_KIT)],
                step,
            )

        assert sum(result.counts().values()) == len(result.identities)
        assert result.confidence == pytest.approx(1.0)
        assert any("followed" in reason for reason in result.reasons)

    def test_no_players_is_reported_not_crashed(self, tracker):
        result = tracker.update(frame_of(make_scene(), 0), [])

        assert result.identities == []
        assert result.confidence == 0.0
        assert result.reasons


def textured_scene(seed: int = 11) -> np.ndarray:
    """Enough visual detail for optical flow to register a pan — the flat
    grass `make_scene()` has no corners at all to track, which is also why
    every other test in this file is unaffected by compensation existing at
    all: nothing there gives it a signal, so it silently no-ops."""
    rng = np.random.default_rng(seed)
    image = np.full((*FRAME_SIZE, 3), 60, dtype=np.uint8)
    image[:, :] = GRASS
    for _ in range(400):
        x, y = int(rng.integers(0, FRAME_SIZE[1])), int(rng.integers(0, FRAME_SIZE[0]))
        colour = tuple(int(c) for c in rng.integers(80, 255, size=3))
        cv2.rectangle(image, (x, y), (x + 9, y + 9), colour, -1)
    return image


def panned(image: np.ndarray, dx: float) -> np.ndarray:
    """The same scene as the camera would see it after panning by `dx` px."""
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0]], dtype=np.float64)
    return cv2.warpAffine(
        image, matrix, (FRAME_SIZE[1], FRAME_SIZE[0]), borderMode=cv2.BORDER_REFLECT
    )


class TestCameraMotionCompensation:
    """The failure this exists to fix: a broadcast pan moves every player's
    box at once, and without compensation that reads as every track having
    lost its player, not as the camera having moved."""

    #: A player who is not moving on the pitch, filmed through a camera
    #: panning 70px/frame — comfortably enough to drop IoU below the default
    #: 0.3 match threshold for a 30x90 box if the prediction does not know
    #: the whole scene just shifted.
    _PAN_PER_FRAME = 70

    def _run_pan(self, tracker: IdentityTracker):
        base = textured_scene()
        result = None
        first_id = None
        states = []
        for step in range(6):
            dx = step * self._PAN_PER_FRAME
            scene = panned(base, dx)
            # The player's own position on the pitch never changes — only
            # the camera moves — so in image space they appear to pan with
            # everything else.
            poses = [add_player(scene, 400 + dx, 400, RED_KIT)]
            result = tracker.update(frame_of(scene, step), poses)
            if step == 0:
                first_id = result.identities[0].track_id
            states.append(result.identities[0].state)
        return result, first_id, states

    def test_a_hard_pan_does_not_break_the_track(self):
        tracker = IdentityTracker(detect_camera_cuts=False)
        result, first_id, states = self._run_pan(tracker)

        assert result.identities[0].track_id == first_id
        assert result.identities[0].state is IdentityState.CONFIRMED
        # Never lost, so never had to be recovered — a clean, continuous
        # follow throughout, not a lucky re-identification at the end.
        assert IdentityState.RECOVERED not in states

    def test_disabling_compensation_reproduces_the_old_failure(self):
        """Proof the fix does something: with compensation off, the pan is
        big enough that plain IoU misses the player outright at least once —
        the track only survives at all because position-and-kit recovery
        (`_recover_lost`) picks it back up, with lower confidence and a flag
        asking the operator to double check. With compensation on, that
        recovery is never even needed (see the test above)."""
        tracker = IdentityTracker(detect_camera_cuts=False, camera_motion_compensation=False)
        _, _, states = self._run_pan(tracker)

        assert IdentityState.RECOVERED in states

    def test_a_still_camera_needs_no_compensation_to_agree(self):
        """Compensation must not change anything when there is nothing to
        compensate for — a static shot should behave exactly as before."""
        scene = textured_scene()
        with_motion = IdentityTracker(detect_camera_cuts=False)
        without_motion = IdentityTracker(
            detect_camera_cuts=False, camera_motion_compensation=False
        )

        for step in range(6):
            x = 400 + step * 5
            a = with_motion.update(frame_of(scene, step), [add_player(scene, x, 400, RED_KIT)])
            b = without_motion.update(
                frame_of(scene, step), [add_player(scene, x, 400, RED_KIT)]
            )

        assert a.identities[0].state == b.identities[0].state == IdentityState.CONFIRMED
