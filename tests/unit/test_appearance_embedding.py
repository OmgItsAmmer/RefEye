"""A general appearance embedding — the signal kit colour cannot be.

Two properties matter: the vector is genuinely discriminative (two visually
different crops land far apart, the same crop lands on itself), and the
extractor degrades honestly (a degenerate crop is None, never a fabricated
vector standing in for "could not measure").
"""

from __future__ import annotations

import numpy as np
import pytest

from offside.body_keypoints.ground_point import GroundPoint
from offside.body_keypoints.keypoints import SOURCE_ANKLE, PlayerPose
from offside.player_identity.appearance_embedding import AppearanceEmbedder

pytest.importorskip("timm")


def pose_at(box: tuple[float, float, float, float]) -> PlayerPose:
    x1, _y1, x2, y2 = box
    return PlayerPose(
        frame_id=0,
        bbox_xyxy=box,
        detection_confidence=0.9,
        ground_point=GroundPoint(
            xy=((x1 + x2) / 2.0, y2), confidence=0.9, source=SOURCE_ANKLE, reason="test"
        ),
        source_model="test",
        keypoints={},
    )


@pytest.fixture(scope="module")
def embedder() -> AppearanceEmbedder:
    """Loaded once — a real (small, CPU) model download/inference, not free
    to repeat per test."""
    instance = AppearanceEmbedder()
    instance.load()
    return instance


def solid_scene(colour: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:, :] = colour
    return image


class TestDiscriminativePower:
    def test_the_same_crop_is_its_own_nearest_neighbour(self, embedder):
        scene = solid_scene((40, 40, 200))
        box = (50.0, 40.0, 90.0, 160.0)
        pose = pose_at(box)

        first = embedder.extract(scene, pose)
        second = embedder.extract(scene, pose)

        assert first is not None and second is not None
        assert np.allclose(first, second)

    def test_visually_different_crops_land_far_apart(self, embedder):
        """Not just 'not identical' — genuinely separated, well beyond what
        two repeated measurements of the same crop show as noise."""
        red_scene = solid_scene((30, 30, 200))
        checker_scene = np.zeros((200, 200, 3), dtype=np.uint8)
        for row in range(0, 200, 10):
            for col in range(0, 200, 10):
                if (row // 10 + col // 10) % 2 == 0:
                    checker_scene[row : row + 10, col : col + 10] = (200, 200, 200)

        box = (50.0, 40.0, 90.0, 160.0)
        red = embedder.extract(red_scene, pose_at(box))
        checker = embedder.extract(checker_scene, pose_at(box))
        red_again = embedder.extract(red_scene, pose_at(box))

        assert red is not None and checker is not None
        different = np.linalg.norm(red - checker)
        repeated = np.linalg.norm(red - red_again)
        assert different > repeated + 0.1

    def test_every_vector_is_unit_normalised(self, embedder):
        scene = solid_scene((90, 150, 60))
        vector = embedder.extract(scene, pose_at((20.0, 20.0, 60.0, 140.0)))

        assert vector is not None
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-4)


class TestDegradingHonestly:
    def test_a_box_entirely_off_frame_is_none_not_fabricated(self, embedder):
        scene = solid_scene((50, 50, 50))
        pose = pose_at((500.0, 500.0, 540.0, 600.0))

        assert embedder.extract(scene, pose) is None

    def test_a_zero_area_box_is_none_not_a_crash(self, embedder):
        scene = solid_scene((50, 50, 50))
        pose = pose_at((100.0, 100.0, 100.0, 100.0))

        assert embedder.extract(scene, pose) is None

    def test_batch_extraction_keeps_none_in_place_for_bad_crops(self, embedder):
        scene = solid_scene((50, 50, 50))
        poses = [
            pose_at((50.0, 40.0, 90.0, 160.0)),
            pose_at((900.0, 900.0, 950.0, 990.0)),  # off-frame
            pose_at((60.0, 40.0, 100.0, 160.0)),
        ]

        results = embedder.extract_batch(scene, poses)

        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None
