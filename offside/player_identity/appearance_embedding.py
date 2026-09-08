"""A general appearance embedding — separating players kit colour cannot.

## Why this exists alongside the kit-colour signal, not instead of it

`jersey_color.py`'s two-dominant-Lab-colours signature is fast, deterministic,
and exactly the right tool for its job: catching a cross-team swap. But it is
*structurally* blind between two players on the same team — they are dressed
identically by design, so no amount of tuning the colour extractor closes
that gap.

This module is a different axis of information: build, posture, hair, boots,
sock height, sleeve length — everything about a crop that colour clustering
throws away. It cannot replace the kit-colour veto (colour's calibrated
30-unit threshold is proven; an embedding distance has no such history yet),
but it can resolve some of what colour is structurally unable to even
attempt — specifically the *teammate* ambiguity `IdentityTracker` currently
can only ever report as "contested", never resolve.

## Why an ImageNet backbone, not a person-ReID checkpoint

A model actually trained for re-identification (OSNet, FastReID) would
separate people better than a generic classifier. But that means sourcing and
trusting a third-party checkpoint from outside this project's own pipeline,
which is a bigger decision than reusing infrastructure already vetted here:
`timm.create_model(..., pretrained=True)` is exactly how T-DEED's own
backbone is loaded (`ai/action_spotting/tdeed/_vendor/model/model.py`), so
this reuses a download path and a dependency the project already trusts,
rather than adding a new one. An ImageNet backbone's features still separate
people by build and appearance well above chance — just not as well as a
purpose-trained one would. If crowded-box contested rates stay high after
this, swapping in a real ReID checkpoint is a drop-in change: everything
downstream only depends on `AppearanceEmbedder.extract` returning an
L2-normalised vector, not on which model produced it.
"""

from __future__ import annotations

import numpy as np

from core.errors.exceptions import ModelLoadError
from observability.logging.setup import get_logger
from offside.body_keypoints.keypoints import PlayerPose

logger = get_logger(__name__)

Box = tuple[float, float, float, float]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class AppearanceEmbedder:
    """One L2-normalised embedding per player crop, batched across a frame.

    Lazily loaded — constructing this must not require torch/timm to already
    be imported or a model already downloaded, the same rule every other
    model wrapper in this project follows (`load()` on first real use, not
    at construction).
    """

    def __init__(
        self,
        *,
        model_name: str = "mobilenetv3_small_100",
        crop_height: int = 128,
        crop_width: int = 64,
        crop_width_shrink: float = 0.75,
        device: str = "cpu",
    ):
        self._model_name = model_name
        self._crop_height = crop_height
        self._crop_width = crop_width
        # Pulled in from the box's own edges before cropping — a crowded box
        # is exactly where two players' boxes overlap most, and the box edge
        # is exactly where a neighbour's own pixels bleed in. Only the width
        # shrinks: standing players are captured tightly top-to-bottom by
        # M1's detector already, but shoulder-to-shoulder is where crowding
        # happens.
        self._crop_width_shrink = crop_width_shrink
        self._device = device

        self._model = None
        self._torch = None
        self._mean = None
        self._std = None

    @classmethod
    def from_config(cls, config, *, device: str = "cpu"):
        return cls(
            model_name=config.embedding_model,
            crop_height=config.embedding_crop_height,
            crop_width=config.embedding_crop_width,
            device=device,
        )

    @property
    def model_name(self) -> str:
        return f"timm:{self._model_name}"

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import timm
            import torch
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ModelLoadError("timm/torch is not installed") from exc

        try:
            model = timm.create_model(self._model_name, pretrained=True, num_classes=0)
        except Exception as exc:
            raise ModelLoadError(
                f"could not load appearance embedding model {self._model_name}: {exc}"
            ) from exc

        model.eval()
        model.to(self._device)
        self._torch = torch
        self._model = model
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self._device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self._device).view(1, 3, 1, 1)

        logger.info(
            "model_ready",
            component="appearance_embedder",
            model=self.model_name,
            device=self._device,
        )

    def extract_batch(
        self, image: np.ndarray, poses: list[PlayerPose]
    ) -> list[np.ndarray | None]:
        """One embedding per pose, in one forward pass.

        Batched deliberately: this runs once per frame over every player on
        it, and the triggered path already re-analyses a whole review strip
        (`OffsidePipeline.warm_up`) — a dozen-plus separate model calls per
        frame would add up fast for no benefit over one.
        """
        if not poses:
            return []
        if self._model is None:
            self.load()

        crops: list[np.ndarray | None] = [self._crop(image, pose) for pose in poses]
        usable = [(index, crop) for index, crop in enumerate(crops) if crop is not None]
        if not usable:
            return [None] * len(poses)

        import cv2

        tensors = []
        for _, crop in usable:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self._crop_width, self._crop_height))
            tensors.append(resized)

        batch = self._torch.from_numpy(np.stack(tensors)).float()
        batch = batch.permute(0, 3, 1, 2) / 255.0
        batch = (batch.to(self._device) - self._mean) / self._std

        with self._torch.no_grad():
            output = self._model(batch)

        vectors = output.cpu().numpy()
        results: list[np.ndarray | None] = [None] * len(poses)
        for (index, _), vector in zip(usable, vectors):
            norm = float(np.linalg.norm(vector))
            results[index] = vector / norm if norm > 1e-9 else vector
        return results

    def extract(self, image: np.ndarray, pose: PlayerPose) -> np.ndarray | None:
        return self.extract_batch(image, [pose])[0]

    def _crop(self, image: np.ndarray, pose: PlayerPose) -> np.ndarray | None:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = pose.bbox_xyxy
        pad = (x2 - x1) * (1.0 - self._crop_width_shrink) / 2.0
        x1i, y1i = max(0, int(x1 + pad)), max(0, int(y1))
        x2i, y2i = min(width, int(x2 - pad)), min(height, int(y2))
        if x2i <= x1i or y2i <= y1i:
            return None
        return image[y1i:y2i, x1i:x2i]
