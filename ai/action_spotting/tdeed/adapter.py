"""T-DEED action spotter adapter — SoccerNetBall_challenge1.

Wraps the vendored T-DEED architecture (`_vendor/model/`, copied from
https://github.com/arturxe2/T-DEED) around the `checkpoint_best.pt` weights
in `models/tdeed/`, behind the same `ActionSpotter` protocol every other
spotter uses. Nothing outside this file touches T-DEED types, label indices,
or tensor shapes (architecture.md section 19).

Verified against the actual checkpoint (not assumed): `strict=True` state-dict
load raises on any shape mismatch, so if this adapter loads without error, the
architecture below is genuinely the one the weights were trained with.

## What this checkpoint is

SoccerNetBall_challenge1 — 1st place, 2024 SoccerNet Ball Action Spotting
Challenge. Trained on **broadcast footage at 796x448**, temporal stride 2,
100-frame clips, 12 ball-action classes plus a SoccerNet-pretrained auxiliary
head (17 classes) used only during training, not at inference.

## Preprocessing (must match training exactly, or the model degrades)

RGB (not BGR), resized to 796x448, ImageNet-normalized, sliding 100-frame
windows with 75-frame overlap (STRIDE_SNB=2, matching `T-DEED/inference.py`).
Deviating from this is a silent-accuracy-loss bug, not a crash — so every
constant here is named after the source file/constant it mirrors.

## Known upstream issue worked around here

`TDEEDModel.Impl.update_pred_head()` hardcodes `.cuda()`, which crashes on a
CPU-only machine. Rather than patch the vendored file (see `_vendor/README.md`
for why), this adapter reimplements the two lines it needs without the
hardcoded device — `_attach_dual_head()` below.
"""

from __future__ import annotations

import json
import types
import uuid
from collections import defaultdict

import numpy as np

from ai.action_spotting.common.actions import SOCCERNET_TO_DOMAIN, SUPPORTED_ACTIONS
from core.config.paths import resolve
from core.domain.models import ActionCandidate
from core.errors.exceptions import ModelInferenceError, ModelLoadError
from core.interfaces.ai import AnalysisClip
from observability.logging.setup import get_logger

logger = get_logger(__name__)

MODEL_NAME = "tdeed"

# From T-DEED/inference.py: SoccerNetBall uses temporal stride 2, and the
# non-max-suppression window widths defined as WINDOWS_SNB there.
_STRIDE_SNB = 2
_NMS_WINDOW_FRAMES = 12  # WINDOWS_SNB[1] in the source

# Bundled alongside the vendored architecture: the exact training config,
# copied verbatim from T-DEED/config/SoccerNetBall/SoccerNetBall_challenge1.json
# so clip length / feature arch / frame size can never drift from what the
# checkpoint expects.
_CONFIG_PATH = "ai/action_spotting/tdeed/_vendor/SoccerNetBall_challenge1.json"

# 12 SoccerNetBall classes + background, in the exact order T-DEED's
# data/soccernetball/class.txt defines them — the checkpoint's output index
# ordering depends on this exact list.
_SOCCERNETBALL_CLASSES = [
    "PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", "THROW IN",
    "SHOT", "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL",
]
# SoccerNet (17-class) pretrain head size — needed to reconstruct the dual
# prediction head shape the checkpoint was saved with, even though only the
# SoccerNetBall head's output is used at inference.
_PRETRAIN_CLASS_COUNT = 17


class TDeedActionSpotter:
    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        score_threshold: float = 0.15,
        max_candidates: int = 12,
    ):
        self._checkpoint = checkpoint
        self._device = device
        self._score_threshold = score_threshold
        self._max_candidates = max_candidates

        self._model = None
        self._args = None
        self._version = "SoccerNetBall_challenge1"

    # -- ActionSpotter protocol --------------------------------------------

    @property
    def model_name(self) -> str:
        return MODEL_NAME

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def supported_actions(self) -> set[str]:
        mapped = {
            SOCCERNET_TO_DOMAIN[label]
            for label in _SOCCERNETBALL_CLASSES
            if label in SOCCERNET_TO_DOMAIN
        }
        return mapped or set(SUPPORTED_ACTIONS)

    def load(self) -> None:
        checkpoint_path = resolve(self._checkpoint)
        if not checkpoint_path.exists():
            raise ModelLoadError(
                f"T-DEED checkpoint not found at {checkpoint_path}. "
                "Download SoccerNetBall_challenge1's checkpoint_best.pt from "
                "the T-DEED repository release and place it there; the "
                "application will fall back to the configured baseline spotter "
                "until it is present."
            )

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is declared
            raise ModelLoadError("torch is not installed") from exc

        try:
            from ai.action_spotting.tdeed._vendor.model.model import TDEEDModel
        except ImportError as exc:
            raise ModelLoadError(
                f"Vendored T-DEED architecture could not be imported: {exc}"
            ) from exc

        self._args = self._load_config()
        try:
            self._model = TDEEDModel(device=self._device, args=self._args)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"Could not construct T-DEED model: {exc}") from exc

        self._attach_dual_head()

        try:
            state_dict = torch.load(
                str(checkpoint_path), map_location=self._device, weights_only=False
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"Could not read T-DEED checkpoint {checkpoint_path}: {exc}"
            ) from exc

        try:
            # strict=True is deliberate: a shape mismatch means this adapter's
            # assumptions about the checkpoint are wrong, and silently loading
            # a subset of weights would produce confidently wrong candidates
            # rather than a clear failure.
            self._model._model.load_state_dict(state_dict, strict=True)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"T-DEED checkpoint {checkpoint_path} does not match the vendored "
                f"model architecture: {exc}"
            ) from exc

        self._model._model.eval()

        logger.info(
            "model_ready",
            component="action_spotter",
            model=self.model_name,
            version=self._version,
            device=self._device,
            checkpoint=str(checkpoint_path),
        )

    def warmup(self) -> None:
        """One dummy clip so the operator's first trigger doesn't pay CUDA
        context / cuDNN algorithm-selection cost (architecture.md section 37)."""
        if self._model is None:
            raise ModelLoadError("warmup() called before load()")

        import time

        import torch

        width, height = self._frame_size()
        dummy = torch.zeros((1, self._args.clip_len, 3, height, width))
        started = time.perf_counter()
        with torch.no_grad():
            self._model.predict(dummy, use_amp=(self._device == "cuda"))

        logger.info(
            "model_warmed_up",
            component="action_spotter",
            model=self.model_name,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    def infer(self, clip: AnalysisClip) -> list[ActionCandidate]:
        if self._model is None:
            raise ModelLoadError("infer() called before load()")

        frames = [f for f in clip.frames if f.image is not None]
        if len(frames) < self._args.clip_len:
            # T-DEED pads short clips with zero frames internally in its own
            # dataset loader; a triggered analysis window that is genuinely
            # this short has nothing meaningful to spot regardless.
            logger.debug(
                "tdeed_clip_too_short",
                available=len(frames),
                required=self._args.clip_len,
            )
            return []

        try:
            scores = self._run_sliding_windows(frames)
            return self._to_candidates(scores, frames)
        except Exception as exc:  # noqa: BLE001
            raise ModelInferenceError(f"T-DEED inference failed: {exc}") from exc

    # -- internals ----------------------------------------------------------

    def _load_config(self):
        config_path = resolve(_CONFIG_PATH)
        if not config_path.exists():
            raise ModelLoadError(f"T-DEED training config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        args = types.SimpleNamespace(**raw)
        args.crop_dim = None if args.crop_dim <= 0 else args.crop_dim
        return args

    def _attach_dual_head(self) -> None:
        """Equivalent of TDEEDModel.Impl.update_pred_head(), without the
        hardcoded .cuda() that upstream method contains (see module docstring
        and _vendor/README.md)."""
        from ai.action_spotting.tdeed._vendor.model.modules import FC2Layers

        num_classes = [len(_SOCCERNETBALL_CLASSES) + 1, _PRETRAIN_CLASS_COUNT + 1]
        impl = self._model._model
        impl._pred_fine = FC2Layers(impl._feat_dim, num_classes).to(self._device)
        impl._double_head = True
        self._model._num_classes = int(np.sum(num_classes))

    def _frame_size(self) -> tuple[int, int]:
        """(width, height) the checkpoint was trained at."""
        return (796, 448)

    def _run_sliding_windows(self, frames: list) -> np.ndarray:
        """Mirrors T-DEED/util/eval.py:inference() — accumulate per-frame
        class-probability scores across overlapping clip windows, averaged by
        how many windows covered each frame."""
        import cv2
        import torch

        width, height = self._frame_size()
        clip_len = self._args.clip_len
        overlap_len = clip_len // 4 * 3  # matches inference.py's non-soccernet branch

        rgb_frames = [
            cv2.resize(cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB), (width, height))
            for f in frames[:: _STRIDE_SNB]
        ]
        real_frame_count = len(rgb_frames)

        # A window shorter than clip_len is padded at the FRONT, matching
        # T-DEED's own ActionSpotInferenceDataset. `_to_candidates` needs to
        # know exactly how much padding was added so it can map a score-array
        # index back to the real frame it came from — getting this offset
        # wrong silently points confirmed candidates at the wrong frame.
        pad_count = max(0, clip_len - real_frame_count)
        if pad_count:
            pad = [np.zeros((height, width, 3), dtype=np.uint8)] * pad_count
            rgb_frames = pad + rgb_frames

        num_classes = len(_SOCCERNETBALL_CLASSES) + 1
        video_len = len(rgb_frames)
        scores = np.zeros((video_len, num_classes), dtype=np.float32)
        support = np.zeros((video_len,), dtype=np.int32)

        step = clip_len - overlap_len
        start = 0
        while start < video_len:
            window = rgb_frames[start : start + clip_len]
            if len(window) < clip_len:
                window = window + [window[-1]] * (clip_len - len(window))

            tensor = torch.from_numpy(np.stack(window)).permute(0, 3, 1, 2)
            tensor = tensor.unsqueeze(0).float()

            with torch.no_grad():
                _, window_scores = self._model.predict(
                    tensor, use_amp=(self._device == "cuda")
                )
            window_scores = window_scores[0][:, :num_classes]

            end = min(start + clip_len, video_len)
            scores[start:end] += window_scores[: end - start]
            support[start:end] += 1

            if start + clip_len >= video_len:
                break
            start += step

        support[support == 0] = 1
        scores /= support[:, None]

        # Drop the padded rows now, at the source, so every caller downstream
        # can assume scores[i] corresponds to strided_frames[i] with no offset.
        return scores[pad_count:]

    def _to_candidates(
        self, scores: np.ndarray, frames: list
    ) -> list[ActionCandidate]:
        """Score array -> events (per-class local maxima) -> ActionCandidates.

        Mirrors process_frame_predictions_inference + soft_non_maximum_supression
        from T-DEED/util/eval.py, simplified to run in-process without their
        file-based storage step.
        """
        # Map back from the stride-subsampled score index to the real frame.
        strided_frames = frames[:: _STRIDE_SNB]
        strided_frames = strided_frames[: scores.shape[0]] or frames[:1]

        events_by_class: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for frame_index in range(scores.shape[0]):
            for class_index in range(1, scores.shape[1]):  # skip background (0)
                score = float(scores[frame_index, class_index])
                if score >= self._score_threshold:
                    events_by_class[class_index].append((frame_index, score))

        candidates: list[ActionCandidate] = []
        for class_index, events in events_by_class.items():
            native_label = _SOCCERNETBALL_CLASSES[class_index - 1]
            domain_action = SOCCERNET_TO_DOMAIN.get(native_label)
            if domain_action is None:
                continue

            for frame_index, score in _soft_nms(events, window=_NMS_WINDOW_FRAMES):
                if frame_index >= len(strided_frames):
                    continue
                frame = strided_frames[frame_index]
                candidates.append(
                    ActionCandidate(
                        candidate_id=uuid.uuid4().hex[:10],
                        action_type=domain_action,
                        anchor_frame_id=frame.frame_id,
                        anchor_timestamp_ms=frame.timestamp_ms,
                        model_score=round(score, 4),
                        source_model=f"{MODEL_NAME}:{self._version}",
                        metadata={"native_label": native_label},
                    )
                )

        candidates.sort(key=lambda c: c.model_score, reverse=True)
        return candidates[: self._max_candidates]


def _soft_nms(
    events: list[tuple[int, float]], window: int
) -> list[tuple[int, float]]:
    """Suppress nearby duplicate detections within one class, softly.

    Direct translation of T-DEED's soft_non_maximum_supression for a single
    class's event list: repeatedly take the best-scoring event, decay the
    score of everything within `window` frames of it by squared distance, and
    stop once the best remaining score falls below the original threshold
    that produced the event list.
    """
    if not events:
        return []

    remaining = list(events)
    kept: list[tuple[int, float]] = []
    floor = min(score for _, score in events)

    while remaining:
        best_frame, best_score = max(remaining, key=lambda e: e[1])
        if best_score < floor:
            break
        kept.append((best_frame, best_score))

        decayed = []
        for frame, score in remaining:
            if frame == best_frame:
                continue
            distance = abs(frame - best_frame)
            if distance <= window:
                score = score * (distance**2) / max(window**2, 1)
            decayed.append((frame, score))
        remaining = decayed

    return kept
