"""ModelRegistry — resolves detector / tracker / action spotter from config.

Architecture.md section 20. Swapping a model is a YAML edit, never a code
edit: nothing outside this module names a concrete implementation.

Load policy (architecture.md sections 37, 38, 49):
  * Weights load once at startup and are warmed up before first use — never
    reloaded per analysis trigger.
  * If a model cannot load, the registry records the reason and degrades:
    the action spotter falls back to the configured baseline, and if the
    detector fails the AI layer reports itself Unavailable while the video
    pipeline keeps running.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from core.config.paths import resolve
from core.config.schema import AppSettings
from core.errors.exceptions import ModelLoadError
from observability.logging.setup import get_logger
from vision.detection.classes import PERSON_CLASSES
from vision.features.feature_cache import FeatureCache

logger = get_logger(__name__)


class ModelStatus(str, Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"      # running, but not on the configured primary model
    UNAVAILABLE = "unavailable"


@dataclass
class ModelInfo:
    """Provenance for every result the system produces (section 45, 54)."""

    name: str
    version: str
    provider: str
    checkpoint: str | None = None
    sha256: str | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "provider": self.provider,
            "checkpoint": self.checkpoint,
            "sha256": self.sha256,
        }


@dataclass
class RegistryState:
    status: ModelStatus = ModelStatus.NOT_LOADED
    message: str = ""
    models: dict[str, ModelInfo] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class ModelRegistry:
    def __init__(self, settings: AppSettings, feature_cache: FeatureCache):
        self._settings = settings
        self._feature_cache = feature_cache

        self._detector = None
        self._tracker = None
        self._ball_tracker = None
        self._spotter = None
        self._state = RegistryState()
        self._resolved_device: str | None = None

    # -- state --------------------------------------------------------------

    @property
    def state(self) -> RegistryState:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._state.status in (ModelStatus.READY, ModelStatus.DEGRADED)

    @property
    def resolved_device(self) -> str | None:
        """What actually ran the models — "cuda" or "cpu" — after GPU
        auto-detection and fallback (`_resolve_device`). None before the
        first load. Distinct from `settings.runtime.device`, which is only
        the *request* ("cuda" means "prefer GPU, fall back to CPU")."""
        return self._resolved_device

    def model_info(self) -> dict[str, dict]:
        return {key: info.as_dict() for key, info in self._state.models.items()}

    # -- loading ------------------------------------------------------------

    def load_all(self) -> RegistryState:
        """Load every model once. Safe to call from a background thread."""
        self._state = RegistryState(status=ModelStatus.LOADING)
        self._resolved_device = None
        logger.info("model_loading", device=self._settings.runtime.device)

        detector_ok = self._load_detector()
        self._load_trackers()
        spotter_ok = self._load_spotter()

        if not detector_ok:
            self._state.status = ModelStatus.UNAVAILABLE
            self._state.message = (
                "AI analysis is unavailable: the object detector could not be loaded."
            )
        elif not spotter_ok:
            self._state.status = ModelStatus.UNAVAILABLE
            self._state.message = "AI analysis is unavailable: no action spotter could be loaded."
        elif self._state.warnings:
            self._state.status = ModelStatus.DEGRADED
            self._state.message = self._state.warnings[0]
        else:
            self._state.status = ModelStatus.READY
            self._state.message = "AI ready"

        logger.info(
            "model_ready" if self.is_ready else "model_error",
            status=self._state.status.value,
            message=self._state.message,
            models=self.model_info(),
        )
        return self._state

    def _load_detector(self) -> bool:
        cfg = self._settings.ai.detector
        try:
            detector = self._build_detector(cfg)
            detector.load()
            detector.warmup()
        except ModelLoadError as exc:
            logger.error("model_error", component="detector", error=str(exc))
            self._state.warnings.append(str(exc))
            return False

        self._detector = detector
        self._state.models["detector"] = ModelInfo(
            name=detector.model_name,
            version=detector.model_version,
            provider=cfg.provider,
            checkpoint=cfg.checkpoint,
            sha256=_sha256_of(cfg.checkpoint),
        )
        return True

    def _build_detector(self, cfg):
        if cfg.provider == "yolo":
            from vision.detection.yolo_detector import YoloObjectDetector

            return YoloObjectDetector(
                checkpoint=cfg.checkpoint,
                confidence_threshold=cfg.confidence_threshold,
                ball_confidence_threshold=cfg.ball_confidence_threshold,
                device=self._resolve_device(),
                imgsz=cfg.imgsz,
            )
        if cfg.provider == "fixture":
            # Development fixture only — see fixture_detector.py.
            from vision.detection.fixture_detector import FixtureColorDetector

            self._state.warnings.append(
                "Using the synthetic fixture detector — valid only for the "
                "generated development clip, not real footage."
            )
            return FixtureColorDetector(confidence_threshold=cfg.confidence_threshold)

        raise ModelLoadError(f"Unknown detector provider: '{cfg.provider}'")

    def _load_trackers(self) -> None:
        from vision.tracking.ball_tracker import BallTracker
        from vision.tracking.iou_tracker import ByteTracker

        cfg = self._settings.ai.tracker
        self._tracker = ByteTracker(
            high_threshold=cfg.high_threshold,
            low_threshold=cfg.low_threshold,
            match_iou=cfg.match_iou,
            max_misses=cfg.max_misses,
            track_classes=PERSON_CLASSES,
        )
        self._ball_tracker = BallTracker(max_gap_frames=cfg.ball_max_gap_frames)

        self._state.models["tracker"] = ModelInfo(
            name="bytetrack-iou", version="1.0", provider=cfg.provider
        )

    def _load_spotter(self) -> bool:
        """Load the configured spotter, falling back if it is unavailable."""
        provider = self._settings.ai.action_spotter.provider

        if provider != "kinematic":
            try:
                spotter = self._build_spotter(provider)
                spotter.load()
                spotter.warmup()
                self._spotter = spotter
                self._state.models["action_spotter"] = ModelInfo(
                    name=spotter.model_name,
                    version=spotter.model_version,
                    provider=provider,
                    checkpoint=self._checkpoint_for(provider),
                    sha256=_sha256_of(self._checkpoint_for(provider)),
                )
                return True
            except (ModelLoadError, NotImplementedError) as exc:
                fallback = self._settings.ai.action_spotter.fallback_provider
                logger.warning(
                    "action_spotter_fallback",
                    requested=provider,
                    fallback=fallback,
                    reason=str(exc),
                )
                self._state.warnings.append(
                    f"'{provider}' is unavailable — running on the "
                    f"'{fallback}' baseline spotter instead."
                )
                provider = fallback

        return self._load_kinematic(provider)

    def _load_kinematic(self, provider: str) -> bool:
        from ai.action_spotting.kinematic.spotter import KinematicActionSpotter

        cfg = self._settings.ai.action_spotter
        resolution = self._settings.video.analysis_resolution

        spotter = KinematicActionSpotter(
            feature_cache=self._feature_cache,
            frame_width=resolution.width,
            frame_height=resolution.height,
            min_direction_change_deg=cfg.min_direction_change_deg,
            min_speed_ratio=cfg.min_speed_ratio,
            proximity_radius_px=cfg.proximity_radius_px,
            min_score=cfg.min_score,
        )
        spotter.warmup()
        self._spotter = spotter

        self._state.models["action_spotter"] = ModelInfo(
            name=spotter.model_name,
            version=spotter.model_version,
            provider=provider,
        )
        return True

    def _build_spotter(self, provider: str):
        if provider == "tdeed":
            from ai.action_spotting.tdeed.adapter import TDeedActionSpotter

            return TDeedActionSpotter(
                checkpoint=self._checkpoint_for(provider),
                device=self._resolve_device(),
            )
        if provider in ("footpass", "soccernet"):
            # Deliberately not implemented in M1 — these adapters belong to
            # the Phase 2 benchmarking milestone (STACK.md section 11).
            raise NotImplementedError(
                f"The '{provider}' adapter is scheduled for the model-benchmarking "
                "milestone and is not implemented in M1."
            )
        raise ModelLoadError(f"Unknown action_spotter provider: '{provider}'")

    def _checkpoint_for(self, provider: str) -> str | None:
        entry = self._settings.ai.providers.get(provider)
        return entry.checkpoint if entry else None

    def _resolve_device(self) -> str:
        """Honour the configured device, but never fail because CUDA is absent.

        Cached after the first resolution — called once per loaded model
        (detector, spotter), and re-running the CUDA probe each time would
        also re-append the same fallback warning to `self._state.warnings`.
        """
        if self._resolved_device is not None:
            return self._resolved_device

        requested = self._settings.runtime.device
        if requested != "cuda":
            self._resolved_device = requested
            return self._resolved_device

        try:
            import torch

            if torch.cuda.is_available():
                self._resolved_device = "cuda"
                return self._resolved_device
        except Exception:  # noqa: BLE001
            pass

        logger.warning("cuda_unavailable", falling_back_to="cpu")
        self._state.warnings.append("No CUDA device found — running AI on CPU.")
        self._resolved_device = "cpu"
        return self._resolved_device

    # -- accessors ----------------------------------------------------------

    def get_detector(self):
        return self._detector

    def get_tracker(self):
        return self._tracker

    def get_ball_tracker(self):
        return self._ball_tracker

    def get_action_spotter(self):
        return self._spotter


def _sha256_of(path: str | None) -> str | None:
    """Hash a model asset so a result can always be traced to its weights."""
    if not path:
        return None
    file_path = resolve(path)
    if not file_path.exists():
        return None

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(registry: ModelRegistry, destination: Path) -> None:
    """Record which weights produced which results (architecture.md section 54)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(registry.model_info(), indent=2, sort_keys=True), encoding="utf-8"
    )
