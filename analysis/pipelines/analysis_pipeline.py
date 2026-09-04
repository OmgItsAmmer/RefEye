"""The triggered analysis pipeline.

Implements architecture.md section 40:

    clip -> action spotting -> de-duplication -> contact refinement
         -> ranking -> AnalysisResult

Runs entirely on the analysis worker thread. Model work is submitted through
the inference scheduler at HIGH priority so background feature extraction
cannot delay the operator (section 34).

Every stage reports progress into the request state machine, so the UI shows
what is happening rather than an opaque spinner, and every failure produces a
structured result rather than an exception reaching the UI (section 42).
"""

from __future__ import annotations

import time

from ai.candidate_ranking.ranker import ScoreCandidateRanker
from ai.contact_refinement.refiner import ContactRefiner
from ai.inference_runtime.scheduler import (
    InferencePriority,
    InferenceScheduler,
    SchedulerBusy,
)
from ai.model_registry.registry import ModelRegistry
from analysis.event_chain.chain import build_event_chain
from analysis.request_manager.manager import ProgressReporter
from analysis.request_manager.states import RequestState
from analysis.results.deduplication import deduplicate
from analysis.results.review_session import build_review_session
from core.config.schema import AppSettings
from core.domain.models import AnalysisRequest, AnalysisResult, RefinedCandidate
from core.errors.exceptions import (
    AnalysisWindowUnavailableError,
    ModelInferenceError,
)
from core.interfaces.ai import AnalysisClip, AnalysisContext
from observability.logging.setup import get_logger
from vision.features.feature_cache import FeatureCache

logger = get_logger(__name__)

#: Shown to the operator when the AI cannot produce a result. Deliberately
#: free of technical detail — that goes to the log (architecture.md §47).
OPERATOR_FAILURE_MESSAGE = (
    "AI analysis could not complete. The live video is still available. Please retry."
)


class AnalysisPipeline:
    def __init__(
        self,
        settings: AppSettings,
        video_service,
        registry: ModelRegistry,
        feature_cache: FeatureCache,
        scheduler: InferenceScheduler,
        live_pipeline=None,
    ):
        self._settings = settings
        self._video = video_service
        self._registry = registry
        self._cache = feature_cache
        self._scheduler = scheduler
        self._live = live_pipeline

        self._refiner = ContactRefiner(settings.ai.contact_refinement, feature_cache)
        self._ranker = ScoreCandidateRanker(settings.ai.candidate_ranking)
        #: Set on every completed analysis; the UI reads it to drive review.
        self.last_review_session = None

    def __call__(
        self, request: AnalysisRequest, progress: ProgressReporter
    ) -> AnalysisResult:
        """Entry point installed into the request manager."""
        started = time.perf_counter()
        warnings: list[str] = []
        diagnostics: dict = {
            "models": self._registry.model_info(),
            "pipeline": "m1",
        }

        progress.advance(RequestState.PREPARING)
        clip = self._prepare_clip(request)
        diagnostics["clip_frames"] = len(clip.frames)
        diagnostics["clip_window_ms"] = clip.end_timestamp_ms - clip.start_timestamp_ms

        # --- action spotting ------------------------------------------------
        progress.advance(RequestState.SPOTTING_ACTIONS)
        spotting_started = time.perf_counter()
        raw_candidates = self._spot_actions(clip)
        diagnostics["spotting_ms"] = _elapsed_ms(spotting_started)
        diagnostics["raw_candidates"] = len(raw_candidates)

        spotter = self._registry.get_action_spotter()
        logger.info(
            "action_spotting_completed",
            model=spotter.model_name,
            model_version=getattr(spotter, "model_version", "unknown"),
            candidates=len(raw_candidates),
            duration_ms=diagnostics["spotting_ms"],
        )

        if not raw_candidates:
            # Finding nothing is a valid outcome, not a failure — but the
            # request must still walk its remaining states, or the manager
            # cannot legally reach COMPLETED and would report a failure to
            # the operator.
            progress.advance(RequestState.REFINING)
            progress.advance(RequestState.RANKING)
            return self._empty_result(
                request,
                diagnostics,
                warnings
                + ["No ball-contact actions were found in the recent play."],
                started,
            )

        # --- refinement -----------------------------------------------------
        progress.advance(RequestState.REFINING)
        refine_started = time.perf_counter()
        refined = self._refine(raw_candidates)
        diagnostics["refinement_ms"] = _elapsed_ms(refine_started)

        logger.info(
            "refinement_completed",
            candidates=len(refined),
            duration_ms=diagnostics["refinement_ms"],
            adjusted=sum(1 for c in refined if c.evidence.get("frame_shift")),
        )

        # --- de-duplication + ranking ---------------------------------------
        progress.advance(RequestState.RANKING)
        deduplicated = deduplicate(
            refined, self._settings.ai.candidate_ranking.dedup_temporal_distance_frames
        )
        diagnostics["deduplicated_candidates"] = len(deduplicated)

        ranked = self._ranker.rank(
            deduplicated, AnalysisContext(request=request, clip=clip)
        )

        chain = build_event_chain(ranked)
        diagnostics["event_chain"] = chain.describe()
        diagnostics["total_ms"] = _elapsed_ms(started)

        if any(not c.evidence.get("refined", False) for c in ranked):
            warnings.append(
                "Some candidates could not be refined because the ball was not "
                "visible; the model's original frame was kept for those."
            )

        result = AnalysisResult(
            request_id=request.request_id,
            candidates=ranked,
            selected_candidate_index=0,
            status=RequestState.COMPLETED.value,
            warnings=warnings,
            diagnostics=diagnostics,
        )

        # Freeze the frames the operator will step through. The live buffer
        # keeps evicting while they review, so review must not depend on it.
        self.last_review_session = build_review_session(
            result,
            clip.frames,
            frames_each_side=self._settings.ai.candidate_ranking.review_frames_each_side,
        )
        diagnostics["review_memory_kb"] = self.last_review_session.total_bytes() // 1024

        return result

    # -- stages -------------------------------------------------------------

    def _prepare_clip(self, request: AnalysisRequest) -> AnalysisClip:
        clip = self._video.get_recent_clip(
            end_timestamp_ms=request.triggered_at_ms,
            duration_ms=request.requested_window_ms,
            request_id=request.request_id,
        )

        # The background loop only samples every Nth frame. Fill the gaps for
        # this window so the trigger is not limited by that stride
        # (architecture.md section 4.5: deeper processing around the moment).
        if self._live is not None and self._live.enabled:
            self._live.process_now(clip.frames)

        return clip

    def _spot_actions(self, clip: AnalysisClip):
        spotter = self._registry.get_action_spotter()
        if spotter is None:
            raise ModelInferenceError("No action spotter is loaded")

        try:
            job = self._scheduler.submit(
                lambda: spotter.infer(clip),
                priority=InferencePriority.HIGH,
                label="action_spotting",
            )
        except SchedulerBusy as exc:
            raise ModelInferenceError(str(exc)) from exc

        return job.wait(timeout=60.0)

    def _refine(self, candidates) -> list[RefinedCandidate]:
        if not self._settings.ai.contact_refinement.enabled:
            return [self._refiner.keep_anchor(c, "refinement disabled") for c in candidates]

        refined: list[RefinedCandidate] = []
        for candidate in candidates:
            result = self._refiner.refine(candidate)
            # Carry forward context the ranker and the UI need but that
            # RefinedCandidate has no dedicated field for.
            result.evidence.setdefault("player_track_id", candidate.player_track_id)
            result.evidence.setdefault("timestamp_ms", candidate.anchor_timestamp_ms)
            refined.append(result)
        return refined

    def _empty_result(
        self,
        request: AnalysisRequest,
        diagnostics: dict,
        warnings: list[str],
        started: float,
    ) -> AnalysisResult:
        diagnostics["total_ms"] = _elapsed_ms(started)
        return AnalysisResult(
            request_id=request.request_id,
            candidates=[],
            selected_candidate_index=0,
            status=RequestState.COMPLETED.value,
            warnings=warnings,
            diagnostics=diagnostics,
        )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
