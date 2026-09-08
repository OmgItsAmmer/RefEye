"""One detailed JSON file per offside pipeline run.

The existing session log (`observability/logging/setup.py`) is a firehose —
every subsystem in the app writes into the same growing `session_<id>.jsonl`:
video buffering, model loading, UI navigation, every triggered analysis, all
interleaved in the order it happened to occur. That file answers "what did
this session do." Finding out "what happened on *this* frame" — the question
that actually matters when a call looks wrong — means grepping megabytes of
unrelated lines for the right `frame_id`, by hand.

This is that second question answered directly: one self-contained JSON file
per pipeline run (`OffsidePipeline.analyse()`, triggered when the operator
confirms a frame), holding everything a developer or an agent needs without
cross-referencing anything else —

* every stage's state, its full detail lines (not just the one-line summary
  the UI shows), and how long it took,
* the final verdict and the whole confidence chain behind it — the same
  `ConfidenceSignal`s the review panel renders, in the weakest-link order
  that actually decided the number,
* enough raw counts (players found, poses measured, team split, calibration
  level) to spot "22 players but only 13 placed on a team" at a glance,
  without re-deriving it from the detail strings.

This module owns the file format; `offside/pipeline.py` owns *when* it gets
called (once per `analyse()`, wrapping the same per-stage loop that already
drives the review screen's live checklist — one measurement point, not two).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from core.config.paths import resolve
from core.config.schema import PipelineRunLogConfig
from observability.logging.setup import get_logger

logger = get_logger(__name__)


@dataclass
class _StageEntry:
    key: str
    phase: str
    title: str
    state: str
    summary: str
    details: list[str]
    duration_ms: float


@dataclass
class PipelineRunLog:
    """Collects one `analyse()` call's stage timings, then writes the whole
    run as one JSON file. One instance per run — `OffsidePipeline.analyse()`
    builds a fresh one at the top of every call, the same way it builds a
    fresh `FrameAnalysis`."""

    config: PipelineRunLogConfig
    frame_id: int
    _run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _started_perf: float = field(default_factory=time.perf_counter)
    _started_wall: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _stages: list[_StageEntry] = field(default_factory=list)

    def record_stage(self, report, duration_ms: float) -> None:
        """Called once per `StageReport` a pipeline stage produced, with how
        long the *whole stage call* took — not sub-divided further, since in
        practice one stage method always produces exactly one report, and
        splitting hairs below that granularity would not tell anyone
        anything a slow model call doesn't already explain."""
        self._stages.append(
            _StageEntry(
                key=report.key,
                phase=report.phase,
                title=report.title,
                state=report.state.value,
                summary=report.summary,
                details=list(report.details),
                duration_ms=round(duration_ms, 2),
            )
        )

    def write(self, analysis) -> Path | None:
        """Serialise everything collected plus the finished `FrameAnalysis`,
        and write it. Returns the path written, or None when disabled —
        never raises: a logging failure must not take down an offside call
        that otherwise succeeded (same "degrade, don't crash" rule every
        model in this codebase already follows)."""
        if not self.config.enabled:
            return None

        try:
            record = self._build_record(analysis)
            directory = resolve(self.config.directory)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"run_{self.frame_id}_{self._run_id}.json"
            path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
            return path
        except Exception as exc:  # noqa: BLE001 — logging must never break the pipeline
            logger.warning("pipeline_run_log_failed", frame_id=self.frame_id, error=str(exc))
            return None

    # -- assembly -------------------------------------------------------

    def _build_record(self, analysis) -> dict[str, Any]:
        finished_wall = datetime.now(timezone.utc)
        return {
            "run_id": self._run_id,
            "session_id": structlog.contextvars.get_contextvars().get("session_id"),
            "frame_id": self.frame_id,
            "started_at": self._started_wall.isoformat(),
            "finished_at": finished_wall.isoformat(),
            "duration_ms": round((time.perf_counter() - self._started_perf) * 1000, 2),
            "stages": [
                {
                    "key": s.key,
                    "phase": s.phase,
                    "title": s.title,
                    "state": s.state,
                    "summary": s.summary,
                    "details": s.details,
                    "duration_ms": s.duration_ms,
                }
                for s in self._stages
            ],
            "verdict": _verdict_section(analysis),
            "signals": _signals_section(analysis),
            "counts": _counts_section(analysis),
        }


def _verdict_section(analysis) -> dict[str, Any] | None:
    explanation = getattr(analysis, "explanation", None)
    decision = getattr(analysis, "offside", None)
    if explanation is None and decision is None:
        return None
    section: dict[str, Any] = {}
    if decision is not None:
        section["geometry_verdict"] = decision.verdict.value
        section["geometry_confidence"] = decision.confidence
        section["warnings"] = list(decision.warnings)
        section["reasons"] = list(decision.reasons)
    if explanation is not None:
        section["published_verdict"] = explanation.verdict.value
        section["confidence"] = explanation.confidence
        section["band"] = explanation.band.value
        section["headline"] = explanation.headline
        section["detail"] = explanation.detail
        section["withheld"] = explanation.withheld
        section["limits"] = list(explanation.limits)
        section["actions"] = list(explanation.actions)
        if explanation.weakest is not None:
            section["weakest_link"] = {
                "key": explanation.weakest.key,
                "phase": explanation.weakest.phase,
                "label": explanation.weakest.label,
                "score": explanation.weakest.score,
                "reason": explanation.weakest.reason,
            }
    return section


def _signals_section(analysis) -> list[dict[str, Any]]:
    explanation = getattr(analysis, "explanation", None)
    if explanation is None:
        return []
    return [
        {
            "key": s.key,
            "phase": s.phase,
            "label": s.label,
            "available": s.available,
            "score": s.value,
            "blocking": s.blocking,
            "reason": s.reason,
            "action": s.action,
        }
        for s in explanation.signals
    ]


def _counts_section(analysis) -> dict[str, Any]:
    """Raw numbers pulled out of the stage details so a run can be scanned
    (or aggregated across many runs) without re-parsing summary sentences —
    the same numbers every ad hoc debugging script in this project's history
    has ended up recomputing by hand from a live pipeline run."""
    counts: dict[str, Any] = {}

    detections = getattr(analysis, "detections", None) or []
    players = getattr(analysis, "players", None)
    if players is not None:
        counts["players_detected"] = len(players)
    ball = getattr(analysis, "ball", None)
    counts["ball_found"] = ball is not None
    if detections and players is not None:
        counts["raw_detections"] = len(detections)

    poses = getattr(analysis, "poses", None) or []
    if poses:
        measured = sum(1 for p in poses if p.ground_point.is_measured)
        counts["poses_total"] = len(poses)
        counts["poses_measured"] = measured
    torso_confident = getattr(analysis, "torso_confident_count", None)
    if torso_confident is not None:
        # A player can be counted in `poses_measured` (good foot) and not
        # here (no usable torso) — the two are independent, and a gap
        # between them is exactly what predicts an M2.3 kit-sampling
        # failure a stage before it happens.
        counts["poses_torso_confident"] = torso_confident

    calibration = getattr(analysis, "calibration", None)
    if calibration is not None:
        counts["calibration_level"] = calibration.level.value
        counts["calibration_confidence"] = calibration.confidence
        counts["calibration_source"] = calibration.source

    auto_points = getattr(analysis, "auto_landmark_points", None)
    if auto_points is not None:
        counts["auto_landmark_points_found"] = len(auto_points)

    teams = getattr(analysis, "teams", None)
    if teams is not None:
        counts["teams"] = teams.counts()
        counts["attacking_side_known"] = teams.sides_are_known
        counts["team_confidence"] = teams.confidence

    identities = getattr(analysis, "identities", None)
    if identities is not None:
        counts["identity_counts"] = identities.counts()
        counts["identity_confidence"] = identities.confidence

    return counts
