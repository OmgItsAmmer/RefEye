"""One JSON file per offside pipeline run.

The failures worth protecting here: a disabled config must write nothing, a
write must never raise (logging is not allowed to break a call that
otherwise succeeded), and the file actually contains the things a developer
or an agent would open it to find — stage timing, the full confidence chain,
and the raw counts, not just a summary sentence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from core.config.schema import PipelineRunLogConfig
from observability.logging.pipeline_run_log import PipelineRunLog
from offside.decision_support.explainer import DecisionExplainer
from offside.pipeline import StageReport, StageState
from tests.unit.test_decision_support import healthy


@dataclass
class FakeAnalysisForLog:
    """The subset of `FrameAnalysis` the logger actually reads, kept
    separate from the real dataclass so this test doesn't need a real
    detector/pose/calibration run just to check the logging shape."""

    detections: list = field(default_factory=list)
    players: list = field(default_factory=list)
    ball: object = None
    poses: list = field(default_factory=list)
    torso_confident_count: int | None = None
    calibration: object = None
    auto_landmark_points: list = field(default_factory=list)
    teams: object = None
    identities: object = None
    offside: object = None
    explanation: object = None


def make_report(key: str, state: StageState = StageState.OK) -> StageReport:
    return StageReport(
        key=key, phase="M1", title=f"stage {key}", state=state,
        summary=f"{key} summary", details=[f"{key} detail one", f"{key} detail two"],
    )


def make_config(tmp_path, *, enabled: bool = True) -> PipelineRunLogConfig:
    return PipelineRunLogConfig(enabled=enabled, directory=str(tmp_path))


def test_disabled_config_writes_nothing(tmp_path):
    log = PipelineRunLog(make_config(tmp_path, enabled=False), frame_id=1)
    log.record_stage(make_report("detection"), 12.0)

    path = log.write(FakeAnalysisForLog())

    assert path is None
    assert list(tmp_path.iterdir()) == []


def test_an_enabled_run_writes_exactly_one_file(tmp_path):
    log = PipelineRunLog(make_config(tmp_path), frame_id=42)
    log.record_stage(make_report("detection"), 12.0)

    path = log.write(FakeAnalysisForLog())

    assert path is not None
    assert path.exists()
    files = list(tmp_path.iterdir())
    assert len(files) == 1


def test_the_filename_carries_the_frame_id(tmp_path):
    log = PipelineRunLog(make_config(tmp_path), frame_id=999)
    path = log.write(FakeAnalysisForLog())
    assert "999" in path.name


def test_stages_are_recorded_with_state_summary_details_and_timing(tmp_path):
    log = PipelineRunLog(make_config(tmp_path), frame_id=1)
    log.record_stage(make_report("detection", StageState.OK), 15.5)
    log.record_stage(make_report("pitch_calibration", StageState.DEGRADED), 200.25)

    path = log.write(FakeAnalysisForLog())
    record = json.loads(path.read_text(encoding="utf-8"))

    assert len(record["stages"]) == 2
    first = record["stages"][0]
    assert first["key"] == "detection"
    assert first["state"] == "ok"
    assert first["summary"] == "detection summary"
    assert first["details"] == ["detection detail one", "detection detail two"]
    assert first["duration_ms"] == 15.5
    assert record["stages"][1]["state"] == "degraded"


def test_stages_stay_in_the_order_they_ran(tmp_path):
    log = PipelineRunLog(make_config(tmp_path), frame_id=1)
    for key in ["detection", "body_keypoints", "pitch_calibration", "offside_line"]:
        log.record_stage(make_report(key), 1.0)

    path = log.write(FakeAnalysisForLog())
    record = json.loads(path.read_text(encoding="utf-8"))

    assert [s["key"] for s in record["stages"]] == [
        "detection", "body_keypoints", "pitch_calibration", "offside_line"
    ]


def test_a_write_failure_does_not_raise(tmp_path, monkeypatch):
    """Logging must never take down a call that otherwise succeeded."""
    log = PipelineRunLog(make_config(tmp_path), frame_id=1)
    monkeypatch.setattr(log, "_build_record", lambda analysis: 1 / 0)

    result = log.write(FakeAnalysisForLog())  # must not raise

    assert result is None


def test_run_and_frame_ids_are_present(tmp_path):
    log = PipelineRunLog(make_config(tmp_path), frame_id=7)
    path = log.write(FakeAnalysisForLog())
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["frame_id"] == 7
    assert record["run_id"]
    assert record["started_at"]
    assert record["finished_at"]
    assert record["duration_ms"] >= 0


# -- the verdict / signals / counts sections, against a real explanation ---


def explain_real():
    inputs = healthy()
    decision = inputs.pop("decision")
    explanation = DecisionExplainer().explain(decision, **inputs)
    return decision, explanation


def test_the_verdict_section_carries_the_weakest_link(tmp_path):
    decision, explanation = explain_real()
    analysis = FakeAnalysisForLog(offside=decision, explanation=explanation)

    log = PipelineRunLog(make_config(tmp_path), frame_id=1)
    path = log.write(analysis)
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["verdict"]["published_verdict"] == explanation.verdict.value
    assert record["verdict"]["confidence"] == pytest.approx(explanation.confidence)
    assert record["verdict"]["headline"] == explanation.headline
    assert "weakest_link" in record["verdict"]


def test_the_signals_section_mirrors_every_explanation_signal(tmp_path):
    _, explanation = explain_real()
    analysis = FakeAnalysisForLog(explanation=explanation)

    log = PipelineRunLog(make_config(tmp_path), frame_id=1)
    path = log.write(analysis)
    record = json.loads(path.read_text(encoding="utf-8"))

    assert len(record["signals"]) == len(explanation.signals)
    keys = {s["key"] for s in record["signals"]}
    assert keys == {s.key for s in explanation.signals}


def test_no_verdict_yet_is_recorded_as_null_not_omitted_or_crashing(tmp_path):
    log = PipelineRunLog(make_config(tmp_path), frame_id=1)
    path = log.write(FakeAnalysisForLog())
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["verdict"] is None
    assert record["signals"] == []


def test_counts_section_reads_real_numbers_not_just_summaries(tmp_path):
    poses = [
        type("P", (), {"ground_point": type("G", (), {"is_measured": True})()})(),
        type("P", (), {"ground_point": type("G", (), {"is_measured": False})()})(),
    ]
    analysis = FakeAnalysisForLog(players=[object(), object()], poses=poses)

    log = PipelineRunLog(make_config(tmp_path), frame_id=1)
    path = log.write(analysis)
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["counts"]["players_detected"] == 2
    assert record["counts"]["poses_total"] == 2
    assert record["counts"]["poses_measured"] == 1
    assert record["counts"]["ball_found"] is False


def test_counts_section_carries_the_torso_yield(tmp_path):
    """The exact number that explained the frame_819 failure: good feet, bad
    shirts — this is the count that predicts an M2.3 kit-sampling failure a
    stage before it happens, so it must be readable without re-deriving it
    from the M2.3 detail text."""
    fake_pose = type("P", (), {"ground_point": type("G", (), {"is_measured": True})()})()
    analysis = FakeAnalysisForLog(poses=[fake_pose] * 19, torso_confident_count=4)

    log = PipelineRunLog(make_config(tmp_path), frame_id=1)
    path = log.write(analysis)
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["counts"]["poses_torso_confident"] == 4
    assert record["counts"]["poses_total"] == 19


def test_torso_yield_is_omitted_not_zero_before_the_stage_has_run(tmp_path):
    """None (never computed) and 0 (computed, found nobody) are different
    facts — collapsing them would make an unrun stage look like a failed one."""
    fake_pose = type("P", (), {"ground_point": type("G", (), {"is_measured": True})()})()
    analysis = FakeAnalysisForLog(poses=[fake_pose] * 3, torso_confident_count=None)

    log = PipelineRunLog(make_config(tmp_path), frame_id=1)
    path = log.write(analysis)
    record = json.loads(path.read_text(encoding="utf-8"))

    assert "poses_torso_confident" not in record["counts"]
