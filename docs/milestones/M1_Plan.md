# Milestone 1 (M1) — MVP Delivery Plan

## 1. Goal of M1

## APPlication name : **RefEye(be consistent everywhere)**

## Deliver a **working, professionally polished Windows desktop MVP** that demonstrates the core product loop end-to-end:

> live/replay video → continuous lightweight background CV → operator hotkey trigger → AI candidate actions → contact-frame refinement → ranked candidates → operator review UI → confirmed frame.

This is a **trial engagement**. The client has not yet committed to the full product. M1 exists to prove:

1. We can build a real, responsive, non-toy Windows application (not a Jupyter notebook / CLI script).
2. The core "trigger → AI candidates → operator picks the right frame" workflow actually works on real broadcast footage.
3. The codebase is clean, modular, and worth investing in further (source-code handover credibility).
4. The UI looks and feels like a professional product, not a prototype.

M1 is **not** required to include multi-model benchmarking, offside logic, incident export, or full stabilization hardening — those are follow-on milestones once the trial is approved. M1 picks **one** action-spotting model, integrates it properly behind the replaceable-model architecture, and makes the whole pipeline feel solid.

## 2. Scope Boundaries for M1

**In scope:**

- Local video file playback as the primary input source (`LocalFileInput`), with the `VideoInput` interface built so a capture-card/RTSP source can be added later without refactoring.
- Rolling encoded + decoded buffers with bounded memory.
- Live preview UI, responsive at all times.
- One integrated action-spotting model (T-DEED, chosen as primary — see [STACK.md](../architecture/STACK.md)) behind the common `ActionSpotter` interface.
- Ball/player detection + lightweight tracking sufficient to support contact refinement.
- Global configurable hotkey to trigger analysis on the recent buffer window.
- Contact-frame refinement, de-duplication, and candidate ranking (simplified scoring, weights in config).
- Professional review UI: best candidate front and center, alternate candidate navigation, frame-by-frame navigation, manual confirm.
- Structured logging + basic diagnostics panel.
- Config externalized via YAML.
- Packaged as a runnable Windows build (PyInstaller) for demo delivery.

**Explicitly out of scope for M1** (deferred to later milestones per architecture §5, §57 Phase 2/6/7):

- Multi-model benchmarking harness output/report (interface must still allow swapping models).
- Capture card / RTSP / NDI live inputs (interface ready, adapter not required).
- Incident save/export, SQLite persistence of incidents.
- Event-chain reasoning beyond simple chronological candidate ordering.
- Full long-session stabilization / stress testing.
- Offside module.
- Installer polish (Inno Setup) — a working PyInstaller folder build is sufficient for the trial demo.

## 3. Phase Breakdown

### Phase M1.1 — Foundation & Project Skeleton

**Objective:** A running, empty-but-real application shell that proves the architecture before any AI is added.

- Repository structure per architecture §7 (already scaffolded).
- `core/config`: typed config loader (pydantic models over YAML), environment override support.
- `observability/logging`: structured logging (module, session id, request id fields), log file rotation.
- `video/inputs`: `VideoInput` protocol + `LocalFileInput` adapter using PyAV/FFmpeg.
- `video/decoding` + `video/timestamps`: decoder producing `FramePacket`s on a stable timeline (PTS-based).
- `video/buffering`: bounded `EncodedRingBuffer` and `DecodedRingBuffer` with eviction.
- `apps/desktop`: PySide6 shell — main window, live video panel rendering decoded frames off the UI thread, stream status indicator.
- `apps/desktop/shortcuts`: configurable global hotkey registration (no analysis wired yet — just an event).
- `analysis/request_manager`: `AnalysisRequest` state machine (CREATED → QUEUED → ... ) with no real work behind it yet (stub COMPLETED).
- Basic pytest suite for buffers, timestamp normalization, config loading, state machine transitions.

**Exit criteria:** App opens, plays a local video file smoothly, hotkey fires and logs a request id, memory stays bounded during playback, tests pass.

### Phase M1.2 — AI Model Integration (Single Model, Replaceable)

**Objective:** Real AI in the loop, isolated behind adapters so the model is swappable later.

- `ai/model_registry`: `ModelRegistry` resolving detector / tracker / action spotter from config.
- `ai/action_spotting/common`: `ActionSpotter` protocol, `AnalysisClip`, `ActionCandidate` schema.
- `ai/action_spotting/tdeed`: `TDeedActionSpotter` adapter wrapping the T-DEED checkpoint (warmup on load, not per-trigger).
- `vision/detection`: `ObjectDetector` protocol + a practical ball/player detector (evaluated against sample client-like footage; accuracy expectations kept configurable, not hard-coded).
- `vision/tracking`: lightweight multi-object tracker (ByteTrack-class) behind `MultiObjectTracker` protocol.
- `vision/features`: bounded `FeatureCache` (ball position/velocity, nearest player, scene-cut flag) updated continuously at low cost.
- `ai/inference_runtime` + `analysis/pipelines`: `InferenceScheduler` with HIGH/MEDIUM/LOW priority so triggered analysis is never starved by background feature work.
- Model asset manifest (`deployment/model_assets`) recording model id/version/hash.

**Exit criteria:** Pressing the hotkey runs real inference on the recent buffer and returns a list of raw `ActionCandidate`s (unranked, unrefined) within an acceptable latency, logged with model name/version and request id.

### Phase M1.3 — Contact Refinement, De-duplication & Ranking

**Objective:** Turn raw model output into a small, trustworthy, ordered set of candidates.

- `ai/contact_refinement`: `ContactRefiner` — local search window around each anchor frame using ball proximity, velocity/direction change, track consistency; weights defined in config, not hard-coded.
- `analysis/results`: candidate de-duplication (cluster by temporal distance + action type + track id).
- `ai/candidate_ranking`: `CandidateRanker` producing an ordered `list[RefinedCandidate]`, never collapsing to a single result.
- `analysis/event_chain`: minimal chronological ordering of candidates (full event-chain reasoning deferred).
- Missing-ball and camera-cut handling per architecture §25–§26 (short-gap tolerance, no cross-cut velocity).
- Unit tests for de-dup clustering, ranking order, refinement scoring, missing-ball fallback.

**Exit criteria:** A single hotkey trigger reliably returns 3–5 distinct, meaningfully ordered candidates with refined frame numbers on representative test clips.

### Phase M1.4 — Review UI (Professional Polish)

**Objective:** This is the phase the client actually sees and judges. Must look and feel like a finished product.

- `apps/desktop/ui`: `CandidateReviewPanel` — frame view, action label, confidence/metadata, candidate index ("2 of 4"), best-candidate auto-selected first.
- Frame navigation (prev/next frame) and candidate navigation (prev/next candidate) as distinct, clearly bound controls.
- `AnalysisStatus`, `ModelStatus`, `StreamStatus` indicators reacting to signals (§32 UI event model), not polling.
- Confirm-frame action with a clear success state.
- Retry-on-failure flow with a friendly, non-technical error message (§47) while the live stream keeps running.
- Applied visual design system: consistent spacing, color palette, iconography, dark-themed professional look (see STACK.md UI section), custom app icon.
- `apps/desktop/ui/theme`: centralized QSS stylesheet, no inline ad-hoc styling scattered across widgets.

**Exit criteria:** A non-technical stakeholder can trigger analysis, browse candidates and frames, and confirm a frame without explanation. The app *looks* like a paid product.

### Phase M1.5 — Hardening, Diagnostics & Demo Packaging

**Objective:** Make the trial build safe to hand to the client and demo live.

- `observability/diagnostics`: lightweight diagnostics panel (FPS, buffer duration, last request latency, GPU/CPU/RAM) — developer-facing, hideable.
- Graceful degradation: app still opens and plays video if GPU/model load fails; AI shown as "Unavailable" (§38, §49).
- Stream-failure handling for the local-file path (corrupt/unsupported file → clear error, app stays alive).
- Memory-bounded soak test: run a full sample match end-to-end, confirm no unbounded growth.
- `deployment/windows` + `deployment/packaging`: PyInstaller build producing a runnable folder (`SoccerAnalysis.exe` + `runtime/` + `models/` + `config/` + `logs/`), smoke-tested on a clean machine/VM.
- README / short operator quick-start (hotkeys, how to load a clip, how to confirm a frame).
- Tag the milestone build for client review.

**Exit criteria:** A zipped Windows build runs on a machine without a dev environment, demonstrates the full loop end-to-end, and survives a full test-clip playthrough without crashing or degrading.

## 4. Cross-Cutting Rules for M1 (carried from architecture §58)

- No decode or inference on the UI thread, ever.
- No unbounded buffers/queues/caches.
- Models loaded once, not per-trigger.
- Every `AnalysisRequest` gets a unique id; every result logs model name + version.
- Always return multiple candidates, never collapse to one.
- Config-driven: hotkeys, model choice, buffer sizes, refinement weights — nothing tuning-related hard-coded.
- Unit tests written for pure logic (buffers, ranking, refinement, state machine) before/alongside UI wiring.

## 5. M1 Definition of Done

- [ ] Windows build starts reliably from a packaged folder, no dev environment required.
- [ ] Local video file loads and plays smoothly with bounded memory.
- [ ] Configurable hotkey triggers analysis on the recent buffer window.
- [ ] Real model inference returns candidates within acceptable latency, logged with model/version + request id.
- [ ] Contact refinement adjusts anchor frames using real evidence (not just the raw model frame).
- [ ] Duplicate candidates are consolidated; ranked list always has multiple entries when evidence supports it.
- [ ] Review UI: best candidate shown first, frame nav and candidate nav both work, manual confirm works.
- [ ] AI/model failure does not crash the app; live preview remains usable.
- [ ] All tuning values (hotkeys, model choice, buffer duration, refinement weights) live in `config/`, not code.
- [ ] Structured logs + diagnostics panel available for demo/debugging.
- [ ] Full sample-match playthrough completes without crash or unbounded memory growth.
- [ ] Client-ready packaged build + short operator guide delivered.

## 6. What Happens After M1 (not built now, just so scope stays honest)

If the trial is approved, subsequent milestones (per architecture §57 Phases 2, 6, 7) would add: multi-model benchmark harness and model selection evidence, live capture-card/RTSP input, incident save/export, event-chain reasoning modes, long-session stabilization, and full handover packaging with signed installer. M1 deliberately keeps these out so effort stays concentrated on a convincing, demo-ready core loop.
