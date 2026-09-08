# RefEye pipeline — findings and fixes log

Session date: 2026-09-08. Everything below was found and fixed in one continuous
working session, in the order it happened. Each entry states the symptom the
user actually saw, the root cause traced through the real code, the fix, and
how it was verified (tests, real footage, or both).

---

## 1. "Team colours: 0%" was a mislabelled signal, not a real failure

**Symptom.** Screenshot of the app showed "Team colours: 0% POOR" on a frame
where the two kits were visually obvious to the human eye.

**Root cause.** `offside/decision_support/signals.py`'s `team_signal()` zeroed
out the *entire* team-colours score whenever `not teams.sides_are_known` —
i.e. it conflated two independent facts: "could the two kits be told apart"
(a colour-clustering question) and "do we know which side is attacking" (a
ball-possession question). A frame with perfectly clean kit clustering but no
ball detection was reporting the clustering itself as failed.

**Fix.** Split into two independent `ConfidenceSignal`s:
- `team_signal` — now only reflects `teams.color_model.confidence` (the
  clustering fit quality), never sides-known status.
- `attacking_side_signal` (new) — reflects whether the attacking side could
  be established, separately.

`collect_signals()` now returns 6 signals instead of 5. The M2.7 review panel
was updated to show them as two separate rows (`RequirementRow`), so a clean
kit-clustering frame with an unknown attacking side now correctly reads
"Team colours: GOOD" / "Attacking side: POOR" instead of one blended failure.

**Files.** `offside/decision_support/signals.py`, `__init__.py`,
`explainer.py`; `apps/desktop/ui/widgets/offside_review.py` (full redesign,
see §4); `tests/unit/test_decision_support.py`,
`tests/unit/test_offside_review_panel.py`.

**Verified.** New tests: `test_unknown_sides_block_the_attacking_side_signal_not_team_colours`,
`test_a_kit_clustering_failure_still_blocks_team_colours`,
`test_team_colours_and_attacking_side_are_reported_separately`.

---

## 2. Weak detection models — upgraded nano → medium checkpoints

**Symptom.** User: "how the f*** can't models diff teams... we don't have a
computational power issue."

**Root cause.** `config/local.yaml` **and** `.env`
(`SOCCER_AI__DETECTOR__CHECKPOINT`) both independently pinned the old
`yolo11n.pt` (nano) checkpoint, silently overriding `config/default.yaml`
even after upgrading the default. Nano models under-detect and under-measure
kit colour on small broadcast player boxes.

**Fix.** Downloaded and switched to medium checkpoints across all three
config layers:
- `models/detector/yolo11m.pt` (detector)
- `models/pose/yolo11m-pose.pt` (body keypoints)

**Files.** `config/default.yaml`, `config/local.yaml`, `.env`.

**Verified.** Confirmed via `load_settings()` that the resolved checkpoint
path actually changed; re-ran the real-clip integration test.

---

## 3. Ball misdetection — white spots mistaken for the ball

**Symptom.** User: "there are some white spots on which that is considered
as ball."

**Root cause.** The detector's highest-confidence "ball" detection was
trusted outright. Measured on real clips: more than one "ball" detection
appeared on 4 of every 10 sampled frames (advertising boards, players' white
boots/socks, crowd). Confidence alone doesn't distinguish them.

**Fix.** New module `offside/ball_selection.py`: `select_ball()` filters
"ball"-class detections by size *relative to the median player box height on
the same frame* (a zoom-invariant reference — the ball is always a fixed
real-world size relative to a player, regardless of camera zoom). Returns
`None` — never a best-effort guess — when nothing passes the size-sanity
window.

**Config.** `DetectorConfig.ball_min_size_ratio` (default 0.06),
`ball_max_size_ratio` (default 0.22).

**Files.** `offside/ball_selection.py` (new), `offside/pipeline.py`
(`FrameAnalysis.ball` property now routes through it), `core/config/schema.py`.

**Verified.** `tests/unit/test_ball_selection.py`.

---

## 4. UI redesign — deterministic requirements checklist + suggestion box

**Ask.** User wanted results "not necessarily a description" — a
deterministic checklist of what's satisfied, a percentage confidence per
requirement, and a suggestion box, rather than prose.

**Fix.** Full rewrite of `apps/desktop/ui/widgets/offside_review.py`:
- `RequirementRow` — coloured dot + label + mini progress bar +
  `"{score%} {GOOD|WEAK|POOR|N/A}"`, one per `ConfidenceSignal`
  (thresholds: ≥0.7 GOOD, ≥0.45 WEAK, else POOR; `blocking=True` always
  reads POOR regardless of its raw number).
- `SuggestionBox` — bordered, tinted card listing concrete next actions
  (`explanation.actions`); hidden entirely when there's nothing to say.
- Confidence badge now shows the **band** (HIGH/MEDIUM/LOW/NONE), not the
  verdict — the verdict lives in the big word (OFFSIDE / ONSIDE / NO CALL).
- Confidence shown as an integer percentage, not a 0–1 decimal.

**Files.** `apps/desktop/ui/widgets/offside_review.py`,
`apps/desktop/ui/theme/stylesheet.py`.

**Verified.** `tests/unit/test_offside_review_panel.py` rewritten, 27 tests —
covers the badge-shows-band-not-verdict distinction, blocking-reads-POOR
even with a nonzero score, suggestion box hidden/shown correctly, rows reused
not stacked across frames (headless-Qt `isVisibleTo(panel)` gotcha, not
`isVisible()`).

---

## 5. Auto pitch-landmark detection, with manual fallback

**Ask.** User: "I WANT AN AUTO POINT FINDER... if everything fails then
fallback to manual marking... force user to select marks before deciding."

**Fix.**
- `offside/pitch_calibration/auto_landmarks.py` (new) — `AutoLandmarkDetector`
  wraps a YOLO11-pose model fine-tuned on SoccerNet pitch-calibration
  keypoints (`Adit-jain/Soccana_Keypoint`, downloaded via `huggingface_hub`).
  Maps its 29 model keypoints onto 21 of this project's own
  `PitchModel.landmarks()` names.
- `offside/pipeline.py`'s calibration priority chain: operator marks (≥4) →
  auto-landmarks (if ≥6 confident points, `calibrate_manual(...,
  source="auto_landmarks")`) → directional fallback.
- `apps/desktop/ui/widgets/landmark_marking_dialog.py` (new) — manual
  fallback UI: click-to-place landmark dialog, `_ClickableFrame` custom-painted
  widget, Done disabled until 4 marks placed.
- `apps/desktop/ui/widgets/pitch_map.py` — "Mark landmarks" button, only
  enabled once a frame has been analysed; caption reads "Not calibrated on
  this frame — mark landmarks below for a precise line."
- Deliberately **not** a hard block on Confirm — matches this app's
  established "never trap the operator" rule. The confidence-floor system
  already withholds a confident verdict without real calibration.

**Validated (before wiring into production).** 3 real clips, 0.90–0.95 fit
confidence, sub-20cm reprojection error, visually confirmed by projecting the
pitch-model's line drawing back through the fit and checking it lands on the
real painted lines pixel-precisely. (Caught and corrected my own initial
misjudgement here — 2 of 8 points looked "wrong" by eye due to perspective
distortion in a wide broadcast shot; the numbers said they were correct, and
they were.)

**Files.** `offside/pitch_calibration/auto_landmarks.py`,
`offside/pitch_calibration/calibrator.py` (`calibrate_manual(..., source=)`
param threaded through), `models/pitch_keypoints/README.md`,
`apps/desktop/ui/widgets/landmark_marking_dialog.py`,
`apps/desktop/ui/main_window.py` (`_on_mark_landmarks_requested`),
`apps/desktop/ui/widgets/pitch_map.py`.

**Verified.** `tests/unit/test_auto_landmarks.py`,
`tests/integration/test_auto_landmarks_footage.py`,
`tests/unit/test_landmark_marking_dialog.py`,
`tests/unit/test_pitch_map_panel.py`.

---

## 6. Structured per-run pipeline logger

**Ask.** User: "create a logger module that logs for each pipeline run... it
must log each step in detail so agents and developers observe what's
currently odd in the pipeline."

**Fix.** `observability/logging/pipeline_run_log.py` (new) —
`PipelineRunLog`: one JSON file per triggered run, written to
`logs/CLI/run_<frame_id>_<run_id>.json`. Records, per stage: state, summary,
detail lines, timing (ms). Also records: the full confidence-signal chain,
the published verdict, and a `counts` block (players/poses/torso-confident/
calibration level+confidence+source/team counts/attacking-side-known/
identity state counts). `write()` never raises — logging must not break an
otherwise-successful call — and returns `None` cleanly when disabled.

Wired into `offside/pipeline.py::analyse()`: built once at the top of every
call, every stage's timing captured via the existing `run()` closure,
written just before returning.

**Config.** `LoggingConfig.pipeline_run_log: PipelineRunLogConfig` (`enabled:
bool = True`, `directory: str = "./logs/CLI"`).

**Files.** `observability/logging/pipeline_run_log.py`,
`offside/pipeline.py`, `core/config/schema.py`, `.gitignore`
(`logs/CLI/*.json`).

**Verified.** `tests/unit/test_pipeline_run_log.py`, 13 tests — disabled
config writes nothing, a write failure never raises, the verdict/signals/
counts sections are populated correctly from a real `DecisionExplainer`
output, `None` vs `0` distinguished correctly for not-yet-run stages.

This logger is what made every diagnosis after this point possible —
several of the findings below (§8, §12) were only provable by reading real
`logs/CLI/run_*.json` files rather than guessing from a screenshot.

---

## 7. `run_819` diagnosis: torso-keypoint yield was invisible one stage too late

**Symptom.** User opened `logs/CLI/run_819_006752147d72.json` directly and
asked "read this and tell me what's wrong." Team assignment had failed
completely: 0/19 players placed on a team.

**Root cause.** M2.2 (body keypoints) reported "19/19 players with a measured
foot position" — perfect. But M2.3 (team assignment) could only get a usable
shirt colour from 4 of 19 players. Feet and torso are **independently found**
keypoint groups from the same pose model — a side-on player, a crowded box,
or a shoulder cut off by a neighbour can leave someone with a confident ankle
and zero usable shoulders. Nothing surfaced this gap until it silently
produced "0 players placed on a team" three stages later, which sent the
operator hunting for a cause with no lead.

**Fix.**
- `offside/body_keypoints/keypoints.py` — `TORSO_KEYPOINTS` constant,
  `PlayerPose.has_confident_torso(min_confidence)` method.
- `offside/pipeline.py::_run_body_keypoints()` — now computes and reports
  torso-confident count *in M2.2's own stage report*: `"4/19 players have a
  confident torso (both shoulders and both hips) — this is what M2.3's
  shirt-colour sampler needs, and a good foot position does not guarantee
  it."` M2.2 now goes **DEGRADED** when torso yield is low, even if feet are
  fine.
- `FrameAnalysis.torso_confident_count` field; surfaced in the run log as
  `counts.poses_torso_confident` (omitted, not `0`, before the stage runs —
  `None` and "measured zero" are different facts).

**Files.** `offside/body_keypoints/keypoints.py`, `offside/pipeline.py`,
`observability/logging/pipeline_run_log.py`.

**Verified.** `tests/unit/test_body_keypoints.py` (`TestHasConfidentTorso`,
5 tests — including "a good ankle does not imply a good torso"),
`tests/unit/test_pipeline_run_log.py` (torso-yield present/omitted cases),
`tests/integration/test_pipeline_debugger.py` (real-clip assertion that the
M2.2 detail line and count are both populated).

---

## 8. `min_players_for_clustering`: 6 → 2 (the real mathematical floor)

**Ask.** Same conversation as §7: "why is min = 6 a must? we can do it on 2
as well. fix it too."

**Root cause.** The schema already documented 2 as the true minimum (a
2-means fit needs at least one point per cluster:
`Field(default=6, ge=2)`), but the *default* was set at 6 — a hard cliff.
Below 6 usable shirt colours, the stage refused outright and returned
nothing, even when 4–5 genuinely would have supported a fit. This
contradicted the module's own stated philosophy ("degrade honestly, never
refuse outright") — the confidence formula already scales down smoothly with
thin evidence (`len(usable) / confident_player_count`); the floor of 6 was
refusing to even attempt what the rest of the module already handles
gracefully.

**Fix.** `min_players_for_clustering` default `6 → 2` in all four places it's
declared: `core/config/schema.py`, `config/default.yaml`,
`offside/team_assignment/assigner.py`, `offside/team_assignment/clustering.py`.

**Verified live before shipping.** n=2 → real 0.2-confidence
`TeamColorModel` (not a crash, not a refusal). The exact `run_819` scenario
(4 usable colours) → real 0.4-confidence fit instead of an outright refusal.

**Tests.** `test_too_few_players_is_a_refusal_not_a_guess` (asserted 2
players → refusal) was the *wrong* test for the new intended behaviour —
replaced with `test_two_players_is_the_floor_not_a_refusal` (2 → real
low-confidence result) and `test_below_the_mathematical_floor_is_still_a_refusal`
(1 player → still refuses; the actual floor didn't move, only the
artificial cliff above it did).

---

## 9. Goalkeeper misidentified from the crowd

**Ask.** User: "a person in crowd can be misunderstood as goalkeeper... find
the outlier closer to the goal net... but before that we need to eliminate
crowd noise."

**Root cause, traced precisely.** Every COCO "person" detection — steward,
photographer, spectator standing close to the pitch — becomes a `player`
detection (`vision/detection/classes.py`); there was **no** on-pitch
filtering anywhere before team assignment. `TeamAssigner._identify_goalkeeper()`
ranks odd-kit outliers by how extreme their position is along the
goal-to-goal axis and picks whoever sits at the far end — which the user
correctly guessed is *already* equivalent to "closest to a goal net". The
actual bug: a crowd member standing **outside the pitch entirely** projects
to a *more* extreme position on that axis than any real keeper ever would
(a real keeper stands near their own goal line; a spectator in the stands is
metres further out) — so instead of merely being miscounted, the crowd
detection would **beat the real goalkeeper for the title, every time**. I
also found a `Pitch.contains()` method whose docstring claimed "M2.5 uses
this as a sanity check" — it didn't; that check only existed in the
rendering/overlay code, never in the real decision path.

**Fix.** `offside/team_assignment/assigner.py::_identify_goalkeepers()` now
uses `Pitch.contains(position, margin=goalkeeper_pitch_margin_m)` (new
config, default 8m) to disqualify any odd-kit candidate that projects well
outside the pitch, before ranking — when calibration is metric. The search
continues to the next candidate rather than stopping, so the real keeper
still wins if present. `TeamAssigner` now takes a `PitchModel` (defaults to
a standard 105×68m pitch if none is wired in).

**Files.** `offside/team_assignment/assigner.py`, `core/config/schema.py`
(`goalkeeper_pitch_margin_m`), `offside/pipeline.py` (passes `self._pitch`
through).

**Verified.** New tests reproduce the failure directly, then prove the fix:
`test_a_crowd_detection_far_outside_the_pitch_is_not_called_the_goalkeeper`
(crowd detection alone → zero goalkeepers found, not a false positive),
`test_the_real_goalkeeper_still_wins_over_a_crowd_detection` (both present →
real keeper correctly identified, crowd correctly excluded).

**Known remaining gap.** Only works when pitch calibration is metric — on an
uncalibrated frame there's no pitch-space position to check against, so this
risk remains open there.

---

## 10. Identity tracking was structurally incapable of reaching CONFIRMED

**Symptom.** User: "the current player identity tracking is super weak...
it just simply not identifying." Screenshot showed `IDENTITY: done,
degraded`.

**Root cause, proven from a real run log (`run_2710_*.json`).** All three
separate runs of the same frame showed identical results: `16 tentative, 0
confirmed, 0 recovered, 0 contested`. `IdentityState.CONFIRMED` requires 5
**consecutive** matched frames (`min_frames_to_confirm`), but
`OffsidePipeline.analyse()` was only ever called **once**, on the single
confirmed frame (`apps/desktop/viewmodels/offside_runner.py`,
`main_window.py`). Every track therefore always started at `hits=1` and was
judged immediately — `CONFIRMED` was unreachable in the shipped app's real
usage pattern, regardless of how clean the kit colours were. This had
nothing to do with kit colour or camera angle, despite how it looked.

**Fix.** `OffsidePipeline.warm_up(frames)` (new method) — runs
detection → body_keypoints → identity → calibration → team_assignment
(skipping offside_line/decision_support) over a list of preceding frames,
purely to build real track continuity, before `analyse()` judges the
confirmed frame. `ReviewSession`/`ReviewStrip` already freezes a ±15-frame
window around every candidate for the review UI (`analysis/results/review_session.py`)
— added `ReviewStrip.frames_before(frame_id)` to pull exactly the preceding
run-up out of that existing buffer, no new data source needed. Wired through
`OffsideRunner.analyse()` → `MainViewModel.check_offside()` →
`main_window._on_confirmed()`. No `PipelineRunLog` is written for warm-up
frames — they're priming, not a result.

**Files.** `offside/pipeline.py` (`warm_up`), `apps/desktop/viewmodels/offside_runner.py`,
`apps/desktop/viewmodels/main_viewmodel.py`, `apps/desktop/ui/main_window.py`,
`analysis/results/review_session.py`.

**Verified.** Synthetic reproduction with a stub detector/pose estimator
(no real model weights needed): `test_without_warm_up_every_identity_is_tentative`
reproduces the exact bug (0 confirmed after one frame);
`test_warm_up_lets_identities_reach_confirmed` proves the fix (all 6 players
reach CONFIRMED, `confidence: 1.0`, after warming up on 6 preceding frames).
`test_warm_up_writes_no_pipeline_run_log` confirms no log spam.

---

## 11. Player-identity tracker rebuilt — Tier 1: mechanics (camera motion + optimal assignment)

**Ask.** "The current player identification algorithm is not efficient."
Full design discussion landed on a 3-tier hybrid (mechanics → appearance
embeddings → jersey-number OCR); user chose to implement Tier 1 + Tier 2
now, Tier 3 deferred (see §14).

### 11a. No camera-motion compensation (the single biggest gap)

**Root cause.** A broadcast camera pans/tilts/zooms constantly. The
tracker's only defence was a smoothed constant-velocity term that *conflates
player motion with camera motion* — during a real pan, every player's box
shifts together by tens of pixels, IoU collapses under the match threshold,
and tracks die en masse. The codebase already solved exactly this problem
elsewhere: `offside/pitch_calibration/tracking.py::CalibrationFollower`
computes real inter-frame camera homographies via masked optical flow +
RANSAC, specifically to carry pitch marks across a pan. Identity tracking
simply didn't use it.

**Fix.** Extracted the shared primitive into
`vision/tracking/camera_motion.py` (`CameraMotionEstimator`,
`transform_box`, `transform_point`, `to_gray`) — a pure, behaviour-preserving
extraction from `CalibrationFollower`'s internals (its 11 existing tests
pass unchanged after the refactor). `IdentityTracker` now:
- warps every track's predicted box by the estimated camera motion before
  measuring IoU (`_Track.predict(motion_matrix)`), and
- computes player velocity *net of* camera motion, not conflating the two
  (`_Track.observe(..., motion_matrix)` — warps the old centre forward by
  the camera's own motion before computing the residual delta).

**Config.** `PlayerIdentityConfig.camera_motion_compensation: bool = True`,
`camera_motion_redetect_below: int = 120`.

**Proven directly.** `test_disabling_compensation_reproduces_the_old_failure`
— a 70px/frame pan needs the lower-confidence `_recover_lost` path to
survive at all without compensation.
`test_a_hard_pan_does_not_break_the_track` — the identical pan, same player,
stays continuously `CONFIRMED` with compensation on, no recovery needed.

### 11b. Greedy assignment → Hungarian (optimal)

**Root cause.** The old `_associate` took the global argmax, locked that
row/column, repeated. One locally-best pairing could force two *other*
tracks into a mutual swap — exactly the failure this phase exists to
prevent.

**Fix.** Replaced with `scipy.optimize.linear_sum_assignment` over a cost
matrix that embeds the kit-colour veto as a large-but-finite penalty (so
Hungarian routes around a vetoed pairing when a better option exists,
achieving the old greedy-loop's retry behaviour as one global optimum
instead of a sequence of local ones). Contested/ambiguous detection logic
preserved exactly, now evaluated against the true optimal assignment.

**Dependency.** Added `scipy>=1.11,<1.14` to `pyproject.toml` and
`requirements.txt`. **Caught during install**: `pip install scipy` silently
upgraded `numpy` from the project's pinned `1.26.4` to `2.5.3`, which would
have broken torch/opencv ABI compatibility across the whole project — fixed
immediately by reinstalling `numpy<2.0` and a compatible `scipy<1.14`.

**Files.** `vision/tracking/camera_motion.py` (new),
`offside/pitch_calibration/tracking.py` (refactored to compose it),
`offside/player_identity/tracker.py`, `core/config/schema.py`,
`pyproject.toml`, `requirements.txt`.

**Verified.** `tests/unit/test_camera_motion.py` (6 tests, new module in
isolation), `tests/unit/test_calibration_follow.py` (11/11 unchanged after
extraction), `tests/unit/test_player_identity.py` (18→21 tests).

---

## 12. Player-identity tracker — Tier 2: appearance embeddings (and two bugs it took real testing to catch)

**Ask.** Add a real appearance signal beyond kit colour, since colour is
*structurally* blind between two players on the same team (dressed
identically by design) — exactly the case a crowded-box "contested"
association can never resolve today.

**Design.** `offside/player_identity/appearance_embedding.py` (new) —
`AppearanceEmbedder` wraps a `timm` ImageNet backbone
(`mobilenetv3_small_100`, loaded the same way T-DEED's own backbone already
is — no new class of dependency) as a batched, L2-normalised per-crop
embedding. Deliberately *not* a purpose-trained person-ReID checkpoint
(OSNet/FastReID): that would mean trusting a third-party checkpoint from
outside anything this project already vets, versus reusing infrastructure
(`timm.create_model(..., pretrained=True)`) already proven here. Swappable
later without touching any caller — everything downstream only depends on
getting an L2-normalised vector back.

### 12a. First design was wrong: embeddings must never *resolve* doubt, only add it

**What I built first.** Let a confident-looking embedding match positively
clear a position-based "contested" ambiguity between two teammates — the
exact gap kit colour can't close.

**What testing found.** `test_ambiguous_overlap_between_teammates_is_reported_not_resolved`
(pre-existing test, protecting a real design guarantee) failed: two
teammates in identical kit, boxes overlapping 87%, got confidently
*resolved* instead of correctly staying `CONTESTED`. Traced it to actual
pixel content: at that much box overlap, each player's crop was measurably
contaminated by the *other* player's own paint bleeding across the box edge
(confirmed directly — 84 differing pixels between two crops that should have
been visually identical). The embedding wasn't reading a real appearance
difference; it was reading crop contamination, and reporting it as
confidence.

**Fix.** Reverted the "resolve" direction entirely. The embedding is now
strictly **one-directional, exactly like the kit-colour veto**: it may
refuse a pairing IoU and colour both accepted, but it can never clear a
doubt they raised. Safe regardless of calibration precision — a veto can
only make the system *more* cautious, never falsely confident. Removed the
now-dead `_embedding_resolves` method and its `embedding_resolve_margin`
config entirely rather than leave inert code behind.

### 12b. Second bug: the veto was too aggressive for *ordinary* frame-to-frame matching

**What I built next.** Folded the embedding veto into `_associate`'s main
per-frame cost matrix, alongside kit colour — applied to *every* match, not
just recoveries.

**What testing found.** All synthetic unit tests passed, but the real-footage
integration test (`tests/integration/test_player_identity_footage.py`, 25
real frames from the reference clip, real detector/pose models) failed:
`test_most_players_become_confirmed_as_evidence_accumulates` — only 8 of 18
players reached `CONFIRMED` (needed ≥9), with 8 forced through the slower
`_recover_lost` path instead. Confirmed via `git stash` that this test
passed cleanly (6/6) on the exact same clip before today's tracker changes.

**Root cause.** A generic ImageNet backbone's features genuinely shift
frame-to-frame on real broadcast footage — motion blur, a stride's change of
pose, a partial turn — even for the same player. Gating *every* IoU+colour
match on that additionally agreeing was asking a noisy signal to out-vote an
already-solid agreement, exactly where the extra caution wasn't needed.

**Fix.** Narrowed the embedding veto to `_recover_lost` only — occlusion
recovery, where a multi-frame gap has already weakened position's own claim
and "is this really the same player" is worth asking again. Ordinary
frame-to-frame association is back to kit colour as the sole veto (plus
Hungarian + camera-motion compensation from Tier 1).

**Files.** `offside/player_identity/appearance_embedding.py` (new),
`offside/player_identity/tracker.py`, `offside/player_identity/identity.py`
(`PlayerIdentity.embedding_distance` field, for observability),
`core/config/schema.py` (`use_appearance_embedding`, `embedding_model`,
`embedding_memory`, `embedding_max_distance`), `offside/pipeline.py` (passes
a resolved device through).

**Verified.**
- `tests/unit/test_appearance_embedding.py` (6 tests) — the embedding is
  genuinely discriminative (same crop → itself; visually different crops →
  well-separated, beyond repeat-measurement noise), degrades honestly
  (off-frame/zero-area crop → `None`, never fabricated).
- `tests/unit/test_player_identity.py` — `TestAppearanceEmbeddingVeto` (3
  tests, using a scripted fake embedder to sidestep the crop-contamination
  risk entirely): a wrong-build reappearance is correctly refused as a
  recovery; a genuine match still recovers normally (the veto isn't a second,
  stricter gate that blocks ordinary recoveries); disabling the feature
  falls back to exactly the old colour-and-position-only behaviour.
- `tests/integration/test_player_identity_footage.py` — real clip, 6/6
  passing again after the fix.

**Full suite after all of Tier 1 + Tier 2**: 554/559 passing. The remaining
5 are the same pre-existing `test_main_window.py` failures present all
session (an unrelated `_analyze_button` attribute gap), confirmed unrelated
throughout.

---

## 13. Frame 1162: "BALL" visibly on screen, but verdict says "no ball detected"

**Symptom.** Screenshot showed a clearly labelled "BALL" box near the
goalkeeper, `Attacking side: 0% POOR`, `Offside line: 0% POOR`, verdict
withheld.

**Diagnosis, from the real run log (`run_1162_b659e9c88b70.json`).** M2's own
detector call for this frame found `raw_detections: 16`, all players, **zero
ball detections** — the stage detail literally says `"16 players, no ball
this frame"`. That correctly cascades: no ball → no automatic signal for
which side is attacking → offside line can't be drawn →
`decision_support` reports `weakest link: Attacking side (M2.3) at 0.00` and
correctly withholds rather than guesses.

**The genuinely confusing part.** The visible "BALL" label did **not** come
from the same detector pass that produced the verdict. The M2.2 stage detail
line says it outright: `"detection size: 1280px (live preview uses 640px)"`
— two separate detector runs happen, a fast 640px pass for the live/browsing
overlay and a slower, more accurate 1280px pass M2 actually decides from. In
a goalmouth scramble (ball tight against the keeper's hands, motion blur
from a save) the two legitimately disagreed: the 640px pass caught it, the
1280px pass that matters didn't. Nothing in the UI currently discloses that
these are two different detections, so an operator sees a labelled ball
while being told "no ball was detected" — which reads as a bug even though
it's fully explainable from the logs.

**Status: diagnosed, not yet fixed.** Two options put to the user and not
yet decided: (a) make the overlay honestly disclose which detector pass a
box came from, or (b) feed the live-preview's ball detection through as a
fallback hint when M2's own higher-resolution pass misses. The panel's own
built-in suggestion — "set the attacking team by hand" — is the correct
immediate workaround and already exists as an operator override.

---

## Cross-cutting notes

- **"Degrade honestly, never guess" stayed the north star throughout.**
  Every fix above either widens what the pipeline can confidently measure
  (§2, §5, §9, §10, §11, §12) or makes an existing refusal/degradation more
  honest and specific (§1, §3, §7, §8, §13) — never papers over missing
  evidence with a guess. §12a is the sharpest example: the first embedding
  design would have quietly turned crop contamination into false confidence,
  and it was reverted specifically because a veto-only signal can't do that.
- **The pipeline run logger (§6) was a force multiplier.** Findings in §7,
  §10, and §13 were only provable — not just guessable — because a real
  per-stage JSON log existed to read. Built early in the session, used
  repeatedly afterward.
- **Every fix above was verified two ways where a real clip was available**:
  a synthetic/unit test proving the mechanism in isolation, and (where
  applicable) a real-footage integration test. §12b exists specifically
  because the synthetic tests alone would have shipped a real regression —
  the real-clip test is what caught it.
- **Full regression posture**: 554/559 passing after the most recent full
  suite run, with the 5 failures being the same pre-existing, unrelated
  `test_main_window.py` gap confirmed present since before this session
  started.
