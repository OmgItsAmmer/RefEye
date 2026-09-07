# Detector checkpoint (players + ball, M1/M2)

This directory holds `yolo11m.pt` — a general-purpose YOLOv11 object
detector, remapped from COCO classes onto "player" / "ball" (see
`vision/detection/classes.py`). **Not committed to git** (a binary under
`models/**/*.pt`; see `.gitignore`).

## Getting it

```bash
curl -L -o models/detector/yolo11m.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt
```

(~41 MB. Ultralytics, AGPL-3.0. `yolo11n.pt`, the nano variant this project
shipped with originally, still works as a smaller/faster fallback.)

## Why medium, not nano

`yolo11n.pt` (nano, ~5.6 MB) was the original choice. It was picked for the
Milestone 1 *live* preview path, where the detector runs continuously on a
video stream and speed matters directly. Nobody revisited it when the
offside path (Milestone 2) was added — a path that is **triggered once per
confirmed frame**, not run continuously, where the trade-off is completely
different: a slower, more accurate model costs a fraction of a second on one
still frame, not an ongoing per-frame tax.

Both `yolo11n.pt` and `yolo11m.pt` remain generic — pretrained on COCO, a
general-purpose photo dataset, not football broadcast footage. This upgrade
buys detection *capacity* (better recall on small, distant, occluded, or
motion-blurred players — exactly YOLO11-medium's advertised strength over
nano at the cost of inference time), not football-domain specificity. That
second gap is still open; see "A football-specific checkpoint" below.

## The ball problem this does *not* fix on its own

The ball confidence threshold (`ai.detector.ball_confidence_threshold`,
default 0.15) is deliberately loose — missing the real ball silently breaks
the attacking-side call (M2.3) and the "beyond the ball" offside check
(M2.5), and that's worse than an occasional false positive. Measured on
client footage before any size filtering: **more than one "ball" detection
appeared on 4 of every 10 sampled frames**, with sizes up to 31% of the
median player's height on screen (a real football is ~12-13%). A bigger
model changes *how many* candidates pass the confidence bar; it does not, by
itself, stop the wrong one from being picked.

That part is fixed separately, in `offside/ball_selection.py` — a size
sanity gate against every player's box on the same frame, run regardless of
which detector checkpoint is loaded. See that module's docstring for the
measured before/after.

## A football-specific checkpoint (not yet wired in)

Public football-finetuned YOLO checkpoints exist and would close the
remaining gap — a model trained on broadcast football has seen the specific
failure cases a generic model hasn't (kit occlusion, players bunched at a
free kick, ball-sized objects on ad boards) and typically ships with
separate player / goalkeeper / referee / ball classes instead of one generic
"person" class. Candidates worth evaluating on this project's own clips
before committing to one:

- [Roboflow: football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) — YOLO-format, player/goalkeeper/referee/ball classes.
- SoccerNet-finetuned YOLOv8l checkpoints (player detection and a separate
  dedicated ball detector) referenced in published SoccerNet tooling.

This is a bigger step than the nano→medium swap above: it needs validating
against real client footage (recall/precision, not just "it loads") before
it should replace the general-purpose checkpoint in `config/default.yaml`,
because a wrong class mapping or a model trained on a visually different
league/broadcast style can silently underperform a generic model rather than
beat it. Not done in this pass — flagged here as the next concrete step.

## The action spotter (Phase A) already does this

`ai/action_spotting/` (T-DEED, `models/tdeed/`) is *already* football-specific
— trained on SoccerNet, a football broadcast dataset — and is proof this
kind of upgrade is practical in this codebase. It is a different model for a
different job (finding *when* something happened, not detecting boxes on one
frame), so it doesn't help stage 1 directly, but it's the existing precedent
for "a domain-specific model is worth the integration effort here."
