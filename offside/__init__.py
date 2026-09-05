"""Offside decision support (Milestone 2).

The module boundary reserved in architecture.md section 60. Everything in
here answers one question: *at the confirmed contact frame, where exactly is
everyone standing relative to each other and the goal line?*

Layout (phases from docs/milestones/M2_Plan.md):

    body_keypoints/        M2.2 — per-player body points, esp. feet
    pitch_calibration/     M2.1 — image -> top-down pitch map      (not built)
    field_geometry/        M2.1 — pitch landmark reference model   (not built)
    team_assignment/       M2.3 — attacker / defender / keeper     (not built)
    second_last_defender/  M2.5 — defender ranking                 (not built)
    offside_line/          M2.5 — line computation                 (not built)
    decision_support/      M2.6 — confidence + reasoning           (not built)

Two rules bind every module here (M2_Plan section 7):

  * Every stage reports its own confidence and a plain-language reason. A
    result that had to be guessed says so; it is never dressed up as a
    measurement.
  * Nothing tuning-related is hard-coded. Thresholds and penalties come from
    `AppSettings.offside`, so the pipeline can be re-tuned for a different
    broadcaster without a code change.
"""
