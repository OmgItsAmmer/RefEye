"""Visual debugger for the offside pipeline (development tool).

    python -m tools.pipeline_debugger [clip.mp4]

Shows each implemented stage's real output on real footage, and lists the
stages that are not built yet so the gap stays visible. Not part of the
shipped application — it exists to make M2's accuracy inspectable while it is
being built and tuned.
"""
