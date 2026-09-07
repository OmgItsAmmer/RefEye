"""Re-exports the product's offside pipeline for the debug tool.

The pipeline itself moved to `offside/pipeline.py` — it is the product's
pipeline, not the debugger's, and living under `tools/` was making that untrue
whenever the shipped app finally wanted to run it. This module stays so
`from tools.pipeline_debugger.pipeline import ...` keeps working for the
debugger and its tests without a second implementation to drift out of sync.
"""

from offside.pipeline import (
    FrameAnalysis,
    OffsidePipeline,
    StageReport,
    StageState,
)

__all__ = ["FrameAnalysis", "OffsidePipeline", "StageReport", "StageState"]
