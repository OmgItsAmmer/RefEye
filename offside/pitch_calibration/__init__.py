"""Camera-to-pitch calibration (M2.1).

Read `calibrator.py`'s docstring before extending this: it records which
approaches were tried against real broadcast footage and why the automatic
path stops where it does.

    PitchCalibrator      builds calibrations (manual = metric, auto = directional)
    PitchCalibration     what a frame supports, with confidence and reasons
    CalibrationLevel     NONE / DIRECTIONAL / METRIC
    detect_pitch_lines   the painted markings, as line segments
    pitch_line_mask      what the line finder sees (for the debug view)
"""

from offside.pitch_calibration.calibrator import (
    CalibrationLevel,
    LineFamily,
    PitchCalibration,
    PitchCalibrator,
)
from offside.pitch_calibration.homography import (
    HomographyFit,
    PointCorrespondence,
    line_through_vanishing_point,
    solve_homography,
)
from offside.pitch_calibration.line_detection import (
    DetectedLine,
    VanishingPoint,
    detect_pitch_lines,
    estimate_vanishing_point,
    merge_lines,
    pitch_line_mask,
    split_by_orientation,
)

__all__ = [
    "CalibrationLevel",
    "DetectedLine",
    "HomographyFit",
    "LineFamily",
    "PitchCalibration",
    "PitchCalibrator",
    "PointCorrespondence",
    "VanishingPoint",
    "detect_pitch_lines",
    "estimate_vanishing_point",
    "line_through_vanishing_point",
    "merge_lines",
    "pitch_line_mask",
    "solve_homography",
    "split_by_orientation",
]
