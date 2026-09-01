"""Domain/application exceptions. See architecture.md section 47.

Detailed technical detail belongs in logs; messages shown to the UI must
stay understandable to a non-technical operator.
"""


class SoccerAnalysisError(Exception):
    """Base class for all application/domain errors."""


class StreamUnavailableError(SoccerAnalysisError):
    pass


class ModelInferenceError(SoccerAnalysisError):
    pass


class AnalysisWindowUnavailableError(SoccerAnalysisError):
    pass


class ConfigurationError(SoccerAnalysisError):
    pass


class ModelLoadError(SoccerAnalysisError):
    pass
