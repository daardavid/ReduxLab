"""
Custom exception hierarchy for PCA Application.

All application-specific exceptions inherit from ``PCAAppError`` so
callers can catch a single base type.  Each subclass carries structured
context (file path, operation name, original error) that the logging
system and UI error dialogs can consume directly.
"""


class PCAAppError(Exception):
    """Base exception for all PCA application errors."""

    def __init__(self, message: str, *, context: dict | None = None):
        self.context = context or {}
        super().__init__(message)

    def __str__(self):
        base = super().__str__()
        if self.context:
            details = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{base} [{details}]"
        return base


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

class DataLoadError(PCAAppError):
    """Raised when a data file cannot be loaded or parsed."""

    def __init__(self, message: str, *, file_path: str = "", **ctx):
        super().__init__(message, context={"file_path": file_path, **ctx})
        self.file_path = file_path


class DataValidationError(PCAAppError):
    """Raised when loaded data fails validation checks."""

    def __init__(self, message: str, *, column: str = "", **ctx):
        super().__init__(message, context={"column": column, **ctx})
        self.column = column


class PreprocessingError(PCAAppError):
    """Raised during data cleaning / imputation / standardisation."""

    def __init__(self, message: str, *, step: str = "", **ctx):
        super().__init__(message, context={"step": step, **ctx})
        self.step = step


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class AnalysisError(PCAAppError):
    """Raised when a PCA or statistical analysis step fails."""

    def __init__(self, message: str, *, analysis_type: str = "", **ctx):
        super().__init__(message, context={"analysis_type": analysis_type, **ctx})
        self.analysis_type = analysis_type


class InsufficientDataError(AnalysisError):
    """Raised when there is not enough data to perform an analysis."""

    def __init__(self, message: str, *, rows: int = 0, cols: int = 0, **ctx):
        super().__init__(message, rows=rows, cols=cols, **ctx)
        self.rows = rows
        self.cols = cols


# ---------------------------------------------------------------------------
# Visualisation & Export
# ---------------------------------------------------------------------------

class VisualizationError(PCAAppError):
    """Raised when a plot or chart cannot be generated."""

    def __init__(self, message: str, *, viz_type: str = "", **ctx):
        super().__init__(message, context={"viz_type": viz_type, **ctx})
        self.viz_type = viz_type


class ExportError(PCAAppError):
    """Raised when exporting results (Excel, image, …) fails."""

    def __init__(self, message: str, *, dest_path: str = "", **ctx):
        super().__init__(message, context={"dest_path": dest_path, **ctx})
        self.dest_path = dest_path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConfigurationError(PCAAppError):
    """Raised for invalid or missing configuration."""

    def __init__(self, message: str, *, key: str = "", **ctx):
        super().__init__(message, context={"key": key, **ctx})
        self.key = key
