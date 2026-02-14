"""
Frames package for PCA Application.

Each analysis frame lives in its own module for maintainability.
This __init__.py re-exports every public frame class so existing
``from frames import X`` (or ``from refactored_frames import X``)
continues to work.
"""

from frontend.frames.series_frame import SeriesAnalysisFrame
from frontend.frames.cross_section_frame import CrossSectionAnalysisFrame
from frontend.frames.panel_frame import PanelAnalysisFrame
from frontend.frames.biplot_frame import BiplotAnalysisFrame
from frontend.frames.scatter_frame import ScatterAnalysisFrame
from frontend.frames.hierarchical_frame import HierarchicalClusteringFrame
from frontend.frames.correlation_frame import CorrelationAnalysisFrame

__all__ = [
    "SeriesAnalysisFrame",
    "CrossSectionAnalysisFrame",
    "PanelAnalysisFrame",
    "BiplotAnalysisFrame",
    "ScatterAnalysisFrame",
    "HierarchicalClusteringFrame",
    "CorrelationAnalysisFrame",
]
