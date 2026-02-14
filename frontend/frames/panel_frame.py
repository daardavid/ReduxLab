"""
Panel Analysis Frame for PCA Application

Contains the PanelAnalysisFrame class for 3D visualization of research unit
trajectories in PCA space over time.
"""

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from frontend.base_analysis_frame import BaseAnalysisFrame


class PanelAnalysisFrame(BaseAnalysisFrame):
    """Refactored frame for panel analysis."""

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.setup_ui()

    def setup_ui(self):
        desc = ttk.Label(
            self,
            text="3D visualization of research unit trajectories in PCA space over time.",
            wraplength=800,
            style='secondary.TLabel',
            font=("Helvetica", 11)
        )
        desc.pack(pady=(0, 30), anchor=W)

        # Configuration cards
        self.create_config_card("Data File", self.setup_file_config)
        self.create_config_card("Indicators", self.setup_indicators_config)
        self.create_config_card("Research Units", self.setup_units_config)
        self.create_config_card("Analysis Period", self.setup_years_config)

    def setup_units_config(self, parent):
        """Setup units selection configuration."""
        units_frame = ttk.Frame(parent)
        units_frame.pack(fill=X, pady=(0, 10))

        ttk.Button(
            units_frame,
            text="🏢 Select Units",
            style='outline.TButton',
            command=self.select_units
        ).pack(side=LEFT, padx=(0, 20))

        self.units_status = ttk.Label(
            units_frame,
            text="None selected",
            style='secondary.TLabel'
        )
        self.units_status.pack(side=LEFT)

    def setup_years_config(self, parent):
        """Setup years selection configuration."""
        years_frame = ttk.Frame(parent)
        years_frame.pack(fill=X, pady=(0, 10))

        ttk.Button(
            years_frame,
            text="📅 Select Years",
            style='outline.TButton',
            command=self.select_years
        ).pack(side=LEFT, padx=(0, 20))

        self.years_status = ttk.Label(
            years_frame,
            text="None selected",
            style='secondary.TLabel'
        )
        self.years_status.pack(side=LEFT)

    def select_units(self):
        """Select research units for PCA 3D."""
        super().select_units(title="Select Units for PCA 3D")

    def get_config(self):
        """Get configuration for panel analysis."""
        if (hasattr(self, 'file_entry') and self.file_entry.get().strip() and
            self.selected_indicators and
            self.selected_units and
            self.selected_years):

            return {
                'data_file': self.file_entry.get().strip(),
                'selected_sheet_names': self.selected_indicators,
                'selected_countries': self.selected_units,
                'years_range': (min(self.selected_years), max(self.selected_years))
            }
        return None
