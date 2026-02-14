"""
Scatter Analysis Frame for PCA Application

Contains the ScatterAnalysisFrame class for interactive scatter plot analysis
with principal components and customization options.
"""

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import messagebox
from frontend.base_analysis_frame import BaseAnalysisFrame
import pandas as pd
import os


class ScatterAnalysisFrame(BaseAnalysisFrame):
    """Refactored frame for scatter plot analysis."""

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.setup_ui()

    def setup_ui(self):
        desc = ttk.Label(
            self,
            text="Interactive scatter plot with principal components and customization options.",
            wraplength=800,
            style='secondary.TLabel',
            font=("Helvetica", 11)
        )
        desc.pack(pady=(0, 30), anchor=W)

        # Configuration cards
        self.create_config_card("Data File", self.setup_file_config)
        self.create_config_card("Indicators", self.setup_indicators_config)
        self.create_config_card("Research Units", self.setup_units_config)
        self.create_config_card("Analysis Year", self.setup_year_config)
        self.create_config_card("Visual Configuration", self.setup_visual_config)

        # Initialize state
        self.scatter_config = {
            'pc_x': 1,
            'pc_y': 2,
            'use_cmap': False,
            'cmap': 'viridis',
            'density': False,
            'gradient': '',
            'alpha': 0.7,
            'point_size': 30,
            'edgecolor': 'None',
            'HT2': False,
            'SPE': False,
            'show_labels': False
        }

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

    def setup_year_config(self, parent):
        """Setup year selection configuration."""
        year_frame = ttk.Frame(parent)
        year_frame.pack(fill=X, pady=(0, 10))

        ttk.Button(
            year_frame,
            text="📅 Select Year",
            style='outline.TButton',
            command=self.select_year
        ).pack(side=LEFT, padx=(0, 20))

        self.year_status = ttk.Label(
            year_frame,
            text="None selected",
            style='secondary.TLabel'
        )
        self.year_status.pack(side=LEFT)

    def setup_visual_config(self, parent):
        """Setup visual configuration."""
        ttk.Button(
            parent,
            text="🟢 Configure Scatterplot",
            style='outline.TButton',
            command=self.configure_scatter
        ).pack(anchor=W)

    def select_units(self):
        """Select research units."""
        super().select_units(title="Select Research Units/Countries")


    def configure_scatter(self):
        """Configure scatterplot options."""
        dialog = tk.Toplevel(self)
        dialog.title("Configure PCA Scatterplot")
        dialog.geometry("500x500")
        dialog.transient(self)
        dialog.grab_set()

        # Title
        ttk.Label(dialog, text="Scatterplot Configuration Options:", font=("Helvetica", 14, "bold")).pack(pady=10)

        # Main frame
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=BOTH, expand=YES, padx=20, pady=(0, 20))

        # Principal components
        ttk.Label(main_frame, text="Principal Components:", font=("Helvetica", 11, "bold")).pack(anchor=W, pady=(10, 5))

        pc_frame = ttk.Frame(main_frame)
        pc_frame.pack(fill=X, pady=(0, 10))

        ttk.Label(pc_frame, text="PC X:").grid(row=0, column=0, sticky=W, padx=(0, 5))
        pc_x_var = tk.IntVar(value=self.scatter_config.get('pc_x', 1))
        pc_x_combo = ttk.Combobox(pc_frame, textvariable=pc_x_var, values=[1, 2, 3, 4, 5], state="readonly", width=5)
        pc_x_combo.grid(row=0, column=1, padx=(0, 20))

        ttk.Label(pc_frame, text="PC Y:").grid(row=0, column=2, sticky=W, padx=(0, 5))
        pc_y_var = tk.IntVar(value=self.scatter_config.get('pc_y', 2))
        pc_y_combo = ttk.Combobox(pc_frame, textvariable=pc_y_var, values=[1, 2, 3, 4, 5], state="readonly", width=5)
        pc_y_combo.grid(row=0, column=3)

        # Color scheme
        ttk.Label(main_frame, text="Color Scheme:", font=("Helvetica", 11, "bold")).pack(anchor=W, pady=(10, 5))

        color_frame = ttk.Frame(main_frame)
        color_frame.pack(fill=X, pady=(0, 10))

        use_cmap_var = tk.BooleanVar(value=self.scatter_config.get('use_cmap', False))
        ttk.Checkbutton(color_frame, text="Use color map", variable=use_cmap_var).pack(anchor=W)

        ttk.Label(color_frame, text="Color map:").pack(anchor=W, pady=(5, 0))
        cmap_var = tk.StringVar(value=self.scatter_config.get('cmap', 'viridis'))
        cmap_combo = ttk.Combobox(color_frame, textvariable=cmap_var,
                                 values=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm'], state="readonly")
        cmap_combo.pack(fill=X, pady=(0, 5))

        # Visualization options
        ttk.Label(main_frame, text="Visualization Options:", font=("Helvetica", 11, "bold")).pack(anchor=W, pady=(10, 5))

        show_labels_var = tk.BooleanVar(value=self.scatter_config.get('show_labels', False))
        ttk.Checkbutton(main_frame, text="Show country labels", variable=show_labels_var).pack(anchor=W, pady=2)

        density_var = tk.BooleanVar(value=self.scatter_config.get('density', False))
        ttk.Checkbutton(main_frame, text="Show density", variable=density_var).pack(anchor=W, pady=2)

        # Statistical controls
        ttk.Label(main_frame, text="Statistical Controls:", font=("Helvetica", 11, "bold")).pack(anchor=W, pady=(10, 5))

        ht2_var = tk.BooleanVar(value=self.scatter_config.get('HT2', False))
        ttk.Checkbutton(main_frame, text="Show Hotelling T² ellipse", variable=ht2_var).pack(anchor=W, pady=2)

        spe_var = tk.BooleanVar(value=self.scatter_config.get('SPE', False))
        ttk.Checkbutton(main_frame, text="Show SPE limits", variable=spe_var).pack(anchor=W, pady=2)

        # Appearance
        ttk.Label(main_frame, text="Appearance:", font=("Helvetica", 11, "bold")).pack(anchor=W, pady=(10, 5))

        size_frame = ttk.Frame(main_frame)
        size_frame.pack(fill=X, pady=(0, 5))

        ttk.Label(size_frame, text="Point size:").grid(row=0, column=0, sticky=W)
        point_size_var = tk.IntVar(value=self.scatter_config.get('point_size', 30))
        size_scale = ttk.Scale(size_frame, from_=10, to=100, variable=point_size_var, orient=HORIZONTAL)
        size_scale.grid(row=0, column=1, sticky=EW, padx=(10, 0))
        ttk.Label(size_frame, textvariable=point_size_var).grid(row=0, column=2, padx=(5, 0))

        alpha_frame = ttk.Frame(main_frame)
        alpha_frame.pack(fill=X, pady=(5, 10))

        ttk.Label(alpha_frame, text="Transparency:").grid(row=0, column=0, sticky=W)
        alpha_var = tk.DoubleVar(value=self.scatter_config.get('alpha', 0.7))
        alpha_scale = ttk.Scale(alpha_frame, from_=0.1, to=1.0, variable=alpha_var, orient=HORIZONTAL)
        alpha_scale.grid(row=0, column=1, sticky=EW, padx=(10, 0))
        ttk.Label(alpha_frame, textvariable=alpha_var, font=("Consolas", 9)).grid(row=0, column=2, padx=(5, 0))

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=X, padx=20, pady=10)

        def apply_config():
            self.scatter_config.update({
                'pc_x': pc_x_var.get(),
                'pc_y': pc_y_var.get(),
                'use_cmap': use_cmap_var.get(),
                'cmap': cmap_var.get(),
                'density': density_var.get(),
                'alpha': alpha_var.get(),
                'point_size': point_size_var.get(),
                'HT2': ht2_var.get(),
                'SPE': spe_var.get(),
                'show_labels': show_labels_var.get()
            })
            dialog.destroy()
            tk.messagebox.showinfo("Configuration", "Scatterplot configuration applied successfully")

        ttk.Button(
            button_frame,
            text="Apply",
            style='success.TButton',
            command=apply_config
        ).pack(side=RIGHT, padx=(10, 0))

        ttk.Button(
            button_frame,
            text="Cancel",
            command=dialog.destroy
        ).pack(side=RIGHT)

    def get_config(self):
        """
        Get scatterplot analysis configuration from the UI state.
        Uses BaseAnalysisFrame attributes: file_entry, selected_indicators,
        selected_countries, selected_year.
        """
        # Validate data file
        data_file = self.file_entry.get().strip() if hasattr(self, 'file_entry') else ''
        if not data_file:
            raise ValueError("No se ha seleccionado un archivo de datos. Selecciona un archivo primero.")

        # Validate indicators
        if not self.selected_indicators:
            raise ValueError("No se han seleccionado indicadores. Selecciona al menos uno.")

        # Validate units
        if not self.selected_countries:
            raise ValueError("No se han seleccionado unidades de investigación. Selecciona al menos una.")

        # Validate year
        if not self.selected_year:
            raise ValueError("No se ha seleccionado un año. Selecciona un año objetivo.")

        return {
            'data_file': data_file,
            'selected_indicators': self.selected_indicators,
            'selected_countries': self.selected_countries,
            'target_year': self.selected_year,
            'analysis_type': 'cross_section',
            'scatter_config': self.scatter_config
        }
