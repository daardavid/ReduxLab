"""
Biplot Analysis Frame for PCA Application

Contains the BiplotAnalysisFrame class for advanced biplot analysis
with customizable markers, colors, and unit categorization.
"""

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import messagebox
from frontend.base_analysis_frame import BaseAnalysisFrame
from backend.group_analysis_mixin import GroupAnalysisMixin
import pandas as pd
import os


class BiplotAnalysisFrame(BaseAnalysisFrame, GroupAnalysisMixin):
    """Unified frame for cross-section / biplot analysis with group filtering."""

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.setup_ui()

    def setup_ui(self):
        desc = ttk.Label(
            self,
            text="Cross-section biplot analysis with customizable markers, groups, and unit categorization.",
            wraplength=800,
            style='secondary.TLabel',
            font=("Helvetica", 11)
        )
        desc.pack(pady=(0, 30), anchor=W)

        # Initialize transformation variables FIRST (before creating config cards)
        self.apply_transformations = tk.BooleanVar(value=False)
        self.transformation_method = tk.StringVar(value='auto')
        self.skewness_threshold = tk.DoubleVar(value=1.0)
        self.arrow_scale_str = tk.StringVar(value='0.0 (Auto)')

        # Initialize biplot state
        self.biplot_config = {
            'categorization_scheme': 'continents',
            'marker_scheme': 'classic',
            'color_scheme': 'viridis',
            'show_arrows': True,
            'show_labels': True,
            'alpha': 0.7
        }

        # Configuration cards
        self.create_config_card("Data File", self.setup_file_config)
        self.create_config_card("Indicators", self.setup_indicators_config)
        self.create_config_card("Research Units", self.setup_units_config)
        self.create_config_card("Analysis Year", self.setup_year_config)
        self.create_config_card("Advanced Options", self.setup_advanced_config)
        self.create_config_card("Visual Configuration", self.setup_visual_config)

        # Add group management section (from GroupAnalysisMixin)
        self.setup_group_integration(self)

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
            text="🎨 Configure Biplot",
            style='outline.TButton',
            command=self.configure_biplot
        ).pack(anchor=W)

    def select_units(self):
        """Select research units."""
        super().select_units(title="Select Research Units/Countries")
    
    def setup_advanced_config(self, parent):
        """Setup advanced configuration for transformations."""
        # ✅ Use pack() consistently instead of grid() to avoid conflicts
        
        # Transformations section
        ttk.Label(
            parent, 
            text="Data Transformations:", 
            font=("Helvetica", 10, "bold")
        ).pack(anchor=W, pady=(0, 5))
        
        ttk.Checkbutton(
            parent,
            text="Apply automatic transformations",
            variable=self.apply_transformations
        ).pack(anchor=W, pady=2)
        
        # Method selection
        method_frame = ttk.Frame(parent)
        method_frame.pack(fill=X, pady=2)
        ttk.Label(method_frame, text="Method:").pack(side=LEFT, padx=(20, 5))
        method_combo = ttk.Combobox(
            method_frame, 
            textvariable=self.transformation_method,
            values=['auto', 'log', 'log1p', 'sqrt', 'box-cox', 'yeo-johnson'],
            state="readonly",
            width=15
        )
        method_combo.pack(side=LEFT)
        
        # Skewness threshold
        threshold_frame = ttk.Frame(parent)
        threshold_frame.pack(fill=X, pady=2)
        ttk.Label(threshold_frame, text="Skewness threshold:").pack(side=LEFT, padx=(20, 5))
        threshold_scale = ttk.Scale(
            threshold_frame, 
            from_=0.5, 
            to=2.0,
            variable=self.skewness_threshold,
            orient=tk.HORIZONTAL,
            length=150
        )
        threshold_scale.pack(side=LEFT)
        
        threshold_label = ttk.Label(
            threshold_frame, 
            text=f"{self.skewness_threshold.get():.1f}"
        )
        threshold_label.pack(side=LEFT, padx=5)
        
        # Update label when slider changes
        def update_threshold_label(*args):
            threshold_label.config(text=f"{self.skewness_threshold.get():.1f}")
        self.skewness_threshold.trace_add('write', update_threshold_label)
        
        # Arrow scale section
        ttk.Label(
            parent,
            text="Biplot Vector Scale:",
            font=("Helvetica", 10, "bold")
        ).pack(anchor=W, pady=(15, 5))
        
        arrow_frame = ttk.Frame(parent)
        arrow_frame.pack(fill=X, pady=2)
        ttk.Label(arrow_frame, text="Arrow scale:").pack(side=LEFT)
        arrow_combo = ttk.Combobox(
            arrow_frame,
            textvariable=self.arrow_scale_str,
            values=['0.0 (Auto)', '0.2', '0.3', '0.4', '0.5', '0.8', '1.0'],
            state="readonly",
            width=15
        )
        arrow_combo.pack(side=LEFT, padx=(5, 0))
        arrow_combo.set('0.0 (Auto)')
        
        ttk.Label(
            parent,
            text="ℹ️ Auto-calculates optimal scale for vector visibility",
            font=("Helvetica", 8),
            foreground="gray"
        ).pack(anchor=W, pady=(2, 0))

    def get_config(self):
        """Get configuration for biplot analysis with group filtering support."""
        # Use group-filtered units if available, otherwise fall back to all selected
        current_units = self.get_current_units() if hasattr(self, 'get_current_units') else self.selected_countries

        if (hasattr(self, 'file_entry') and self.file_entry.get().strip() and
            self.selected_indicators and
            current_units and
            self.selected_year):

            # Parse arrow_scale from string
            arrow_str = self.arrow_scale_str.get()
            if 'Auto' in arrow_str:
                arrow_scale = None
            else:
                try:
                    arrow_scale = float(arrow_str)
                except ValueError:
                    arrow_scale = None

            # Get filtered units based on group selection
            filtered_units = self.get_filtered_units_for_analysis() if hasattr(self, 'get_filtered_units_for_analysis') else self.selected_countries

            base_config = {
                'data_file': self.file_entry.get().strip(),
                'selected_sheet_names': self.selected_indicators,
                'selected_countries': filtered_units,
                'target_year': self.selected_year,
                'biplot_config': self.biplot_config,
                # Advanced options
                'apply_transformations': self.apply_transformations.get(),
                'transformation_method': self.transformation_method.get(),
                'skewness_threshold': self.skewness_threshold.get(),
                'arrow_scale': arrow_scale
            }

            # Enhance with group information if mixin is active
            if hasattr(self, 'get_group_enhanced_config'):
                return self.get_group_enhanced_config(base_config)
            return base_config
        return None

    def configure_biplot(self):
        """Configure biplot options."""
        dialog = tk.Toplevel(self)
        dialog.title("Configure Advanced Biplot")
        dialog.geometry("500x600")
        dialog.transient(self)
        dialog.grab_set()

        # Title
        ttk.Label(dialog, text="Biplot Configuration Options:", font=("Helvetica", 14, "bold")).pack(pady=10)

        # Main frame with scrollbar
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=BOTH, expand=YES, padx=20, pady=(0, 20))

        # Canvas and scrollbar
        canvas = tk.Canvas(main_frame, height=500)  # Increased height
        scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Categorization scheme
        ttk.Label(scrollable_frame, text="Categorization Scheme:", font=("Helvetica", 11, "bold")).pack(anchor=W, pady=(10, 5))
        cat_var = tk.StringVar(value=self.biplot_config.get('categorization_scheme', 'continents'))
        cat_combo = ttk.Combobox(scrollable_frame, textvariable=cat_var,
                                values=['continents', 'income_groups', 'regions', 'custom'], state="readonly")
        cat_combo.pack(fill=X, pady=(0, 10))

        # Marker scheme
        ttk.Label(scrollable_frame, text="Marker Scheme:", font=("Helvetica", 11, "bold")).pack(anchor=W, pady=(10, 5))
        marker_var = tk.StringVar(value=self.biplot_config.get('marker_scheme', 'classic'))
        marker_combo = ttk.Combobox(scrollable_frame, textvariable=marker_var,
                                   values=['classic', 'shapes', 'numbers', 'letters'], state="readonly")
        marker_combo.pack(fill=X, pady=(0, 10))

        # Color scheme
        ttk.Label(scrollable_frame, text="Color Scheme:", font=("Helvetica", 11, "bold")).pack(anchor=W, pady=(10, 5))
        color_var = tk.StringVar(value=self.biplot_config.get('color_scheme', 'viridis'))
        color_combo = ttk.Combobox(scrollable_frame, textvariable=color_var,
                                  values=['viridis', 'plasma', 'inferno', 'magma', 'tab10', 'tab20'], state="readonly")
        color_combo.pack(fill=X, pady=(0, 10))

        # Visualization options
        ttk.Label(scrollable_frame, text="Visualization Options:", font=("Helvetica", 11, "bold")).pack(anchor=W, pady=(10, 5))

        show_arrows_var = tk.BooleanVar(value=self.biplot_config.get('show_arrows', True))
        ttk.Checkbutton(scrollable_frame, text="Show variable arrows", variable=show_arrows_var).pack(anchor=W, pady=2)

        show_labels_var = tk.BooleanVar(value=self.biplot_config.get('show_labels', True))
        ttk.Checkbutton(scrollable_frame, text="Show country labels", variable=show_labels_var).pack(anchor=W, pady=2)

        # Transparency
        ttk.Label(scrollable_frame, text="Transparency (Alpha):", font=("Helvetica", 11, "bold")).pack(anchor=W, pady=(10, 5))
        alpha_var = tk.DoubleVar(value=self.biplot_config.get('alpha', 0.7))
        alpha_scale = ttk.Scale(scrollable_frame, from_=0.1, to=1.0, variable=alpha_var, orient=HORIZONTAL)
        alpha_scale.pack(fill=X, pady=(0, 5))
        ttk.Label(scrollable_frame, textvariable=alpha_var, font=("Consolas", 9)).pack(anchor=W)

        canvas.pack(side=LEFT, fill=BOTH, expand=YES)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=X, padx=20, pady=10)

        def apply_config():
            self.biplot_config.update({
                'categorization_scheme': cat_var.get(),
                'marker_scheme': marker_var.get(),
                'color_scheme': color_var.get(),
                'show_arrows': show_arrows_var.get(),
                'show_labels': show_labels_var.get(),
                'alpha': alpha_var.get()
            })
            dialog.destroy()
            tk.messagebox.showinfo("Configuration", "Biplot configuration applied successfully")

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
