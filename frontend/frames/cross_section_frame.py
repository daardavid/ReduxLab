"""
Cross-Section Analysis Frame for PCA Application

Contains the CrossSectionAnalysisFrame class for comparative analysis
between multiple research units at a specific time, with group filtering support.
"""

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import messagebox
from frontend.base_analysis_frame import BaseAnalysisFrame
from backend.group_analysis_mixin import GroupAnalysisMixin
from backend import data_loader_module as dl
import pandas as pd
import os


class CrossSectionAnalysisFrame(BaseAnalysisFrame, GroupAnalysisMixin):
    """Refactored frame for cross-section analysis with group filtering."""

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.setup_ui()

    def setup_ui(self):
        desc = ttk.Label(
            self,
            text="Comparative analysis between multiple research units at a specific time.",
            wraplength=800,
            style='secondary.TLabel',
            font=("Helvetica", 11)
        )
        desc.pack(pady=(0, 30), anchor=W)

        # ✅ Initialize transformation variables FIRST (before creating config cards)
        self.apply_transformations = tk.BooleanVar(value=False)
        self.transformation_method = tk.StringVar(value='auto')
        self.skewness_threshold = tk.DoubleVar(value=1.0)
        self.arrow_scale_str = tk.StringVar(value='0.0 (Auto)')  # String for combobox

        # Configuration cards
        self.create_config_card("Data File", self.setup_file_config)
        self.create_config_card("Indicators", self.setup_indicators_config)
        self.create_config_card("Research Units", self.setup_units_config)
        self.create_config_card("Analysis Year", self.setup_year_config)
        self.create_config_card("Advanced Options", self.setup_advanced_config)
        
        # Add group integration
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

    def select_units(self):
        """Select research units."""
        super().select_units(title="Select Research Units/Countries")
    
    def setup_advanced_config(self, parent):
        """Setup advanced configuration for transformations and visualization."""
        # ✅ Use pack() consistently instead of grid() to avoid conflicts
        
        # Transformations section
        trans_label = ttk.Label(
            parent, 
            text="Data Transformations:", 
            font=("Helvetica", 10, "bold")
        )
        trans_label.pack(anchor=W, pady=(0, 5))
        
        ttk.Checkbutton(
            parent,
            text="Apply automatic transformations (for financial data)",
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
        ).pack(anchor=W, padx=(20, 0), pady=(0, 5))
        
        # Analysis button
        ttk.Button(
            parent,
            text="📊 Analyze Data Distribution",
            style='outline.TButton',
            command=self.analyze_distribution
        ).pack(anchor=W, pady=(10, 0))

    def analyze_distribution(self):
        """Analyze data distribution to recommend transformations."""
        try:
            if not hasattr(self, 'file_entry') or not self.file_entry.get().strip():
                messagebox.showwarning("Warning", "Please load a data file first.")
                return
            
            if not self.selected_indicators:
                messagebox.showwarning("Warning", "Please select indicators first.")
                return
            
            # Import here to avoid circular dependencies
            from backend.data_transformations import analyze_data_distribution
            from backend import data_loader_module as dl
            
            # Load data
            file_path = self.file_entry.get().strip()
            df_dict = dl.cargar_datos_multiples(file_path, self.selected_indicators)
            
            if not df_dict:
                messagebox.showerror("Error", "Failed to load data.")
                return
            
            # Merge data from all indicators
            df_merged = pd.concat([df_dict[ind] for ind in df_dict.keys()], axis=1)
            
            # Analyze
            analysis = analyze_data_distribution(
                df_merged, 
                skewness_threshold=self.skewness_threshold.get()
            )
            
            # Show results in dialog
            dialog = tk.Toplevel(self)
            dialog.title("Data Distribution Analysis")
            dialog.geometry("700x500")
            dialog.transient(self)
            
            text_widget = tk.Text(dialog, wrap=tk.WORD, font=("Consolas", 9))
            text_widget.pack(fill=BOTH, expand=YES, padx=10, pady=10)
            
            scrollbar = ttk.Scrollbar(dialog, command=text_widget.yview)
            scrollbar.pack(side=RIGHT, fill=Y)
            text_widget.config(yscrollcommand=scrollbar.set)
            
            # Format results
            text_widget.insert(tk.END, "📊 DATA DISTRIBUTION ANALYSIS\n", "header")
            text_widget.insert(tk.END, "="*70 + "\n\n")
            
            transform_count = 0
            for col, info in analysis.items():
                text_widget.insert(tk.END, f"📌 {col}\n", "bold")
                text_widget.insert(tk.END, f"   Type: {info['type']}\n")
                text_widget.insert(tk.END, f"   Skewness: {info['skewness']:.2f}\n")
                
                if info['needs_transform']:
                    text_widget.insert(tk.END, f"   ⚠️ Transformation recommended\n", "warning")
                    transform_count += 1
                else:
                    text_widget.insert(tk.END, f"   ✓ Distribution acceptable\n", "ok")
                
                text_widget.insert(tk.END, "\n")
            
            text_widget.insert(tk.END, "="*70 + "\n")
            text_widget.insert(tk.END, f"\nSummary: {transform_count}/{len(analysis)} columns need transformation\n", "summary")
            
            if transform_count > 0:
                text_widget.insert(tk.END, "\n💡 Recommendation: Enable 'Apply automatic transformations'\n", "recommendation")
            
            # Configure tags
            text_widget.tag_config("header", font=("Consolas", 11, "bold"), foreground="#2563eb")
            text_widget.tag_config("bold", font=("Consolas", 9, "bold"))
            text_widget.tag_config("warning", foreground="#dc2626")
            text_widget.tag_config("ok", foreground="#16a34a")
            text_widget.tag_config("summary", font=("Consolas", 10, "bold"))
            text_widget.tag_config("recommendation", foreground="#2563eb", font=("Consolas", 9, "italic"))
            
            text_widget.config(state=tk.DISABLED)
            
        except ImportError:
            messagebox.showerror(
                "Module Not Found",
                "data_transformations module not available.\nPlease ensure it's in the same directory."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze distribution:\n{str(e)}")

    def get_config(self):
        """Get configuration for cross-section analysis with group filtering."""
        if (hasattr(self, 'file_entry') and self.file_entry.get().strip() and
            self.selected_indicators and
            self.get_current_units() and
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

            base_config = {
                'data_file': self.file_entry.get().strip(),
                'selected_sheet_names': self.selected_indicators,
                'selected_countries': self.get_filtered_units_for_analysis(),
                'target_year': self.selected_year,
                # Advanced options
                'apply_transformations': self.apply_transformations.get(),
                'transformation_method': self.transformation_method.get(),
                'skewness_threshold': self.skewness_threshold.get(),
                'arrow_scale': arrow_scale
            }
            
            # Enhance with group information
            return self.get_group_enhanced_config(base_config)
        return None
