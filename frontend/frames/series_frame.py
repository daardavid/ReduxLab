"""
Series Analysis Frame for PCA Application

Contains the SeriesAnalysisFrame class for time series analysis
of a specific research unit over time.
"""

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import messagebox
from frontend.base_analysis_frame import BaseAnalysisFrame
from backend import data_loader_module as dl
import os


class SeriesAnalysisFrame(BaseAnalysisFrame):
    """Refactored frame for series analysis."""

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.setup_ui()

    def setup_ui(self):
        # Description
        desc = ttk.Label(
            self,
            text="Time series analysis for a specific research unit over time.",
            wraplength=800,
            style='secondary.TLabel',
            font=("Helvetica", 11)
        )
        desc.pack(pady=(0, 30), anchor=W)

        # Configuration cards
        self.create_config_card("Data File", self.setup_file_config)
        self.create_config_card("Indicators", self.setup_indicators_config)
        self.create_config_card("Research Unit", self.setup_unit_config)
        self.create_config_card("Analysis Period", self.setup_years_config)

        # Initialize state variables
        self.selected_country = None

        # Update initial button state
        self._update_button_state()

    def get_config(self) -> dict:
        """
        Recopila la configuración de la UI para el Análisis de Serie de Tiempo.
        Cumple con la interfaz estandarizada de 'get_config'.
        """
        # 1. Validar que los datos estén cargados (¡Crítico!)
        if not hasattr(self, 'file_entry') or not self.file_entry.get().strip():
            messagebox.showerror("Error de Configuración", "Por favor, cargue un archivo de datos primero.")
            return None  # Devolver None para cancelar el análisis

        # 2. Recopilar todos los parámetros de la UI
        config_dict = {
            "data_file": self.file_entry.get().strip(),
            "selected_sheet_names": self.selected_indicators,
            "country_to_analyze": self.selected_country,  # Variable correcta según el código
            "years_range": (min(self.selected_years), max(self.selected_years)) if self.selected_years else None,
        }

        # 3. Validar la configuración
        if not config_dict["country_to_analyze"]:
            messagebox.showerror("Error de Configuración", "Por favor, seleccione una Unidad de Investigación.")
            return None

        if not config_dict["selected_sheet_names"]:
            messagebox.showerror("Error de Configuración", "Por favor, seleccione al menos un indicador.")
            return None

        if not config_dict["years_range"]:
            messagebox.showerror("Error de Configuración", "Por favor, seleccione al menos un año.")
            return None

        # Log de configuración obtenida
        if hasattr(self.app, 'logger'):
            self.app.logger.info(f"Configuración de Serie de Tiempo obtenida: {config_dict}")
        else:
            print(f"Configuración de Serie de Tiempo obtenida: {config_dict}")

        return config_dict

    def _update_button_state(self):
        """Update run button state based on configuration."""
        is_ready = (
            hasattr(self, 'file_entry') and self.file_entry.get().strip() and
            self.selected_indicators and
            self.selected_country and
            self.selected_years
        )

        if hasattr(self.app, 'btn_run'):
            self.app.btn_run.config(state=NORMAL if is_ready else DISABLED)

    def setup_unit_config(self, parent):
        """Setup unit selection configuration."""
        unit_frame = ttk.Frame(parent)
        unit_frame.pack(fill=X, pady=(0, 10))

        ttk.Button(
            unit_frame,
            text="🏢 Select Unit",
            style='outline.TButton',
            command=self.select_unit
        ).pack(side=LEFT, padx=(0, 20))

        self.unit_status = ttk.Label(
            unit_frame,
            text="None selected",
            style='secondary.TLabel'
        )
        self.unit_status.pack(side=LEFT)

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

    def select_unit(self):
        """Select research unit/country."""
        try:
            if hasattr(self, 'file_entry') and self.file_entry.get().strip():
                file_path = self.file_entry.get().strip()
                if os.path.exists(file_path):
                    all_sheets_data = dl.load_excel_file(file_path)
                    if all_sheets_data:
                        # For time series, need transformed data
                        first_sheet = list(all_sheets_data.values())[0]
                        transformed_df = dl.transformar_df_indicador_v1(first_sheet)

                        if transformed_df is not None and not transformed_df.empty:
                            available_countries = list(transformed_df.columns)

                            # Create selection dialog
                            dialog = tk.Toplevel(self)
                            dialog.title("Select Research Unit/Country")
                            dialog.geometry("400x400")
                            dialog.transient(self)
                            dialog.grab_set()

                            ttk.Label(dialog, text="Available countries/units:", font=("Helvetica", 12, "bold")).pack(pady=10)

                            # Listbox with scrollbar
                            list_frame = ttk.Frame(dialog)
                            list_frame.pack(fill=BOTH, expand=YES, padx=20, pady=(0, 20))

                            listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE, height=10)
                            scrollbar = ttk.Scrollbar(list_frame, orient=VERTICAL, command=listbox.yview)
                            listbox.configure(yscrollcommand=scrollbar.set)

                            # Add countries to list
                            for country in available_countries:
                                listbox.insert(tk.END, country)

                            listbox.pack(side=LEFT, fill=BOTH, expand=YES)
                            scrollbar.pack(side=RIGHT, fill=Y)

                            # Buttons
                            button_frame = ttk.Frame(dialog)
                            button_frame.pack(fill=X, padx=20, pady=10)

                            def confirm_selection():
                                selection = listbox.curselection()
                                if selection:
                                    country = available_countries[selection[0]]
                                    self.unit_status.config(text=country)
                                    self.selected_country = country
                                    dialog.destroy()
                                    self._update_button_state()
                                else:
                                    tk.messagebox.showwarning("Warning", "Select a country/unit")

                            ttk.Button(
                                button_frame,
                                text="Confirm",
                                style='success.TButton',
                                command=confirm_selection
                            ).pack(side=RIGHT, padx=(10, 0))

                            ttk.Button(
                                button_frame,
                                text="Cancel",
                                command=dialog.destroy
                            ).pack(side=RIGHT)

                            return

            tk.messagebox.showwarning("Warning", "Select a data file first")

        except Exception as e:
            tk.messagebox.showerror("Error", f"Error loading countries: {str(e)}")
