"""
Base Analysis Frame for PCA Application

Provides common functionality for all analysis frames.
"""

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import pandas as pd
from backend import data_loader_module as dl


class BaseAnalysisFrame(ttk.Frame):
    """Base class for analysis frames with common functionality."""

    def __init__(self, parent, app):
        
        # --- MODIFICACIÓN IMPORTANTE ---
        # Ya no creamos un Canvas ni un Scrollbar aquí.
        # Simplemente inicializamos este Frame. El 'parent' que recibe
        # (que es el ScrolledFrame de ui_manager) se encargará del scroll.
        super().__init__(parent)
        
        self.app = app
        
        # Estas variables están bien, son de estado
        self.selected_indicators = []
        self.selected_countries = []
        self.selected_year = None
        self.selected_years = []
        self.selected_units = []

        # --- TODO EL CÓDIGO DEL CANVAS Y SCROLLBAR HA SIDO ELIMINADO ---
        # (Ya no se necesita: ni canvas, ni scrollbar, ni canvas_window,
        # ni update_scrollregion, ni binds, ni mousewheel, ni self.after)
        # --- FIN DE LA MODIFICACIÓN ---
    
    def create_config_card(self, title, content_callback, fill=X):
        """Create a configuration card."""
        card = ttk.LabelFrame(self, text=f"⚙️ {title}", padding=20)
        card.pack(fill=fill, pady=(0, 20))

        content_frame = ttk.Frame(card)
        content_frame.pack(fill=fill)

        content_callback(content_frame)
        return card

    def setup_file_config(self, parent):
        """Setup file selection configuration."""
        file_frame = ttk.Frame(parent)
        file_frame.pack(fill=X, pady=(0, 10))

        self.file_entry = ttk.Entry(file_frame, width=50)
        self.file_entry.pack(side=LEFT, fill=X, expand=YES, padx=(0, 10))

        ttk.Button(
            file_frame,
            text="📁 Select File",
            style='outline.TButton',
            command=self.select_file
        ).pack(side=RIGHT)

    def setup_indicators_config(self, parent):
        """Setup indicators selection configuration."""
        indicators_frame = ttk.Frame(parent)
        indicators_frame.pack(fill=X, pady=(0, 10))

        ttk.Button(
            indicators_frame,
            text="📊 Select Indicators",
            style='outline.TButton',
            command=self.select_indicators
        ).pack(side=LEFT, padx=(0, 20))

        self.indicators_status = ttk.Label(
            indicators_frame,
            text="None selected",
            style='secondary.TLabel'
        )
        self.indicators_status.pack(side=LEFT)

    def select_file(self):
        """Select a file."""
        file_path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("Parquet files", "*.parquet"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            # Validate file
            if self.app.file_handler.validate_file_path(file_path):
                self.file_entry.delete(0, END)
                self.file_entry.insert(0, file_path)
                # Update button state if necessary
                if hasattr(self.app, 'sync_gui_from_cfg'):
                    self.app.sync_gui_from_cfg()

    def select_indicators(self):
        """Select indicators with multi-select dialog."""
        try:
            # Load data from file to get available indicators
            if hasattr(self, 'file_entry') and self.file_entry.get().strip():
                file_path = self.file_entry.get().strip()
                if os.path.exists(file_path):
                    all_sheets_data = dl.load_excel_file(file_path)
                    if all_sheets_data:
                        available_indicators = list(all_sheets_data.keys())

                        # Calculate dialog size based on number of indicators
                        n_indicators = len(available_indicators)
                        dialog_height = min(700, max(550, 200 + n_indicators * 25))  # Dynamic height
                        dialog_width = 500  # Slightly wider for better readability

                        # Create selection dialog
                        dialog = tk.Toplevel(self)
                        dialog.title("Select Indicators")
                        dialog.geometry(f"{dialog_width}x{dialog_height}")
                        dialog.transient(self)
                        dialog.grab_set()

                        # Title
                        ttk.Label(dialog, text="Available indicators:", font=("Helvetica", 12, "bold")).pack(pady=(10, 5))

                        # Select all checkbox
                        select_all_var = tk.BooleanVar()
                        select_all_cb = ttk.Checkbutton(
                            dialog,
                            text="Select All",
                            variable=select_all_var,
                            command=lambda: self._toggle_select_all(select_all_var, indicator_vars, available_indicators)
                        )
                        select_all_cb.pack(pady=(0, 10))

                        # Scrollable list frame
                        list_frame = ttk.Frame(dialog)
                        list_frame.pack(fill=BOTH, expand=YES, padx=20, pady=(0, 20))

                        # Canvas and scrollbar - dynamic height based on content
                        canvas_height = min(500, max(300, n_indicators * 25))  # Dynamic canvas height
                        canvas = tk.Canvas(list_frame, height=canvas_height)
                        scrollbar = ttk.Scrollbar(list_frame, orient=VERTICAL, command=canvas.yview)
                        scrollable_frame = ttk.Frame(canvas)

                        scrollable_frame.bind(
                            "<Configure>",
                            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
                        )

                        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
                        canvas.configure(yscrollcommand=scrollbar.set)

                        # Variables for individual checkboxes
                        indicator_vars = {}

                        # Create checkboxes for each indicator
                        for indicator in available_indicators:
                            var = tk.BooleanVar()
                            indicator_vars[indicator] = var
                            cb = ttk.Checkbutton(
                                scrollable_frame,
                                text=indicator,
                                variable=var,
                                command=lambda: self._update_select_all_state(select_all_var, indicator_vars, available_indicators)
                            )
                            cb.pack(anchor=W, pady=1)

                        canvas.pack(side=LEFT, fill=BOTH, expand=YES)
                        scrollbar.pack(side=RIGHT, fill=Y)

                        # Buttons
                        button_frame = ttk.Frame(dialog)
                        button_frame.pack(fill=X, padx=20, pady=10)

                        def confirm_selection():
                            selected = [ind for ind in available_indicators if indicator_vars[ind].get()]
                            if selected:
                                # Update status if exists
                                if hasattr(self, 'indicators_status'):
                                    self.indicators_status.config(text=f"{len(selected)} selected")
                                self.selected_indicators = selected
                                dialog.destroy()
                                # Sync button state
                                if hasattr(self.app, 'sync_gui_from_cfg'):
                                    self.app.sync_gui_from_cfg()
                            else:
                                messagebox.showwarning("Warning", "Select at least one indicator")

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

            # If no file, show message
            messagebox.showwarning("Warning", "Select a data file first")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading indicators: {str(e)}")

    def _toggle_select_all(self, select_all_var, item_vars, available_items):
        """Toggle select all for items."""
        select_all = select_all_var.get()
        for item in available_items:
            item_vars[item].set(select_all)

    def _update_select_all_state(self, select_all_var, item_vars, available_items):
        """Update select all state based on individual selections."""
        all_selected = all(item_vars[item].get() for item in available_items)
        none_selected = not any(item_vars[item].get() for item in available_items)

        if all_selected:
            select_all_var.set(True)
        elif none_selected:
            select_all_var.set(False)
        # If some selected, leave state as is

    def select_units(self, title="Select Units/Countries", allow_multiple=True):
        """Select units/countries with multi-select dialog and search filter."""
        try:
            # Load data from file to get available countries
            if hasattr(self, 'file_entry') and self.file_entry.get().strip():
                file_path = self.file_entry.get().strip()
                if os.path.exists(file_path):
                    # Read directly from Excel
                    excel_data = pd.read_excel(file_path, sheet_name=None)
                    if excel_data:
                        # Use first sheet
                        first_sheet_name = list(excel_data.keys())[0]
                        df = excel_data[first_sheet_name]

                        # Get countries from "Unnamed: 0" column
                        if "Unnamed: 0" in df.columns:
                            available_countries = sorted(df["Unnamed: 0"].dropna().unique())
                        else:
                            messagebox.showerror("Error", "Column 'Unnamed: 0' with countries not found")
                            return

                        # Create display names using CODE_TO_NAME mapping
                        from backend.constants import CODE_TO_NAME
                        display_names = {
                            unit: f"{CODE_TO_NAME.get(unit, str(unit))} ({unit})" 
                            for unit in available_countries
                        }

                        # Calculate dialog size based on number of countries
                        n_countries = len(available_countries)
                        dialog_height = min(700, max(550, 200 + n_countries * 25))
                        dialog_width = 550  # Wider for search bar

                        # Create selection dialog
                        dialog = tk.Toplevel(self)
                        dialog.title(title)
                        dialog.geometry(f"{dialog_width}x{dialog_height}")
                        dialog.transient(self)
                        dialog.grab_set()

                        # Header
                        header_frame = ttk.Frame(dialog)
                        header_frame.pack(fill=X, padx=20, pady=(10, 5))
                        
                        ttk.Label(
                            header_frame, 
                            text="Available countries/units:", 
                            font=("Helvetica", 12, "bold")
                        ).pack(anchor=W)

                        # ✅ Search/filter bar (Excel-style)
                        search_frame = ttk.Frame(dialog)
                        search_frame.pack(fill=X, padx=20, pady=(5, 10))
                        
                        search_var = tk.StringVar()
                        search_var.trace_add("write", lambda *args: filter_countries())
                        
                        search_icon = ttk.Label(search_frame, text="🔍", font=("Helvetica", 12))
                        search_icon.pack(side=LEFT, padx=(0, 5))
                        
                        search_entry = ttk.Entry(
                            search_frame, 
                            textvariable=search_var,
                            font=("Helvetica", 10),
                            width=40
                        )
                        search_entry.pack(side=LEFT, fill=X, expand=YES)
                        search_entry.focus_set()
                        
                        # Clear search button
                        def clear_search():
                            search_var.set("")
                            search_entry.focus_set()
                        
                        clear_btn = ttk.Button(
                            search_frame,
                            text="✕",
                            style='secondary.Outline.TButton',
                            width=3,
                            command=clear_search
                        )
                        clear_btn.pack(side=LEFT, padx=(5, 0))
                        
                        # Results counter
                        results_label = ttk.Label(
                            dialog,
                            text=f"Showing {n_countries} of {n_countries} units",
                            font=("Helvetica", 9),
                            style='secondary.TLabel'
                        )
                        results_label.pack(padx=20, pady=(0, 5))

                        # Select all checkbox
                        select_all_var = tk.BooleanVar()
                        select_all_cb = ttk.Checkbutton(
                            dialog,
                            text="Select All (visible)",
                            variable=select_all_var,
                            command=lambda: toggle_select_all()
                        )
                        select_all_cb.pack(padx=20, pady=(0, 10))

                        # Scrollable list frame
                        list_frame = ttk.Frame(dialog)
                        list_frame.pack(fill=BOTH, expand=YES, padx=20, pady=(0, 20))

                        canvas_height = min(500, max(300, n_countries * 25))
                        canvas = tk.Canvas(list_frame, height=canvas_height)
                        scrollbar = ttk.Scrollbar(list_frame, orient=VERTICAL, command=canvas.yview)
                        scrollable_frame = ttk.Frame(canvas)

                        scrollable_frame.bind(
                            "<Configure>",
                            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
                        )

                        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
                        canvas.configure(yscrollcommand=scrollbar.set)

                        # Variables for checkboxes
                        country_vars = {}
                        checkbox_widgets = {}

                        # Create checkboxes for each country
                        for unit in available_countries:
                            var = tk.BooleanVar()
                            country_vars[unit] = var
                            cb = ttk.Checkbutton(
                                scrollable_frame,
                                text=display_names[unit],
                                variable=var,
                                command=lambda: update_select_all_state()
                            )
                            cb.pack(anchor=W, pady=1)
                            checkbox_widgets[unit] = cb

                        canvas.pack(side=LEFT, fill=BOTH, expand=YES)
                        scrollbar.pack(side=RIGHT, fill=Y)

                        # ✅ Filter function (like Excel)
                        def filter_countries():
                            search_text = search_var.get().lower()
                            visible_count = 0
                            
                            for unit in available_countries:
                                # Check if search text is in unit code OR display name
                                unit_text = f"{unit} {display_names[unit]}".lower()
                                matches = search_text in unit_text
                                
                                if matches:
                                    checkbox_widgets[unit].pack(anchor=W, pady=1)
                                    visible_count += 1
                                else:
                                    checkbox_widgets[unit].pack_forget()
                            
                            # Update results counter
                            results_label.config(
                                text=f"Showing {visible_count} of {n_countries} units"
                            )
                            
                            # Update select all state
                            update_select_all_state()
                        
                        def toggle_select_all():
                            """Toggle all VISIBLE checkboxes."""
                            search_text = search_var.get().lower()
                            state = select_all_var.get()
                            
                            for unit in available_countries:
                                unit_text = f"{unit} {display_names[unit]}".lower()
                                if search_text in unit_text:  # Only visible ones
                                    country_vars[unit].set(state)
                        
                        def update_select_all_state():
                            """Update select all checkbox based on visible items."""
                            search_text = search_var.get().lower()
                            visible_units = [
                                unit for unit in available_countries
                                if search_text in f"{unit} {display_names[unit]}".lower()
                            ]
                            
                            if not visible_units:
                                select_all_var.set(False)
                                return
                            
                            all_selected = all(country_vars[unit].get() for unit in visible_units)
                            select_all_var.set(all_selected)

                        # Buttons
                        button_frame = ttk.Frame(dialog)
                        button_frame.pack(fill=X, padx=20, pady=10)

                        def confirm_selection():
                            selected = [unit for unit in available_countries if country_vars[unit].get()]
                            if selected:
                                if hasattr(self, 'units_status'):
                                    self.units_status.config(text=f"{len(selected)} selected")
                                self.selected_countries = selected
                                self.selected_units = selected  # For compatibility
                                dialog.destroy()
                                # Sync button state
                                if hasattr(self.app, 'sync_gui_from_cfg'):
                                    self.app.sync_gui_from_cfg()
                            else:
                                messagebox.showwarning("Warning", "Select at least one unit/country")

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

            # If no file, show message
            messagebox.showwarning("Warning", "Select a data file first")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading countries: {str(e)}")

    def _toggle_select_all_countries(self, select_all_var, country_vars, available_countries):
        """Toggle select all for countries."""
        select_all = select_all_var.get()
        for country in available_countries:
            country_vars[country].set(select_all)

    def _update_select_all_countries_state(self, select_all_var, country_vars, available_countries):
        """Update select all state for countries."""
        all_selected = all(country_vars[country].get() for country in available_countries)
        none_selected = not any(country_vars[country].get() for country in available_countries)

        if all_selected:
            select_all_var.set(True)
        elif none_selected:
            select_all_var.set(False)
        # If some selected, leave state as is

    def select_year(self, title="Select Year"):
        """Select a year."""
        try:
            # Load data from file to get available years
            if hasattr(self, 'file_entry') and self.file_entry.get().strip():
                file_path = self.file_entry.get().strip()
                if os.path.exists(file_path):
                    all_sheets_data = dl.load_excel_file(file_path)
                    if all_sheets_data:
                        # Get years from first sheet
                        first_sheet = list(all_sheets_data.values())[0]
                        # Years are in columns excluding "Unnamed: 0"
                        year_columns = [col for col in first_sheet.columns if col != "Unnamed: 0" and str(col).isdigit()]
                        available_years = sorted([int(col) for col in year_columns])

                        if not available_years:
                            messagebox.showwarning("Warning", "No years found in data")
                            return

                        # Create selection dialog
                        dialog = tk.Toplevel(self)
                        dialog.title(title)
                        dialog.geometry("300x200")
                        dialog.transient(self)
                        dialog.grab_set()

                        ttk.Label(dialog, text=f"Available years: {min(available_years)} - {max(available_years)}",
                                  font=("Helvetica", 12, "bold")).pack(pady=10)

                        # Year combobox
                        year_var = tk.StringVar(value=str(available_years[-1]))  # Last year as default
                        year_combo = ttk.Combobox(dialog, textvariable=year_var,
                                                values=[str(y) for y in available_years], state="readonly", width=10)
                        year_combo.pack(pady=10)

                        # Buttons
                        button_frame = ttk.Frame(dialog)
                        button_frame.pack(fill=X, padx=20, pady=10)

                        def confirm_selection():
                            try:
                                selected_year = int(year_var.get())
                                if selected_year in available_years:
                                    # Update status if exists
                                    if hasattr(self, 'year_status'):
                                        self.year_status.config(text=str(selected_year))
                                    self.selected_year = selected_year
                                    dialog.destroy()
                                    # Sync button state
                                    if hasattr(self.app, 'sync_gui_from_cfg'):
                                        self.app.sync_gui_from_cfg()
                                else:
                                    messagebox.showerror("Error", "Invalid year selected")
                            except ValueError:
                                messagebox.showerror("Error", "Select a valid year")

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

            # If no file, show message
            messagebox.showwarning("Warning", "Select a data file first")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading years: {str(e)}")

    def select_years(self, title="Select Years"):
        """Select multiple years."""
        try:
            # Load data from file to get available years
            if hasattr(self, 'file_entry') and self.file_entry.get().strip():
                file_path = self.file_entry.get().strip()
                if os.path.exists(file_path):
                    all_sheets_data = dl.load_excel_file(file_path)
                    if all_sheets_data:
                        # Get years from first sheet
                        first_sheet = list(all_sheets_data.values())[0]
                        # Years are in columns excluding "Unnamed: 0"
                        year_columns = [col for col in first_sheet.columns if col != "Unnamed: 0" and str(col).isdigit()]
                        available_years = sorted([int(col) for col in year_columns])

                        if not available_years:
                            messagebox.showwarning("Warning", "No years found in data")
                            return

                        # Calculate dialog size based on number of years
                        n_years = len(available_years)
                        dialog_height = min(600, max(500, 200 + n_years * 25))  # Dynamic height
                        dialog_width = 400

                        # Create selection dialog
                        dialog = tk.Toplevel(self)
                        dialog.title(title)
                        dialog.geometry(f"{dialog_width}x{dialog_height}")
                        dialog.transient(self)
                        dialog.grab_set()

                        ttk.Label(dialog, text="Available years:", font=("Helvetica", 12, "bold")).pack(pady=(10, 5))

                        # Select all checkbox
                        select_all_var = tk.BooleanVar()
                        select_all_cb = ttk.Checkbutton(
                            dialog,
                            text="Select All",
                            variable=select_all_var,
                            command=lambda: self._toggle_select_all_years(select_all_var, year_vars, available_years)
                        )
                        select_all_cb.pack(pady=(0, 10))

                        # Scrollable list frame
                        list_frame = ttk.Frame(dialog)
                        list_frame.pack(fill=BOTH, expand=YES, padx=20, pady=(0, 20))

                        # Canvas and scrollbar - dynamic height based on content
                        canvas_height = min(400, max(300, n_years * 25))  # Dynamic canvas height
                        canvas = tk.Canvas(list_frame, height=canvas_height)
                        scrollbar = ttk.Scrollbar(list_frame, orient=VERTICAL, command=canvas.yview)
                        scrollable_frame = ttk.Frame(canvas)

                        scrollable_frame.bind(
                            "<Configure>",
                            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
                        )

                        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
                        canvas.configure(yscrollcommand=scrollbar.set)

                        # Variables for checkboxes
                        year_vars = {}

                        # Create checkboxes for each year
                        for year in available_years:
                            var = tk.BooleanVar()
                            year_vars[year] = var
                            cb = ttk.Checkbutton(
                                scrollable_frame,
                                text=str(year),
                                variable=var,
                                command=lambda: self._update_select_all_years_state(select_all_var, year_vars, available_years)
                            )
                            cb.pack(anchor=W, pady=1)

                        canvas.pack(side=LEFT, fill=BOTH, expand=YES)
                        scrollbar.pack(side=RIGHT, fill=Y)

                        # Buttons
                        button_frame = ttk.Frame(dialog)
                        button_frame.pack(fill=X, padx=20, pady=10)

                        def confirm_selection():
                            selected = [year for year in available_years if year_vars[year].get()]
                            if selected:
                                if hasattr(self, 'years_status'):
                                    self.years_status.config(text=f"{len(selected)} selected")
                                self.selected_years = selected
                                dialog.destroy()
                                # Sync button state
                                if hasattr(self.app, 'sync_gui_from_cfg'):
                                    self.app.sync_gui_from_cfg()
                            else:
                                messagebox.showwarning("Warning", "Select at least one year")

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

            # If no file, show message
            messagebox.showwarning("Warning", "Select a data file first")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading years: {str(e)}")

    def _toggle_select_all_years(self, select_all_var, year_vars, available_years):
        """Toggle select all for years."""
        select_all = select_all_var.get()
        for year in available_years:
            year_vars[year].set(select_all)

    def _update_select_all_years_state(self, select_all_var, year_vars, available_years):
        """Update select all state for years."""
        all_selected = all(year_vars[year].get() for year in available_years)
        none_selected = not any(year_vars[year].get() for year in available_years)

        if all_selected:
            select_all_var.set(True)
        elif none_selected:
            select_all_var.set(False)
        # If some selected, leave state as is