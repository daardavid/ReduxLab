"""Hierarchical Clustering Frame for PCA Application."""

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import messagebox
from frontend.base_analysis_frame import BaseAnalysisFrame
from backend import data_loader_module as dl
import pandas as pd
import os


class HierarchicalClusteringFrame(BaseAnalysisFrame):
    """Refactored frame for hierarchical clustering analysis."""

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.setup_ui()

    def setup_ui(self):
        # Initialize variables BEFORE card creation so setup callbacks
        # can reference them, and so we don't overwrite Labels created by callbacks.
        self.raw_data = None
        self.pca_data = None
        self.fig_canvas = None
        self.groups = {}
        self.group_colors = {}
        self.selected_indicators = []
        self.selected_units = []

        # Description
        desc = ttk.Label(
            self,
            text="Hierarchical clustering analysis on research units using PCA components.",
            wraplength=800,
            style='secondary.TLabel',
            font=("Helvetica", 11)
        )
        desc.pack(pady=(0, 30), anchor=W)

        # Configuration cards (these set self.indicators_status, self.units_status, etc.)
        self.create_config_card("Data File", self.setup_file_config)
        self.create_config_card("Indicators", self.setup_indicators_config)
        self.create_config_card("Research Units", self.setup_units_config)
        self.create_config_card("Clustering Parameters", self.setup_params_config)
        self.create_config_card("Visualization", self.setup_viz_config)

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

    def setup_units_config(self, parent):
        """Setup units display configuration."""
        units_frame = ttk.Frame(parent)
        units_frame.pack(fill=X, pady=(0, 10))

        ttk.Label(units_frame, text="🏢 Research Units:", style='secondary.TLabel').pack(side=LEFT, padx=(0, 10))

        self.units_status = ttk.Label(
            units_frame,
            text="Load data file first",
            style='secondary.TLabel'
        )
        self.units_status.pack(side=LEFT)

    def setup_params_config(self, parent):
        """Setup clustering parameters configuration."""
        params_frame = ttk.Frame(parent)
        params_frame.pack(fill=X, pady=(0, 10))

        # Linkage method
        ttk.Label(params_frame, text="Linkage Method:").grid(row=0, column=0, sticky=W, pady=5)
        self.linkage_var = tk.StringVar(value='ward')
        linkage_combo = ttk.Combobox(
            params_frame,
            textvariable=self.linkage_var,
            values=['ward', 'complete', 'average', 'single'],
            state="readonly",
            width=15
        )
        linkage_combo.grid(row=0, column=1, padx=(10, 0), pady=5)

        # Distance metric
        ttk.Label(params_frame, text="Distance Metric:").grid(row=1, column=0, sticky=W, pady=5)
        self.metric_var = tk.StringVar(value='euclidean')
        metric_combo = ttk.Combobox(
            params_frame,
            textvariable=self.metric_var,
            values=['euclidean', 'cityblock', 'cosine'],
            state="readonly",
            width=15
        )
        metric_combo.grid(row=1, column=1, padx=(10, 0), pady=5)

    def setup_viz_config(self, parent):
        """Setup visualization configuration."""
        viz_frame = ttk.Frame(parent)
        viz_frame.pack(fill=X, pady=(0, 10))

        # Unit selection button
        ttk.Button(
            viz_frame,
            text="🏢 Select Units",
            style='outline.TButton',
            command=self.select_units
        ).pack(side=LEFT, padx=(0, 20))

        # PCA components selection button
        ttk.Button(
            viz_frame,
            text="📊 Select PCA Components",
            style='outline.TButton',
            command=self.select_variables
        ).pack(side=LEFT, padx=(0, 20))

        # Group assignment button
        ttk.Button(
            viz_frame,
            text="👥 Assign Groups",
            style='info.TButton',
            command=self.assign_groups
        ).pack(side=LEFT, padx=(0, 20))

        # Run clustering button
        ttk.Button(
            viz_frame,
            text="▶ Run Clustering",
            style='success.TButton',
            command=self.run_clustering
        ).pack(side=LEFT, padx=(0, 20))

        # Status labels frame
        status_frame = ttk.Frame(viz_frame)
        status_frame.pack(side=LEFT, padx=(20, 0))

        # Visualization-specific units status (separate from the Research Units card status)
        self.viz_units_status = ttk.Label(
            status_frame,
            text="All units selected",
            style='secondary.TLabel'
        )
        self.viz_units_status.pack(anchor=W)

        # Variables status label
        self.variables_status = ttk.Label(
            status_frame,
            text="All variables selected",
            style='secondary.TLabel'
        )
        self.variables_status.pack(anchor=W)

        # Initialize status
        self.update_units_status()
        self.update_variables_status()

    def select_file(self):
        """Select a file and automatically load all indicators and units."""
        file_path = self.app.file_handler.select_file(
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
                self.file_entry.delete(0, tk.END)
                self.file_entry.insert(0, file_path)

                # Automatically load all indicators and units
                self._auto_load_indicators_and_units(file_path)

                # Update button state
                if hasattr(self.app, 'sync_gui_from_cfg'):
                    self.app.sync_gui_from_cfg()

    def _auto_load_indicators_and_units(self, file_path):
        """Automatically load all available indicators and units from the selected file."""
        try:
            # Load Excel file to get sheet names (indicators)
            excel_data = pd.read_excel(file_path, sheet_name=None, header=None)
            if not excel_data:
                return

            # Load all indicators (sheet names)
            all_indicators = list(excel_data.keys())
            self.selected_indicators = all_indicators

            # Update indicators status
            if hasattr(self, 'indicators_status'):
                self.indicators_status.config(text=f"{len(all_indicators)} loaded")

            # Load all units from all sheets
            all_units = set()
            for sheet_name, df in excel_data.items():
                if df.empty:
                    continue
                # Units are in column A, starting from row 1 (after header row)
                units_in_sheet = df.iloc[1:, 0].dropna().unique()
                all_units.update(units_in_sheet)

            unique_units = sorted(list(all_units))
            self.selected_units = unique_units
            self.selected_countries = unique_units  # For compatibility

            # Update units status
            if hasattr(self, 'units_status'):
                self.units_status.config(text=f"{len(unique_units)} loaded")

        except Exception as e:
            tk.messagebox.showerror("Error", f"Error loading data: {str(e)}")

    def setup_canvas_frame(self):
        """Setup canvas frame for dendrogram display."""
        # Canvas frame
        self.canvas_frame = ttk.LabelFrame(self, text="Dendrogram", padding=10)
        self.canvas_frame.pack(fill=BOTH, expand=YES, pady=(20, 0))

        # Placeholder
        self.placeholder_label = ttk.Label(
            self.canvas_frame,
            text="Dendrogram will appear here after running analysis",
            style='secondary.TLabel'
        )
        self.placeholder_label.pack(expand=YES)

    def select_indicators(self):
        """Select indicators for hierarchical clustering analysis (sheet names in multi-sheet format)."""
        try:
            # Load data from file to get available indicators (sheet names)
            if hasattr(self, 'file_entry') and self.file_entry.get().strip():
                file_path = self.file_entry.get().strip()
                if os.path.exists(file_path):
                    # Load Excel file to get sheet names (indicators)
                    excel_data = pd.read_excel(file_path, sheet_name=None, header=None)
                    if not excel_data:
                        tk.messagebox.showerror("Error", "Could not load Excel file")
                        return

                    # Sheet names are the indicator names
                    indicator_cols = list(excel_data.keys())

                    if not indicator_cols:
                        tk.messagebox.showerror("Error", "No sheets (indicators) found in Excel file")
                        return

                    # Calculate dialog size based on number of indicators
                    n_indicators = len(indicator_cols)
                    dialog_height = min(700, max(550, 200 + n_indicators * 25))
                    dialog_width = 500

                    # Create selection dialog
                    dialog = tk.Toplevel(self)
                    dialog.title("Select Indicators")
                    dialog.geometry(f"{dialog_width}x{dialog_height}")
                    dialog.transient(self)
                    dialog.grab_set()

                    # Title
                    ttk.Label(dialog, text="Available indicators (sheet names):", font=("Helvetica", 12, "bold")).pack(pady=(10, 5))

                    # Select all checkbox
                    select_all_var = tk.BooleanVar()
                    select_all_cb = ttk.Checkbutton(
                        dialog,
                        text="Select All",
                        variable=select_all_var,
                        command=lambda: self._toggle_select_all(select_all_var, indicator_vars, indicator_cols)
                    )
                    select_all_cb.pack(pady=(0, 10))

                    # Scrollable list frame
                    list_frame = ttk.Frame(dialog)
                    list_frame.pack(fill=BOTH, expand=YES, padx=20, pady=(0, 20))

                    # Canvas and scrollbar
                    canvas_height = min(500, max(300, n_indicators * 25))
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

                    # Create checkboxes for each indicator (pre-selected by default)
                    for indicator in indicator_cols:
                        var = tk.BooleanVar(value=True)  # Pre-select all by default
                        indicator_vars[indicator] = var
                        cb = ttk.Checkbutton(
                            scrollable_frame,
                            text=indicator,
                            variable=var,
                            command=lambda: self._update_select_all_state(select_all_var, indicator_vars, indicator_cols)
                        )
                        cb.pack(anchor=W, pady=1)

                    canvas.pack(side=LEFT, fill=BOTH, expand=YES)
                    scrollbar.pack(side=RIGHT, fill=Y)

                    # Buttons
                    button_frame = ttk.Frame(dialog)
                    button_frame.pack(fill=X, padx=20, pady=10)

                    def confirm_selection():
                        selected = [ind for ind in indicator_cols if indicator_vars[ind].get()]
                        if selected:
                            self.selected_indicators = selected
                            # Update status
                            if hasattr(self, 'indicators_status'):
                                self.indicators_status.config(text=f"{len(selected)}/{len(indicator_cols)} selected")
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

    def assign_groups(self):
        """Open dialog to assign research units to groups with colors."""
        if self.pca_data is None:
            tk.messagebox.showwarning("Warning", "Load PCA data first")
            return

        # Create group assignment dialog
        dialog = tk.Toplevel(self)
        dialog.title("Assign Groups to Research Units")
        dialog.geometry("600x700")
        dialog.transient(self)
        dialog.grab_set()

        # Title
        ttk.Label(dialog, text="Assign Research Units to Groups:", font=("Helvetica", 14, "bold")).pack(pady=10)

        # Main frame with scrollbar
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=BOTH, expand=YES, padx=20, pady=(0, 20))

        # Canvas and scrollbar
        canvas = tk.Canvas(main_frame, height=400)
        scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Get available units (companies)
        available_units = list(self.pca_data.index)

        # Group management section
        group_frame = ttk.LabelFrame(scrollable_frame, text="Group Management", padding=10)
        group_frame.pack(fill=X, pady=(0, 20))

        # Group name entry
        ttk.Label(group_frame, text="New Group Name:").grid(row=0, column=0, sticky=W, pady=5)
        group_name_var = tk.StringVar()
        group_name_entry = ttk.Entry(group_frame, textvariable=group_name_var, width=20)
        group_name_entry.grid(row=0, column=1, padx=(10, 0), pady=5)

        # Color selection
        ttk.Label(group_frame, text="Color:").grid(row=0, column=2, sticky=W, padx=(20, 0), pady=5)
        color_var = tk.StringVar(value="#FF6B6B")
        color_combo = ttk.Combobox(group_frame, textvariable=color_var,
                                  values=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"],
                                  state="readonly", width=10)
        color_combo.grid(row=0, column=3, padx=(10, 0), pady=5)

        # Create group button
        def create_group():
            group_name = group_name_var.get().strip()
            if group_name and group_name not in self.group_colors:
                self.group_colors[group_name] = color_var.get()
                update_group_list()
                # Update the assign group combo box with new groups
                update_group_combo()
                group_name_var.set("")
                tk.messagebox.showinfo("Success", f"Group '{group_name}' created")
            else:
                tk.messagebox.showwarning("Warning", "Enter a unique group name")

        ttk.Button(group_frame, text="Create Group", command=create_group).grid(row=0, column=4, padx=(20, 0), pady=5)

        # Groups list
        ttk.Label(group_frame, text="Created Groups:").grid(row=1, column=0, sticky=W, pady=(20, 5))
        groups_listbox = tk.Listbox(group_frame, height=3, width=50)
        groups_listbox.grid(row=2, column=0, columnspan=5, pady=(0, 10))

        def update_group_list():
            groups_listbox.delete(0, tk.END)
            for group, color in self.group_colors.items():
                groups_listbox.insert(tk.END, f"{group} - {color}")

        # Unit assignment section
        assign_frame = ttk.LabelFrame(scrollable_frame, text="Assign Units to Groups", padding=10)
        assign_frame.pack(fill=X, pady=(0, 20))

        # Unit selection (multiple selection listbox)
        ttk.Label(assign_frame, text="Select Units:").grid(row=0, column=0, sticky=W, pady=5)

        # Frame for unit listbox with scrollbar
        unit_frame = ttk.Frame(assign_frame)
        unit_frame.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        unit_listbox = tk.Listbox(unit_frame, selectmode=tk.MULTIPLE, height=20, width=50)
        unit_scrollbar = ttk.Scrollbar(unit_frame, orient=VERTICAL, command=unit_listbox.yview)
        unit_listbox.configure(yscrollcommand=unit_scrollbar.set)

        # Function to populate unit listbox with assignment status
        def populate_unit_listbox():
            unit_listbox.delete(0, tk.END)
            for unit in available_units:
                if unit in self.groups:
                    group = self.groups[unit]
                    display_text = f"{unit} ({group})"
                else:
                    display_text = unit
                unit_listbox.insert(tk.END, display_text)

        # Populate the listbox initially
        populate_unit_listbox()

        unit_listbox.pack(side=LEFT, fill=BOTH, expand=YES)
        unit_scrollbar.pack(side=RIGHT, fill=Y)

        # Group selection
        ttk.Label(assign_frame, text="Assign to Group:").grid(row=0, column=2, sticky=W, padx=(20, 0), pady=5)
        assign_group_var = tk.StringVar()
        assign_group_combo = ttk.Combobox(assign_frame, textvariable=assign_group_var,
                                         values=list(self.group_colors.keys()), state="readonly", width=20)
        assign_group_combo.grid(row=0, column=3, padx=(10, 0), pady=5)

        # Function to update group combo values
        def update_group_combo():
            assign_group_combo['values'] = list(self.group_colors.keys())

        # Assign button
        def assign_units():
            selected_indices = unit_listbox.curselection()
            group = assign_group_var.get()
            if selected_indices and group:
                assigned_count = 0
                for index in selected_indices:
                    # Get the actual unit name (before the group indicator)
                    display_text = unit_listbox.get(index)
                    unit = display_text.split(" (")[0]  # Remove group indicator if present
                    if unit not in self.groups:  # Only assign if not already assigned
                        self.groups[unit] = group
                        assigned_count += 1

                if assigned_count > 0:
                    update_assignments_list()
                    populate_unit_listbox()  # Refresh the unit list to show new assignments
                    # Clear selection
                    unit_listbox.selection_clear(0, tk.END)
                    assign_group_var.set("")
                    tk.messagebox.showinfo("Success", f"Assigned {assigned_count} units to group '{group}'")
                else:
                    tk.messagebox.showwarning("Warning", "Selected units are already assigned to groups")
            else:
                tk.messagebox.showwarning("Warning", "Select units and group")

        ttk.Button(assign_frame, text="Assign", command=assign_units).grid(row=0, column=4, padx=(20, 0), pady=5)

        # Current assignments section (separate from unit selection)
        assignments_frame = ttk.LabelFrame(scrollable_frame, text="Current Assignments", padding=10)
        assignments_frame.pack(fill=X, pady=(10, 20))

        assignments_listbox = tk.Listbox(assignments_frame, height=8, width=80)
        assignments_scrollbar = ttk.Scrollbar(assignments_frame, orient=VERTICAL, command=assignments_listbox.yview)
        assignments_listbox.configure(yscrollcommand=assignments_scrollbar.set)

        assignments_listbox.pack(side=LEFT, fill=BOTH, expand=YES)
        assignments_scrollbar.pack(side=RIGHT, fill=Y)

        def update_assignments_list():
            assignments_listbox.delete(0, tk.END)
            for unit, group in self.groups.items():
                color = self.group_colors.get(group, "#000000")
                assignments_listbox.insert(tk.END, f"{unit} -> {group} ({color})")

        # Remove assignment button
        def remove_assignment():
            selection = assignments_listbox.curselection()
            if selection:
                assignment_text = assignments_listbox.get(selection[0])
                unit = assignment_text.split(" -> ")[0]
                if unit in self.groups:
                    del self.groups[unit]
                    update_assignments_list()
                    populate_unit_listbox()  # Refresh unit list to show removed assignments

        ttk.Button(assignments_frame, text="Remove Selected Assignment", command=remove_assignment).pack(anchor=E, pady=(5, 0))

        # Initialize lists
        update_group_list()
        update_assignments_list()

        canvas.pack(side=LEFT, fill=BOTH, expand=YES)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=X, padx=20, pady=10)

        def apply_groups():
            if not self.groups:
                tk.messagebox.showwarning("Warning", "No groups assigned")
                return
            dialog.destroy()
            tk.messagebox.showinfo("Success", f"Groups assigned to {len(self.groups)} units")

        ttk.Button(
            button_frame,
            text="Apply Groups",
            style='success.TButton',
            command=apply_groups
        ).pack(side=RIGHT, padx=(10, 0))

        ttk.Button(
            button_frame,
            text="Cancel",
            command=dialog.destroy
        ).pack(side=RIGHT)

    def select_units(self):
        """Select research units for hierarchical clustering analysis (from multi-sheet format)."""
        try:
            # Load data from file to get available units
            if hasattr(self, 'file_entry') and self.file_entry.get().strip():
                file_path = self.file_entry.get().strip()
                if os.path.exists(file_path):
                    # Load Excel file to get units from all sheets
                    excel_data = pd.read_excel(file_path, sheet_name=None, header=None)
                    if not excel_data:
                        tk.messagebox.showerror("Error", "Could not load Excel file")
                        return

                    # Collect unique units from all sheets
                    all_units = set()
                    for sheet_name, df in excel_data.items():
                        if df.empty:
                            continue
                        # Units are in column A, starting from row 1 (after header row)
                        units_in_sheet = df.iloc[1:, 0].dropna().unique()
                        all_units.update(units_in_sheet)

                    available_units = sorted(list(all_units))

                    if not available_units:
                        tk.messagebox.showerror("Error", "No units found in any sheet")
                        return

                    # Calculate dialog size based on number of units
                    n_units = len(available_units)
                    dialog_height = min(700, max(550, 200 + n_units * 25))
                    dialog_width = 500

                    # Create selection dialog
                    dialog = tk.Toplevel(self)
                    dialog.title("Select Research Units for Hierarchical Clustering")
                    dialog.geometry(f"{dialog_width}x{dialog_height}")
                    dialog.transient(self)
                    dialog.grab_set()

                    ttk.Label(dialog, text="Available research units:", font=("Helvetica", 12, "bold")).pack(pady=(10, 5))

                    # Select all checkbox
                    select_all_var = tk.BooleanVar()
                    select_all_cb = ttk.Checkbutton(
                        dialog,
                        text="Select All",
                        variable=select_all_var,
                        command=lambda: self._toggle_select_all_countries(select_all_var, unit_vars, available_units)
                    )
                    select_all_cb.pack(pady=(0, 10))

                    # Scrollable list frame
                    list_frame = ttk.Frame(dialog)
                    list_frame.pack(fill=BOTH, expand=YES, padx=20, pady=(0, 20))

                    # Canvas and scrollbar
                    canvas_height = min(500, max(300, n_units * 25))
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
                    unit_vars = {}
                    selected_units = []

                    # Create checkboxes for each unit (pre-selected by default)
                    for unit in available_units:
                        var = tk.BooleanVar(value=True)  # Pre-select all by default
                        unit_vars[unit] = var
                        cb = ttk.Checkbutton(
                            scrollable_frame,
                            text=str(unit),
                            variable=var,
                            command=lambda: self._update_select_all_countries_state(select_all_var, unit_vars, available_units)
                        )
                        cb.pack(anchor=W, pady=1)

                    canvas.pack(side=LEFT, fill=BOTH, expand=YES)
                    scrollbar.pack(side=RIGHT, fill=Y)

                    # Buttons
                    button_frame = ttk.Frame(dialog)
                    button_frame.pack(fill=X, padx=20, pady=10)

                    def confirm_selection():
                        selected = [unit for unit in available_units if unit_vars[unit].get()]
                        if selected:
                            self.selected_countries = selected
                            self.selected_units = selected  # For compatibility
                            # Update status
                            if hasattr(self, 'units_status'):
                                self.units_status.config(text=f"{len(selected)}/{len(available_units)} selected")
                            dialog.destroy()
                            # Sync button state
                            if hasattr(self.app, 'sync_gui_from_cfg'):
                                self.app.sync_gui_from_cfg()
                        else:
                            messagebox.showwarning("Warning", "Select at least one unit")

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
            messagebox.showerror("Error", f"Error loading units: {str(e)}")

    def update_units_status(self):
        """Update the units status labels (both Research Units card and Visualization card)."""
        try:
            text = f"{len(self.selected_units)} units selected" if self.selected_units else "No units selected"
            if hasattr(self, 'units_status') and self.units_status is not None and self.units_status.winfo_exists():
                self.units_status.config(text=text)
            if hasattr(self, 'viz_units_status') and self.viz_units_status is not None and self.viz_units_status.winfo_exists():
                self.viz_units_status.config(text=text)
        except (AttributeError, tk.TclError):
            pass

    def select_variables(self):
        """Open dialog to select which PCA variables to include in clustering."""
        if self.pca_data is None:
            tk.messagebox.showwarning("Warning", "Load PCA data first")
            return

        # Create variable selection dialog
        dialog = tk.Toplevel(self)
        dialog.title("Select Variables for Clustering")
        dialog.geometry("500x600")
        dialog.transient(self)
        dialog.grab_set()

        # Title
        ttk.Label(dialog, text="Select PCA Variables to Include:", font=("Helvetica", 14, "bold")).pack(pady=10)

        # Main frame with scrollbar
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=BOTH, expand=YES, padx=20, pady=(0, 20))

        # Canvas and scrollbar
        canvas = tk.Canvas(main_frame, height=400)
        scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Get available variables (PCA components)
        available_variables = list(self.pca_data.columns)

        # Selection controls
        controls_frame = ttk.LabelFrame(scrollable_frame, text="Selection Controls", padding=10)
        controls_frame.pack(fill=X, pady=(0, 20))

        # Select all/none buttons
        def select_all():
            for var in available_variables:
                if var in variable_vars:
                    variable_vars[var].set(True)

        def select_none():
            for var in available_variables:
                if var in variable_vars:
                    variable_vars[var].set(False)

        ttk.Button(controls_frame, text="Select All", command=select_all).pack(side=LEFT, padx=(0, 10))
        ttk.Button(controls_frame, text="Select None", command=select_none).pack(side=LEFT)

        # Variables list
        variables_frame = ttk.LabelFrame(scrollable_frame, text="Available Variables", padding=10)
        variables_frame.pack(fill=X, pady=(0, 20))

        # Variables checkboxes
        variable_vars = {}
        for variable in available_variables:
            var = tk.BooleanVar(value=variable in self.selected_variables)
            variable_vars[variable] = var
            ttk.Checkbutton(
                variables_frame,
                text=f"{variable}",
                variable=var
            ).pack(anchor=W, pady=2)

        canvas.pack(side=LEFT, fill=BOTH, expand=YES)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=X, padx=20, pady=10)

        def apply_selection():
            selected = [var for var in available_variables if variable_vars[var].get()]
            if not selected:
                tk.messagebox.showwarning("Warning", "Select at least one variable")
                return

            self.selected_variables = selected
            self.update_variables_status()
            dialog.destroy()
            tk.messagebox.showinfo("Success", f"Selected {len(selected)} variables for clustering")

        ttk.Button(
            button_frame,
            text="Apply Selection",
            style='success.TButton',
            command=apply_selection
        ).pack(side=RIGHT, padx=(10, 0))

        ttk.Button(
            button_frame,
            text="Cancel",
            command=dialog.destroy
        ).pack(side=RIGHT)

    def update_variables_status(self):
        """Update the variables status label."""
        try:
            if hasattr(self, 'variables_status') and self.variables_status is not None and self.variables_status.winfo_exists():
                if self.selected_variables:
                    total_vars = len(self.pca_data.columns) if self.pca_data is not None else 0
                    self.variables_status.config(
                        text=f"{len(self.selected_variables)}/{total_vars} variables selected"
                    )
                else:
                    self.variables_status.config(text="No variables selected")
        except (AttributeError, tk.TclError):
            # Label doesn't exist or has been destroyed, skip update
            pass

    def run_clustering(self):
        """Run hierarchical clustering."""
        if not hasattr(self, 'file_entry') or not self.file_entry.get().strip():
            tk.messagebox.showwarning("Warning", "Select a data file first")
            return

        if not self.selected_indicators:
            tk.messagebox.showwarning("Warning", "Select at least one indicator")
            return

        if not self.selected_units:
            tk.messagebox.showwarning("Warning", "Select at least one unit")
            return

        try:
            # Load raw data
            file_path = self.file_entry.get().strip()
            all_sheets_data = dl.load_excel_file(file_path)
            if not all_sheets_data:
                tk.messagebox.showerror("Error", "Could not load data file")
                return

            # Filter selected indicators
            selected_data = {k: v for k, v in all_sheets_data.items() if k in self.selected_indicators}
            if not selected_data:
                tk.messagebox.showerror("Error", "No selected indicators found in data")
                return

            # Transform data for each indicator
            data_transformada_indicadores = {}
            for sheet_name, df in selected_data.items():
                df_transformado = dl.transformar_df_indicador_v1(df)
                if df_transformado is not None:
                    data_transformada_indicadores[sheet_name] = df_transformado

            if not data_transformada_indicadores:
                tk.messagebox.showerror("Error", "Could not transform data")
                return

            # Consolidate data for selected units
            # Use the first indicator to get available units, then filter
            first_indicator = list(data_transformada_indicadores.keys())[0]
            df_first = data_transformada_indicadores[first_indicator]
            available_units = [unit for unit in self.selected_units if unit in df_first.columns]

            if len(available_units) < 2:
                tk.messagebox.showerror("Error", f"Need at least 2 units with data. Only {len(available_units)} available.")
                return

            # Consolidate data for available units
            df_consolidado = dl.consolidate_data_for_country(
                data_transformada_indicadores,
                available_units[0],  # Use first unit to get structure
                list(data_transformada_indicadores.keys())
            )

            if df_consolidado.empty:
                tk.messagebox.showerror("Error", "Could not consolidate data")
                return

            # Filter to only selected units that have data
            df_consolidado = df_consolidado[[col for col in df_consolidado.columns if col in available_units]]

            if df_consolidado.shape[1] < 2:
                tk.messagebox.showerror("Error", f"Need at least 2 units with data. Only {df_consolidado.shape[1]} have complete data.")
                return

            # Perform PCA
            import preprocessing_module as dl_prep
            import pca_module as pca_mod

            df_estandarizado = dl_prep.preprocess_data(df_consolidado)
            pca_model, df_componentes = pca_mod.realizar_pca(df_estandarizado)

            if pca_model is None or df_componentes is None:
                tk.messagebox.showerror("Error", "PCA analysis failed")
                return

            # Store PCA data for clustering
            self.pca_data = df_componentes
            self.selected_variables = list(df_componentes.columns)  # All PCA components

            # Run clustering
            method = self.linkage_var.get()
            metric = self.metric_var.get()

            from backend.analysis_logic import perform_hierarchical_clustering
            fig = perform_hierarchical_clustering(
                dataframe=self.pca_data,
                method=method,
                metric=metric,
                groups=self.groups if self.groups else None,
                group_colors=self.group_colors if self.group_colors else None,
                selected_variables=self.selected_variables,
                selected_units=list(self.pca_data.index)
            )

            # Show dendrogram in separate matplotlib window
            if fig is not None:
                import matplotlib.pyplot as plt
                plt.show()
                tk.messagebox.showinfo("Success", "Dendrogram displayed in separate window")
            else:
                tk.messagebox.showerror("Error", "Could not generate dendrogram")

        except Exception as e:
            tk.messagebox.showerror("Error", f"Clustering error: {str(e)}")

    def get_config(self):
        """Get configuration for hierarchical clustering."""
        if hasattr(self, 'file_entry') and self.file_entry.get().strip():
            return {
                'data_file': self.file_entry.get().strip(),
                'selected_indicators': self.selected_indicators,
                'selected_units': self.selected_units,
                'method': self.linkage_var.get(),
                'metric': self.metric_var.get(),
                'groups': self.groups,
                'group_colors': self.group_colors
            }
        return None
