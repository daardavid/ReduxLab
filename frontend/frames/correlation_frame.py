"""Correlation and Network Analysis Frame for PCA Application."""

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import messagebox
from frontend.ui_components import ResponsiveAnalysisFrame
from backend.constants import CODE_TO_NAME
from backend import data_loader_module as dl
import pandas as pd
import os


class CorrelationAnalysisFrame(ResponsiveAnalysisFrame):
    """Refactored frame for correlation and network analysis."""

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.setup_ui()

        # Initialize variables
        self.similarity_matrix = None
        self.histogram_fig = None
        self.heatmap_fig = None
        self.network_fig = None

    def setup_ui(self):
        # Initialize state variables first
        self.selected_file = None
        self.correlation_method = tk.StringVar(value='pearson')
        self.time_aggregated = tk.BooleanVar(value=True)
        self.similarity_threshold = tk.DoubleVar(value=0.3)
        self.visualization_type = tk.StringVar(value='heatmap')

        # Initialize visualization config variables
        self.heatmap_cmap = tk.StringVar(value='coolwarm')
        self.network_layout = tk.StringVar(value='spring')
        self.node_size = tk.IntVar(value=20)

        # Initialize selection variables
        self.selected_indicators = []
        self.selected_units = []
        self.groups = {}  # unit -> group mapping
        self.group_colors = {}  # group -> color mapping
        self.analysis_mode = tk.StringVar(value="all_units")  # Default to analyze all units

        # Description
        desc = ttk.Label(
            self.content_frame,
            text="Analyze relationships and similarities between research units using correlation analysis and network visualization.",
            wraplength=800,
            style='secondary.TLabel',
            font=("Helvetica", 11)
        )
        desc.pack(pady=(0, 30), anchor=W)

        # Configuration cards
        self.create_config_card("Data File", self.setup_file_config)
        self.create_config_card("Indicators", self.setup_indicators_config)
        self.create_config_card("Research Units", self.setup_units_config)
        self.create_config_card("Groups & Colors", self.setup_groups_config)
        self.create_config_card("Analysis Parameters", self.setup_analysis_config)
        self.create_config_card("Visualization", self.setup_visualization_config, fill=BOTH)

        # Update initial button state
        self._update_button_state()

    def _update_button_state(self):
        """Update run button state based on configuration."""
        # Get filtered units for validation
        filtered_units = self.get_filtered_units_for_analysis() if hasattr(self, 'get_filtered_units_for_analysis') else self.selected_units
        
        # CRITICAL PROTECTION: Ensure we always have enough units for analysis
        # If filtered units are insufficient, use all selected units as fallback
        if hasattr(self, 'selected_units') and self.selected_units:
            if not filtered_units or len(filtered_units) < 2:
                filtered_units = self.selected_units
                # Log the fallback for debugging
                if hasattr(self.app, 'logger'):
                    self.app.logger.warning(f"Button validation: Using fallback to all selected units ({len(self.selected_units)})")

        is_ready = (
            hasattr(self, 'file_entry') and self.file_entry.get().strip() and
            self.selected_indicators and
            filtered_units and  # Use filtered units instead of all selected units
            len(filtered_units) >= 2 and  # Ensure minimum units for correlation
            self.correlation_method.get() and
            self.visualization_type.get()
        )

        if hasattr(self.app, 'btn_run'):
            self.app.btn_run.config(state=NORMAL if is_ready else DISABLED)
            
        # Also update degree analysis button state
        if hasattr(self, 'update_degree_analysis_button_state'):
            self.update_degree_analysis_button_state()
            
        # Log current state for debugging
        if hasattr(self.app, 'logger') and hasattr(self, 'selected_units'):
            self.app.logger.info(f"Button state: {'ENABLED' if is_ready else 'DISABLED'}, "
                               f"Units: {len(filtered_units) if filtered_units else 0}, "
                               f"Total selected: {len(self.selected_units) if self.selected_units else 0}")

        return is_ready

    def select_file(self):
        """Override base select_file to auto-load indicators and units after file selection."""
        from tkinter import filedialog
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
            if self.app.file_handler.validate_file_path(file_path):
                self.file_entry.delete(0, tk.END)
                self.file_entry.insert(0, file_path)
                # Auto-load indicators and units
                self._auto_load_indicators_and_units(file_path)
                self._update_button_state()

    def _auto_load_indicators_and_units(self, file_path):
        """Automatically load all available indicators and units from the selected file."""
        try:
            excel_data = pd.read_excel(file_path, sheet_name=None, header=None)
            if not excel_data:
                return

            # Load all indicators (sheet names)
            all_indicators = list(excel_data.keys())
            self.selected_indicators = all_indicators
            if hasattr(self, 'indicators_status') and self.indicators_status:
                self.indicators_status.config(text=f"{len(all_indicators)} indicators loaded")

            # Load all units from all sheets
            all_units = set()
            for sheet_name, df in excel_data.items():
                if df.empty:
                    continue
                units_in_sheet = df.iloc[1:, 0].dropna().unique()
                all_units.update(units_in_sheet)

            unique_units = sorted(list(all_units))
            self.selected_units = unique_units

            if hasattr(self, 'units_status') and self.units_status:
                self.units_status.config(text=f"{len(unique_units)} units loaded")

            self._update_button_state()

        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {str(e)}")

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
            text="Load data file first",
            style='secondary.TLabel'
        )
        self.indicators_status.pack(side=LEFT)

    def setup_units_config(self, parent):
        """Setup units selection configuration."""
        units_frame = ttk.Frame(parent)
        units_frame.pack(fill=X, pady=(0, 10))

        ttk.Button(
            units_frame,
            text="🏢 Select Research Units",
            style='outline.TButton',
            command=self.select_units
        ).pack(side=LEFT, padx=(0, 20))

        self.units_status = ttk.Label(
            units_frame,
            text="None selected",
            style='secondary.TLabel'
        )
        self.units_status.pack(side=LEFT)

    def setup_groups_config(self, parent):
        """Setup groups and colors configuration with universal group manager integration."""
        groups_frame = ttk.Frame(parent)
        groups_frame.pack(fill=X, pady=(0, 10))

        # Universal groups button
        ttk.Button(
            groups_frame,
            text="⚙️ Manage Universal Groups",
            style='primary.TButton',
            command=self.open_universal_groups
        ).pack(side=LEFT, padx=(0, 10))

        # Load current groups button
        ttk.Button(
            groups_frame,
            text="🔄 Load Current Groups",
            style='outline.TButton',
            command=self.load_current_groups
        ).pack(side=LEFT, padx=(0, 20))

        # Groups status
        self.groups_status = ttk.Label(
            groups_frame,
            text="No groups loaded",
            style='secondary.TLabel'
        )
        self.groups_status.pack(side=LEFT)

        # Group details frame (initially hidden)
        self.group_details_frame = ttk.Frame(parent)
        self.group_details_frame.pack(fill=X, pady=(10, 0))
        
        # Group selection frame (for analysis filtering)
        self.group_selection_frame = ttk.Frame(parent)
        self.group_selection_frame.pack(fill=X, pady=(10, 0))
        
        # This will be populated when groups are loaded
        self.update_groups_display()

    def setup_group_analysis_filter(self):
        """Setup group-based analysis filtering controls."""
        # Sync groups from universal manager first
        self.sync_groups_format()
        
        # Clear existing widgets
        for widget in self.group_selection_frame.winfo_children():
            widget.destroy()
        
        if not self.groups or not self.group_colors:
            return
        
        # Create analysis filter frame
        filter_frame = ttk.LabelFrame(self.group_selection_frame, text="🎯 Analyze Specific Groups", padding=10)
        filter_frame.pack(fill=X, pady=(5, 0))
        
        # Filter mode selection
        filter_mode_frame = ttk.Frame(filter_frame)
        filter_mode_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(filter_mode_frame, text="Analysis Mode:").pack(side=LEFT, padx=(0, 10))
        
        self.analysis_mode = tk.StringVar(value="all_units")
        
        # Radio buttons for analysis mode
        modes_frame = ttk.Frame(filter_mode_frame)
        modes_frame.pack(side=LEFT)
        
        ttk.Radiobutton(
            modes_frame,
            text="All Units",
            variable=self.analysis_mode,
            value="all_units",
            command=self.on_analysis_mode_change
        ).pack(side=LEFT, padx=(0, 15))
        
        ttk.Radiobutton(
            modes_frame,
            text="Selected Groups",
            variable=self.analysis_mode,
            value="selected_groups",
            command=self.on_analysis_mode_change
        ).pack(side=LEFT, padx=(0, 15))
        
        ttk.Radiobutton(
            modes_frame,
            text="Exclude Groups",
            variable=self.analysis_mode,
            value="exclude_groups",
            command=self.on_analysis_mode_change
        ).pack(side=LEFT)
        
        # Group selection area (initially hidden)
        self.group_checkboxes_frame = ttk.Frame(filter_frame)
        self.group_checkboxes_frame.pack(fill=X, pady=(10, 0))
        
        # Initialize group selection checkboxes
        self.group_selection_vars = {}
        self.setup_group_checkboxes()

    def setup_group_checkboxes(self):
        """Setup checkboxes for group selection."""
        # Clear existing checkboxes
        for widget in self.group_checkboxes_frame.winfo_children():
            widget.destroy()
        
        if not self.groups:
            return
        
        # Get unique groups
        unique_groups = set(self.groups.values())
        if not unique_groups:
            return
        
        # Create checkboxes frame
        checkboxes_container = ttk.Frame(self.group_checkboxes_frame)
        checkboxes_container.pack(fill=X)
        
        # Title
        title_frame = ttk.Frame(checkboxes_container)
        title_frame.pack(fill=X, pady=(0, 5))
        
        ttk.Label(
            title_frame,
            text="Select Groups for Analysis:",
            font=("Helvetica", 10, "bold")
        ).pack(side=LEFT)
        
        # Select all/none buttons
        select_buttons_frame = ttk.Frame(title_frame)
        select_buttons_frame.pack(side=RIGHT)
        
        ttk.Button(
            select_buttons_frame,
            text="Select All",
            command=self.select_all_groups,
            style="info.Outline.TButton"
        ).pack(side=LEFT, padx=(0, 5))
        
        ttk.Button(
            select_buttons_frame,
            text="Clear All",
            command=self.clear_all_groups_selection,
            style="secondary.Outline.TButton"
        ).pack(side=LEFT)
        
        # Create checkboxes for each group
        groups_grid_frame = ttk.Frame(checkboxes_container)
        groups_grid_frame.pack(fill=X, pady=(5, 0))
        
        # Organize groups by rows
        groups_list = sorted(unique_groups)
        max_cols = 3
        
        for i, group_name in enumerate(groups_list):
            row = i // max_cols
            col = i % max_cols
            
            # Create frame for this group checkbox
            group_cb_frame = ttk.Frame(groups_grid_frame)
            group_cb_frame.grid(row=row, column=col, sticky='w', padx=(0, 20), pady=2)
            
            # Color indicator
            color_label = tk.Label(
                group_cb_frame,
                width=2,
                height=1,
                bg=self.group_colors.get(group_name, '#CCCCCC'),
                relief="solid",
                borderwidth=1
            )
            color_label.pack(side=LEFT, padx=(0, 5))
            
            # Checkbox with group name and unit count
            units_in_group = [u for u, g in self.groups.items() if g == group_name]
            var = tk.BooleanVar(value=True)  # Default to selected
            self.group_selection_vars[group_name] = var
            
            checkbox_text = f"{group_name} ({len(units_in_group)} units)"
            ttk.Checkbutton(
                group_cb_frame,
                text=checkbox_text,
                variable=var,
                command=self.on_group_selection_change
            ).pack(side=LEFT)
        
        # Initially hide the group selection area
        self.group_checkboxes_frame.pack_forget()

    def on_analysis_mode_change(self):
        """Handle analysis mode change."""
        try:
            mode = self.analysis_mode.get()
            
            if hasattr(self.app, 'logger'):
                self.app.logger.info(f"Analysis mode changed to: {mode}")
            
            if mode == "all_units":
                # Hide group selection
                self.group_checkboxes_frame.pack_forget()
            elif mode in ["selected_groups", "exclude_groups"]:
                # Show group selection
                self.group_checkboxes_frame.pack(fill=X, pady=(10, 0))
            
            # CRITICAL: Force button state update with delay to ensure UI is ready
            def delayed_update():
                self.update_analysis_mode_status()
                self._update_button_state()
                
                # ADDITIONAL PROTECTION: Force enable if we have enough units
                if hasattr(self, 'selected_units') and self.selected_units and len(self.selected_units) >= 2:
                    filtered_units = self.get_filtered_units_for_analysis()
                    if filtered_units and len(filtered_units) >= 2:
                        # Force enable the button as final protection
                        if hasattr(self.app, 'btn_run'):
                            self.app.btn_run.config(state=NORMAL)
                        if hasattr(self.app, 'logger'):
                            self.app.logger.info(f"Mode change completed: {len(filtered_units)} units available for analysis")
            
            # Execute update with small delay to ensure UI rendering is complete
            if hasattr(self, 'after'):
                self.after(10, delayed_update)
            else:
                delayed_update()
                
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Error in on_analysis_mode_change: {e}")
            # Force fallback to enabled state
            if hasattr(self.app, 'btn_run'):
                self.app.btn_run.config(state=NORMAL)

    def update_analysis_mode_status(self):
        """Update the status text based on current analysis mode."""
        mode = self.analysis_mode.get()
        
        if mode == "all_units":
            total_units = len(self.selected_units) if hasattr(self, 'selected_units') else 0
            status_text = f"Analyzing all {total_units} units"
        
        elif mode == "selected_groups":
            selected_groups = [name for name, var in self.group_selection_vars.items() if var.get()]
            selected_units = []
            for group_name in selected_groups:
                units_in_group = [u for u, g in self.groups.items() if g == group_name]
                selected_units.extend(units_in_group)
            
            status_text = f"Analyzing {len(selected_groups)} groups ({len(selected_units)} units)"
        
        elif mode == "exclude_groups":
            excluded_groups = [name for name, var in self.group_selection_vars.items() if var.get()]
            excluded_units = []
            for group_name in excluded_groups:
                units_in_group = [u for u, g in self.groups.items() if g == group_name]
                excluded_units.extend(units_in_group)
            
            total_units = len(self.selected_units) if hasattr(self, 'selected_units') else 0
            remaining_units = total_units - len(excluded_units)
            
            status_text = f"Excluding {len(excluded_groups)} groups ({remaining_units} units for analysis)"
        
        # Update groups status if it exists
        if hasattr(self, 'groups_status'):
            current_text = self.groups_status.cget('text')
            if " - " in current_text:
                base_text = current_text.split(" - ")[0]
            else:
                base_text = current_text
            
            self.groups_status.config(text=f"{base_text} - {status_text}")

    def select_all_groups(self):
        """Select all groups for analysis."""
        for var in self.group_selection_vars.values():
            var.set(True)
        self.on_group_selection_change()

    def clear_all_groups_selection(self):
        """Clear all group selections."""
        for var in self.group_selection_vars.values():
            var.set(False)
        self.on_group_selection_change()

    def on_group_selection_change(self):
        """Handle group selection change."""
        try:
            if hasattr(self.app, 'logger'):
                mode = self.analysis_mode.get() if hasattr(self, 'analysis_mode') else 'unknown'
                selected_count = len([v for v in self.group_selection_vars.values() if v.get()]) if hasattr(self, 'group_selection_vars') else 0
                self.app.logger.info(f"Group selection changed in mode {mode}: {selected_count} groups selected")
            
            self.update_analysis_mode_status()
            self._update_button_state()
            
            # ADDITIONAL PROTECTION: Ensure button is enabled if analysis is viable
            if hasattr(self, 'selected_units') and self.selected_units and len(self.selected_units) >= 2:
                filtered_units = self.get_filtered_units_for_analysis()
                if filtered_units and len(filtered_units) >= 2:
                    if hasattr(self.app, 'btn_run'):
                        self.app.btn_run.config(state=NORMAL)
                    if hasattr(self.app, 'logger'):
                        self.app.logger.info(f"Group selection validated: {len(filtered_units)} units available")
                        
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Error in on_group_selection_change: {e}")
            # Force fallback to enabled state if we have basic requirements
            if hasattr(self, 'selected_units') and self.selected_units and len(self.selected_units) >= 2:
                if hasattr(self.app, 'btn_run'):
                    self.app.btn_run.config(state=NORMAL)

    def get_filtered_units_for_analysis(self):
        """Get the list of units to include in analysis based on current filter mode."""
        if not hasattr(self, 'selected_units') or not self.selected_units:
            return []
        
        try:
            # Sync groups from UniversalGroupManager before processing
            self.sync_groups_format()
            
            mode = getattr(self, 'analysis_mode', None)
            if not mode:
                return self.selected_units
            
            current_mode = mode.get()
            
            if current_mode == "all_units":
                # Return all selected units, regardless of group status
                return self.selected_units
            
            elif current_mode == "selected_groups":
                if not hasattr(self, 'group_selection_vars') or not self.group_selection_vars:
                    # If no groups are configured, fallback to all units
                    if hasattr(self.app, 'logger'):
                        self.app.logger.warning("selected_groups mode: No group_selection_vars - fallback to all units")
                    return self.selected_units
                
                # Get units from selected groups
                selected_groups = [name for name, var in self.group_selection_vars.items() if var.get()]
                
                # If no groups are selected, fallback to all units to prevent analysis blocking
                if not selected_groups:
                    if hasattr(self.app, 'logger'):
                        self.app.logger.warning("selected_groups mode: No groups selected - fallback to all units")
                    return self.selected_units
                
                filtered_units = []
                for group_name in selected_groups:
                    units_in_group = [u for u, g in self.groups.items() if g == group_name and u in self.selected_units]
                    filtered_units.extend(units_in_group)
                
                # CRITICAL FIX: Always fallback to all units if no matches found
                unique_filtered = list(set(filtered_units))
                
                if hasattr(self.app, 'logger'):
                    self.app.logger.info(f"selected_groups mode: {len(selected_groups)} groups selected, "
                                       f"{len(unique_filtered)} units found, "
                                       f"{len(self.selected_units)} total units")
                
                # If no units found in selected groups, ALWAYS fallback to all units
                if not unique_filtered:
                    if hasattr(self.app, 'logger'):
                        self.app.logger.warning("selected_groups mode: No units found in groups - FALLBACK to all units")
                    return self.selected_units
                
                # If fewer than 2 units found, fallback to all units for analysis viability
                if len(unique_filtered) < 2:
                    if hasattr(self.app, 'logger'):
                        self.app.logger.warning(f"selected_groups mode: Only {len(unique_filtered)} units found - FALLBACK to all units")
                    return self.selected_units
                
                return unique_filtered
            
            elif current_mode == "exclude_groups":
                if not hasattr(self, 'group_selection_vars') or not self.group_selection_vars:
                    # If no groups are configured, return all units
                    if hasattr(self.app, 'logger'):
                        self.app.logger.warning("exclude_groups mode: No group_selection_vars - return all units")
                    return self.selected_units
                
                # Get units from excluded groups
                excluded_groups = [name for name, var in self.group_selection_vars.items() if var.get()]
                
                if hasattr(self.app, 'logger'):
                    self.app.logger.info(f"exclude_groups mode: {len(excluded_groups)} groups to exclude")
                
                excluded_units = []
                for group_name in excluded_groups:
                    units_in_group = [u for u, g in self.groups.items() if g == group_name]
                    excluded_units.extend(units_in_group)
                
                # Return units not in excluded groups (including ungrouped units)
                filtered_units = [u for u in self.selected_units if u not in excluded_units]
                
                if hasattr(self.app, 'logger'):
                    self.app.logger.info(f"exclude_groups mode: {len(excluded_units)} units excluded, "
                                       f"{len(filtered_units)} units remaining")
                
                # Ensure at least 2 units remain for analysis
                if len(filtered_units) < 2:
                    if hasattr(self.app, 'logger'):
                        self.app.logger.warning(f"exclude_groups mode: Only {len(filtered_units)} units remaining - FALLBACK to all units")
                    return self.selected_units
                
                return filtered_units
            
            # Fallback to all units for any unexpected mode
            return self.selected_units
            
        except Exception as e:
            # CRITICAL PROTECTION: If anything fails, return all selected units
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Error in get_filtered_units_for_analysis: {e}")
            return self.selected_units

    def open_universal_groups(self):
        """Open the universal group manager."""
        try:
            from group_manager import show_group_manager_gui
            
            # Get available units from current data
            available_units = []
            if hasattr(self, 'selected_units') and self.selected_units:
                available_units = self.selected_units.copy()
            elif hasattr(self, 'file_entry') and self.file_entry.get().strip():
                # Try to load units from file
                try:
                    file_path = self.file_entry.get().strip()
                    if file_path and os.path.exists(file_path):
                        if file_path.endswith('.csv'):
                            df = pd.read_csv(file_path)
                            if 'Unit' in df.columns:
                                available_units = list(df['Unit'].unique())
                            elif 'Empresa' in df.columns:
                                available_units = list(df['Empresa'].unique())
                            else:
                                available_units = list(df.index)
                except Exception as e:
                    self.app.logger.warning(f"Could not load units for group manager: {e}")
            
            # Show group manager
            show_group_manager_gui(self.app, available_units)
            
            # Refresh groups after closing manager
            self.app.after(1000, self.load_current_groups)  # Delayed refresh
            
        except Exception as e:
            self.app.logger.error(f"Error opening group manager: {e}")
            tk.messagebox.showerror("Error", f"Error opening group manager: {e}")

    def load_current_groups(self):
        """Load groups from universal group manager for current units."""
        try:
            # Get current units
            if not hasattr(self, 'selected_units') or not self.selected_units:
                self.groups_status.config(text="Select units first to load groups")
                return

            # Log initial state
            if hasattr(self.app, 'logger'):
                self.app.logger.info(f"Loading groups for {len(self.selected_units)} selected units")
            
            # Sync groups from universal manager
            self.sync_groups_format()
            
            # Get universal group manager
            group_manager = self.app.group_manager
            
            # Get groups for current units
            unit_groups = group_manager.get_groups_for_units(self.selected_units)
            
            groups_found = 0
            units_grouped = 0
            
            for group_name, units_in_group in unit_groups.items():
                if group_name != 'Ungrouped' and units_in_group:
                    groups_found += 1
                    units_grouped += len(units_in_group)
                        
                    # Increment usage counter
                    group_manager.increment_usage(group_name)
            
            # Update display
            if groups_found > 0:
                self.groups_status.config(
                    text=f"{groups_found} groups loaded ({units_grouped}/{len(self.selected_units)} units grouped)"
                )
                self.app.logger.info(f"Loaded {groups_found} groups for correlation analysis")
            else:
                self.groups_status.config(text="No matching groups found")
                if hasattr(self.app, 'logger'):
                    self.app.logger.warning("No matching groups found - all units will be treated as ungrouped")
            
            self.update_groups_display()
            self.setup_group_analysis_filter()  # Setup group-based filtering
            
            # CRITICAL PROTECTION: Ensure analysis mode stays in "all_units" after loading groups
            # This prevents the analysis from being disabled when groups are loaded
            if hasattr(self, 'analysis_mode'):
                previous_mode = self.analysis_mode.get()
                self.analysis_mode.set("all_units")
                
                # Log mode change for debugging
                if hasattr(self.app, 'logger'):
                    self.app.logger.info(f"Analysis mode changed from '{previous_mode}' to 'all_units' after loading groups")
                
                self.on_analysis_mode_change()  # Update UI visibility
            
            # ADDITIONAL PROTECTION: Force button state update with extra validation
            if hasattr(self, '_update_button_state'):
                button_state = self._update_button_state()
                
                # If button is still disabled, force enable it if we have enough units
                if not button_state and len(self.selected_units) >= 2:
                    if hasattr(self.app, 'logger'):
                        self.app.logger.warning(f"Forcing button enable - have {len(self.selected_units)} units")
                    
                    # Force enable the button as a last resort
                    if hasattr(self.app, 'btn_run'):
                        self.app.btn_run.config(state=NORMAL)
            
            # Log final state
            if hasattr(self.app, 'logger'):
                final_mode = self.analysis_mode.get() if hasattr(self, 'analysis_mode') else 'unknown'
                self.app.logger.info(f"Groups loading completed - Final mode: {final_mode}")
            
        except Exception as e:
            self.app.logger.error(f"Error loading groups: {e}")
            tk.messagebox.showerror("Error", f"Error loading groups: {e}")
            
            # Even if group loading fails, ensure analysis can proceed
            if hasattr(self, 'analysis_mode'):
                self.analysis_mode.set("all_units")
            if hasattr(self, '_update_button_state'):
                self._update_button_state()

    def update_groups_display(self):
        """Update the visual display of loaded groups."""
        # Sync groups from universal manager first
        self.sync_groups_format()
        
        # Clear existing widgets
        for widget in self.group_details_frame.winfo_children():
            widget.destroy()
        
        if not self.groups or not self.group_colors:
            return
        
        # Group summary frame
        summary_frame = ttk.LabelFrame(self.group_details_frame, text="🏷️ Loaded Groups", padding=5)
        summary_frame.pack(fill=X, pady=(5, 0))
        
        # Create a frame for group display
        groups_display_frame = ttk.Frame(summary_frame)
        groups_display_frame.pack(fill=X)
        
        # Display each group with its color
        displayed_groups = set()
        row = 0
        col = 0
        max_cols = 3
        
        for unit, group_name in self.groups.items():
            if group_name not in displayed_groups:
                displayed_groups.add(group_name)
                
                # Create group indicator
                group_frame = ttk.Frame(groups_display_frame)
                group_frame.grid(row=row, column=col, padx=5, pady=2, sticky='w')
                
                # Color indicator
                color_label = tk.Label(
                    group_frame,
                    width=2,
                    height=1,
                    bg=self.group_colors.get(group_name, '#CCCCCC'),
                    relief="solid",
                    borderwidth=1
                )
                color_label.pack(side=LEFT, padx=(0, 5))
                
                # Group name and count
                units_in_group = [u for u, g in self.groups.items() if g == group_name]
                group_text = f"{group_name} ({len(units_in_group)})"
                ttk.Label(
                    group_frame,
                    text=group_text,
                    font=("Helvetica", 9)
                ).pack(side=LEFT)
                
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1

    def select_indicators(self):
        """Select indicators for correlation analysis (sheet names in multi-sheet format)."""
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

                    # Create checkboxes for each indicator
                    for indicator in indicator_cols:
                        var = tk.BooleanVar()
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

    def select_units(self):
        """Select research units for correlation analysis with search/filter functionality."""
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

                    # Create display names using CODE_TO_NAME mapping
                    display_names = {
                        unit: f"{CODE_TO_NAME.get(unit, str(unit))} ({unit})"
                        for unit in available_units
                    }

                    # Calculate dialog size based on number of units
                    n_units = len(available_units)
                    dialog_height = min(700, max(550, 200 + n_units * 25))
                    dialog_width = 550  # Wider for search bar

                    # Create selection dialog
                    dialog = tk.Toplevel(self)
                    dialog.title("Select Research Units for Correlation Analysis")
                    dialog.geometry(f"{dialog_width}x{dialog_height}")
                    dialog.transient(self)
                    dialog.grab_set()

                    # Header
                    header_frame = ttk.Frame(dialog)
                    header_frame.pack(fill=X, padx=20, pady=(10, 5))

                    ttk.Label(
                        header_frame,
                        text="Available research units:",
                        font=("Helvetica", 12, "bold")
                    ).pack(anchor=W)

                    # ✅ Search/filter bar (Excel-style)
                    search_frame = ttk.Frame(dialog)
                    search_frame.pack(fill=X, padx=20, pady=(5, 10))

                    search_var = tk.StringVar()
                    search_var.trace_add("write", lambda *args: filter_units())

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
                        text=f"Showing {n_units} of {n_units} units",
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
                    checkbox_widgets = {}

                    # Create checkboxes for each unit
                    for unit in available_units:
                        var = tk.BooleanVar()
                        unit_vars[unit] = var
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
                    def filter_units():
                        search_text = search_var.get().lower()
                        visible_count = 0

                        for unit in available_units:
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
                            text=f"Showing {visible_count} of {n_units} units"
                        )

                        # Update select all state
                        update_select_all_state()

                    def toggle_select_all():
                        """Toggle all VISIBLE checkboxes."""
                        search_text = search_var.get().lower()
                        state = select_all_var.get()

                        for unit in available_units:
                            unit_text = f"{unit} {display_names[unit]}".lower()
                            if search_text in unit_text:  # Only visible ones
                                unit_vars[unit].set(state)

                    def update_select_all_state():
                        """Update select all checkbox based on visible items."""
                        search_text = search_var.get().lower()
                        visible_units = [
                            unit for unit in available_units
                            if search_text in f"{unit} {display_names[unit]}".lower()
                        ]

                        if not visible_units:
                            select_all_var.set(False)
                            return

                        all_selected = all(unit_vars[unit].get() for unit in visible_units)
                        select_all_var.set(all_selected)

                    # Buttons
                    button_frame = ttk.Frame(dialog)
                    button_frame.pack(fill=X, padx=20, pady=10)

                    def confirm_selection():
                        selected = [unit for unit in available_units if unit_vars[unit].get()]
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

    def create_groups(self):
        """Create groups and assign colors for research units."""
        if not self.selected_units:
            tk.messagebox.showwarning("Warning", "Select research units first")
            return

        # Create group assignment dialog
        dialog = tk.Toplevel(self)
        dialog.title("Create Groups & Assign Colors")
        dialog.geometry("600x700")
        dialog.transient(self)
        dialog.grab_set()

        # Title
        ttk.Label(dialog, text="Create Groups and Assign Colors:", font=("Helvetica", 14, "bold")).pack(pady=10)

        # Main frame with scrollbar
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=BOTH, expand=YES, padx=20, pady=(0, 20))

        # Canvas and scrollbar
        canvas = tk.Canvas(main_frame, height=500)
        scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Group management section
        group_frame = ttk.LabelFrame(scrollable_frame, text="Group Management", padding=10)
        group_frame.pack(fill=X, pady=(0, 20))

        # Group name entry
        name_frame = ttk.Frame(group_frame)
        name_frame.pack(fill=X, pady=(0, 10))

        ttk.Label(name_frame, text="Group Name:").pack(side=LEFT)
        group_name_var = tk.StringVar()
        ttk.Entry(name_frame, textvariable=group_name_var, width=20).pack(side=LEFT, padx=(10, 0))

        # Color picker
        color_frame = ttk.Frame(group_frame)
        color_frame.pack(fill=X, pady=(0, 10))

        ttk.Label(color_frame, text="Color:").pack(side=LEFT)
        color_var = tk.StringVar(value="#1f77b4")
        ttk.Entry(color_frame, textvariable=color_var, width=10).pack(side=LEFT, padx=(10, 0))

        # Predefined colors
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        colors_frame = ttk.Frame(group_frame)
        colors_frame.pack(fill=X, pady=(0, 10))

        def set_color(c):
            color_var.set(c)

        for i, c in enumerate(colors):
            ttk.Button(colors_frame, text="■", style=f"outline.TButton",
                      command=lambda c=c: set_color(c)).grid(row=i//5, column=i%5, padx=2, pady=2)

        # Add group button
        def add_group():
            name = group_name_var.get().strip()
            color = color_var.get().strip()
            if name and color:
                self.group_colors[name] = color
                update_group_list()
                group_name_var.set("")
            else:
                tk.messagebox.showwarning("Warning", "Enter group name and color")

        ttk.Button(group_frame, text="Add Group", command=add_group).pack(anchor=E)

        # Groups list
        groups_list_frame = ttk.LabelFrame(scrollable_frame, text="Defined Groups", padding=10)
        groups_list_frame.pack(fill=X, pady=(0, 20))

        groups_listbox = tk.Listbox(groups_list_frame, height=5)
        groups_scrollbar = ttk.Scrollbar(groups_list_frame, orient=VERTICAL, command=groups_listbox.yview)
        groups_listbox.configure(yscrollcommand=groups_scrollbar.set)

        groups_listbox.pack(side=LEFT, fill=BOTH, expand=YES)
        groups_scrollbar.pack(side=RIGHT, fill=Y)

        def update_group_list():
            groups_listbox.delete(0, tk.END)
            for group, color in self.group_colors.items():
                groups_listbox.insert(tk.END, f"{group}: {color}")

        # Unit assignment section
        assign_frame = ttk.LabelFrame(scrollable_frame, text="Assign Units to Groups", padding=10)
        assign_frame.pack(fill=X, pady=(0, 20))

        # Units list
        units_list_frame = ttk.Frame(assign_frame)
        units_list_frame.pack(fill=X, pady=(0, 10))

        ttk.Label(units_list_frame, text="Available Units:").pack(anchor=W)
        units_listbox = tk.Listbox(units_list_frame, selectmode=tk.MULTIPLE, height=8)
        units_scrollbar = ttk.Scrollbar(units_list_frame, orient=VERTICAL, command=units_listbox.yview)
        units_listbox.configure(yscrollcommand=units_scrollbar.set)

        for unit in self.selected_units:
            units_listbox.insert(tk.END, unit)

        units_listbox.pack(side=LEFT, fill=BOTH, expand=YES)
        units_scrollbar.pack(side=RIGHT, fill=Y)

        # Assignment controls
        controls_frame = ttk.Frame(assign_frame)
        controls_frame.pack(fill=X, pady=(0, 10))

        def assign_to_group():
            unit_selections = units_listbox.curselection()
            group_selection = groups_listbox.curselection()

            if not unit_selections:
                tk.messagebox.showwarning("Warning", "Select at least one unit first")
                return

            if not group_selection:
                tk.messagebox.showwarning("Warning", "Select a group first")
                return

            group_name = list(self.group_colors.keys())[group_selection[0]]
            assigned_count = 0

            for selection_idx in unit_selections:
                unit = self.selected_units[selection_idx]
                if unit not in self.groups:  # Only assign if not already assigned
                    self.groups[unit] = group_name
                    assigned_count += 1

            if assigned_count > 0:
                update_assignments_list()
                # Refresh units list to show new assignments
                units_listbox.delete(0, tk.END)
                for unit in self.selected_units:
                    units_listbox.insert(tk.END, unit)
                units_listbox.selection_clear(0, tk.END)  # Clear unit selection
                tk.messagebox.showinfo("Success", f"Assigned {assigned_count} units to group '{group_name}'")
            else:
                tk.messagebox.showwarning("Warning", "Selected units are already assigned to groups")

        ttk.Button(controls_frame, text="→ Assign to Group", command=assign_to_group).pack()

        # Assignments list
        assignments_frame = ttk.Frame(assign_frame)
        assignments_frame.pack(fill=X)

        ttk.Label(assignments_frame, text="Assignments:").pack(anchor=W)
        assignments_listbox = tk.Listbox(assignments_frame, height=8)
        assignments_scrollbar = ttk.Scrollbar(assignments_frame, orient=VERTICAL, command=assignments_listbox.yview)
        assignments_listbox.configure(yscrollcommand=assignments_scrollbar.set)

        assignments_listbox.pack(side=LEFT, fill=BOTH, expand=YES)
        assignments_scrollbar.pack(side=RIGHT, fill=Y)

        def update_assignments_list():
            assignments_listbox.delete(0, tk.END)
            for unit, group in self.groups.items():
                assignments_listbox.insert(tk.END, f"{unit} → {group}")

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
                tk.messagebox.showwarning("Warning", "No units assigned to groups")
                return

            self.update_groups_status()
            dialog.destroy()
            tk.messagebox.showinfo("Success", f"Groups created with {len(self.groups)} unit assignments")

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

    def update_groups_status(self):
        """Update groups status label."""
        if self.groups:
            self.groups_status.config(text=f"{len(set(self.groups.values()))} groups, {len(self.groups)} assignments")
        else:
            self.groups_status.config(text="No groups defined")

    def setup_analysis_config(self, parent):
        """Setup analysis parameters configuration."""
        # Correlation method
        ttk.Label(parent, text="Correlation Method:", font=("Helvetica", 10, "bold")).pack(anchor=W, pady=(10, 5))

        method_frame = ttk.Frame(parent)
        method_frame.pack(fill=X, pady=(0, 10))

        ttk.Radiobutton(
            method_frame, text="Pearson", variable=self.correlation_method, value='pearson'
        ).pack(side=LEFT, padx=(0, 20))

        ttk.Radiobutton(
            method_frame, text="Spearman", variable=self.correlation_method, value='spearman'
        ).pack(side=LEFT, padx=(0, 20))

        ttk.Radiobutton(
            method_frame, text="Kendall", variable=self.correlation_method, value='kendall'
        ).pack(side=LEFT, padx=(0, 20))

        ttk.Radiobutton(
            method_frame, text="DTW (Time Series)", variable=self.correlation_method, value='dtw'
        ).pack(side=LEFT)

        # Time aggregation
        ttk.Checkbutton(
            parent, text="Aggregate data by time period (recommended for most analyses)",
            variable=self.time_aggregated
        ).pack(anchor=W, pady=(10, 5))

        # Similarity threshold with histogram preview
        ttk.Label(parent, text="Similarity Threshold:", font=("Helvetica", 10, "bold")).pack(anchor=W, pady=(10, 5))

        threshold_frame = ttk.Frame(parent)
        threshold_frame.pack(fill=X, pady=(0, 10))

        ttk.Label(threshold_frame, text="Threshold:").grid(row=0, column=0, sticky=W, padx=(0, 10))
        threshold_scale = ttk.Scale(
            threshold_frame, from_=0.0, to=1.0, variable=self.similarity_threshold,
            orient=HORIZONTAL, length=200
        )
        threshold_scale.grid(row=0, column=1, padx=(0, 10))

        threshold_label = ttk.Label(threshold_frame, textvariable=self.similarity_threshold, width=5)
        threshold_label.grid(row=0, column=2)

        # Preview histogram button
        ttk.Button(
            threshold_frame, text="📊 Preview Distribution",
            style='info.Outline.TButton', command=self.show_histogram_preview
        ).grid(row=0, column=3, padx=(20, 0))

    def setup_visualization_config(self, parent):
        """Setup visualization configuration."""
        # Visualization type
        ttk.Label(parent, text="Visualization Type:", font=("Helvetica", 10, "bold")).pack(anchor=W, pady=(10, 5))

        viz_frame = ttk.Frame(parent)
        viz_frame.pack(fill=X, pady=(0, 10))

        ttk.Radiobutton(
            viz_frame, text="Heatmap", variable=self.visualization_type, value='heatmap'
        ).pack(side=LEFT, padx=(0, 20))

        ttk.Radiobutton(
            viz_frame, text="Network Graph", variable=self.visualization_type, value='network'
        ).pack(side=LEFT, padx=(0, 20))

        ttk.Radiobutton(
            viz_frame, text="Hierarchical View", variable=self.visualization_type, value='hierarchical'
        ).pack(side=LEFT, padx=(0, 20))

        ttk.Radiobutton(
            viz_frame, text="Both", variable=self.visualization_type, value='both'
        ).pack(side=LEFT)

        # Create a scrollable container for visualization options
        self.viz_container_frame = ttk.Frame(parent)
        self.viz_container_frame.pack(fill=BOTH, expand=YES, pady=(10, 0))

        # Canvas and scrollbar for visualization options - let it expand naturally
        self.viz_canvas = tk.Canvas(self.viz_container_frame)  # Remove fixed height to allow expansion
        self.viz_scrollbar = ttk.Scrollbar(self.viz_container_frame, orient=VERTICAL, command=self.viz_canvas.yview)
        self.viz_options_frame = ttk.Frame(self.viz_canvas)

        # Ensure the scroll region updates when the frame size changes
        def update_scroll_region(event=None):
            self.viz_canvas.configure(scrollregion=self.viz_canvas.bbox("all"))

        self.viz_options_frame.bind("<Configure>", update_scroll_region)

        self.viz_canvas.create_window((0, 0), window=self.viz_options_frame, anchor="nw")
        self.viz_canvas.configure(yscrollcommand=self.viz_scrollbar.set)

        # Pack canvas to fill available space
        self.viz_canvas.pack(side=LEFT, fill=BOTH, expand=YES)
        self.viz_scrollbar.pack(side=RIGHT, fill=Y)

        # Force initial scroll region update
        self.viz_container_frame.after(100, update_scroll_region)

        # Bind to update options when visualization type changes
        self.visualization_type.trace('w', lambda *args: self.update_visualization_options())

        # Initialize options
        self.update_visualization_options()

    def update_visualization_options(self):
        """Update visualization options based on selected type."""
        # Clear existing options
        for widget in self.viz_options_frame.winfo_children():
            widget.destroy()

        viz_type = self.visualization_type.get()

        if viz_type in ['heatmap', 'both']:
            # Heatmap options
            heatmap_frame = ttk.LabelFrame(self.viz_options_frame, text="Heatmap Options", padding=10)
            heatmap_frame.pack(fill=X, pady=(0, 10))

            # Outlier filtering options for heatmap
            self.heatmap_filter_outliers_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                heatmap_frame, text="🧹 Filter outlier units",
                variable=self.heatmap_filter_outliers_var
            ).grid(row=0, column=0, columnspan=2, sticky=W, pady=2)

            # Minimum degree for heatmap
            ttk.Label(heatmap_frame, text="Min connections:").grid(row=1, column=0, sticky=W, pady=2)
            self.heatmap_min_degree_var = tk.IntVar(value=1)
            heatmap_degree_combo = ttk.Combobox(
                heatmap_frame, textvariable=self.heatmap_min_degree_var,
                values=[0, 1, 2, 3, 5], state="readonly", width=5
            )
            heatmap_degree_combo.grid(row=1, column=1, padx=(10, 0), pady=2)

            # Minimum weighted degree for heatmap
            ttk.Label(heatmap_frame, text="Min correlation sum:").grid(row=2, column=0, sticky=W, pady=2)
            self.heatmap_min_weighted_degree_var = tk.DoubleVar(value=0.5)
            heatmap_weighted_scale = ttk.Scale(
                heatmap_frame, from_=0.0, to=2.0, variable=self.heatmap_min_weighted_degree_var,
                orient=HORIZONTAL, length=100
            )
            heatmap_weighted_scale.grid(row=2, column=1, padx=(10, 0), pady=2)

            ttk.Label(heatmap_frame, text="Color Scheme:").grid(row=3, column=0, sticky=W, pady=2)
            self.heatmap_cmap = tk.StringVar(value='coolwarm')
            cmap_combo = ttk.Combobox(
                heatmap_frame, textvariable=self.heatmap_cmap,
                values=['coolwarm', 'RdYlBu', 'viridis', 'plasma', 'Blues', 'Reds'],
                state="readonly", width=15
            )
            cmap_combo.grid(row=3, column=1, padx=(10, 0), pady=2)

            # Interactive heatmap option
            self.interactive_heatmap_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                heatmap_frame, text="Interactive (Plotly)",
                variable=self.interactive_heatmap_var
            ).grid(row=4, column=0, columnspan=2, sticky=W, pady=2)

            # Force horizontal labels option
            self.force_horizontal_labels_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                heatmap_frame, text="🔄 Force X-axis diagonal labels",
                variable=self.force_horizontal_labels_var
            ).grid(row=5, column=0, columnspan=2, sticky=W, pady=2)

            ttk.Checkbutton(
                heatmap_frame, text="Show values in cells", variable=tk.BooleanVar(value=False)
            ).grid(row=6, column=0, columnspan=2, sticky=W, pady=2)

        if viz_type in ['network', 'both']:
            # Network options
            network_frame = ttk.LabelFrame(self.viz_options_frame, text="Network Options", padding=10)
            network_frame.pack(fill=X, pady=(0, 10))

            # Info text about network features
            info_text = "💡 Network features: Edge thickness represents correlation strength, node size represents connectivity"
            ttk.Label(network_frame, text=info_text, font=("Helvetica", 9), foreground="blue").grid(row=0, column=0, columnspan=3, sticky=W, pady=(0, 10))

            # === LAYOUT OPTIMIZATION SECTION ===
            layout_opt_frame = ttk.LabelFrame(network_frame, text="🎨 Layout Optimization", padding=5)
            layout_opt_frame.grid(row=1, column=0, columnspan=3, sticky=EW, pady=(0, 5))

            # Layout algorithm with improved options
            ttk.Label(layout_opt_frame, text="Layout Algorithm:").grid(row=0, column=0, sticky=W, pady=2)
            self.network_layout = tk.StringVar(value='spring_optimized')
            layout_combo = ttk.Combobox(
                layout_opt_frame, textvariable=self.network_layout,
                values=['spring_optimized', 'kamada_kawai', 'spring_default', 'circular', 'fruchterman_reingold'],
                state="readonly", width=18
            )
            layout_combo.grid(row=0, column=1, padx=(10, 0), pady=2)

            # Layout info button
            layout_info_btn = ttk.Button(
                layout_opt_frame, text="?", width=3,
                command=self.show_layout_info,
                style="info.TButton"
            )
            layout_info_btn.grid(row=0, column=2, padx=(5, 0), pady=2)

            # Spring layout repulsion factor (k parameter)
            ttk.Label(layout_opt_frame, text="Node Separation (k):").grid(row=1, column=0, sticky=W, pady=2)
            self.spring_k_var = tk.DoubleVar(value=1.0)
            spring_k_scale = ttk.Scale(
                layout_opt_frame, from_=0.3, to=3.0, variable=self.spring_k_var,
                orient=HORIZONTAL, length=120
            )
            spring_k_scale.grid(row=1, column=1, padx=(10, 0), pady=2)
            
            # K value label
            self.k_value_label = ttk.Label(layout_opt_frame, text="1.0", width=6)
            self.k_value_label.grid(row=1, column=2, padx=(5, 0), pady=2)
            self.spring_k_var.trace('w', self.update_k_value_label)

            # Layout iterations
            ttk.Label(layout_opt_frame, text="Iterations:").grid(row=2, column=0, sticky=W, pady=2)
            self.layout_iterations_var = tk.IntVar(value=100)
            ttk.Spinbox(
                layout_opt_frame, from_=20, to=500, textvariable=self.layout_iterations_var, 
                width=8, increment=20
            ).grid(row=2, column=1, sticky=W, padx=(10, 0), pady=2)

            # === VISUAL HIERARCHY SECTION ===
            hierarchy_frame = ttk.LabelFrame(network_frame, text="👑 Visual Hierarchy", padding=5)
            hierarchy_frame.grid(row=2, column=0, columnspan=3, sticky=EW, pady=(0, 5))

            # Node sizing strategy
            ttk.Label(hierarchy_frame, text="Node Size Strategy:").grid(row=0, column=0, sticky=W, pady=2)
            self.node_size_strategy_var = tk.StringVar(value='degree_based')
            size_strategy_combo = ttk.Combobox(
                hierarchy_frame, textvariable=self.node_size_strategy_var,
                values=['uniform', 'degree_based', 'betweenness_based', 'eigenvector_based'],
                state="readonly", width=16
            )
            size_strategy_combo.grid(row=0, column=1, padx=(10, 0), pady=2)

            # Strategy info button
            strategy_info_btn = ttk.Button(
                hierarchy_frame, text="?", width=3,
                command=self.show_node_strategy_info,
                style="info.TButton"
            )
            strategy_info_btn.grid(row=0, column=2, padx=(5, 0), pady=2)

            # Node size range
            ttk.Label(hierarchy_frame, text="Size Range:").grid(row=1, column=0, sticky=W, pady=2)
            size_range_frame = ttk.Frame(hierarchy_frame)
            size_range_frame.grid(row=1, column=1, columnspan=2, sticky=W, padx=(10, 0), pady=2)

            ttk.Label(size_range_frame, text="Min:").pack(side=LEFT)
            self.node_size_min_var = tk.IntVar(value=15)
            ttk.Spinbox(size_range_frame, from_=5, to=50, textvariable=self.node_size_min_var, width=5).pack(side=LEFT, padx=(5, 10))

            ttk.Label(size_range_frame, text="Max:").pack(side=LEFT)
            self.node_size_max_var = tk.IntVar(value=50)
            ttk.Spinbox(size_range_frame, from_=20, to=100, textvariable=self.node_size_max_var, width=5).pack(side=LEFT, padx=5)

            # Edge thickness strategy
            ttk.Label(hierarchy_frame, text="Edge Thickness:").grid(row=2, column=0, sticky=W, pady=2)
            self.edge_thickness_strategy_var = tk.StringVar(value='correlation_based')
            edge_thickness_combo = ttk.Combobox(
                hierarchy_frame, textvariable=self.edge_thickness_strategy_var,
                values=['uniform', 'correlation_based', 'weight_based'],
                state="readonly", width=16
            )
            edge_thickness_combo.grid(row=2, column=1, padx=(10, 0), pady=2)

            # === CANVAS SIZE SECTION ===
            canvas_frame = ttk.LabelFrame(network_frame, text="📐 Canvas & Display", padding=5)
            canvas_frame.grid(row=3, column=0, columnspan=3, sticky=EW, pady=(0, 5))

            # Figure size
            ttk.Label(canvas_frame, text="Figure Size:").grid(row=0, column=0, sticky=W, pady=2)
            self.figure_size_var = tk.StringVar(value='large')
            size_combo = ttk.Combobox(
                canvas_frame, textvariable=self.figure_size_var,
                values=['small', 'medium', 'large', 'xlarge', 'fullscreen', 'auto-detect', 'custom'],
                state="readonly", width=12
            )
            size_combo.grid(row=0, column=1, padx=(10, 0), pady=2)

            # Size info button
            size_info_btn = ttk.Button(
                canvas_frame, text="?", width=3,
                command=self.show_size_info,
                style="info.TButton"
            )
            size_info_btn.grid(row=0, column=2, padx=(5, 0), pady=2)

            # Custom size inputs (initially hidden)
            self.custom_size_frame = ttk.Frame(canvas_frame)
            self.custom_size_frame.grid(row=1, column=0, columnspan=3, sticky=W, pady=2)

            ttk.Label(self.custom_size_frame, text="Width:").pack(side=LEFT)
            self.custom_width_var = tk.IntVar(value=1600)
            ttk.Entry(self.custom_size_frame, textvariable=self.custom_width_var, width=6).pack(side=LEFT, padx=(5, 10))

            ttk.Label(self.custom_size_frame, text="Height:").pack(side=LEFT)
            self.custom_height_var = tk.IntVar(value=900)
            ttk.Entry(self.custom_size_frame, textvariable=self.custom_height_var, width=6).pack(side=LEFT, padx=5)

            # Initially hide custom size
            self.custom_size_frame.grid_remove()

            # Bind size selection change
            size_combo.bind('<<ComboboxSelected>>', self.on_figure_size_change)

            # Zoom and pan options
            self.enable_zoom_pan_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                canvas_frame, text="Enable zoom & pan",
                variable=self.enable_zoom_pan_var
            ).grid(row=2, column=0, columnspan=2, sticky=W, pady=2)

            # Fullscreen and layout options
            fullscreen_frame = ttk.Frame(canvas_frame)
            fullscreen_frame.grid(row=3, column=0, columnspan=3, sticky=EW, pady=2)

            self.minimize_margins_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                fullscreen_frame, text="Minimize margins",
                variable=self.minimize_margins_var
            ).pack(side=LEFT)

            self.responsive_layout_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                fullscreen_frame, text="Responsive layout",
                variable=self.responsive_layout_var
            ).pack(side=LEFT, padx=(20, 0))

            # Browser optimization
            browser_frame = ttk.Frame(canvas_frame)
            browser_frame.grid(row=4, column=0, columnspan=3, sticky=EW, pady=2)

            self.browser_fullscreen_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                browser_frame, text="📱 Optimize for browser fullscreen",
                variable=self.browser_fullscreen_var
            ).pack(side=LEFT)

            self.hide_toolbar_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                browser_frame, text="Hide plot toolbar",
                variable=self.hide_toolbar_var
            ).pack(side=LEFT, padx=(20, 0))

            # === DENSITY MANAGEMENT SECTION ===
            density_frame = ttk.LabelFrame(network_frame, text="🌐 Density Management", padding=5)
            density_frame.grid(row=4, column=0, columnspan=3, sticky=EW, pady=(0, 5))

            # Auto-grouping option (kept from original)
            self.auto_group_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                density_frame, text="🔍 Auto-group by communities (Louvain)",
                variable=self.auto_group_var
            ).grid(row=0, column=0, columnspan=2, sticky=W, pady=2)

            # Intelligent filtering
            self.intelligent_filtering_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                density_frame, text="🧠 Intelligent density reduction",
                variable=self.intelligent_filtering_var
            ).grid(row=1, column=0, columnspan=2, sticky=W, pady=2)

            # Backbone extraction
            self.backbone_extraction_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                density_frame, text="🦴 Extract network backbone",
                variable=self.backbone_extraction_var
            ).grid(row=2, column=0, columnspan=2, sticky=W, pady=2)

            # Community highlight
            self.highlight_communities_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                density_frame, text="🏘️ Highlight community structures",
                variable=self.highlight_communities_var
            ).grid(row=3, column=0, columnspan=2, sticky=W, pady=2)

            # === LEGACY FILTERING OPTIONS ===
            legacy_frame = ttk.LabelFrame(network_frame, text="🔧 Manual Filtering", padding=5)
            legacy_frame.grid(row=5, column=0, columnspan=3, sticky=EW, pady=(0, 5))

            # Outlier filtering options (kept from original)
            self.filter_outliers_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                legacy_frame, text="🧹 Filter outlier units",
                variable=self.filter_outliers_var
            ).grid(row=0, column=0, columnspan=2, sticky=W, pady=2)

            # Minimum degree
            ttk.Label(legacy_frame, text="Min connections:").grid(row=1, column=0, sticky=W, pady=2)
            self.min_degree_var = tk.IntVar(value=1)
            degree_combo = ttk.Combobox(
                legacy_frame, textvariable=self.min_degree_var,
                values=[0, 1, 2, 3, 5], state="readonly", width=5
            )
            degree_combo.grid(row=1, column=1, padx=(10, 0), pady=2)

            # Minimum weighted degree
            ttk.Label(legacy_frame, text="Min correlation sum:").grid(row=2, column=0, sticky=W, pady=2)
            self.min_weighted_degree_var = tk.DoubleVar(value=0.5)
            weighted_scale = ttk.Scale(
                legacy_frame, from_=0.0, to=2.0, variable=self.min_weighted_degree_var,
                orient=HORIZONTAL, length=100
            )
            weighted_scale.grid(row=2, column=1, padx=(10, 0), pady=2)

            # Top K connections option
            self.top_k_connections_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                legacy_frame, text="Show only top K connections",
                variable=self.top_k_connections_var
            ).grid(row=3, column=0, columnspan=2, sticky=W, pady=2)

            # K value selector
            ttk.Label(legacy_frame, text="K value:").grid(row=4, column=0, sticky=W, pady=2)
            self.k_value_var = tk.IntVar(value=5)
            ttk.Spinbox(
                legacy_frame, from_=1, to=20, textvariable=self.k_value_var, width=5
            ).grid(row=4, column=1, sticky=W, pady=2)

            # Show labels
            self.show_labels_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                legacy_frame, text="Show node labels", variable=self.show_labels_var
            ).grid(row=5, column=0, columnspan=2, sticky=W, pady=2)

            # === EDGE COLORING CONFIGURATION ===
            edge_color_frame = ttk.LabelFrame(self.viz_options_frame, text="🎨 Edge Coloring Options", padding=10)
            edge_color_frame.pack(fill=X, pady=(0, 10))

            # Enable/disable edge coloring
            self.edge_coloring_enabled_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                edge_color_frame, text="Enable edge coloring by correlation strength",
                variable=self.edge_coloring_enabled_var,
                command=self.toggle_edge_coloring_options
            ).grid(row=0, column=0, columnspan=3, sticky=W, pady=(0, 5))

            # Color scheme selection
            ttk.Label(edge_color_frame, text="Color Scheme:").grid(row=1, column=0, sticky=W, pady=2)
            self.edge_colorscale_var = tk.StringVar(value='Blues')
            
            # Import color schemes from NetworkVisualizer
            try:
                from network_visualization import NetworkVisualizer
                temp_viz = NetworkVisualizer()
                color_schemes = list(temp_viz.get_available_edge_colorschemes().keys())
            except:
                color_schemes = ['Blues', 'Reds', 'Greens', 'Viridis', 'Plasma']
            
            self.edge_colorscale_combo = ttk.Combobox(
                edge_color_frame, textvariable=self.edge_colorscale_var,
                values=color_schemes, state="readonly", width=12
            )
            self.edge_colorscale_combo.grid(row=1, column=1, padx=(10, 0), pady=2)

            # Color scheme info button
            self.color_info_btn = ttk.Button(
                edge_color_frame, text="?", width=3,
                command=self.show_color_scheme_info,
                style="info.TButton"
            )
            self.color_info_btn.grid(row=1, column=2, padx=(5, 0), pady=2)

            # Intensity range
            ttk.Label(edge_color_frame, text="Intensity Range:").grid(row=2, column=0, sticky=W, pady=2)
            intensity_frame = ttk.Frame(edge_color_frame)
            intensity_frame.grid(row=2, column=1, columnspan=2, sticky=W, padx=(10, 0), pady=2)

            ttk.Label(intensity_frame, text="Min:").pack(side=LEFT)
            self.edge_intensity_min_var = tk.DoubleVar(value=0.2)
            ttk.Scale(
                intensity_frame, from_=0.0, to=0.8, variable=self.edge_intensity_min_var,
                orient=HORIZONTAL, length=60
            ).pack(side=LEFT, padx=(5, 10))

            ttk.Label(intensity_frame, text="Max:").pack(side=LEFT)
            self.edge_intensity_max_var = tk.DoubleVar(value=1.0)
            ttk.Scale(
                intensity_frame, from_=0.2, to=1.0, variable=self.edge_intensity_max_var,
                orient=HORIZONTAL, length=60
            ).pack(side=LEFT, padx=5)

            # Opacity
            ttk.Label(edge_color_frame, text="Opacity:").grid(row=3, column=0, sticky=W, pady=2)
            self.edge_opacity_var = tk.DoubleVar(value=0.8)
            ttk.Scale(
                edge_color_frame, from_=0.1, to=1.0, variable=self.edge_opacity_var,
                orient=HORIZONTAL, length=100
            ).grid(row=3, column=1, padx=(10, 0), pady=2)

            # Show colorbar
            self.edge_colorbar_enabled_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                edge_color_frame, text="Show color legend",
                variable=self.edge_colorbar_enabled_var
            ).grid(row=4, column=0, columnspan=2, sticky=W, pady=2)

            # Colorbar title
            ttk.Label(edge_color_frame, text="Legend Title:").grid(row=5, column=0, sticky=W, pady=2)
            self.edge_colorbar_title_var = tk.StringVar(value='Correlation Strength')
            ttk.Entry(
                edge_color_frame, textvariable=self.edge_colorbar_title_var, width=18
            ).grid(row=5, column=1, columnspan=2, padx=(10, 0), pady=2, sticky=W)

            # Default edge color (when coloring disabled)
            ttk.Label(edge_color_frame, text="Default Color:").grid(row=6, column=0, sticky=W, pady=2)
            self.edge_default_color_var = tk.StringVar(value='#888888')
            default_color_frame = ttk.Frame(edge_color_frame)
            default_color_frame.grid(row=6, column=1, columnspan=2, sticky=W, padx=(10, 0), pady=2)

            ttk.Entry(
                default_color_frame, textvariable=self.edge_default_color_var, width=8
            ).pack(side=LEFT)

            # Color preview
            self.color_preview_label = ttk.Label(default_color_frame, text="███", foreground=self.edge_default_color_var.get())
            self.color_preview_label.pack(side=LEFT, padx=(5, 0))

            # Update color preview when color changes
            self.edge_default_color_var.trace('w', self.update_color_preview)

            # Initially disable/enable options based on checkbox
            self.toggle_edge_coloring_options()

            # === NODE DEGREE DISTRIBUTION ANALYSIS ===
            degree_analysis_frame = ttk.LabelFrame(self.viz_options_frame, text="📊 Node Degree Distribution Analysis", padding=10)
            degree_analysis_frame.pack(fill=X, pady=(0, 10))

            # Enable/disable degree analysis
            self.degree_analysis_enabled_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                degree_analysis_frame, text="Enable degree distribution analysis",
                variable=self.degree_analysis_enabled_var,
                command=self.toggle_degree_analysis_options
            ).grid(row=0, column=0, columnspan=3, sticky=W, pady=(0, 5))

            # Visualization type selection
            ttk.Label(degree_analysis_frame, text="Visualization Type:").grid(row=1, column=0, sticky=W, pady=2)
            self.degree_viz_type_var = tk.StringVar(value='bar')
            
            self.degree_viz_combo = ttk.Combobox(
                degree_analysis_frame, textvariable=self.degree_viz_type_var,
                values=['bar', 'line', 'scatter', 'histogram', 'loglog'], 
                state="readonly", width=12
            )
            self.degree_viz_combo.grid(row=1, column=1, padx=(10, 0), pady=2)

            # Visualization type info button
            self.degree_viz_info_btn = ttk.Button(
                degree_analysis_frame, text="?", width=3,
                command=self.show_degree_viz_info,
                style="info.TButton"
            )
            self.degree_viz_info_btn.grid(row=1, column=2, padx=(5, 0), pady=2)

            # Color scheme for degree plot
            ttk.Label(degree_analysis_frame, text="Color Scheme:").grid(row=2, column=0, sticky=W, pady=2)
            self.degree_color_scheme_var = tk.StringVar(value='steelblue')
            
            self.degree_color_combo = ttk.Combobox(
                degree_analysis_frame, textvariable=self.degree_color_scheme_var,
                values=['steelblue', 'viridis', 'plasma', 'red', 'green', 'purple', 'orange'], 
                state="readonly", width=12
            )
            self.degree_color_combo.grid(row=2, column=1, padx=(10, 0), pady=2)

            # Statistics display options
            self.show_degree_stats_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                degree_analysis_frame, text="Show network statistics",
                variable=self.show_degree_stats_var
            ).grid(row=3, column=0, columnspan=2, sticky=W, pady=2)

            self.show_degree_values_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                degree_analysis_frame, text="Show values on bars (≤20 nodes)",
                variable=self.show_degree_values_var
            ).grid(row=4, column=0, columnspan=2, sticky=W, pady=2)

            # Execute button
            self.degree_analysis_btn = ttk.Button(
                degree_analysis_frame,
                text="📊 Show Degree Distribution",
                style='success.TButton',
                command=self.execute_degree_analysis,
                state=DISABLED
            )
            self.degree_analysis_btn.grid(row=5, column=0, columnspan=3, pady=(10, 0), sticky='ew')

            # Initially disable/enable options based on checkbox
            self.toggle_degree_analysis_options()

        # Update scroll region after adding content
        self.viz_container_frame.after(50, lambda: self.viz_canvas.configure(scrollregion=self.viz_canvas.bbox("all")))
        
        # Update degree analysis button state when visualization type changes
        if hasattr(self, 'update_degree_analysis_button_state'):
            self.viz_container_frame.after(100, self.update_degree_analysis_button_state)

    def show_histogram_preview(self):
        """Show histogram preview of similarity values."""
        if not hasattr(self, 'file_entry') or not self.file_entry.get().strip():
            tk.messagebox.showwarning("Warning", "Please select a data file first")
            return

        try:
            # Load data and calculate similarity matrix
            file_path = self.file_entry.get().strip()
            df = dl.load_correlation_data(file_path)
            if df is None:
                tk.messagebox.showerror("Error", "Could not load data file")
                return

            # Calculate similarity matrix
            from correlation_analysis import CorrelationAnalyzer
            analyzer = CorrelationAnalyzer()
            similarity_matrix = analyzer.calculate_similarity_matrix(
                df,
                method=self.correlation_method.get(),
                time_aggregated=self.time_aggregated.get()
            )

            # Create histogram
            from heatmap_visualization import create_similarity_histogram
            fig = create_similarity_histogram(similarity_matrix)

            # Show in a new window
            import matplotlib.pyplot as plt
            plt.figure(fig.number)  # Make sure we're using the right figure
            plt.show()

        except Exception as e:
            tk.messagebox.showerror("Error", f"Could not create histogram preview: {str(e)}")

    def get_config(self):
        """Get configuration for correlation analysis."""
        # 1. Validar que los datos estén cargados (¡Crítico!)
        if not hasattr(self, 'file_entry') or not self.file_entry.get().strip():
            messagebox.showerror("Error de Configuración", "Por favor, cargue un archivo de datos primero.")
            return None  # Devolver None para cancelar el análisis

        # 2. Validar que se hayan seleccionado indicadores
        if not self.selected_indicators:
            messagebox.showerror("Error de Configuración",
                                 "Debe seleccionar al menos un indicador para el análisis de correlación.")
            return None  # Devolver None para cancelar el análisis

        if hasattr(self, 'file_entry') and self.file_entry.get().strip():
            # Get filtered units based on group selection
            filtered_units = self.get_filtered_units_for_analysis()
            
            config = {
                'data_file': self.file_entry.get().strip(),
                'selected_indicators': self.selected_indicators,
                'selected_units': filtered_units,  # Use filtered units instead of all selected units
                'original_selected_units': self.selected_units,  # Keep original selection for reference
                'groups': self.groups,
                'group_colors': self.group_colors,
                'correlation_method': self.correlation_method.get(),
                'time_aggregated': self.time_aggregated.get(),
                'similarity_threshold': self.similarity_threshold.get(),
                'visualization_type': self.visualization_type.get(),
                'analysis_mode': getattr(self, 'analysis_mode', tk.StringVar(value="all_units")).get() if hasattr(self, 'analysis_mode') else "all_units"
            }
            
            # Add group filtering information
            if hasattr(self, 'analysis_mode') and self.analysis_mode.get() != "all_units":
                if hasattr(self, 'group_selection_vars'):
                    if self.analysis_mode.get() == "selected_groups":
                        config['selected_groups'] = [name for name, var in self.group_selection_vars.items() if var.get()]
                    elif self.analysis_mode.get() == "exclude_groups":
                        config['excluded_groups'] = [name for name, var in self.group_selection_vars.items() if var.get()]

            # Add visualization-specific config
            if self.visualization_type.get() in ['heatmap', 'both']:
                config['heatmap_config'] = {
                    'cmap': self.heatmap_cmap.get(),
                    'filter_outliers': self.heatmap_filter_outliers_var.get(),
                    'min_degree': self.heatmap_min_degree_var.get(),
                    'min_weighted_degree': self.heatmap_min_weighted_degree_var.get(),
                    'interactive': self.interactive_heatmap_var.get(),
                    'force_horizontal': self.force_horizontal_labels_var.get()
                }

            if self.visualization_type.get() in ['network', 'both']:
                # Base network config
                base_network_config = {
                    'layout': self.network_layout.get(),
                    'node_size': getattr(self, 'node_size', tk.IntVar(value=20)).get(),
                    'auto_group': self.auto_group_var.get(),
                    'filter_outliers': self.filter_outliers_var.get(),
                    'min_degree': self.min_degree_var.get(),
                    'min_weighted_degree': self.min_weighted_degree_var.get(),
                    'top_k_connections': self.top_k_connections_var.get(),
                    'k_value': self.k_value_var.get(),
                    'groups': self.groups,
                    'group_colors': self.group_colors,
                    # Edge coloring configuration
                    'edge_coloring_enabled': self.edge_coloring_enabled_var.get(),
                    'edge_colorscale': self.edge_colorscale_var.get(),
                    'edge_color_intensity_range': (self.edge_intensity_min_var.get(), self.edge_intensity_max_var.get()),
                    'edge_opacity': self.edge_opacity_var.get(),
                    'edge_colorbar_enabled': self.edge_colorbar_enabled_var.get(),
                    'edge_colorbar_title': self.edge_colorbar_title_var.get(),
                    'edge_default_color': self.edge_default_color_var.get()
                }
                
                # Add advanced network layout config
                advanced_config = self.get_network_layout_config()
                base_network_config.update(advanced_config)
                
                config['network_config'] = base_network_config

            return config
        return None

    def toggle_edge_coloring_options(self):
        """Enable/disable edge coloring options based on checkbox state."""
        state = NORMAL if self.edge_coloring_enabled_var.get() else DISABLED
        
        # Get all widgets in the edge coloring frame that should be toggled
        widgets_to_toggle = [
            self.edge_colorscale_combo,
            self.color_info_btn,
            self.edge_colorbar_enabled_var,
            self.edge_colorbar_title_var
        ]
        
        for widget in widgets_to_toggle:
            try:
                if hasattr(widget, 'config'):
                    widget.config(state=state)
                elif hasattr(widget, 'set'):  # For StringVar, BooleanVar, etc.
                    # These don't have state, but we can track if they should be enabled
                    pass
            except:
                pass

    def update_color_preview(self, *args):
        """Update color preview when default color changes."""
        try:
            color = self.edge_default_color_var.get()
            if hasattr(self, 'color_preview_label'):
                self.color_preview_label.config(foreground=color)
        except:
            pass

    def show_color_scheme_info(self):
        """Show information about available color schemes."""
        try:
            from network_visualization import NetworkVisualizer
            visualizer = NetworkVisualizer()
            schemes = visualizer.get_available_edge_colorschemes()
            
            info_text = "Available Color Schemes:\n\n"
            for name, description in schemes.items():
                info_text += f"• {name}: {description}\n"
            
            info_text += "\nRecommendations:\n"
            info_text += "• Blues: Financial correlation analysis\n"
            info_text += "• Reds: Risk or negative correlation emphasis\n"
            info_text += "• Greens: Growth or positive correlation emphasis\n"
            info_text += "• Viridis: Academic/scientific presentations\n"
            info_text += "• RdBu: Diverging correlations (positive/negative)"
            
            # Create info window
            info_window = tk.Toplevel(self)
            info_window.title("Color Scheme Information")
            info_window.geometry("500x400")
            info_window.resizable(True, True)
            
            # Make it modal
            info_window.transient(self)
            info_window.grab_set()
            
            # Add text widget with scrollbar
            text_frame = ttk.Frame(info_window)
            text_frame.pack(fill=BOTH, expand=YES, padx=10, pady=10)
            
            text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Helvetica", 10))
            scrollbar = ttk.Scrollbar(text_frame, orient=VERTICAL, command=text_widget.yview)
            text_widget.config(yscrollcommand=scrollbar.set)
            
            text_widget.pack(side=LEFT, fill=BOTH, expand=YES)
            scrollbar.pack(side=RIGHT, fill=Y)
            
            text_widget.insert(tk.END, info_text)
            text_widget.config(state=DISABLED)
            
            # Center the window
            info_window.update_idletasks()
            x = (info_window.winfo_screenwidth() // 2) - (info_window.winfo_width() // 2)
            y = (info_window.winfo_screenheight() // 2) - (info_window.winfo_height() // 2)
            info_window.geometry(f"+{x}+{y}")
            
        except Exception as e:
            tk.messagebox.showinfo("Color Schemes", 
                "Available schemes: Blues (financial), Reds (risk), Greens (growth), "
                "Viridis (scientific), RdBu (diverging), and more.")

    def get_filtered_units_for_analysis(self):
        """Get the list of units to include in analysis based on current filter mode."""
        if not hasattr(self, 'selected_units') or not self.selected_units:
            return []
        
        # Sync groups from UniversalGroupManager before processing
        self.sync_groups_format()
        
        mode = getattr(self, 'analysis_mode', None)
        if not mode:
            return self.selected_units
        
        current_mode = mode.get()
        
        if current_mode == "all_units":
            return self.selected_units
        
        elif current_mode == "selected_groups":
            if not hasattr(self, 'group_selection_vars'):
                return self.selected_units
            
            # Get units from selected groups
            selected_groups = [name for name, var in self.group_selection_vars.items() if var.get()]
            filtered_units = []
            
            for group_name in selected_groups:
                units_in_group = [u for u, g in self.groups.items() if g == group_name and u in self.selected_units]
                filtered_units.extend(units_in_group)
            
            return list(set(filtered_units))  # Remove duplicates
        
        elif current_mode == "exclude_groups":
            if not hasattr(self, 'group_selection_vars'):
                return self.selected_units
            
            # Get units from excluded groups
            excluded_groups = [name for name, var in self.group_selection_vars.items() if var.get()]
            excluded_units = []
            
            for group_name in excluded_groups:
                units_in_group = [u for u, g in self.groups.items() if g == group_name]
                excluded_units.extend(units_in_group)
            
            # Return units not in excluded groups
            return [u for u in self.selected_units if u not in excluded_units]
        
        return self.selected_units
    def _toggle_select_all(self, select_all_var, item_vars, available_items):
        """Toggle selection of all items."""
        select_all = select_all_var.get()
        for item in available_items:
            if item in item_vars:
                item_vars[item].set(select_all)

    def _update_select_all_state(self, select_all_var, item_vars, available_items):
        """Update the select all checkbox state based on individual selections."""
        all_selected = all(item_vars.get(item, tk.BooleanVar()).get() for item in available_items)
        select_all_var.set(all_selected)

    def _toggle_select_all_countries(self, select_all_var, country_vars, available_countries):
        """Toggle selection of all countries/units."""
        select_all = select_all_var.get()
        for country in available_countries:
            if country in country_vars:
                country_vars[country].set(select_all)

    def _update_select_all_countries_state(self, select_all_var, country_vars, available_countries):
        """Update the select all countries checkbox state based on individual selections."""
        all_selected = all(country_vars.get(country, tk.BooleanVar()).get() for country in available_countries)
        select_all_var.set(all_selected)

    def toggle_degree_analysis_options(self):
        """Enable/disable degree analysis options based on checkbox state."""
        try:
            enabled = self.degree_analysis_enabled_var.get()
            state = 'normal' if enabled else 'disabled'
            
            # List of widgets to toggle
            widgets_to_toggle = [
                'degree_viz_combo', 'degree_viz_info_btn', 'degree_color_combo',
                'degree_analysis_btn'
            ]
            
            for widget_name in widgets_to_toggle:
                if hasattr(self, widget_name):
                    widget = getattr(self, widget_name)
                    try:
                        widget.config(state=state)
                    except:
                        pass
            
            # Also update button state based on whether graph is available
            self.update_degree_analysis_button_state()
            
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.warning(f"Error toggling degree analysis options: {e}")

    def update_degree_analysis_button_state(self):
        """Update degree analysis button state based on network availability."""
        try:
            if hasattr(self, 'degree_analysis_btn'):
                # Check if degree analysis is enabled
                degree_enabled = getattr(self, 'degree_analysis_enabled_var', tk.BooleanVar(value=True)).get()
                
                # Check if visualization type includes network
                viz_type = getattr(self, 'visualization_type', tk.StringVar(value='')).get()
                network_viz = viz_type in ['network', 'both']
                
                # Check if graph is available
                graph_available = (
                    hasattr(self.app, 'current_network_graph') and
                    self.app.current_network_graph is not None
                )
                
                # Final decision
                enabled = degree_enabled and network_viz and graph_available
                state = 'normal' if enabled else 'disabled'
                
                # Update button
                self.degree_analysis_btn.config(state=state)
                
                # Debug logging
                if hasattr(self.app, 'logger'):
                    self.app.logger.info(f"Degree analysis button update: enabled={enabled} "
                                       f"(degree_enabled={degree_enabled}, network_viz={network_viz}, "
                                       f"graph_available={graph_available}, viz_type='{viz_type}')")
                    
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.warning(f"Error updating degree analysis button state: {e}")
            # Fallback: disable button on error
            if hasattr(self, 'degree_analysis_btn'):
                try:
                    self.degree_analysis_btn.config(state='disabled')
                except:
                    pass

    def show_degree_viz_info(self):
        """Show information about degree visualization types."""
        info_text = """Degree Distribution Visualization Types:

• Bar Chart: Traditional bar chart showing frequency of each degree
  Best for: General overview, small to medium networks

• Line Plot: Connected line graph showing degree distribution
  Best for: Trend analysis, large networks, publication figures

• Scatter Plot: Points showing degree vs frequency
  Best for: Identifying outliers, sparse distributions

• Histogram: Binned frequency distribution
  Best for: Large networks, statistical analysis

• Log-Log Plot: Both axes in logarithmic scale
  Best for: Scale-free networks, power-law distributions

Recommendations:
- Use Bar charts for most general analyses
- Use Log-Log for investigating scale-free properties
- Use Line plots for cleaner publication figures
- Use Histograms for very large networks (>100 nodes)"""

        # Create info window
        info_window = tk.Toplevel(self)
        info_window.title("Degree Visualization Types")
        info_window.geometry("550x450")
        info_window.resizable(True, True)
        
        # Make it modal
        info_window.transient(self)
        info_window.grab_set()
        
        # Add text widget with scrollbar
        text_frame = ttk.Frame(info_window)
        text_frame.pack(fill=BOTH, expand=YES, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Helvetica", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=VERTICAL, command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        text_widget.insert('1.0', info_text)
        text_widget.config(state='disabled')
        
        text_widget.pack(side=LEFT, fill=BOTH, expand=YES)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # Close button
        ttk.Button(
            info_window, text="Close", command=info_window.destroy,
            style="secondary.TButton"
        ).pack(pady=10)

    def execute_degree_analysis(self):
        """Execute degree distribution analysis with selected options."""
        try:
            # Check if graph is available
            if not hasattr(self.app, 'current_network_graph') or self.app.current_network_graph is None:
                messagebox.showwarning(
                    "No Network Available", 
                    "No se ha generado ningún grafo de red.\n"
                    "Por favor, ejecute primero un análisis de correlación con visualización de red."
                )
                return
            
            # Get visualization options
            viz_type = self.degree_viz_type_var.get()
            color_scheme = self.degree_color_scheme_var.get()
            show_stats = self.show_degree_stats_var.get()
            show_values = self.show_degree_values_var.get()
            
            # Call the improved degree distribution function
            if hasattr(self.app, 'plot_degree_distribution_advanced'):
                self.app.plot_degree_distribution_advanced(
                    graph=self.app.current_network_graph,
                    viz_type=viz_type,
                    color_scheme=color_scheme,
                    show_stats=show_stats,
                    show_values=show_values
                )
            else:
                # Fallback to basic version
                self.app.plot_degree_distribution(self.app.current_network_graph)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al ejecutar análisis de distribución de grado:\n{str(e)}")
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Error in degree analysis execution: {e}")

    def update_k_value_label(self, *args):
        """Update the k value label when scale changes."""
        try:
            if hasattr(self, 'k_value_label'):
                k_val = self.spring_k_var.get()
                self.k_value_label.config(text=f"{k_val:.1f}")
        except:
            pass

    def on_figure_size_change(self, event=None):
        """Handle figure size selection change."""
        try:
            if hasattr(self, 'figure_size_var') and hasattr(self, 'custom_size_frame'):
                if self.figure_size_var.get() == 'custom':
                    self.custom_size_frame.grid()
                else:
                    self.custom_size_frame.grid_remove()
        except:
            pass

    def get_figure_dimensions(self):
        """Get figure dimensions based on selection."""
        try:
            size_selection = self.figure_size_var.get()
            
            # Base size mappings
            size_map = {
                'small': (800, 600),
                'medium': (1200, 800),
                'large': (1600, 1000),
                'xlarge': (2000, 1200),
                'fullscreen': (2560, 1440),  # Full HD+ resolution
                'auto-detect': self._detect_screen_size(),
                'custom': (self.custom_width_var.get(), self.custom_height_var.get())
            }
            
            base_dimensions = size_map.get(size_selection, (1600, 1000))
            
            # Apply fullscreen optimizations if browser fullscreen is enabled
            if getattr(self, 'browser_fullscreen_var', tk.BooleanVar(value=False)).get():
                if size_selection in ['fullscreen', 'auto-detect']:
                    # Use even larger dimensions for true fullscreen
                    return (3840, 2160)  # 4K resolution for maximum coverage
                elif size_selection in ['large', 'xlarge']:
                    # Enhance standard sizes for better browser coverage
                    width, height = base_dimensions
                    return (int(width * 1.2), int(height * 1.2))
                    
            return base_dimensions
            
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.warning(f"Error getting figure dimensions: {e}")
            return (1600, 1000)

    def _detect_screen_size(self):
        """Detect screen size and return appropriate dimensions."""
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()  # Hide the window
            
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            
            # Use 90% of screen size for optimal viewing
            opt_width = int(screen_width * 0.9)
            opt_height = int(screen_height * 0.9)
            
            # Ensure minimum reasonable size
            opt_width = max(opt_width, 1200)
            opt_height = max(opt_height, 800)
            
            if hasattr(self.app, 'logger'):
                self.app.logger.info(f"Auto-detected screen size: {screen_width}x{screen_height}, using {opt_width}x{opt_height}")
                
            return (opt_width, opt_height)
            
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.warning(f"Could not auto-detect screen size: {e}")
            return (2000, 1200)  # Fallback to xlarge

    def get_network_layout_config(self):
        """Get optimized layout configuration."""
        try:
            layout_type = self.network_layout.get()
            
            config = {
                'layout_algorithm': layout_type,
                'spring_k': getattr(self, 'spring_k_var', tk.DoubleVar(value=1.0)).get(),
                'iterations': getattr(self, 'layout_iterations_var', tk.IntVar(value=100)).get(),
                'node_size_strategy': getattr(self, 'node_size_strategy_var', tk.StringVar(value='degree_based')).get(),
                'node_size_range': (
                    getattr(self, 'node_size_min_var', tk.IntVar(value=15)).get(),
                    getattr(self, 'node_size_max_var', tk.IntVar(value=50)).get()
                ),
                'edge_thickness_strategy': getattr(self, 'edge_thickness_strategy_var', tk.StringVar(value='correlation_based')).get(),
                'figure_dimensions': self.get_figure_dimensions(),
                'enable_zoom_pan': getattr(self, 'enable_zoom_pan_var', tk.BooleanVar(value=True)).get(),
                'intelligent_filtering': getattr(self, 'intelligent_filtering_var', tk.BooleanVar(value=True)).get(),
                'backbone_extraction': getattr(self, 'backbone_extraction_var', tk.BooleanVar(value=False)).get(),
                'highlight_communities': getattr(self, 'highlight_communities_var', tk.BooleanVar(value=True)).get(),
                'show_labels': getattr(self, 'show_labels_var', tk.BooleanVar(value=True)).get(),
                # New fullscreen options
                'minimize_margins': getattr(self, 'minimize_margins_var', tk.BooleanVar(value=True)).get(),
                'responsive_layout': getattr(self, 'responsive_layout_var', tk.BooleanVar(value=True)).get(),
                'browser_fullscreen': getattr(self, 'browser_fullscreen_var', tk.BooleanVar(value=True)).get(),
                'hide_toolbar': getattr(self, 'hide_toolbar_var', tk.BooleanVar(value=False)).get(),
                'figure_size_mode': getattr(self, 'figure_size_var', tk.StringVar(value='large')).get()
            }
            
            return config
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.warning(f"Error getting network layout config: {e}")
            return {}

    def show_layout_info(self):
        """Show information about layout algorithms."""
        info_text = """Layout Algorithms for Network Visualization:

🎨 spring_optimized: Enhanced spring layout with better node separation
   • Best for: Most networks, good balance of aesthetics and performance
   • Features: Improved repulsion, reduced overlapping

🔬 kamada_kawai: Force-directed layout with energy minimization
   • Best for: Medium networks (20-100 nodes), publication figures
   • Features: Very clean results, mathematically optimal positioning

⚙️ spring_default: Standard spring layout
   • Best for: Quick visualization, small networks
   • Features: Fast computation, basic force simulation

🔄 circular: Arranges nodes in a circle
   • Best for: Highlighting connectivity patterns, small networks
   • Features: Clear structure, easy to interpret

🌀 fruchterman_reingold: Advanced force-directed algorithm
   • Best for: Large networks, complex structures
   • Features: Better handling of dense networks

Recommendations:
• For dense networks ("hairball"): kamada_kawai + high k value
• For quick exploration: spring_optimized
• For presentations: kamada_kawai or circular"""

        self._show_info_dialog("Layout Algorithms", info_text, "600x500")

    def show_node_strategy_info(self):
        """Show information about node sizing strategies."""
        info_text = """Node Sizing Strategies:

📏 uniform: All nodes same size
   • Use for: Focusing on topology, not hierarchy
   • Best when: All nodes equally important

📊 degree_based: Size by number of connections
   • Use for: Highlighting network hubs
   • Best when: Connectivity matters most
   • Formula: Size ∝ node degree

🌉 betweenness_based: Size by bridge importance
   • Use for: Finding key connectors/brokers
   • Best when: Information flow matters
   • Formula: Size ∝ betweenness centrality

🏆 eigenvector_based: Size by influence/prestige
   • Use for: Finding influential nodes
   • Best when: Quality of connections matters
   • Formula: Size ∝ eigenvector centrality

Recommendations:
• Start with degree_based for most analyses
• Use betweenness for flow/bottleneck analysis
• Use eigenvector for influence/prestige studies
• Use uniform to focus purely on structure"""

        self._show_info_dialog("Node Sizing Strategies", info_text, "550x450")

    def show_size_info(self):
        """Show information about figure sizes."""
        info_text = """Figure Size Options:

📱 small (800×600): Quick preview, presentations
📺 medium (1200×800): Standard analysis, reports  
🖥️ large (1600×1000): Detailed analysis, publications
🖼️ xlarge (2000×1200): High-detail, large displays
🌐 fullscreen (2560×1440): Maximum browser coverage
🔍 auto-detect: Adapts to your screen size
⚙️ custom: Your own dimensions

🚀 Browser Fullscreen Options:
• Minimize margins: Removes whitespace around plot
• Responsive layout: Adapts to window resizing
• Optimize for browser fullscreen: Enhanced dimensions
• Hide plot toolbar: Cleaner presentation

📊 Tips for Maximum Screen Usage:
• Use 'fullscreen' + browser optimization for dense networks
• 'auto-detect' automatically uses 90% of your screen
• Press F11 in browser for true fullscreen experience
• Combine with 'minimize margins' for edge-to-edge display

⚡ Performance Notes:
• Larger sizes = slower rendering but better detail
• Browser fullscreen works best with modern browsers
• Interactive features optimal with fullscreen+"""

        self._show_info_dialog("Figure Size Guide", info_text, "550x450")

    def _show_info_dialog(self, title, content, geometry):
        """Helper method to show information dialogs."""
        try:
            info_window = tk.Toplevel(self)
            info_window.title(title)
            info_window.geometry(geometry)
            info_window.resizable(True, True)
            
            # Make it modal
            info_window.transient(self)
            info_window.grab_set()
            
            # Add text widget with scrollbar
            text_frame = ttk.Frame(info_window)
            text_frame.pack(fill=BOTH, expand=YES, padx=10, pady=10)
            
            text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Helvetica", 10))
            scrollbar = ttk.Scrollbar(text_frame, orient=VERTICAL, command=text_widget.yview)
            text_widget.config(yscrollcommand=scrollbar.set)
            
            text_widget.insert('1.0', content)
            text_widget.config(state='disabled')
            
            text_widget.pack(side=LEFT, fill=BOTH, expand=YES)
            scrollbar.pack(side=RIGHT, fill=Y)
            
            # Close button
            ttk.Button(
                info_window, text="Close", command=info_window.destroy,
                style="secondary.TButton"
            ).pack(pady=10)
            
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.warning(f"Error showing info dialog: {e}")
            messagebox.showinfo(title, content)
