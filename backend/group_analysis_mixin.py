"""
Group Analysis Mixin for PCA Frames

Provides universal group management functionality for all analysis frames.
"""

import os
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox


class GroupAnalysisMixin:
    """Mixin class to add group-based analysis filtering to any analysis frame."""
    
    def setup_group_integration(self, parent_frame):
        """Setup group integration for the analysis frame."""
        # Initialize group-related variables
        self.groups = {}
        self.group_colors = {}
        self.analysis_mode = tk.StringVar(value="all_units")
        self.group_selection_vars = {}
        
        # Create group management section
        self.create_group_management_section(parent_frame)
    
    def create_group_management_section(self, parent):
        """Create the group management UI section."""
        print(f"🔧 [DEBUG] create_group_management_section called with parent: {parent}")
        print(f"🔧 [DEBUG] parent type: {type(parent)}")
        
        # Groups configuration card
        groups_card = ttk.LabelFrame(parent, text="🏷️ Groups & Analysis Filtering", padding=15)
        groups_card.pack(fill=X, pady=(10, 0))
        print(f"✅ [DEBUG] Groups card created and packed")
        
        # Group management buttons
        management_frame = ttk.Frame(groups_card)
        management_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Button(
            management_frame,
            text="🏷️ Manage Universal Groups",
            style='primary.TButton',
            command=self.open_universal_groups
        ).pack(side=LEFT, padx=(0, 10))
        
        ttk.Button(
            management_frame,
            text="🔄 Load Current Groups",
            style='outline.TButton',
            command=self.load_current_groups
        ).pack(side=LEFT, padx=(0, 20))
        
        print(f"✅ [DEBUG] Management buttons created")
        
        # Groups status
        self.groups_status = ttk.Label(
            management_frame,
            text="No groups loaded",
            style='secondary.TLabel'
        )
        self.groups_status.pack(side=LEFT)
        
        # Group details display
        self.group_details_frame = ttk.Frame(groups_card)
        self.group_details_frame.pack(fill=X, pady=(10, 0))
        
        # Analysis filtering controls
        self.group_filtering_frame = ttk.Frame(groups_card)
        self.group_filtering_frame.pack(fill=X, pady=(10, 0))
        
        # Initially hidden until groups are loaded
        self.group_filtering_frame.pack_forget()
        
        print(f"✅ [DEBUG] Group management section setup complete")
        
        # ✅ Force scroll region update after creating group section - MULTIPLE TIMES
        if hasattr(parent, 'update_scroll_region'):
            print(f"🔄 [DEBUG] Calling parent.update_scroll_region()")
            parent.update_scroll_region()  # Immediate call
            parent.after(50, parent.update_scroll_region)
            parent.after(150, parent.update_scroll_region)
            parent.after(300, parent.update_scroll_region)
        elif hasattr(self, 'update_scroll_region'):
            print(f"🔄 [DEBUG] Calling self.update_scroll_region()")
            self.update_scroll_region()  # Immediate call
            self.after(50, self.update_scroll_region)
            self.after(150, self.update_scroll_region)
            self.after(300, self.update_scroll_region)
        else:
            print(f"⚠️ [DEBUG] No update_scroll_region method found!")
    
    def open_universal_groups(self):
        """Open the universal group manager."""
        try:
            from group_manager import show_group_manager_gui
            
            # Get available units from current selection
            available_units = []
            if hasattr(self, 'selected_units') and self.selected_units:
                available_units = self.selected_units.copy()
            elif hasattr(self, 'selected_countries') and self.selected_countries:
                available_units = self.selected_countries.copy()
            elif hasattr(self, 'file_entry') and self.file_entry.get().strip():
                # Try to load units from file
                available_units = self.extract_units_from_file()
            
            # Show group manager
            show_group_manager_gui(self.app, available_units)
            
            # Refresh groups after closing manager
            self.app.after(1000, self.load_current_groups)
            
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Error opening group manager: {e}")
            messagebox.showerror("Error", f"Error opening group manager: {e}")
    
    def extract_units_from_file(self):
        """Extract available units from the loaded file."""
        available_units = []
        try:
            file_path = self.file_entry.get().strip()
            if file_path and os.path.exists(file_path):
                import pandas as pd
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    if 'Unit' in df.columns:
                        available_units = list(df['Unit'].unique())
                    elif 'Empresa' in df.columns:
                        available_units = list(df['Empresa'].unique())
                    elif 'Country' in df.columns:
                        available_units = list(df['Country'].unique())
                    else:
                        available_units = list(df.index)
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.warning(f"Could not load units for group manager: {e}")
        
        return available_units
    
    def load_current_groups(self):
        """Load groups from universal group manager."""
        try:
            # Get current units
            current_units = []
            if hasattr(self, 'selected_units') and self.selected_units:
                current_units = self.selected_units
            elif hasattr(self, 'selected_countries') and self.selected_countries:
                current_units = self.selected_countries
            else:
                self.groups_status.config(text="Select units first to load groups")
                return
            
            # Get universal group manager
            if hasattr(self.app, 'group_manager'):
                group_manager = self.app.group_manager
            else:
                from group_manager import get_universal_group_manager
                group_manager = get_universal_group_manager()
            
            # Get groups for current units
            unit_groups = group_manager.get_groups_for_units(current_units)
            
            # Update internal groups and colors
            self.groups = {}
            self.group_colors = {}
            
            groups_found = 0
            units_grouped = 0
            
            for group_name, units_in_group in unit_groups.items():
                if group_name != 'Ungrouped' and units_in_group:
                    groups_found += 1
                    units_grouped += len(units_in_group)
                    
                    # Get group data from manager
                    group_data = group_manager.get_group(group_name)
                    if group_data:
                        color = group_data.get('color', '#FF6B6B')
                        self.group_colors[group_name] = color
                        
                        # Map units to group
                        for unit in units_in_group:
                            self.groups[unit] = group_name
                        
                        # Increment usage counter
                        group_manager.increment_usage(group_name)
            
            # Update display
            if groups_found > 0:
                self.groups_status.config(
                    text=f"{groups_found} groups loaded ({units_grouped}/{len(current_units)} units grouped)"
                )
                if hasattr(self.app, 'logger'):
                    self.app.logger.info(f"Loaded {groups_found} groups for analysis")
                
                # Show group details and filtering controls
                self.update_groups_display()
                self.setup_group_filtering_controls()
                
                # ✅ Update scroll region after adding new widgets - MULTIPLE TIMES WITH LONGER DELAYS
                print(f"📦 [DEBUG] Groups loaded, forcing scroll updates...")
                if hasattr(self, 'update_scroll_region'):
                    self.update_scroll_region()  # Immediate
                    self.after(100, self.update_scroll_region)
                    self.after(300, self.update_scroll_region)
                    self.after(600, self.update_scroll_region)
                    self.after(1000, self.update_scroll_region)
                
            else:
                self.groups_status.config(text="No matching groups found")
            
            # Update button state
            if hasattr(self, '_update_button_state'):
                self._update_button_state()
            
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Error loading groups: {e}")
            messagebox.showerror("Error", f"Error loading groups: {e}")
    
    def update_groups_display(self):
        """Update the visual display of loaded groups."""
        # Clear existing widgets
        for widget in self.group_details_frame.winfo_children():
            widget.destroy()
        
        if not self.groups or not self.group_colors:
            return
        
        # Group summary frame
        summary_frame = ttk.Frame(self.group_details_frame)
        summary_frame.pack(fill=X, pady=(5, 0))
        
        # Display each group with its color
        displayed_groups = set()
        row = 0
        col = 0
        max_cols = 4
        
        for unit, group_name in self.groups.items():
            if group_name not in displayed_groups:
                displayed_groups.add(group_name)
                
                # Create group indicator
                group_frame = ttk.Frame(summary_frame)
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
        
        # ✅ Force scroll region update after displaying groups - MULTIPLE TIMES WITH LONGER DELAYS
        print(f"📊 [DEBUG] Group display updated, forcing scroll updates...")
        if hasattr(self, 'update_scroll_region'):
            self.update_scroll_region()  # Immediate
            self.after(100, self.update_scroll_region)
            self.after(300, self.update_scroll_region)
            self.after(600, self.update_scroll_region)
            self.after(1000, self.update_scroll_region)
    
    def setup_group_filtering_controls(self):
        """Setup group-based analysis filtering controls."""
        # Clear existing widgets
        for widget in self.group_filtering_frame.winfo_children():
            widget.destroy()
        
        if not self.groups:
            self.group_filtering_frame.pack_forget()
            return
        
        # Show the filtering frame
        self.group_filtering_frame.pack(fill=X, pady=(10, 0))
        
        # Filter mode selection
        filter_title = ttk.Label(
            self.group_filtering_frame,
            text="🎯 Analysis Scope:",
            font=("Helvetica", 10, "bold")
        )
        filter_title.pack(anchor=W, pady=(0, 5))
        
        # Radio buttons frame
        radio_frame = ttk.Frame(self.group_filtering_frame)
        radio_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Radiobutton(
            radio_frame,
            text="All Units",
            variable=self.analysis_mode,
            value="all_units",
            command=self.on_analysis_mode_change
        ).pack(side=LEFT, padx=(0, 20))
        
        ttk.Radiobutton(
            radio_frame,
            text="Selected Groups Only",
            variable=self.analysis_mode,
            value="selected_groups",
            command=self.on_analysis_mode_change
        ).pack(side=LEFT, padx=(0, 20))
        
        ttk.Radiobutton(
            radio_frame,
            text="Exclude Groups",
            variable=self.analysis_mode,
            value="exclude_groups",
            command=self.on_analysis_mode_change
        ).pack(side=LEFT)
        
        # Group selection checkboxes (initially hidden)
        self.group_checkboxes_frame = ttk.Frame(self.group_filtering_frame)
        self.group_checkboxes_frame.pack(fill=X, pady=(10, 0))
        
        self.setup_group_checkboxes()
        self.group_checkboxes_frame.pack_forget()  # Initially hidden
        
        # ✅ Force scroll region update after setup
        if hasattr(self, 'update_scroll_region'):
            self.after(100, self.update_scroll_region)
    
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
        
        # Control buttons
        controls_frame = ttk.Frame(self.group_checkboxes_frame)
        controls_frame.pack(fill=X, pady=(0, 5))
        
        ttk.Button(
            controls_frame,
            text="Select All",
            command=self.select_all_groups,
            style="info.Outline.TButton"
        ).pack(side=LEFT, padx=(0, 5))
        
        ttk.Button(
            controls_frame,
            text="Clear All",
            command=self.clear_all_groups_selection,
            style="secondary.Outline.TButton"
        ).pack(side=LEFT)
        
        # Status label
        self.group_filter_status = ttk.Label(
            controls_frame,
            text="",
            style='info.TLabel'
        )
        self.group_filter_status.pack(side=RIGHT)
        
        # Checkboxes grid
        checkboxes_frame = ttk.Frame(self.group_checkboxes_frame)
        checkboxes_frame.pack(fill=X, pady=(5, 0))
        
        # Create checkboxes for each group
        groups_list = sorted(unique_groups)
        max_cols = 3
        self.group_selection_vars = {}
        
        for i, group_name in enumerate(groups_list):
            row = i // max_cols
            col = i % max_cols
            
            # Create frame for this group
            group_frame = ttk.Frame(checkboxes_frame)
            group_frame.grid(row=row, column=col, sticky='w', padx=(0, 20), pady=2)
            
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
            
            # Checkbox
            var = tk.BooleanVar(value=True)
            self.group_selection_vars[group_name] = var
            
            units_in_group = [u for u, g in self.groups.items() if g == group_name]
            checkbox_text = f"{group_name} ({len(units_in_group)})"
            
            ttk.Checkbutton(
                group_frame,
                text=checkbox_text,
                variable=var,
                command=self.update_group_filter_status
            ).pack(side=LEFT)
    
    def on_analysis_mode_change(self):
        """Handle analysis mode change."""
        mode = self.analysis_mode.get()
        
        if mode == "all_units":
            self.group_checkboxes_frame.pack_forget()
        else:
            self.group_checkboxes_frame.pack(fill=X, pady=(10, 0))
        
        self.update_group_filter_status()
        
        # ✅ Update scroll region when showing/hiding checkboxes - MULTIPLE TIMES WITH LONGER DELAYS
        print(f"🔄 [DEBUG] Analysis mode changed, forcing scroll updates...")
        if hasattr(self, 'update_scroll_region'):
            self.update_scroll_region()  # Immediate
            self.after(100, self.update_scroll_region)
            self.after(300, self.update_scroll_region)
            self.after(600, self.update_scroll_region)
            self.after(1000, self.update_scroll_region)
        
        if hasattr(self, '_update_button_state'):
            self._update_button_state()
    
    def select_all_groups(self):
        """Select all groups."""
        for var in self.group_selection_vars.values():
            var.set(True)
        self.update_group_filter_status()
    
    def clear_all_groups_selection(self):
        """Clear all group selections."""
        for var in self.group_selection_vars.values():
            var.set(False)
        self.update_group_filter_status()
    
    def update_group_filter_status(self):
        """Update the group filter status display."""
        if not hasattr(self, 'group_filter_status'):
            return
        
        mode = self.analysis_mode.get()
        
        if mode == "all_units":
            current_units = self.get_current_units()
            self.group_filter_status.config(text=f"All {len(current_units)} units")
        
        elif mode in ["selected_groups", "exclude_groups"]:
            if hasattr(self, 'group_selection_vars'):
                selected_groups = [name for name, var in self.group_selection_vars.items() if var.get()]
                filtered_units = self.get_filtered_units_for_analysis()
                
                if mode == "selected_groups":
                    self.group_filter_status.config(
                        text=f"{len(selected_groups)} groups → {len(filtered_units)} units"
                    )
                else:  # exclude_groups
                    total_units = len(self.get_current_units())
                    self.group_filter_status.config(
                        text=f"Excluding {len(selected_groups)} groups → {len(filtered_units)} units"
                    )
        
        if hasattr(self, '_update_button_state'):
            self._update_button_state()
    
    def get_current_units(self):
        """Get the current list of units (abstracted for different frame types)."""
        if hasattr(self, 'selected_units') and self.selected_units:
            return self.selected_units
        elif hasattr(self, 'selected_countries') and self.selected_countries:
            return self.selected_countries
        else:
            return []
    
    def get_filtered_units_for_analysis(self):
        """Get filtered units based on current group selection mode."""
        current_units = self.get_current_units()
        
        if not current_units:
            return []
        
        mode = self.analysis_mode.get()
        
        if mode == "all_units":
            return current_units
        
        if not hasattr(self, 'group_selection_vars') or not self.group_selection_vars:
            return current_units
        
        if mode == "selected_groups":
            # Include only units from selected groups
            selected_groups = [name for name, var in self.group_selection_vars.items() if var.get()]
            filtered_units = []
            
            for group_name in selected_groups:
                units_in_group = [u for u, g in self.groups.items() if g == group_name and u in current_units]
                filtered_units.extend(units_in_group)
            
            return list(set(filtered_units))
        
        elif mode == "exclude_groups":
            # Exclude units from selected groups
            excluded_groups = [name for name, var in self.group_selection_vars.items() if var.get()]
            excluded_units = []
            
            for group_name in excluded_groups:
                units_in_group = [u for u, g in self.groups.items() if g == group_name]
                excluded_units.extend(units_in_group)
            
            return [u for u in current_units if u not in excluded_units]
        
        return current_units
    
    def get_group_enhanced_config(self, base_config):
        """Enhance base configuration with group filtering information."""
        if not base_config:
            return None
        
        # Add group filtering information
        filtered_units = self.get_filtered_units_for_analysis()
        
        enhanced_config = base_config.copy()
        enhanced_config.update({
            'groups': self.groups,
            'group_colors': self.group_colors,
            'analysis_mode': self.analysis_mode.get(),
            'original_units': self.get_current_units(),
            'filtered_units': filtered_units
        })
        
        # Add specific group selection info
        if hasattr(self, 'group_selection_vars') and self.analysis_mode.get() != "all_units":
            if self.analysis_mode.get() == "selected_groups":
                enhanced_config['selected_groups'] = [
                    name for name, var in self.group_selection_vars.items() if var.get()
                ]
            elif self.analysis_mode.get() == "exclude_groups":
                enhanced_config['excluded_groups'] = [
                    name for name, var in self.group_selection_vars.items() if var.get()
                ]
        
        return enhanced_config