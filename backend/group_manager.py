"""
Universal Group Manager for PCA Application

Manages persistent groups of research units with universal scope across all analyses.
Provides GUI for group creation, editing, and automatic integration.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import tkinter as tk
from tkinter import messagebox, filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import pandas as pd
from datetime import datetime
from difflib import SequenceMatcher

# Import logging
try:
    from backend.logging_config import get_logger
    logger = get_logger("group_manager")
except ImportError:
    import logging
    logger = logging.getLogger("group_manager")


class UniversalGroupManager:
    """Universal group manager with persistence and cross-analysis integration."""
    
    def __init__(self, app=None):
        self.app = app
        _config_dir = Path(__file__).resolve().parent.parent / "config"
        self.groups_file = _config_dir / "universal_groups.json"
        self.groups_history_file = _config_dir / "groups_history.json"
        
        # Ensure config directory exists
        _config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data structures
        self.groups: Dict[str, Dict[str, Any]] = {}
        self.group_history: List[Dict[str, Any]] = []
        
        # Smart matching settings
        self.smart_matching_enabled = True  # Enable by default
        self.smart_matching_threshold = 0.75  # 75% similarity threshold
        self.smart_matching_cache: Dict[str, Tuple[str, str]] = {}  # cache: unit -> (matched_group_unit, group_name)
        
        # Color generation counter for unique colors
        self._color_counter = 0
        
        # Enhanced color palette with better distribution
        self.color_palette = [
            "#FF6B6B",  # Red
            "#4ECDC4",  # Teal
            "#45B7D1",  # Blue
            "#96CEB4",  # Mint
            "#FECA57",  # Yellow
            "#FF9FF3",  # Pink
            "#54A0FF",  # Light Blue
            "#5F27CD",  # Purple
            "#00D2D3",  # Cyan
            "#FF9F43",  # Orange
            "#EE5A24",  # Dark Orange
            "#009432",  # Green
            "#0652DD",  # Dark Blue
            "#9C88FF",  # Lavender
            "#FFC312",  # Gold
            "#C44569",  # Rose
            "#40407A",  # Navy
            "#706FD3",  # Purple Light
            "#F97F51",  # Coral
            "#1B9CFC",  # Sky Blue
            "#6C5CE7",  # Purple Blue
            "#A29BFE",  # Periwinkle
            "#FD79A8",  # Hot Pink
            "#FDCB6E",  # Peach
            "#6C5CE7",  # Amethyst
            "#74B9FF",  # Light Sky Blue
            "#00B894",  # Mint Green
            "#E17055",  # Terra Cotta
            "#81ECEC",  # Aqua
            "#FAB1A0"   # Light Salmon
        ]
        
        # Load existing groups
        self.load_groups()
        self.load_history()
        
        logger.info("Universal Group Manager initialized")
    
    def load_groups(self) -> bool:
        """Load groups from persistent storage."""
        try:
            if self.groups_file.exists():
                with open(self.groups_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.groups = data.get('groups', {})
                    
                    # Load smart matching settings if available
                    smart_settings = data.get('smart_matching_settings', {})
                    self.smart_matching_enabled = smart_settings.get('enabled', True)
                    self.smart_matching_threshold = smart_settings.get('threshold', 0.75)
                    
                    logger.info(f"Loaded {len(self.groups)} groups from storage")
                    logger.info(f"Smart matching: {'enabled' if self.smart_matching_enabled else 'disabled'} (threshold: {self.smart_matching_threshold})")
                    return True
            else:
                logger.info("No existing groups file found, starting with empty groups")
                return False
        except Exception as e:
            logger.error(f"Error loading groups: {e}")
            messagebox.showerror("Error", f"Error loading groups: {e}")
            return False
    
    def save_groups(self) -> bool:
        """Save groups to persistent storage."""
        try:
            data = {
                'groups': self.groups,
                'smart_matching_settings': {
                    'enabled': self.smart_matching_enabled,
                    'threshold': self.smart_matching_threshold
                },
                'last_updated': datetime.now().isoformat(),
                'total_groups': len(self.groups),
                'total_units': sum(len(group_data.get('units', [])) for group_data in self.groups.values())
            }
            
            with open(self.groups_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.groups)} groups to storage")
            return True
        except Exception as e:
            logger.error(f"Error saving groups: {e}")
            messagebox.showerror("Error", f"Error saving groups: {e}")
            return False
    
    def load_history(self) -> bool:
        """Load operation history."""
        try:
            if self.groups_history_file.exists():
                with open(self.groups_history_file, 'r', encoding='utf-8') as f:
                    self.group_history = json.load(f).get('history', [])
                    logger.info(f"Loaded {len(self.group_history)} history entries")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            return False
    
    def save_history(self) -> bool:
        """Save operation history."""
        try:
            # Keep only last 1000 entries
            if len(self.group_history) > 1000:
                self.group_history = self.group_history[-1000:]
            
            data = {
                'history': self.group_history,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.groups_history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            logger.error(f"Error saving history: {e}")
            return False
    
    def log_operation(self, operation: str, details: Dict[str, Any]):
        """Log group operation to history."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details
        }
        
        self.group_history.append(entry)
        self.save_history()
        logger.info(f"Group operation logged: {operation}")
    
    def create_group(self, group_name: str, units: List[str], 
                    description: str = "", color: str = None) -> bool:
        """Create a new group."""
        try:
            if group_name in self.groups:
                messagebox.showerror("Error", f"Group '{group_name}' already exists")
                return False
            
            if not color:
                color = self.get_next_available_color()
            
            group_data = {
                'name': group_name,
                'units': list(set(units)),  # Remove duplicates
                'description': description,
                'color': color,
                'created': datetime.now().isoformat(),
                'last_modified': datetime.now().isoformat(),
                'usage_count': 0
            }
            
            self.groups[group_name] = group_data
            
            # Log operation
            self.log_operation('create_group', {
                'group_name': group_name,
                'unit_count': len(units),
                'description': description,
                'color': color
            })
            
            # Save to storage
            if self.save_groups():
                logger.info(f"Created group '{group_name}' with {len(units)} units")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error creating group: {e}")
            messagebox.showerror("Error", f"Error creating group: {e}")
            return False
    
    def edit_group(self, group_name: str, new_name: str = None, new_units: List[str] = None,
                  new_description: str = None, new_color: str = None) -> bool:
        """Edit an existing group."""
        try:
            if group_name not in self.groups:
                messagebox.showerror("Error", f"Group '{group_name}' does not exist")
                return False
            
            group_data = self.groups[group_name]
            changes = {}
            
            # Handle name change
            if new_name is not None and new_name != group_name:
                if new_name in self.groups:
                    messagebox.showerror("Error", f"Group '{new_name}' already exists")
                    return False
                
                # Move the group data to new name
                self.groups[new_name] = group_data.copy()
                self.groups[new_name]['name'] = new_name
                del self.groups[group_name]
                group_data = self.groups[new_name]
                changes['name'] = {'old': group_name, 'new': new_name}
            
            if new_units is not None:
                old_units = group_data['units'].copy()
                group_data['units'] = list(set(new_units))
                changes['units'] = {'old': old_units, 'new': group_data['units']}
            
            if new_description is not None:
                old_desc = group_data['description']
                group_data['description'] = new_description
                changes['description'] = {'old': old_desc, 'new': new_description}
            
            if new_color is not None:
                old_color = group_data['color']
                group_data['color'] = new_color
                changes['color'] = {'old': old_color, 'new': new_color}
            
            group_data['last_modified'] = datetime.now().isoformat()
            
            # Log operation
            self.log_operation('edit_group', {
                'group_name': new_name if new_name else group_name,
                'changes': changes
            })
            
            # Save to storage
            if self.save_groups():
                logger.info(f"Edited group '{new_name if new_name else group_name}'")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error editing group: {e}")
            messagebox.showerror("Error", f"Error editing group: {e}")
            return False
    
    def delete_group(self, group_name: str) -> bool:
        """Delete a group."""
        try:
            if group_name not in self.groups:
                messagebox.showerror("Error", f"Group '{group_name}' does not exist")
                return False
            
            # Confirm deletion
            response = messagebox.askyesno(
                "Confirm Deletion",
                f"Are you sure you want to delete group '{group_name}'?\n"
                f"This action cannot be undone."
            )
            
            if not response:
                return False
            
            group_data = self.groups[group_name].copy()
            del self.groups[group_name]
            
            # Log operation
            self.log_operation('delete_group', {
                'group_name': group_name,
                'deleted_data': group_data
            })
            
            # Save to storage
            if self.save_groups():
                logger.info(f"Deleted group '{group_name}'")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting group: {e}")
            messagebox.showerror("Error", f"Error deleting group: {e}")
            return False
    
    def get_group(self, group_name: str) -> Optional[Dict[str, Any]]:
        """Get group data."""
        return self.groups.get(group_name)
    
    def get_all_groups(self) -> Dict[str, Dict[str, Any]]:
        """Get all groups."""
        return self.groups.copy()
    
    def get_group_names(self) -> List[str]:
        """Get all group names."""
        return list(self.groups.keys())
    
    def get_units_in_group(self, group_name: str) -> List[str]:
        """Get units in a specific group."""
        group_data = self.groups.get(group_name)
        return group_data['units'] if group_data else []
    
    def normalize_name(self, name: str) -> str:
        """
        Normalize a unit name for smart matching.
        - Converts to lowercase
        - Removes extra whitespace
        - Removes special characters (keeps alphanumeric and spaces)
        """
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Remove special characters but keep spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def calculate_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two names using SequenceMatcher.
        Returns a value between 0.0 and 1.0
        """
        norm1 = self.normalize_name(name1)
        norm2 = self.normalize_name(name2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # Use SequenceMatcher for fuzzy matching
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def contains_match(self, name1: str, name2: str) -> bool:
        """
        Check if one name contains the other (useful for cases like 'Huawei 2011' contains 'Huawei')
        Enhanced to require significant word overlap, not just any substring.
        """
        norm1 = self.normalize_name(name1)
        norm2 = self.normalize_name(name2)
        
        if not norm1 or not norm2:
            return False
        
        # Split into words
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        # Check if one set of words is a subset of the other
        # This handles "Huawei 2011" containing "Huawei" but not random substrings
        if words2.issubset(words1) or words1.issubset(words2):
            return True
        
        # Also check for direct substring match if one name is very short (likely a brand/core name)
        shorter = norm1 if len(norm1) < len(norm2) else norm2
        longer = norm2 if len(norm1) < len(norm2) else norm1
        
        # If the shorter name is >= 4 chars and appears as a word in the longer name
        if len(shorter) >= 4 and shorter in longer:
            # Make sure it's a word boundary match, not just substring
            import re
            if re.search(r'\b' + re.escape(shorter) + r'\b', longer):
                return True
        
        return False
    
    def find_smart_match(self, unit_name: str, use_cache: bool = True) -> Optional[Tuple[str, str]]:
        """
        Find a smart match for a unit name against all group units.
        
        Args:
            unit_name: The unit name to match
            use_cache: Whether to use cached results
            
        Returns:
            Tuple of (matched_group_unit, group_name) if found, None otherwise
        """
        if not self.smart_matching_enabled:
            return None
        
        # Check cache first
        if use_cache and unit_name in self.smart_matching_cache:
            return self.smart_matching_cache[unit_name]
        
        best_match = None
        best_score = 0.0
        best_group = None
        
        # Normalize the input name
        norm_input = self.normalize_name(unit_name)
        
        # Search through all groups and their units
        for group_name, group_data in self.groups.items():
            units = group_data.get('units', [])
            
            for group_unit in units:
                # Check for contains match first (higher priority)
                if self.contains_match(unit_name, group_unit):
                    # Exact substring match - use it immediately with high score
                    score = 0.95
                    if score > best_score:
                        best_score = score
                        best_match = group_unit
                        best_group = group_name
                else:
                    # Calculate similarity score
                    score = self.calculate_similarity(unit_name, group_unit)
                    
                    if score > best_score:
                        best_score = score
                        best_match = group_unit
                        best_group = group_name
        
        # Only return match if score exceeds threshold
        if best_score >= self.smart_matching_threshold and best_match and best_group:
            result = (best_match, best_group)
            # Cache the result
            self.smart_matching_cache[unit_name] = result
            logger.debug(f"Smart match found: '{unit_name}' -> '{best_match}' (group: {best_group}, score: {best_score:.2f})")
            return result
        
        return None
    
    def get_group_for_unit(self, unit_name: str) -> Optional[str]:
        """Find which group contains a specific unit (with smart matching support)."""
        # First, try exact match
        for group_name, group_data in self.groups.items():
            if unit_name in group_data['units']:
                return group_name
        
        # If smart matching is enabled, try fuzzy match
        if self.smart_matching_enabled:
            match_result = self.find_smart_match(unit_name)
            if match_result:
                _, group_name = match_result
                return group_name
        
        return None
    
    def set_smart_matching(self, enabled: bool, threshold: float = None):
        """
        Enable or disable smart matching.
        
        Args:
            enabled: Whether to enable smart matching
            threshold: Optional similarity threshold (0.0 to 1.0)
        """
        self.smart_matching_enabled = enabled
        
        if threshold is not None:
            if 0.0 <= threshold <= 1.0:
                self.smart_matching_threshold = threshold
            else:
                logger.warning(f"Invalid threshold {threshold}, must be between 0.0 and 1.0")
        
        # Clear cache when settings change
        self.smart_matching_cache.clear()
        
        logger.info(f"Smart matching {'enabled' if enabled else 'disabled'} (threshold: {self.smart_matching_threshold})")
    
    def clear_smart_matching_cache(self):
        """Clear the smart matching cache."""
        self.smart_matching_cache.clear()
        logger.info("Smart matching cache cleared")
    
    def get_smart_matches_for_units(self, units: List[str]) -> Dict[str, Tuple[str, str, float]]:
        """
        Get smart matches for a list of units.
        
        Returns:
            Dictionary mapping unit -> (matched_unit, group_name, similarity_score)
        """
        if not self.smart_matching_enabled:
            return {}
        
        matches = {}
        
        for unit in units:
            match_result = self.find_smart_match(unit, use_cache=False)
            if match_result:
                matched_unit, group_name = match_result
                # Recalculate score for display
                score = self.calculate_similarity(unit, matched_unit)
                # Check if it was a contains match
                if self.contains_match(unit, matched_unit):
                    score = 0.95  # Mark as high-confidence contains match
                matches[unit] = (matched_unit, group_name, score)
        
        return matches
    
    def get_next_available_color(self) -> str:
        """Get next available color from palette with smart selection."""
        used_colors = {group_data.get('color', '') for group_data in self.groups.values()}
        
        # If we have fewer groups than colors in palette, use palette sequentially  
        if len(self.groups) < len(self.color_palette):
            # Find the next unused color in order
            for i, color in enumerate(self.color_palette):
                if color not in used_colors:
                    return color
        
        # If all main colors are used, or for generating multiple colors,
        # use the golden ratio method
        import colorsys
        
        # Increment counter for unique generation
        self._color_counter += 1
        
        # Generate a color that's visually distinct using golden ratio
        hue = (self._color_counter * 0.618033988749895) % 1.0
        saturation = 0.8  # High saturation for vibrant colors
        value = 0.9       # High value for bright colors
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        
        # Ensure the color is not already used
        attempts = 0
        while hex_color in used_colors and attempts < 50:
            attempts += 1
            self._color_counter += 1
            hue = (self._color_counter * 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )
        
        return hex_color
    
    def get_sequential_color(self) -> str:
        """Get the next color in sequence for UI purposes."""
        # This is used for previewing colors in the UI
        color_index = self._color_counter % len(self.color_palette)
        self._color_counter += 1
        return self.color_palette[color_index]
    
    def get_contrasting_colors(self, base_color: str, count: int) -> List[str]:
        """Generate contrasting colors based on a base color."""
        import colorsys
        
        # Convert hex to RGB
        base_color = base_color.lstrip('#')
        rgb = tuple(int(base_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        hsv = colorsys.rgb_to_hsv(*rgb)
        
        colors = []
        for i in range(count):
            # Rotate hue by golden ratio for maximum contrast
            new_hue = (hsv[0] + (i * 0.618033988749895)) % 1.0
            new_rgb = colorsys.hsv_to_rgb(new_hue, hsv[1], hsv[2])
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(new_rgb[0] * 255),
                int(new_rgb[1] * 255),
                int(new_rgb[2] * 255)
            )
            colors.append(hex_color)
        
        return colors
    
    def export_groups(self, filepath: str) -> bool:
        """Export groups to external file."""
        try:
            export_data = {
                'export_info': {
                    'exported_at': datetime.now().isoformat(),
                    'total_groups': len(self.groups),
                    'total_units': sum(len(group_data.get('units', [])) for group_data in self.groups.values()),
                    'source': 'PCA-SS Universal Group Manager'
                },
                'groups': self.groups
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.log_operation('export_groups', {
                'filepath': filepath,
                'group_count': len(self.groups)
            })
            
            logger.info(f"Exported {len(self.groups)} groups to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting groups: {e}")
            messagebox.showerror("Error", f"Error exporting groups: {e}")
            return False
    
    def import_groups(self, filepath: str, overwrite: bool = False) -> bool:
        """Import groups from external file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            imported_groups = import_data.get('groups', {})
            conflicts = []
            imported_count = 0
            
            for group_name, group_data in imported_groups.items():
                if group_name in self.groups and not overwrite:
                    conflicts.append(group_name)
                else:
                    # Update timestamps
                    group_data['last_modified'] = datetime.now().isoformat()
                    if overwrite or group_name not in self.groups:
                        group_data['created'] = datetime.now().isoformat()
                    
                    self.groups[group_name] = group_data
                    imported_count += 1
            
            if conflicts and not overwrite:
                conflict_msg = f"Found {len(conflicts)} conflicting group names:\n" + \
                              "\n".join(conflicts[:10])
                if len(conflicts) > 10:
                    conflict_msg += f"\n... and {len(conflicts) - 10} more"
                
                response = messagebox.askyesno(
                    "Import Conflicts",
                    f"{conflict_msg}\n\nDo you want to overwrite existing groups?"
                )
                
                if response:
                    return self.import_groups(filepath, overwrite=True)
                else:
                    # Import only non-conflicting groups
                    if imported_count > 0:
                        self.save_groups()
                    
                    messagebox.showinfo(
                        "Import Complete",
                        f"Imported {imported_count} groups.\n"
                        f"Skipped {len(conflicts)} conflicting groups."
                    )
                    return True
            
            # Save imported groups
            if self.save_groups():
                self.log_operation('import_groups', {
                    'filepath': filepath,
                    'imported_count': imported_count,
                    'conflicts': len(conflicts),
                    'overwrite': overwrite
                })
                
                logger.info(f"Imported {imported_count} groups from {filepath}")
                messagebox.showinfo(
                    "Import Successful",
                    f"Successfully imported {imported_count} groups."
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error importing groups: {e}")
            messagebox.showerror("Error", f"Error importing groups: {e}")
            return False
    
    def clear_all_groups(self) -> bool:
        """Clear all groups (with confirmation)."""
        try:
            if not self.groups:
                messagebox.showinfo("Info", "No groups to clear.")
                return True
            
            response = messagebox.askyesno(
                "Confirm Clear All",
                f"Are you sure you want to delete all {len(self.groups)} groups?\n"
                f"This action cannot be undone."
            )
            
            if not response:
                return False
            
            old_groups = self.groups.copy()
            self.groups = {}
            
            self.log_operation('clear_all_groups', {
                'cleared_count': len(old_groups),
                'cleared_groups': list(old_groups.keys())
            })
            
            if self.save_groups():
                logger.info(f"Cleared all {len(old_groups)} groups")
                messagebox.showinfo("Clear Complete", f"Cleared all {len(old_groups)} groups.")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error clearing groups: {e}")
            messagebox.showerror("Error", f"Error clearing groups: {e}")
            return False
    
    def export_groups_to_csv(self, filepath: str) -> bool:
        """
        Export groups to CSV format.
        Format: Column 1 = Unit, Column 2 = Group
        """
        try:
            # Prepare data for export
            export_data = []
            
            for group_name, group_data in self.groups.items():
                units = group_data.get('units', [])
                for unit in units:
                    export_data.append({
                        'Unit': unit,
                        'Group': group_name
                    })
            
            # Sort by unit name for consistency
            export_data.sort(key=lambda x: x['Unit'])
            
            # Create DataFrame and export
            df = pd.DataFrame(export_data)
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            self.log_operation('export_groups_csv', {
                'filepath': filepath,
                'group_count': len(self.groups),
                'unit_count': len(export_data)
            })
            
            logger.info(f"Exported {len(self.groups)} groups ({len(export_data)} units) to CSV: {filepath}")
            messagebox.showinfo(
                "Export Successful",
                f"Successfully exported {len(self.groups)} groups ({len(export_data)} units) to CSV."
            )
            return True
            
        except Exception as e:
            logger.error(f"Error exporting groups to CSV: {e}")
            messagebox.showerror("Export Error", f"Error exporting groups to CSV: {e}")
            return False
    
    def export_groups_to_excel(self, filepath: str) -> bool:
        """
        Export groups to Excel format.
        Format: Column 1 = Unit, Column 2 = Group
        """
        try:
            # Prepare data for export
            export_data = []
            
            for group_name, group_data in self.groups.items():
                units = group_data.get('units', [])
                for unit in units:
                    export_data.append({
                        'Unit': unit,
                        'Group': group_name
                    })
            
            # Sort by unit name for consistency
            export_data.sort(key=lambda x: x['Unit'])
            
            # Create DataFrame and export with additional sheets
            df = pd.DataFrame(export_data)
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Group_Assignments', index=False)
                
                # Summary sheet
                summary_data = []
                for group_name, group_data in self.groups.items():
                    summary_data.append({
                        'Group': group_name,
                        'Unit_Count': len(group_data.get('units', [])),
                        'Description': group_data.get('description', ''),
                        'Color': group_data.get('color', ''),
                        'Created': group_data.get('created', ''),
                        'Last_Modified': group_data.get('last_modified', '')
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Group_Summary', index=False)
            
            self.log_operation('export_groups_excel', {
                'filepath': filepath,
                'group_count': len(self.groups),
                'unit_count': len(export_data)
            })
            
            logger.info(f"Exported {len(self.groups)} groups ({len(export_data)} units) to Excel: {filepath}")
            messagebox.showinfo(
                "Export Successful",
                f"Successfully exported {len(self.groups)} groups ({len(export_data)} units) to Excel.\n"
                f"Created sheets: Group_Assignments, Group_Summary"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error exporting groups to Excel: {e}")
            messagebox.showerror("Export Error", f"Error exporting groups to Excel: {e}")
            return False
    
    def import_groups_from_csv(self, filepath: str, overwrite: bool = False) -> bool:
        """
        Import groups from CSV format.
        Expected format: Column 1 = Unit, Column 2 = Group
        """
        try:
            # Read CSV file
            df = pd.read_csv(filepath, encoding='utf-8')
            
            # Validate format
            if len(df.columns) < 2:
                raise ValueError("CSV file must have at least 2 columns (Unit, Group)")
            
            # Use first two columns regardless of their names
            unit_col = df.columns[0]
            group_col = df.columns[1]
            
            # Group units by group name
            imported_groups = {}
            color_counter = 0  # Track colors for imported groups
            
            for _, row in df.iterrows():
                unit = str(row[unit_col]).strip()
                group = str(row[group_col]).strip()
                
                if pd.isna(row[unit_col]) or pd.isna(row[group_col]):
                    continue  # Skip rows with missing data
                
                if group not in imported_groups:
                    # Get next sequential color for visual distinction
                    if color_counter < len(self.color_palette):
                        color = self.color_palette[color_counter]
                        color_counter += 1
                    else:
                        # Use HSV generation for additional colors
                        import colorsys
                        hue = (color_counter * 0.618033988749895) % 1.0
                        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                        color = "#{:02x}{:02x}{:02x}".format(
                            int(rgb[0] * 255),
                            int(rgb[1] * 255),
                            int(rgb[2] * 255)
                        )
                        color_counter += 1
                    
                    imported_groups[group] = {
                        'name': group,
                        'units': [],
                        'description': f"Imported from CSV: {os.path.basename(filepath)}",
                        'color': color,
                        'created': datetime.now().isoformat(),
                        'last_modified': datetime.now().isoformat(),
                        'usage_count': 0
                    }
                
                if unit not in imported_groups[group]['units']:
                    imported_groups[group]['units'].append(unit)
            
            # Process conflicts and import
            conflicts = []
            imported_count = 0
            
            for group_name, group_data in imported_groups.items():
                if group_name in self.groups and not overwrite:
                    conflicts.append(group_name)
                else:
                    self.groups[group_name] = group_data
                    imported_count += 1
            
            # Handle conflicts
            if conflicts and not overwrite:
                response = messagebox.askyesno(
                    "Import Conflicts",
                    f"Found {len(conflicts)} conflicting group names.\n"
                    f"Do you want to overwrite existing groups?"
                )
                
                if response:
                    return self.import_groups_from_csv(filepath, overwrite=True)
                else:
                    if imported_count > 0:
                        self.save_groups()
                    
                    messagebox.showinfo(
                        "Import Complete",
                        f"Imported {imported_count} groups.\n"
                        f"Skipped {len(conflicts)} conflicting groups."
                    )
                    return True
            
            # Save and log
            if self.save_groups():
                total_units = sum(len(group_data['units']) for group_data in imported_groups.values())
                
                self.log_operation('import_groups_csv', {
                    'filepath': filepath,
                    'imported_count': imported_count,
                    'total_units': total_units,
                    'conflicts': len(conflicts)
                })
                
                logger.info(f"Imported {imported_count} groups ({total_units} units) from CSV: {filepath}")
                messagebox.showinfo(
                    "Import Successful",
                    f"Successfully imported {imported_count} groups ({total_units} units) from CSV."
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error importing groups from CSV: {e}")
            messagebox.showerror("Import Error", f"Error importing groups from CSV: {e}")
            return False
    
    def import_groups_from_excel(self, filepath: str, sheet_name: str = None, overwrite: bool = False) -> bool:
        """
        Import groups from Excel format.
        Expected format: Column 1 = Unit, Column 2 = Group
        """
        try:
            # Read Excel file
            if sheet_name:
                df = pd.read_excel(filepath, sheet_name=sheet_name)
            else:
                # Try common sheet names first
                excel_file = pd.ExcelFile(filepath)
                possible_sheets = ['Group_Assignments', 'Groups', 'Data', excel_file.sheet_names[0]]
                
                df = None
                used_sheet = None
                for sheet in possible_sheets:
                    if sheet in excel_file.sheet_names:
                        df = pd.read_excel(filepath, sheet_name=sheet)
                        used_sheet = sheet
                        break
                
                if df is None:
                    raise ValueError("Could not find a suitable sheet in the Excel file")
                
                logger.info(f"Using sheet '{used_sheet}' for import")
            
            # Validate format
            if len(df.columns) < 2:
                raise ValueError("Excel sheet must have at least 2 columns (Unit, Group)")
            
            # Use first two columns regardless of their names
            unit_col = df.columns[0]
            group_col = df.columns[1]
            
            # Group units by group name
            imported_groups = {}
            color_counter = 0  # Track colors for imported groups
            
            for _, row in df.iterrows():
                unit = str(row[unit_col]).strip()
                group = str(row[group_col]).strip()
                
                if pd.isna(row[unit_col]) or pd.isna(row[group_col]):
                    continue  # Skip rows with missing data
                
                if group not in imported_groups:
                    # Get next sequential color for visual distinction
                    if color_counter < len(self.color_palette):
                        color = self.color_palette[color_counter]
                        color_counter += 1
                    else:
                        # Use HSV generation for additional colors
                        import colorsys
                        hue = (color_counter * 0.618033988749895) % 1.0
                        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                        color = "#{:02x}{:02x}{:02x}".format(
                            int(rgb[0] * 255),
                            int(rgb[1] * 255),
                            int(rgb[2] * 255)
                        )
                        color_counter += 1
                    
                    imported_groups[group] = {
                        'name': group,
                        'units': [],
                        'description': f"Imported from Excel: {os.path.basename(filepath)}",
                        'color': color,
                        'created': datetime.now().isoformat(),
                        'last_modified': datetime.now().isoformat(),
                        'usage_count': 0
                    }
                
                if unit not in imported_groups[group]['units']:
                    imported_groups[group]['units'].append(unit)
            
            # Process conflicts and import
            conflicts = []
            imported_count = 0
            
            for group_name, group_data in imported_groups.items():
                if group_name in self.groups and not overwrite:
                    conflicts.append(group_name)
                else:
                    self.groups[group_name] = group_data
                    imported_count += 1
            
            # Handle conflicts
            if conflicts and not overwrite:
                response = messagebox.askyesno(
                    "Import Conflicts",
                    f"Found {len(conflicts)} conflicting group names.\n"
                    f"Do you want to overwrite existing groups?"
                )
                
                if response:
                    return self.import_groups_from_excel(filepath, sheet_name, overwrite=True)
                else:
                    if imported_count > 0:
                        self.save_groups()
                    
                    messagebox.showinfo(
                        "Import Complete",
                        f"Imported {imported_count} groups.\n"
                        f"Skipped {len(conflicts)} conflicting groups."
                    )
                    return True
            
            # Save and log
            if self.save_groups():
                total_units = sum(len(group_data['units']) for group_data in imported_groups.values())
                
                self.log_operation('import_groups_excel', {
                    'filepath': filepath,
                    'sheet_name': used_sheet if 'used_sheet' in locals() else sheet_name,
                    'imported_count': imported_count,
                    'total_units': total_units,
                    'conflicts': len(conflicts)
                })
                
                logger.info(f"Imported {imported_count} groups ({total_units} units) from Excel: {filepath}")
                messagebox.showinfo(
                    "Import Successful",
                    f"Successfully imported {imported_count} groups ({total_units} units) from Excel."
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error importing groups from Excel: {e}")
            messagebox.showerror("Import Error", f"Error importing groups from Excel: {e}")
            return False
    
    def get_groups_for_units(self, units: List[str]) -> Dict[str, List[str]]:
        """Get group assignments for a list of units."""
        result = {}
        
        for unit in units:
            group_name = self.get_group_for_unit(unit)
            if group_name:
                if group_name not in result:
                    result[group_name] = []
                result[group_name].append(unit)
            else:
                # Ungrouped units
                if 'Ungrouped' not in result:
                    result['Ungrouped'] = []
                result['Ungrouped'].append(unit)
        
        return result
    
    def get_group_colors(self) -> Dict[str, str]:
        """Get mapping of group names to colors."""
        return {name: data['color'] for name, data in self.groups.items()}
    
    def increment_usage(self, group_name: str):
        """Increment usage counter for a group."""
        if group_name in self.groups:
            self.groups[group_name]['usage_count'] = self.groups[group_name].get('usage_count', 0) + 1
            self.save_groups()


class GroupManagerGUI:
    """GUI for managing universal groups."""
    
    def __init__(self, parent, group_manager: UniversalGroupManager):
        self.parent = parent
        self.group_manager = group_manager
        self.dialog = None
        self.available_units = []
        
    def show_manager_dialog(self, available_units: List[str] = None):
        """Show the group manager dialog."""
        if available_units:
            self.available_units = available_units
        
        self.dialog = ttk.Toplevel(self.parent)
        self.dialog.title("🏷️ Universal Group Manager")
        self.dialog.geometry("900x700")
        self.dialog.resizable(True, True)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        self.setup_gui()
        self.refresh_groups_list()
        
    def setup_gui(self):
        """Setup the GUI components."""
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill=BOTH, expand=YES)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="🏷️ Universal Group Manager",
            font=("Helvetica", 16, "bold"),
            bootstyle="primary"
        )
        title_label.pack(pady=(0, 10))
        
        # Info label
        info_text = f"Manage persistent groups across all analyses. Currently loaded: {len(self.group_manager.groups)} groups"
        info_label = ttk.Label(
            main_frame,
            text=info_text,
            font=("Helvetica", 10),
            bootstyle="secondary"
        )
        info_label.pack(pady=(0, 10))
        
        # Smart Matching Settings Frame
        smart_frame = ttk.LabelFrame(main_frame, text="🧠 Smart Matching Settings", padding=10)
        smart_frame.pack(fill=X, pady=(0, 10))
        
        # Smart matching toggle
        smart_row1 = ttk.Frame(smart_frame)
        smart_row1.pack(fill=X, pady=(0, 5))
        
        self.smart_matching_var = tk.BooleanVar(value=self.group_manager.smart_matching_enabled)
        smart_check = ttk.Checkbutton(
            smart_row1,
            text="🎯 Enable Smart Group Detection",
            variable=self.smart_matching_var,
            command=self.toggle_smart_matching,
            bootstyle="success-round-toggle"
        )
        smart_check.pack(side=LEFT)
        
        # Info button
        ttk.Button(
            smart_row1,
            text="ℹ️",
            width=3,
            command=self.show_smart_matching_info,
            bootstyle="info-outline"
        ).pack(side=LEFT, padx=(10, 0))
        
        # Threshold slider
        smart_row2 = ttk.Frame(smart_frame)
        smart_row2.pack(fill=X, pady=(5, 0))
        
        ttk.Label(smart_row2, text="Similarity Threshold:").pack(side=LEFT, padx=(0, 10))
        
        self.threshold_var = tk.DoubleVar(value=self.group_manager.smart_matching_threshold)
        self.threshold_scale = ttk.Scale(
            smart_row2,
            from_=0.5,
            to=0.95,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.update_threshold
        )
        self.threshold_scale.pack(side=LEFT, fill=X, expand=YES, padx=(0, 10))
        
        self.threshold_label = ttk.Label(smart_row2, text=f"{self.threshold_var.get():.0%}")
        self.threshold_label.pack(side=LEFT)
        
        # Test smart matching button
        smart_row3 = ttk.Frame(smart_frame)
        smart_row3.pack(fill=X, pady=(10, 0))
        
        ttk.Button(
            smart_row3,
            text="🔍 Preview Smart Matches",
            command=self.preview_smart_matches,
            bootstyle="info"
        ).pack(side=LEFT)
        
        ttk.Button(
            smart_row3,
            text="🗑️ Clear Cache",
            command=self.clear_cache,
            bootstyle="secondary"
        ).pack(side=LEFT, padx=(5, 0))
        
        # Main content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=BOTH, expand=YES)
        
        # Left panel - Groups list
        left_frame = ttk.LabelFrame(content_frame, text="📋 Existing Groups", padding=10)
        left_frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=(0, 10))
        
        # Groups treeview
        columns = ('Name', 'Units', 'Description', 'Color')
        self.groups_tree = ttk.Treeview(left_frame, columns=columns, show='tree headings', height=15)
        
        # Configure columns
        self.groups_tree.heading('#0', text='Group')
        self.groups_tree.column('#0', width=120)
        
        for col in columns:
            self.groups_tree.heading(col, text=col)
            if col == 'Name':
                self.groups_tree.column(col, width=120)
            elif col == 'Units':
                self.groups_tree.column(col, width=60)
            elif col == 'Description':
                self.groups_tree.column(col, width=200)
            elif col == 'Color':
                self.groups_tree.column(col, width=80)
        
        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(left_frame, orient=VERTICAL, command=self.groups_tree.yview)
        self.groups_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.groups_tree.pack(side=LEFT, fill=BOTH, expand=YES)
        tree_scroll.pack(side=RIGHT, fill=Y)
        
        # Group action buttons
        group_buttons_frame = ttk.Frame(left_frame)
        group_buttons_frame.pack(fill=X, pady=(10, 0))
        
        ttk.Button(
            group_buttons_frame,
            text="✏️ Edit",
            bootstyle="info",
            command=self.edit_selected_group
        ).pack(side=LEFT, padx=(0, 5))
        
        ttk.Button(
            group_buttons_frame,
            text="🗑️ Delete",
            bootstyle="danger",
            command=self.delete_selected_group
        ).pack(side=LEFT, padx=(0, 5))
        
        ttk.Button(
            group_buttons_frame,
            text="📊 Details",
            bootstyle="secondary",
            command=self.show_group_details
        ).pack(side=LEFT)
        
        # Right panel - Create/Edit group
        right_frame = ttk.LabelFrame(content_frame, text="➕ Create New Group", padding=10)
        right_frame.pack(side=RIGHT, fill=BOTH, expand=YES)
        
        # Group name
        ttk.Label(right_frame, text="Group Name:").pack(anchor=W)
        self.group_name_var = tk.StringVar()
        self.group_name_entry = ttk.Entry(right_frame, textvariable=self.group_name_var, width=30)
        self.group_name_entry.pack(fill=X, pady=(5, 10))
        
        # Description
        ttk.Label(right_frame, text="Description (optional):").pack(anchor=W)
        self.description_var = tk.StringVar()
        self.description_entry = ttk.Entry(right_frame, textvariable=self.description_var, width=30)
        self.description_entry.pack(fill=X, pady=(5, 10))
        
        # Color selection
        color_frame = ttk.Frame(right_frame)
        color_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(color_frame, text="Color:").pack(side=LEFT)
        self.selected_color = tk.StringVar(value=self.group_manager.get_next_available_color())
        
        # Color preview
        self.color_preview = tk.Label(
            color_frame,
            width=3,
            height=1,
            bg=self.selected_color.get(),
            relief="solid",
            borderwidth=1
        )
        self.color_preview.pack(side=LEFT, padx=(10, 5))
        
        ttk.Button(
            color_frame,
            text="Choose Color",
            command=self.choose_color,
            bootstyle="secondary"
        ).pack(side=LEFT, padx=(5, 0))
        
        ttk.Button(
            color_frame,
            text="🎨 New Color",
            command=self.generate_new_color,
            bootstyle="info-outline"
        ).pack(side=LEFT, padx=(5, 0))
        
        # Units selection
        ttk.Label(right_frame, text="Select Units:").pack(anchor=W, pady=(10, 5))
        
        # Search frame
        search_frame = ttk.Frame(right_frame)
        search_frame.pack(fill=X, pady=(0, 5))
        
        ttk.Label(search_frame, text="Search:").pack(side=LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.filter_units)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20)
        search_entry.pack(side=LEFT, padx=(5, 0), fill=X, expand=YES)
        
        # Units listbox with checkboxes
        units_frame = ttk.Frame(right_frame)
        units_frame.pack(fill=BOTH, expand=YES)
        
        self.units_listbox = tk.Listbox(
            units_frame,
            selectmode=tk.MULTIPLE,
            height=10
        )
        
        units_scroll = ttk.Scrollbar(units_frame, orient=VERTICAL, command=self.units_listbox.yview)
        self.units_listbox.configure(yscrollcommand=units_scroll.set)
        
        self.units_listbox.pack(side=LEFT, fill=BOTH, expand=YES)
        units_scroll.pack(side=RIGHT, fill=Y)
        
        # Selection buttons
        selection_frame = ttk.Frame(right_frame)
        selection_frame.pack(fill=X, pady=(10, 0))
        
        ttk.Button(
            selection_frame,
            text="Select All",
            command=self.select_all_units,
            bootstyle="info"
        ).pack(side=LEFT, padx=(0, 5))
        
        ttk.Button(
            selection_frame,
            text="Clear Selection",
            command=self.clear_unit_selection,
            bootstyle="secondary"
        ).pack(side=LEFT)
        
        # Create button
        create_frame = ttk.Frame(right_frame)
        create_frame.pack(fill=X, pady=(20, 0))
        
        self.create_button = ttk.Button(
            create_frame,
            text="🏷️ Create Group",
            command=self.create_group,
            bootstyle="success"
        )
        self.create_button.pack(side=LEFT, padx=(0, 10))
        
        self.edit_mode = False
        self.editing_group = None
        
        # Bottom buttons frame
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=X, pady=(20, 0))
        
        # Create two rows for buttons
        buttons_row1 = ttk.Frame(bottom_frame)
        buttons_row1.pack(fill=X, pady=(0, 5))

        buttons_row2 = ttk.Frame(bottom_frame)
        buttons_row2.pack(fill=X)

        # First row - Export buttons
        ttk.Label(buttons_row1, text="Export:", font=("Arial", 9, "bold")).pack(side=LEFT, padx=(0, 5))

        ttk.Button(
            buttons_row1,
            text="📤 JSON",
            command=self.export_groups,
            bootstyle="info"
        ).pack(side=LEFT, padx=(0, 3))

        ttk.Button(
            buttons_row1,
            text="📊 CSV",
            command=self.export_groups_csv,
            bootstyle="info-outline"
        ).pack(side=LEFT, padx=(0, 3))

        ttk.Button(
            buttons_row1,
            text="📈 Excel",
            command=self.export_groups_excel,
            bootstyle="info-outline"
        ).pack(side=LEFT, padx=(0, 10))

        # Second row - Import and other buttons
        ttk.Label(buttons_row2, text="Import:", font=("Arial", 9, "bold")).pack(side=LEFT, padx=(0, 5))

        ttk.Button(
            buttons_row2,
            text="📥 JSON",
            command=self.import_groups,
            bootstyle="success"
        ).pack(side=LEFT, padx=(0, 3))

        ttk.Button(
            buttons_row2,
            text="📊 CSV",
            command=self.import_groups_csv,
            bootstyle="success-outline"
        ).pack(side=LEFT, padx=(0, 3))

        ttk.Button(
            buttons_row2,
            text="📈 Excel",
            command=self.import_groups_excel,
            bootstyle="success-outline"
        ).pack(side=LEFT, padx=(0, 10))

        ttk.Button(
            buttons_row2,
            text="🗑️ Clear All",
            command=self.clear_all_groups,
            bootstyle="danger"
        ).pack(side=LEFT, padx=(10, 0))

        # Close button
        ttk.Button(
            bottom_frame,
            text="✅ Close",
            command=self.dialog.destroy,
            bootstyle="primary"
        ).pack(side=RIGHT)
        
        # Load available units
        self.load_available_units()
    
    def load_available_units(self):
        """Load available units into the listbox."""
        self.units_listbox.delete(0, tk.END)
        
        if self.available_units:
            for unit in sorted(self.available_units):
                self.units_listbox.insert(tk.END, unit)
        else:
            # If no units provided, show a message
            self.units_listbox.insert(tk.END, "No units available. Load data first.")
    
    def filter_units(self, *args):
        """Filter units based on search text."""
        search_text = self.search_var.get().lower()
        self.units_listbox.delete(0, tk.END)
        
        filtered_units = [
            unit for unit in sorted(self.available_units)
            if search_text in unit.lower()
        ]
        
        for unit in filtered_units:
            self.units_listbox.insert(tk.END, unit)
    
    def select_all_units(self):
        """Select all visible units."""
        self.units_listbox.selection_set(0, tk.END)
    
    def clear_unit_selection(self):
        """Clear unit selection."""
        self.units_listbox.selection_clear(0, tk.END)
    
    def choose_color(self):
        """Choose color for group."""
        from tkinter import colorchooser
        
        color = colorchooser.askcolor(
            initialcolor=self.selected_color.get(),
            title="Choose Group Color"
        )
        
        if color[1]:  # color[1] is the hex value
            self.selected_color.set(color[1])
            self.color_preview.config(bg=color[1])
    
    def generate_new_color(self):
        """Generate a new automatic color for the group."""
        new_color = self.group_manager.get_sequential_color()
        self.selected_color.set(new_color)
        self.color_preview.config(bg=new_color)
    
    def create_group(self):
        """Create a new group."""
        group_name = self.group_name_var.get().strip()
        description = self.description_var.get().strip()
        
        if not group_name:
            messagebox.showerror("Error", "Group name is required.")
            return
        
        # Get selected units
        selected_indices = self.units_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "At least one unit must be selected.")
            return
        
        selected_units = [self.units_listbox.get(i) for i in selected_indices]
        
        if self.edit_mode:
            # Edit existing group
            success = self.group_manager.edit_group(
                self.editing_group,
                new_name=group_name if group_name != self.editing_group else None,
                new_units=selected_units,
                new_description=description,
                new_color=self.selected_color.get()
            )
            
            if success:
                messagebox.showinfo("Success", f"Group '{group_name}' updated successfully.")
                self.reset_form()
                self.refresh_groups_list()
        else:
            # Create new group
            success = self.group_manager.create_group(
                group_name,
                selected_units,
                description,
                self.selected_color.get()
            )
            
            if success:
                messagebox.showinfo("Success", f"Group '{group_name}' created successfully.")
                self.reset_form()
                self.refresh_groups_list()
    
    def reset_form(self):
        """Reset the form to initial state."""
        self.group_name_var.set("")
        self.description_var.set("")
        
        # Get a new sequential color for variety in UI
        next_color = self.group_manager.get_sequential_color()
        self.selected_color.set(next_color)
        self.color_preview.config(bg=next_color)
        
        self.units_listbox.selection_clear(0, tk.END)
        self.search_var.set("")
        
        self.edit_mode = False
        self.editing_group = None
        self.create_button.config(text="🏷️ Create Group")
        
        # Update the frame title
        for child in self.dialog.winfo_children():
            if isinstance(child, ttk.Frame):
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, ttk.LabelFrame) and "Create" in str(grandchild['text']):
                        grandchild.config(text="➕ Create New Group")
    
    def edit_selected_group(self):
        """Edit the selected group."""
        selection = self.groups_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a group to edit.")
            return
        
        group_name = self.groups_tree.item(selection[0])['text']
        group_data = self.group_manager.get_group(group_name)
        
        if not group_data:
            messagebox.showerror("Error", f"Group '{group_name}' not found.")
            return
        
        # Switch to edit mode
        self.edit_mode = True
        self.editing_group = group_name
        
        # Populate form with group data
        self.group_name_var.set(group_name)
        self.description_var.set(group_data.get('description', ''))
        self.selected_color.set(group_data.get('color', '#FF6B6B'))
        self.color_preview.config(bg=self.selected_color.get())
        
        # Clear search to show all units
        self.search_var.set("")
        self.filter_units()  # Refresh the listbox with all units
        
        # Select units in the group
        self.units_listbox.selection_clear(0, tk.END)
        group_units = set(group_data.get('units', []))
        
        # Wait a moment for the listbox to update, then select units
        self.dialog.after(50, lambda: self._select_group_units(group_units))
        
        # Update button and frame title
        self.create_button.config(text="💾 Update Group")
        
        # Find and update the frame title
        for child in self.dialog.winfo_children():
            if isinstance(child, ttk.Frame):
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, ttk.LabelFrame) and ("Create" in str(grandchild['text']) or "Edit" in str(grandchild['text'])):
                        grandchild.config(text=f"✏️ Edit Group: {group_name}")
    
    def _select_group_units(self, group_units):
        """Helper method to select units in the listbox."""
        for i in range(self.units_listbox.size()):
            unit = self.units_listbox.get(i)
            if unit in group_units:
                self.units_listbox.selection_set(i)
    
    def delete_selected_group(self):
        """Delete the selected group."""
        selection = self.groups_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a group to delete.")
            return
        
        group_name = self.groups_tree.item(selection[0])['text']
        
        if self.group_manager.delete_group(group_name):
            self.refresh_groups_list()
            if self.edit_mode and self.editing_group == group_name:
                self.reset_form()
    
    def show_group_details(self):
        """Show detailed information about the selected group."""
        selection = self.groups_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a group to view details.")
            return
        
        group_name = self.groups_tree.item(selection[0])['text']
        group_data = self.group_manager.get_group(group_name)
        
        if not group_data:
            messagebox.showerror("Error", f"Group '{group_name}' not found.")
            return
        
        # Create details dialog
        details_dialog = ttk.Toplevel(self.dialog)
        details_dialog.title(f"Group Details: {group_name}")
        details_dialog.geometry("600x500")
        details_dialog.transient(self.dialog)
        details_dialog.grab_set()
        
        main_frame = ttk.Frame(details_dialog, padding=20)
        main_frame.pack(fill=BOTH, expand=YES)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text=f"📊 Group Details: {group_name}",
            font=("Helvetica", 14, "bold"),
            bootstyle="primary"
        )
        title_label.pack(pady=(0, 20))
        
        # Group info
        info_frame = ttk.LabelFrame(main_frame, text="Group Information", padding=10)
        info_frame.pack(fill=X, pady=(0, 10))
        
        info_text = f"""
Name: {group_data.get('name', 'N/A')}
Description: {group_data.get('description', 'No description')}
Color: {group_data.get('color', 'N/A')}
Created: {group_data.get('created', 'N/A')[:19] if group_data.get('created') else 'N/A'}
Last Modified: {group_data.get('last_modified', 'N/A')[:19] if group_data.get('last_modified') else 'N/A'}
Usage Count: {group_data.get('usage_count', 0)}
Total Units: {len(group_data.get('units', []))}
        """.strip()
        
        info_label = ttk.Label(
            info_frame,
            text=info_text,
            font=("Consolas", 10),
            justify=LEFT
        )
        info_label.pack(anchor=W)
        
        # Units list
        units_frame = ttk.LabelFrame(main_frame, text="Units in Group", padding=10)
        units_frame.pack(fill=BOTH, expand=YES, pady=(10, 0))
        
        units_listbox = tk.Listbox(units_frame, height=15)
        units_scroll = ttk.Scrollbar(units_frame, orient=VERTICAL, command=units_listbox.yview)
        units_listbox.configure(yscrollcommand=units_scroll.set)
        
        for unit in sorted(group_data.get('units', [])):
            units_listbox.insert(tk.END, unit)
        
        units_listbox.pack(side=LEFT, fill=BOTH, expand=YES)
        units_scroll.pack(side=RIGHT, fill=Y)
        
        # Close button
        ttk.Button(
            main_frame,
            text="Close",
            command=details_dialog.destroy,
            bootstyle="secondary"
        ).pack(pady=(20, 0))
    
    def refresh_groups_list(self):
        """Refresh the groups treeview."""
        # Clear existing items
        for item in self.groups_tree.get_children():
            self.groups_tree.delete(item)
        
        # Add groups
        for group_name, group_data in self.group_manager.groups.items():
            units_count = len(group_data.get('units', []))
            description = group_data.get('description', '')
            color = group_data.get('color', '#000000')
            
            # Truncate description if too long
            if len(description) > 50:
                description = description[:47] + "..."
            
            self.groups_tree.insert(
                '',
                'end',
                text=group_name,
                values=(group_name, units_count, description, color),
                tags=(color,)
            )
            
            # Configure tag color
            self.groups_tree.tag_configure(color, background=color)
    
    def export_groups(self):
        """Export groups to JSON file."""
        filepath = filedialog.asksaveasfilename(
            title="Export Groups",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            if self.group_manager.export_groups(filepath):
                messagebox.showinfo("Export Successful", f"Groups exported to {filepath}")
    
    def export_groups_csv(self):
        """Export groups to CSV file."""
        filepath = filedialog.asksaveasfilename(
            title="Export Groups to CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            if self.group_manager.export_groups_to_csv(filepath):
                messagebox.showinfo("Export Successful", f"Groups exported to CSV: {filepath}")
    
    def export_groups_excel(self):
        """Export groups to Excel file."""
        filepath = filedialog.asksaveasfilename(
            title="Export Groups to Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if filepath:
            if self.group_manager.export_groups_to_excel(filepath):
                messagebox.showinfo("Export Successful", f"Groups exported to Excel: {filepath}")
    
    def import_groups(self):
        """Import groups from JSON file."""
        filepath = filedialog.askopenfilename(
            title="Import Groups",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            if self.group_manager.import_groups(filepath):
                self.refresh_groups_list()
    
    def import_groups_csv(self):
        """Import groups from CSV file."""
        filepath = filedialog.askopenfilename(
            title="Import Groups from CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            if self.group_manager.import_groups_from_csv(filepath):
                self.refresh_groups_list()
                messagebox.showinfo("Import Complete", "Groups imported successfully from CSV.")
    
    def import_groups_excel(self):
        """Import groups from Excel file."""
        filepath = filedialog.askopenfilename(
            title="Import Groups from Excel",
            filetypes=[("Excel files", "*.xlsx;*.xls"), ("All files", "*.*")]
        )
        
        if filepath:
            # Ask user to specify sheet name (optional)
            sheet_dialog = tk.Toplevel(self.dialog)
            sheet_dialog.title("Select Sheet")
            sheet_dialog.geometry("300x150")
            sheet_dialog.transient(self.dialog)
            sheet_dialog.grab_set()
            
            # Center the dialog
            sheet_dialog.geometry("+%d+%d" % (
                self.dialog.winfo_rootx() + 50,
                self.dialog.winfo_rooty() + 50
            ))
            
            ttk.Label(sheet_dialog, text="Sheet name (leave empty for auto-detect):").pack(pady=10)
            
            sheet_var = tk.StringVar()
            sheet_entry = ttk.Entry(sheet_dialog, textvariable=sheet_var, width=30)
            sheet_entry.pack(pady=5)
            
            result = {"sheet_name": None, "proceed": False}
            
            def on_ok():
                result["sheet_name"] = sheet_var.get().strip() if sheet_var.get().strip() else None
                result["proceed"] = True
                sheet_dialog.destroy()
            
            def on_cancel():
                sheet_dialog.destroy()
            
            button_frame = ttk.Frame(sheet_dialog)
            button_frame.pack(pady=10)
            
            ttk.Button(button_frame, text="OK", command=on_ok).pack(side=LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=LEFT, padx=5)
            
            # Wait for dialog to close
            sheet_dialog.wait_window()
            
            if result["proceed"]:
                if self.group_manager.import_groups_from_excel(filepath, result["sheet_name"]):
                    self.refresh_groups_list()
                    messagebox.showinfo("Import Complete", "Groups imported successfully from Excel.")
    
    def toggle_smart_matching(self):
        """Toggle smart matching on/off."""
        enabled = self.smart_matching_var.get()
        self.group_manager.set_smart_matching(enabled, self.threshold_var.get())
        
        status = "enabled" if enabled else "disabled"
        messagebox.showinfo(
            "Smart Matching",
            f"Smart group detection has been {status}.\n\n"
            f"{'Units like "Huawei 2011" will now match to "Huawei" groups.' if enabled else 'Only exact name matches will be used.'}"
        )
    
    def update_threshold(self, value=None):
        """Update similarity threshold."""
        threshold = self.threshold_var.get()
        self.threshold_label.config(text=f"{threshold:.0%}")
        self.group_manager.set_smart_matching(self.smart_matching_var.get(), threshold)
    
    def show_smart_matching_info(self):
        """Show information about smart matching."""
        info_text = """
🧠 Smart Group Detection

This feature automatically matches similar unit names to existing groups using intelligent pattern matching.

Examples:
• "Huawei 2011" → matches "Huawei"
• "TESLA INC" → matches "Tesla"
• "General Motors Corp" → matches "General Motors"

How it works:
1. Case-insensitive matching
2. Ignores special characters
3. Detects substring matches (e.g., "2011" in name)
4. Uses similarity scoring for fuzzy matches

Threshold Setting:
• Higher (90-95%): Only very similar names
• Medium (75-80%): Balanced (Recommended)
• Lower (50-60%): More permissive matching

💡 Tip: Use "Preview Smart Matches" to see what will be matched before running analysis.
        """
        
        messagebox.showinfo("Smart Matching Information", info_text)
    
    def preview_smart_matches(self):
        """Preview smart matches for available units."""
        if not self.available_units:
            messagebox.showwarning(
                "No Units Available",
                "Please load a data file first to see available units."
            )
            return
        
        if not self.group_manager.smart_matching_enabled:
            messagebox.showwarning(
                "Smart Matching Disabled",
                "Please enable Smart Group Detection first."
            )
            return
        
        # Get smart matches
        matches = self.group_manager.get_smart_matches_for_units(self.available_units)
        
        if not matches:
            messagebox.showinfo(
                "No Matches Found",
                f"No smart matches found with current threshold ({self.threshold_var.get():.0%}).\n\n"
                "Try lowering the threshold or check if your groups contain similar unit names."
            )
            return
        
        # Create preview dialog
        preview_dialog = ttk.Toplevel(self.dialog)
        preview_dialog.title("🔍 Smart Matches Preview")
        preview_dialog.geometry("700x500")
        preview_dialog.transient(self.dialog)
        
        main_frame = ttk.Frame(preview_dialog, padding=10)
        main_frame.pack(fill=BOTH, expand=YES)
        
        # Title
        ttk.Label(
            main_frame,
            text=f"Found {len(matches)} smart matches",
            font=("Helvetica", 14, "bold")
        ).pack(pady=(0, 10))
        
        # Info
        info_text = f"Threshold: {self.threshold_var.get():.0%} | Total units checked: {len(self.available_units)}"
        ttk.Label(main_frame, text=info_text, bootstyle="secondary").pack(pady=(0, 10))
        
        # Treeview for matches
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=BOTH, expand=YES, pady=(0, 10))
        
        columns = ('Database Unit', 'Matched To', 'Group', 'Confidence')
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        tree.heading('Database Unit', text='Unit in Database')
        tree.heading('Matched To', text='Matched To Group Unit')
        tree.heading('Group', text='Group Name')
        tree.heading('Confidence', text='Confidence')
        
        tree.column('Database Unit', width=180)
        tree.column('Matched To', width=180)
        tree.column('Group', width=150)
        tree.column('Confidence', width=100)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=LEFT, fill=BOTH, expand=YES)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # Populate with matches
        for unit, (matched_unit, group_name, score) in sorted(matches.items()):
            confidence = "High ✓" if score >= 0.9 else "Medium" if score >= 0.8 else "Low"
            tree.insert('', 'end', values=(unit, matched_unit, group_name, f"{score:.0%} - {confidence}"))
        
        # Close button
        ttk.Button(
            main_frame,
            text="Close",
            command=preview_dialog.destroy,
            bootstyle="secondary"
        ).pack()
    
    def clear_cache(self):
        """Clear smart matching cache."""
        self.group_manager.clear_smart_matching_cache()
        messagebox.showinfo(
            "Cache Cleared",
            "Smart matching cache has been cleared.\n"
            "New matches will be calculated on next analysis."
        )
    
    def clear_all_groups(self):
        """Clear all groups."""
        if self.group_manager.clear_all_groups():
            self.refresh_groups_list()
            self.reset_form()


# Global instance for universal access
universal_group_manager = None

def get_universal_group_manager(app=None) -> UniversalGroupManager:
    """Get or create the universal group manager instance."""
    global universal_group_manager
    
    if universal_group_manager is None:
        universal_group_manager = UniversalGroupManager(app)
    
    return universal_group_manager

def show_group_manager_gui(parent, available_units: List[str] = None):
    """Show the group manager GUI."""
    group_manager = get_universal_group_manager()
    gui = GroupManagerGUI(parent, group_manager)
    gui.show_manager_dialog(available_units)


if __name__ == "__main__":
    # Test the group manager
    import tkinter as tk
    
    root = tk.Tk()
    root.withdraw()
    
    # Test data
    test_units = [
        "Tesla", "Ford Motor", "General Motors", "BMW Group", "Volkswagen",
        "Toyota Motor", "Nissan Motor", "Stellantis", "BYD", "Geely"
    ]
    
    show_group_manager_gui(root, test_units)
    
    root.mainloop()