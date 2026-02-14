"""
File Handler for PCA Application

Handles file selection, validation, and data loading operations.
"""

import os
from tkinter import filedialog, messagebox
import pandas as pd

# Importar módulos de seguridad
from backend.security_utils import validate_file_path, validate_directory_path, SecurityError
from backend.secure_error_handler import handle_file_operation_error, safe_exception_handler


class FileHandler:
    """Handles file operations and validation for the PCA application."""

    # Configurable row/column limits (can be overridden via config)
    MAX_ROWS = 500_000
    MAX_COLUMNS = 1_000

    def __init__(self, app):
        self.app = app

    def select_file(self, title="Select file", filetypes=None):
        """Select a file using file dialog."""
        if filetypes is None:
            try:
                from data_connectors import FILE_TYPE_FILTERS
                filetypes = FILE_TYPE_FILTERS
            except ImportError:
                filetypes = [
                    ("Excel files", "*.xlsx *.xls"),
                    ("CSV files", "*.csv"),
                    ("Parquet files", "*.parquet"),
                    ("All files", "*.*")
                ]

        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes
        )
        return file_path

    def select_save_file(self, title="Save file", defaultextension=".xlsx", filetypes=None):
        """Select a save file location."""
        if filetypes is None:
            filetypes = [("Excel files", "*.xlsx *.xls")]

        filename = filedialog.asksaveasfilename(
            title=title,
            defaultextension=defaultextension,
            filetypes=filetypes,
        )
        return filename

    @safe_exception_handler
    def validate_file_path(self, file_path):
        """Validate that file exists and is accessible with security checks."""
        try:
            if not file_path or not file_path.strip():
                messagebox.showerror("Validation Error", "File path is required")
                return False

            # Use secure validation
            try:
                try:
                    from data_connectors import SUPPORTED_EXTENSIONS
                    allowed = SUPPORTED_EXTENSIONS
                except ImportError:
                    allowed = ['.xlsx', '.xls', '.csv', '.parquet']
                validated_path = validate_file_path(file_path, allowed_extensions=allowed)
            except SecurityError as e:
                error_msg = handle_file_operation_error(e, file_path, "security validation")
                messagebox.showerror("Security Error", f"File validation failed: {error_msg}")
                return False
            except Exception as e:
                error_msg = handle_file_operation_error(e, file_path, "path validation")
                messagebox.showerror("Validation Error", f"File validation failed: {error_msg}")
                return False

            # Additional permission check (redundant but explicit)
            if not os.access(str(validated_path), os.R_OK):
                messagebox.showerror("Access Denied", f"Cannot read file: {validated_path.name}")
                return False

            return True

        except Exception as e:
            error_msg = handle_file_operation_error(e, file_path, "validation")
            messagebox.showerror("Validation Error", f"Unexpected error during validation: {error_msg}")
            return False

    @safe_exception_handler
    def load_excel_data(self, file_path):
        """Load data from Excel file with security validation."""
        try:
            # Validate path securely first
            validated_path = validate_file_path(file_path, allowed_extensions=['.xlsx', '.xls'])

            # Load with configurable size limits
            excel_data = pd.read_excel(str(validated_path), sheet_name=None, nrows=self.MAX_ROWS)

            # Validate each sheet
            validated_data = {}
            for sheet_name, df in excel_data.items():
                if len(df) >= self.MAX_ROWS:
                    messagebox.showwarning(
                        "Data Truncated",
                        f"Sheet '{sheet_name}' has {len(df)}+ rows and was truncated to {self.MAX_ROWS:,}. "
                        f"Increase FileHandler.MAX_ROWS if you need more."
                    )

                if len(df.columns) > self.MAX_COLUMNS:
                    messagebox.showwarning(
                        "Too Many Columns",
                        f"Sheet '{sheet_name}' has {len(df.columns)} columns (limit: {self.MAX_COLUMNS:,}). Skipping."
                    )
                    continue

                validated_data[sheet_name] = df

            return validated_data if validated_data else None

        except SecurityError as e:
            error_msg = handle_file_operation_error(e, file_path, "loading Excel file")
            messagebox.showerror("Security Error", f"Error loading Excel file: {error_msg}")
            return None
        except Exception as e:
            error_msg = handle_file_operation_error(e, file_path, "loading Excel file")
            messagebox.showerror("Error", f"Error loading Excel file: {error_msg}")
            return None

    @safe_exception_handler
    def load_csv_data(self, file_path):
        """Load data from CSV file with security validation."""
        try:
            # Validate path securely first
            validated_path = validate_file_path(file_path, allowed_extensions=['.csv'])

            # Try different encodings with size limits
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None

            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(str(validated_path), encoding=encoding, nrows=self.MAX_ROWS)
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding")

            # Truncation warning
            if len(df) >= self.MAX_ROWS:
                messagebox.showwarning(
                    "Data Truncated",
                    f"CSV file has {len(df)}+ rows and was truncated to {self.MAX_ROWS:,}. "
                    f"Increase FileHandler.MAX_ROWS if you need more."
                )

            if len(df.columns) > self.MAX_COLUMNS:
                messagebox.showerror(
                    "Too Many Columns",
                    f"CSV file has {len(df.columns)} columns (limit: {self.MAX_COLUMNS:,})."
                )
                return None

            return df

        except SecurityError as e:
            error_msg = handle_file_operation_error(e, file_path, "loading CSV file")
            messagebox.showerror("Security Error", f"Error loading CSV file: {error_msg}")
            return None
        except Exception as e:
            error_msg = handle_file_operation_error(e, file_path, "loading CSV file")
            messagebox.showerror("Error", f"Error loading CSV file: {error_msg}")
            return None

    @safe_exception_handler
    def consolidate_company_data_gui(self):
        """GUI for consolidating company data with security validation."""
        try:
            # Select input file
            input_file = self.select_file(
                title="Select financial data file",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls")]
            )
            if not input_file:
                return

            # Validate input file with security checks
            if not self.validate_file_path(input_file):
                return

            # Select output file with secure path validation
            output_file = self.select_save_file(
                title="Save consolidated data",
                defaultextension=".xlsx",
                filetypes=[
                    ("Excel files", "*.xlsx"),
                    ("CSV files", "*.csv")
                ]
            )
            if not output_file:
                return

            # Validate output path securely
            try:
                validated_output = validate_file_path(output_file, allowed_extensions=['.xlsx', '.csv'], create_parent_dirs=True)
            except SecurityError as e:
                error_msg = handle_file_operation_error(e, output_file, "output path validation")
                messagebox.showerror("Security Error", f"Invalid output path: {error_msg}")
                return

            # Show loading indicator
            self.app.ui_manager.show_loading("Consolidating company data...")

            # Execute consolidation in separate thread with validated paths
            future = self.app.analysis_manager.executor.submit(
                self._run_data_consolidation,
                str(validate_file_path(input_file, allowed_extensions=['.xlsx', '.xls', '.csv'])),
                str(validated_output)
            )

            # Configure callback
            future.add_done_callback(self._on_consolidation_complete)

        except SecurityError as e:
            error_msg = handle_file_operation_error(e, "consolidation operation", "security validation")
            self.app.logger.error(error_msg)
            messagebox.showerror("Security Error", f"Security validation failed: {error_msg}")
        except Exception as e:
            error_msg = handle_file_operation_error(e, "consolidation operation", "starting")
            self.app.logger.error(error_msg)
            messagebox.showerror("Error", f"Error starting consolidation: {error_msg}")

    def _run_data_consolidation(self, input_file, output_file):
        """Execute data consolidation (in separate thread)."""
        try:
            from backend.analysis_logic import consolidate_company_data
            result = consolidate_company_data(input_file, output_file)

            if result['status'] == 'success':
                self.app.logger.info("Consolidation completed successfully")
                return result
            else:
                error_msg = result.get('message', 'Unknown consolidation error')
                self.app.logger.error(f"Consolidation error: {error_msg}")
                raise Exception(error_msg)

        except Exception as e:
            self.app.logger.error(f"Consolidation error: {e}")
            raise Exception(f"Consolidation error: {str(e)}")

    def _on_consolidation_complete(self, future):
        """Callback when consolidation completes."""
        try:
            # Update UI
            self.app.after(0, lambda: self._update_ui_after_consolidation(future))
        except Exception as e:
            self.app.logger.error(f"Error in consolidation callback: {e}")

    def _update_ui_after_consolidation(self, future):
        """Update UI after consolidation."""
        try:
            # Hide loading
            self.app.ui_manager.hide_loading()

            # Check result
            if future.exception():
                error_msg = str(future.exception())
                self.app.logger.error(f"Consolidation failed: {error_msg}")
                messagebox.showerror("Consolidation Error", error_msg)
            else:
                result = future.result()
                messagebox.showinfo("Consolidation Completed",
                                  f"{result['message']}\n\n"
                                  f"Saved file: {result['data']['output_file']}\n"
                                  f"Processed companies: {result['data']['companies_processed']}\n"
                                  f"Generated metrics: {result['data']['indicators_processed'] * 2}")

        except Exception as e:
            self.app.logger.error(f"Error updating UI after consolidation: {e}")
            messagebox.showerror("Error", f"Internal error: {str(e)}")