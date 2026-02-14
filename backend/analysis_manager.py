"""
Analysis Manager for PCA Application

Handles analysis execution, async processing, and result management.
Enhanced with progress tracking, cancellation support, and improved GUI state management.
"""

import logging

import pandas as pd
import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.constants import DISABLED, NORMAL, CENTER, X, Y, BOTH, YES
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed, Future, CancelledError
import matplotlib.pyplot as plt
from tkinter import messagebox, filedialog
import os
import webbrowser
from pathlib import Path
import threading
import time
from typing import Dict, Any, Optional, Callable

# Importar módulos de seguridad
from backend.security_utils import secure_temp_file, secure_temp_directory, validate_file_path, SecurityError
from backend.secure_error_handler import handle_file_operation_error, safe_exception_handler

logger = logging.getLogger(__name__)


class AnalysisManager:
    """Manages analysis execution and async processing with enhanced progress tracking."""

    def __init__(self, app):
        self.app = app
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pca_analysis")

        # Enhanced async management
        self.current_future: Optional[Future] = None
        self.is_analysis_running = False
        self.progress_callback: Optional[Callable] = None
        self.cancellation_requested = False

        # Results storage for export functionality
        self.last_analysis_results: Optional[Dict[str, Any]] = None
        self.last_exportable_data: Optional[Dict[str, Any]] = None
        self.last_analysis_summary: Optional[Dict[str, Any]] = None

        # Progress tracking
        self.progress_steps = {
            'series': ['Loading data', 'Preprocessing', 'PCA analysis', 'Visualization'],
            'cross_section': ['Loading data', 'Preprocessing', 'PCA analysis', 'Clustering', 'Visualization'],
            'panel': ['Loading data', 'Preprocessing', 'PCA analysis', '3D visualization'],
            'biplot': ['Loading data', 'Preprocessing', 'PCA analysis', 'Biplot generation'],
            'scatter': ['Loading data', 'Preprocessing', 'PCA analysis', 'Scatter plot'],
            'correlation': ['Loading data', 'Preprocessing', 'Correlation analysis', 'Visualization'],
            'hierarchical': ['Loading data', 'Preprocessing', 'Hierarchical clustering', 'Visualization']
        }

    def run_current_analysis(self):
        """Execute the current analysis in a separate thread with enhanced progress tracking."""
        try:
            # Check if analysis is already running
            if self.is_analysis_running:
                messagebox.showwarning("Advertencia", "Ya hay un análisis en ejecución. Espere a que termine.")
                return

            analysis_type = self.app.current_analysis_type
            config = None

            # Get configuration based on analysis type
            if analysis_type == "series":
                config = self.app.current_series_frame.get_config() if hasattr(self.app, 'current_series_frame') and self.app.current_series_frame else None
                if not config:
                    messagebox.showwarning("Advertencia", "Configura el análisis de series primero")
                    return

            elif analysis_type == "panel":
                config = self.app.panel_frame.get_config() if hasattr(self.app, 'panel_frame') and self.app.panel_frame else None
                if not config:
                    messagebox.showwarning("Advertencia", "Configura el análisis de panel primero")
                    return

            elif analysis_type == "biplot":
                config = self.app.biplot_frame.get_config() if hasattr(self.app, 'biplot_frame') and self.app.biplot_frame else None
                if not config:
                    messagebox.showwarning("Advertencia", "Configura el análisis de biplot primero")
                    return

            elif analysis_type == "scatter":
                config = self.app.scatter_frame.get_config() if hasattr(self.app, 'scatter_frame') and self.app.scatter_frame else None
                if not config:
                    messagebox.showwarning("Advertencia", "Configura el análisis de scatterplot primero")
                    return

            elif analysis_type == "correlation":
                config = self.app.correlation_frame.get_config() if hasattr(self.app, 'correlation_frame') and self.app.correlation_frame else None
                if not config:
                    messagebox.showwarning("Advertencia", "Configura el análisis de correlación primero")
                    return

            elif analysis_type == "hierarchical":
                config = self.app.hierarchical_frame.get_config() if hasattr(self.app, 'hierarchical_frame') and self.app.hierarchical_frame else None
                if not config:
                    messagebox.showwarning("Advertencia", "Configura el análisis de clustering jerárquico primero")
                    return

            else:
                messagebox.showwarning("Advertencia", "Selecciona un tipo de análisis válido")
                return

            # Start async analysis with progress tracking
            self._run_analysis_async_with_progress(analysis_type, config)

        except Exception as e:
            self.app.logger.error(f"Error preparing analysis: {e}")
            messagebox.showerror("Error", f"Error preparing analysis: {str(e)}")

    def _run_analysis_async_with_progress(self, analysis_type: str, config: Dict[str, Any]):
        """Execute analysis method in background thread with progress tracking."""
        try:
            # Set analysis running state
            self.is_analysis_running = True
            self.cancellation_requested = False

            # Disable run button and update UI state
            self._set_ui_state_running()

            # Show enhanced loading indicator with progress
            steps = self.progress_steps.get(analysis_type, ['Processing'])
            self.app.ui_manager.show_loading_with_progress(
                f"Ejecutando análisis de {analysis_type.replace('_', ' ')}...",
                steps
            )

            # Create progress callback
            def progress_callback(step_index: int, step_name: str):
                if hasattr(self.app.ui_manager, 'update_progress'):
                    self.app.after(0, lambda: self.app.ui_manager.update_progress(step_index, step_name))

            # Execute in separate thread with progress tracking
            self.current_future = self.executor.submit(
                self._run_analysis_with_progress_tracking,
                analysis_type,
                config,
                progress_callback
            )

            # Configure callback for when it completes
            self.current_future.add_done_callback(self._on_analysis_complete)

        except Exception as e:
            self.app.logger.error(f"Error starting async analysis: {e}")
            messagebox.showerror("Error", f"Error starting analysis: {str(e)}")
            self._reset_ui_state()

    def _run_analysis_with_progress_tracking(self, analysis_type: str, config: Dict[str, Any],
                                           progress_callback: Callable[[int, str], None]):
        """Run analysis with progress tracking."""
        try:
            # Update progress: Loading data
            progress_callback(0, "Loading data")

            # Get the appropriate analysis method
            analysis_methods = {
                'series': self._run_series_analysis,
                'cross_section': self._run_cross_section_analysis,  # Legacy compat
                'panel': self._run_panel_analysis,
                'biplot': self._run_biplot_analysis,
                'scatter': self._run_scatter_analysis,
                'correlation': self._run_correlation_analysis,
                'hierarchical': self._run_hierarchical_analysis
            }

            analysis_method = analysis_methods.get(analysis_type)
            if not analysis_method:
                raise ValueError(f"Unknown analysis type: {analysis_type}")

            # Check for cancellation
            if self.cancellation_requested:
                raise CancelledError("Analysis cancelled by user")

            # Update progress: Preprocessing
            progress_callback(1, "Preprocessing")

            # Check for cancellation again
            if self.cancellation_requested:
                raise CancelledError("Analysis cancelled by user")

            # Execute the analysis
            result = analysis_method(config)

            # Handle the new "Paquete de Resultados Completo" structure
            if isinstance(result, dict) and 'display_figures' in result and 'exportable_data' in result and 'summary' in result:
                # New package structure - split it up
                paquete_completo = result

                # Store exportable data separately
                self.last_exportable_data = paquete_completo.get('exportable_data', {})
                self.last_analysis_summary = paquete_completo.get('summary', {})

                # Route display figures to UI via callback
                display_figures = paquete_completo.get('display_figures', {})
                if display_figures and hasattr(self.app, 'ui_manager'):
                    # Use main thread callback to update UI with figures
                    self.app.after(0, lambda: self._display_analysis_figures(display_figures))

                # Update progress: Analysis complete
                steps = self.progress_steps.get(analysis_type, [])
                if len(steps) > 2:
                    progress_callback(2, steps[2])

                return paquete_completo
            else:
                # Legacy structure - store as before for backward compatibility
                if isinstance(result, dict) and result.get('status') == 'success':
                    self.last_analysis_results = result.get('data', {})

                # Update progress: Analysis complete
                steps = self.progress_steps.get(analysis_type, [])
                if len(steps) > 2:
                    progress_callback(2, steps[2])

                return result

        except CancelledError:
            raise
        except Exception as e:
            self.app.logger.error(f"Error in analysis execution: {e}")
            raise

    def cancel_current_analysis(self):
        """Cancel the currently running analysis."""
        if self.is_analysis_running and self.current_future:
            self.cancellation_requested = True
            self.current_future.cancel()
            self.app.logger.info("Analysis cancellation requested")

    def _set_ui_state_running(self):
        """Set UI state to indicate analysis is running."""
        try:
            self.app.btn_run.config(state=DISABLED, text="⏳ Ejecutando...")
            if hasattr(self.app, 'toolbar_btn_run') and self.app.toolbar_btn_run:
                self.app.toolbar_btn_run.config(state=DISABLED)
            # Disable other action buttons during analysis
            if hasattr(self.app, 'btn_export'):
                self.app.btn_export.config(state=DISABLED)
                if hasattr(self.app, 'toolbar_btn_export') and self.app.toolbar_btn_export:
                    self.app.toolbar_btn_export.config(state=DISABLED)
            if hasattr(self.app, 'btn_groups'):
                self.app.btn_groups.config(state=DISABLED)
        except Exception as e:
            self.app.logger.warning(f"Error setting UI state: {e}")

    def _reset_ui_state(self):
        """Reset UI state after analysis completion."""
        try:
            self.is_analysis_running = False
            self.current_future = None
            self.cancellation_requested = False

            self.app.btn_run.config(state=NORMAL, text=f"{self.app.ui_manager.icon_map['run']} Ejecutar Análisis")
            if hasattr(self.app, 'toolbar_btn_run') and self.app.toolbar_btn_run:
                self.app.toolbar_btn_run.config(state=NORMAL)

            # Re-enable other buttons
            if hasattr(self.app, 'btn_export'):
                self.app.btn_export.config(state=NORMAL)
            if hasattr(self.app, 'toolbar_btn_export') and self.app.toolbar_btn_export:
                self.app.toolbar_btn_export.config(state=NORMAL)
            if hasattr(self.app, 'btn_groups'):
                self.app.btn_groups.config(state=NORMAL)

            # Hide loading indicator
            self.app.ui_manager.hide_loading()
        except Exception as e:
            self.app.logger.warning(f"Error resetting UI state: {e}")

    def get_last_analysis_results(self) -> Optional[Dict[str, Any]]:
        """Get the results from the last analysis for export functionality."""
        return self.last_analysis_results

    def get_last_exportable_data(self) -> Optional[Dict[str, Any]]:
        """Get the exportable data from the last analysis."""
        return self.last_exportable_data

    def get_last_analysis_summary(self) -> Optional[Dict[str, Any]]:
        """Get the summary from the last analysis."""
        return self.last_analysis_summary

    def _display_analysis_figures(self, display_figures: Dict[str, Any]):
        """Display analysis figures in the UI."""
        try:
            if not hasattr(self.app, 'ui_manager'):
                self.logger.warning("UI manager not available for displaying figures")
                return

            # Display the main plot if available
            if 'main_plot' in display_figures:
                fig = display_figures['main_plot']
                if fig is not None:
                    self.app.ui_manager.display_figure(fig)
                    self.logger.info("Analysis figure displayed successfully")
                else:
                    self.logger.warning("Main plot figure is None")

            # Handle secondary plots if needed
            for plot_name, fig in display_figures.items():
                if plot_name != 'main_plot' and fig is not None:
                    # Could extend UI manager to handle multiple figures
                    self.logger.info(f"Additional plot '{plot_name}' available but not displayed")

        except Exception as e:
            self.logger.error(f"Error displaying analysis figures: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _on_analysis_complete(self, future):
        """Callback when analysis completes."""
        try:
            # Update UI in main thread
            self.app.after(0, lambda: self._update_ui_after_analysis(future))
        except Exception as e:
            self.app.logger.error(f"Error in analysis callback: {e}")

    def _update_ui_after_analysis(self, future):
        """Update UI after analysis completes."""
        try:
            # Reset UI state first
            self._reset_ui_state()

            # Check for cancellation
            if future.cancelled():
                self.app.logger.info("Analysis was cancelled")
                messagebox.showinfo("Cancelled", "Analysis was cancelled by user")
                return

            # Check for exceptions
            if future.exception():
                exception = future.exception()
                if isinstance(exception, CancelledError):
                    self.app.logger.info("Analysis cancelled via exception")
                    messagebox.showinfo("Cancelled", "Analysis was cancelled")
                else:
                    error_msg = str(exception)
                    self.app.logger.error(f"❌ Analysis failed with exception: {error_msg}", exc_info=True)
                    # Show full traceback to user
                    import traceback
                    full_trace = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
                    messagebox.showerror("Error in Analysis", 
                                       f"Analysis failed:\n\n{error_msg}\n\nCheck console for full traceback")
                    logger.error("FULL ERROR TRACEBACK:\n%s", full_trace)
            else:
                # Get results and validate structure
                results = future.result()
                
                # Log results structure for debugging
                self.app.logger.info(f"📦 Results received. Type: {type(results)}")
                if isinstance(results, dict):
                    self.app.logger.info(f"📋 Results keys: {list(results.keys())}")
                else:
                    self.app.logger.error(f"❌ Results is not a dict! Type: {type(results)}, Value: {results}")
                    messagebox.showerror("Error", f"Invalid results format: expected dict, got {type(results)}")
                    return
                
                # Check for success status
                if results and results.get('status') == 'success':
                    self.app.last_results = results
                    
                    # Store exportable data if present (new package format)
                    if 'exportable_data' in results and 'summary' in results:
                        self.last_exportable_data = results.get('exportable_data', {})
                        self.last_analysis_summary = results.get('summary', {})
                        self.app.logger.info(f"📊 Stored exportable data with {len(self.last_exportable_data)} sheets")
                        self.app.logger.info(f"📋 Summary: {self.last_analysis_summary}")

                    # Handle visualization on main thread
                    self._show_visualization(results)

                    self.app.logger.info("✅ Analysis completed successfully")
                    messagebox.showinfo("Success", "Analysis completed successfully")
                    self.app.btn_export.config(state=NORMAL)
                    if hasattr(self.app, 'toolbar_btn_export') and self.app.toolbar_btn_export:
                        self.app.toolbar_btn_export.config(state=NORMAL)
                else:
                    # Check if it's an error result
                    if results and results.get('status') == 'error':
                        error_msg = results.get('message', 'Unknown error')
                        self.app.logger.error(f"❌ Analysis returned error status: {error_msg}")
                        messagebox.showerror("Analysis Error", error_msg)
                    else:
                        self.app.logger.warning(f"⚠️  Analysis completed but returned no results or invalid status. Results: {results}")
                        messagebox.showwarning("Warning", 
                                             f"Analysis completed but returned unexpected results.\n\n"
                                             f"Status: {results.get('status', 'MISSING')}\n"
                                             f"Keys: {list(results.keys()) if isinstance(results, dict) else 'NOT A DICT'}")

        except Exception as e:
            self.app.logger.error(f"❌ Error updating UI after analysis: {e}", exc_info=True)
            messagebox.showerror("Error", f"Internal error: {str(e)}\n\nCheck console for details")
            import traceback
            traceback.print_exc()
            self._reset_ui_state()

    def _show_visualization(self, result):
        """Show visualization on main thread."""
        try:
            analysis_type = self.app.current_analysis_type
            self.app.logger.info(f"🎨 Showing visualization for: {analysis_type}")
            
            # ✅ NUEVA ARQUITECTURA: Check for visualization_function_name format
            if 'visualization_function_name' in result and 'visualization_params' in result:
                self.app.logger.info("📊 Using NEW visualization_function_name architecture")
                viz_function_name = result.get('visualization_function_name')
                viz_params = result.get('visualization_params', {})
                
                self.app.logger.info(f"🎯 Calling visualization function: {viz_function_name}")
                self.app.logger.info(f"📋 Parameters keys: {list(viz_params.keys())}")
                
                # Dispatcher: Call the appropriate show_* function
                try:
                    # Import all visualization modules with fallback for biplot
                    import visualization_module as dl_viz
                    import scatter_plot
                    
                    # Try to import biplot_advanced, fallback to biplot_simple
                    try:
                        import biplot_advanced
                        biplot_module = biplot_advanced
                        self.app.logger.info("📦 Using biplot_advanced module")
                    except (ImportError, UnicodeEncodeError) as e:
                        self.app.logger.warning(f"⚠️  biplot_advanced import failed: {e}")
                        try:
                            import biplot_simple
                            biplot_module = biplot_simple
                            self.app.logger.info("📦 Using biplot_simple module (fallback)")
                        except ImportError as e2:
                            self.app.logger.error(f"❌ Both biplot modules failed to import: {e2}")
                            biplot_module = None
                    
                    # Find the function in the appropriate module
                    if hasattr(dl_viz, viz_function_name):
                        viz_func = getattr(dl_viz, viz_function_name)
                    elif biplot_module and hasattr(biplot_module, viz_function_name):
                        viz_func = getattr(biplot_module, viz_function_name)
                    elif hasattr(scatter_plot, viz_function_name):
                        viz_func = getattr(scatter_plot, viz_function_name)
                    else:
                        raise AttributeError(f"Visualization function '{viz_function_name}' not found in any module")
                    
                    self.app.logger.info(f"✅ Found function: {viz_func}")
                    
                    # Call the function with parameters in main thread
                    # This will open an interactive matplotlib window with plt.show()
                    viz_func(**viz_params)
                    self.app.logger.info("✅ Visualization function executed successfully")
                    
                except Exception as viz_error:
                    self.app.logger.error(f"❌ Error calling visualization function: {viz_error}", exc_info=True)
                    messagebox.showerror("Visualization Error", 
                                       f"Failed to show visualization:\n\n{str(viz_error)}\n\nCheck console for full traceback")
                    import traceback
                    traceback.print_exc()
            
            # Check for old package format with display_figures (LEGACY)
            elif 'display_figures' in result:
                self.app.logger.warning("⚠️  Using OLD display_figures format (should be migrated)")
                display_figures = result.get('display_figures', {})
                if 'main_plot' in display_figures:
                    fig = display_figures['main_plot']
                    if fig is not None:
                        self.app.logger.info(f"✅ Got main_plot figure: {type(fig)}")
                        # Display the figure using UIManager (OLD embedded approach)
                        if hasattr(self.app, 'ui_manager') and hasattr(self.app.ui_manager, 'display_figure'):
                            self.app.ui_manager.display_figure(fig)
                        else:
                            # Fallback: show with matplotlib
                            import matplotlib.pyplot as plt
                            plt.figure(fig.number)
                            plt.show()
                    else:
                        self.app.logger.warning("⚠️  main_plot figure is None!")
                else:
                    self.app.logger.warning(f"⚠️  No 'main_plot' in display_figures. Keys: {list(display_figures.keys())}")
            
            # Legacy format - get data dict and call specific visualization
            elif 'data' in result:
                self.app.logger.warning("⚠️  Using VERY OLD data format (should be migrated)")
                data = result.get('data', {})
                
                if analysis_type == 'cross_section':
                    self._show_cross_section_visualization(data)
                elif analysis_type == 'series':
                    self._show_series_visualization(data)
                elif analysis_type == 'panel':
                    self._show_panel_visualization(data)
                elif analysis_type == 'biplot':
                    self._show_biplot_visualization(data)
                elif analysis_type == 'scatter':
                    self._show_scatter_visualization(data)
                elif analysis_type == 'correlation':
                    self._show_correlation_visualization(data)
                elif analysis_type == 'hierarchical':
                    self._show_hierarchical_visualization(data)
            else:
                self.app.logger.error(f"❌ Unknown result format! Keys: {list(result.keys())}")

        except Exception as e:
            self.app.logger.error(f"❌ Error showing visualization: {e}", exc_info=True)
            import traceback
            traceback.print_exc()

    def _show_cross_section_visualization(self, data):
        """Show cross-section analysis visualization using biplot_simple."""
        try:
            # ✅ NUEVO: Usar biplot_simple.py que acepta el formato correcto de grupos
            from backend.biplot_simple import create_advanced_biplot_simple
            
            pca_model = data.get('pca_model')
            df_componentes = data.get('components')
            df_estandarizado = data.get('standardized_data')
            indicators = data.get('indicators', [])
            countries = data.get('countries', [])
            year = data.get('year', 'Unknown')
            config = data.get('config', {})
            
            # Obtener información de grupos de la configuración
            groups = data.get('groups', {})
            group_colors = data.get('group_colors', {})
            arrow_scale = config.get('arrow_scale', None)

            if pca_model and df_estandarizado is not None and not df_estandarizado.empty:
                self.app.logger.info(f"📊 Visualizando biplot con {len(countries)} unidades")
                
                if groups and group_colors:
                    self.app.logger.info(f"🎨 Usando {len(set(groups.values()))} grupos configurados")
                    self.app.logger.info(f"Grupos: {set(groups.values())}")
                else:
                    self.app.logger.info("⚪ Sin grupos configurados")
                
                # Configuración para biplot_simple
                biplot_config = {
                    'year': year,
                    'show_arrows': True,
                    'show_labels': True,
                    'alpha': 0.7,
                    'arrow_scale': arrow_scale,
                    'groups': groups,  # ✅ Formato correcto: {unit: group_name}
                    'group_colors': group_colors  # ✅ Formato correcto: {group_name: color}
                }
                
                # Llamar a la función de biplot simple
                success = create_advanced_biplot_simple(df_estandarizado, biplot_config)
                
                if success:
                    self.app.logger.info("✅ Biplot generado exitosamente")
                else:
                    self.app.logger.error("❌ Error al generar biplot")
                    
        except Exception as e:
            self.app.logger.error(f"Error in cross-section visualization: {e}")
            import traceback
            traceback.print_exc()

    def _show_series_visualization(self, data):
        """Show series analysis visualization."""
        try:
            from backend.visualization_module import graficar_componentes_principales_tiempo

            df_componentes = data.get('components')
            pca_model = data.get('pca_model')
            country = data.get('country', 'Unknown')

            if not df_componentes.empty and pca_model:
                varianza_explicada = pca_model.explained_variance_ratio_
                graficar_componentes_principales_tiempo(
                    df_componentes=df_componentes,
                    varianza_explicada=varianza_explicada,
                    titulo=f"Evolución de Componentes Principales - {country}"
                )
        except Exception as e:
            self.app.logger.warning(f"Error in series visualization: {e}")

    def _show_panel_visualization(self, data):
        """Show panel analysis visualization."""
        try:
            from backend.visualization_module import graficar_trayectorias_3d

            df_pc_scores = data.get('pc_scores')
            pca_model = data.get('pca_model')
            countries = data.get('countries', [])

            if not df_pc_scores.empty and pca_model:
                # Crear grupos por defecto
                grupos_paises = {pais: 'Grupo Principal' for pais in countries}
                mapa_colores = {'Grupo Principal': '#1f77b4'}

                graficar_trayectorias_3d(
                    df_pc_scores=df_pc_scores,
                    pca_model=pca_model,
                    grupos_paises=grupos_paises,
                    mapa_de_colores=mapa_colores,
                    titulo="Trayectorias 3D - Análisis Panel"
                )
        except Exception as e:
            self.app.logger.warning(f"Error in panel visualization: {e}")

    def _show_biplot_visualization(self, data):
        """Show biplot analysis visualization."""
        try:
            from backend.visualization_module import graficar_biplot_corte_transversal

            pca_model = data.get('pca_model')
            df_componentes = data.get('components')
            biplot_config = data.get('biplot_config', {})

            if pca_model and not df_componentes.empty:
                # Usar configuración del biplot o valores por defecto
                grupos_paises = biplot_config.get('grupos_paises', ['Grupo Principal'] * len(df_componentes))
                mapa_colores = biplot_config.get('mapa_colores', {'Grupo Principal': '#1f77b4'})
                indicators = biplot_config.get('indicators', [f'Ind{i+1}' for i in range(len(pca_model.components_[0]))])
                countries = biplot_config.get('countries', [f'Obs{i+1}' for i in range(len(df_componentes))])

                graficar_biplot_corte_transversal(
                    pca_model=pca_model,
                    df_pc_scores=df_componentes,
                    nombres_indicadores_originales=indicators,
                    nombres_indicadores_etiquetas=indicators,
                    nombres_individuos_etiquetas=countries,
                    grupos_individuos=grupos_paises,
                    mapa_de_colores=mapa_colores,
                    titulo="Biplot Avanzado",
                    pc_x=0,
                    pc_y=1
                )
        except Exception as e:
            self.app.logger.warning(f"Error in biplot visualization: {e}")

    def _show_scatter_visualization(self, data):
        """Show scatter plot analysis visualization."""
        try:
            from backend.scatter_plot import generate_scatter_plot

            df_estandarizado = data.get('standardized_data')
            pca_model = data.get('pca_model')
            scatter_config = data.get('scatter_config', {})

            if not df_estandarizado.empty and pca_model:
                countries = list(df_estandarizado.index)
                generate_scatter_plot(
                    df_estandarizado, countries, scatter_config, pca_model
                )
        except Exception as e:
            self.app.logger.warning(f"Error in scatter visualization: {e}")

    def _show_correlation_visualization(self, data):
        """Show correlation analysis visualization."""
        try:
            similarity_matrix = data.get('similarity_matrix')
            visualization_type = data.get('visualization_type', 'heatmap')

            if similarity_matrix is None:
                self.app.logger.warning("No similarity matrix available for visualization")
                return

            if visualization_type == 'heatmap':
                heatmap_config = data.get('heatmap_config', {})
                is_interactive = heatmap_config.get('interactive', False)

                if is_interactive:
                    # Create interactive heatmap
                    try:
                        from heatmap_visualization import HeatmapVisualizer
                        visualizer = HeatmapVisualizer()
                        fig = visualizer.create_interactive_heatmap(
                            similarity_matrix,
                            title='Interactive Correlation Heatmap',
                            config=heatmap_config
                        )

                        # Save to secure temporary HTML file and open in browser
                        try:
                            # Create secure temp file
                            temp_file = secure_temp_file(suffix='.html', prefix='pca_heatmap_')

                            fig.write_html(str(temp_file), include_plotlyjs='cdn', full_html=True)
                            webbrowser.open(f'file://{temp_file}')

                            # Track temporary file for cleanup
                            if not hasattr(self.app, 'temp_files'):
                                self.app.temp_files = []
                            self.app.temp_files.append(str(temp_file))

                            messagebox.showinfo("Interactive Heatmap",
                                              "Interactive heatmap opened in your default web browser.\n"
                                              "The temporary file will be cleaned up when you close the application.")

                        except SecurityError as e:
                            error_msg = handle_file_operation_error(e, "temp file creation", "secure heatmap generation")
                            self.app.logger.error(error_msg)
                            messagebox.showerror("Security Error", f"Could not create secure temp file: {error_msg}")
                        except Exception as e:
                            error_msg = handle_file_operation_error(e, "interactive heatmap", "opening in browser")
                            self.app.logger.error(error_msg)
                            messagebox.showerror("Error", f"Could not open interactive heatmap: {error_msg}")

                    except ImportError:
                        self.app.logger.warning("Plotly not available for interactive heatmap, falling back to static")
                        # Fallback to static heatmap
                        from heatmap_visualization import HeatmapVisualizer
                        visualizer = HeatmapVisualizer()
                        fig = visualizer.create_heatmap(similarity_matrix, config=heatmap_config)
                        plt.show()
                else:
                    # Create static heatmap
                    from heatmap_visualization import HeatmapVisualizer
                    visualizer = HeatmapVisualizer()
                    fig = visualizer.create_heatmap(similarity_matrix, config=heatmap_config)
                    plt.show()

            elif visualization_type == 'network':
                try:
                    from network_visualization import create_correlation_network
                    network_config = data.get('network_config', {})
                    graph = create_correlation_network(similarity_matrix, network_config)
                    
                    # Store the graph in the app for degree distribution analysis
                    self.app.current_network_graph = graph
                    
                    # Update button states
                    if hasattr(self.app, 'update_network_graph_reference'):
                        self.app.update_network_graph_reference(graph)
                    
                    # Create interactive visualization
                    from network_visualization import create_interactive_network
                    fig = create_interactive_network(graph, network_config)

                    # Save to secure temporary HTML file and open in browser
                    import webbrowser

                    try:
                        # Create secure temp file
                        temp_file = secure_temp_file(suffix='.html', prefix='pca_network_')

                        fig.write_html(str(temp_file), include_plotlyjs='cdn', full_html=True)
                        webbrowser.open(f'file://{temp_file}')

                        # Track temporary file for cleanup
                        self.app.temp_files.append(str(temp_file))

                        messagebox.showinfo("Network Visualization",
                                          "Network visualization opened in your default web browser.\n"
                                          "The temporary file will be cleaned up when you close the application.")

                    except SecurityError as e:
                        error_msg = handle_file_operation_error(e, "temp file creation", "secure network visualization")
                        self.app.logger.error(error_msg)
                        messagebox.showerror("Security Error", f"Could not create secure temp file: {error_msg}")
                    except Exception as e:
                        error_msg = handle_file_operation_error(e, "network visualization", "opening in browser")
                        self.app.logger.error(error_msg)
                        messagebox.showerror("Error", f"Could not open network visualization: {error_msg}")
                except ImportError as e:
                    self.app.logger.warning(f"Network visualization not available: {e}")
                    messagebox.showwarning(
                        "Network Visualization Unavailable",
                        "Network visualization requires NetworkX and Plotly libraries.\n"
                        "Please install them with: pip install networkx plotly\n"
                        "Falling back to heatmap visualization."
                    )
                    # Fallback to heatmap
                    from heatmap_visualization import HeatmapVisualizer
                    heatmap_config = data.get('heatmap_config', {})
                    visualizer = HeatmapVisualizer()
                    fig = visualizer.create_heatmap(similarity_matrix, config=heatmap_config)
                    plt.show()
                except Exception as e:
                    self.app.logger.error(f"Error in network visualization: {e}")
                    messagebox.showerror("Network Visualization Error", f"Failed to create network visualization: {str(e)}")

            elif visualization_type == 'hierarchical':
                # Show hierarchical network visualization
                self._show_hierarchical_visualization(data)

            elif visualization_type == 'both':
                # Show both visualizations
                from heatmap_visualization import HeatmapVisualizer
                heatmap_config = data.get('heatmap_config', {})
                network_config = data.get('network_config', {})

                # Create heatmap
                visualizer = HeatmapVisualizer()
                fig = visualizer.create_heatmap(similarity_matrix, config=heatmap_config)
                plt.show()

                # Create network with error handling
                try:
                    from network_visualization import create_correlation_network
                    graph = create_correlation_network(similarity_matrix, network_config)
                    
                    # Store the graph in the app for degree distribution analysis
                    self.app.current_network_graph = graph
                    
                    # Update button states
                    if hasattr(self.app, 'update_network_graph_reference'):
                        self.app.update_network_graph_reference(graph)
                    
                    # Create interactive visualization
                    from network_visualization import create_interactive_network
                    fig_network = create_interactive_network(graph, network_config)
                    fig_network.show()
                except ImportError as e:
                    self.app.logger.warning(f"Network visualization not available for 'both' mode: {e}")
                    messagebox.showinfo(
                        "Network Visualization Skipped",
                        "Network visualization requires NetworkX and Plotly libraries.\n"
                        "Heatmap has been displayed. Install additional libraries for network graphs:\n"
                        "pip install networkx plotly"
                    )
                except Exception as e:
                    self.app.logger.error(f"Error in network visualization for 'both' mode: {e}")
                    messagebox.showwarning("Network Visualization Error", f"Network visualization failed: {str(e)}\nHeatmap has been displayed.")

        except Exception as e:
            self.app.logger.warning(f"Error in correlation visualization: {e}")

    def _show_hierarchical_visualization(self, data):
        """Show hierarchical network visualization with drill-down capabilities."""
        try:
            similarity_matrix = data.get('similarity_matrix')
            if similarity_matrix is None:
                messagebox.showwarning("Warning", "No similarity matrix available for hierarchical visualization")
                return

            # Create network from correlation matrix
            from network_visualization import create_correlation_network, HierarchicalNetworkVisualizer

            # Get network config
            network_config = data.get('network_config', {})

            # Create the network graph
            graph_result = create_correlation_network(similarity_matrix, network_config)
            graph = graph_result['graph'] if isinstance(graph_result, dict) else graph_result

            # Detect communities
            visualizer = HierarchicalNetworkVisualizer()
            communities = visualizer.detect_communities(graph, method='auto')

            # Create hierarchical visualization
            fig = visualizer.create_hierarchical_network(graph, communities, network_config)

            # Save to secure temporary HTML file and open in browser
            import webbrowser

            try:
                # Create secure temp file
                temp_file = secure_temp_file(suffix='.html', prefix='pca_hierarchical_')

                fig.write_html(str(temp_file), include_plotlyjs='cdn', full_html=True)
                webbrowser.open(f'file://{temp_file}')

                # Track temporary file for cleanup
                if not hasattr(self.app, 'temp_files'):
                    self.app.temp_files = []
                self.app.temp_files.append(str(temp_file))

                messagebox.showinfo("Hierarchical Network View",
                                  "Hierarchical network visualization opened in your default web browser.\n"
                                  "Shows communities as meta-nodes. Click on a community to explore its internal structure.\n"
                                  "The temporary file will be cleaned up when you close the application.")

            except SecurityError as e:
                error_msg = handle_file_operation_error(e, "temp file creation", "secure hierarchical visualization")
                self.app.logger.error(error_msg)
                messagebox.showerror("Security Error", f"Could not create secure temp file: {error_msg}")
            except Exception as e:
                error_msg = handle_file_operation_error(e, "hierarchical visualization", "opening in browser")
                self.app.logger.error(error_msg)
                messagebox.showerror("Error", f"Could not open hierarchical visualization: {error_msg}")

        except ImportError as e:
            self.app.logger.warning(f"Hierarchical visualization not available: {e}")
            messagebox.showinfo(
                "Hierarchical Visualization Skipped",
                "Hierarchical network visualization requires NetworkX and Plotly libraries.\n"
                "Showing heatmap instead."
            )
            # Fallback to heatmap
            similarity_matrix = data.get('similarity_matrix')
            if similarity_matrix is not None:
                from heatmap_visualization import create_correlation_heatmap
                fig = create_correlation_heatmap(similarity_matrix)
                plt.show()

        except Exception as e:
            self.app.logger.error(f"Error in hierarchical visualization: {e}")
            messagebox.showerror("Hierarchical Visualization Error", f"Failed to create hierarchical visualization: {str(e)}")

    @safe_exception_handler
    def export_results(self):
        """Export analysis results with secure file handling and enhanced functionality."""
        if not self.app.last_results:
            messagebox.showinfo("Export", "No results to export. Run an analysis first.")
            return

        # Create export dialog with multiple options
        self._show_export_dialog()

    def _show_export_dialog(self):
        """Show export options dialog with Excel export and clipboard copy."""
        # Get results from AnalysisManager - use exportable_data for new structure
        results_data = self.get_last_exportable_data()

        if not results_data:
            messagebox.showwarning("Advertencia", "No hay resultados de análisis disponibles para exportar. Ejecuta un análisis primero.")
            return

        dialog = ttk.Toplevel(self.app)
        dialog.title("📤 Exportar Resultados")
        dialog.geometry("450x300")
        dialog.resizable(False, False)
        dialog.transient(self.app)
        dialog.grab_set()

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 225
        y = (dialog.winfo_screenheight() // 2) - 150
        dialog.geometry(f"450x300+{x}+{y}")

        # Title
        title_label = ttk.Label(
            dialog,
            text="📤 Exportar Resultados del Análisis",
            font=("Helvetica", 14, "bold"),
            style='primary.TLabel'
        )
        title_label.pack(pady=(20, 10))

        # Analysis type info
        analysis_type = self.app.current_analysis_type
        type_label = ttk.Label(
            dialog,
            text=f"Tipo de análisis: {analysis_type.replace('_', ' ').title()}",
            font=("Helvetica", 10),
            style='secondary.TLabel'
        )
        type_label.pack(pady=(0, 20))

        # Export options frame
        options_frame = ttk.LabelFrame(dialog, text="Opciones de Exportación", padding=15)
        options_frame.pack(fill=X, padx=20, pady=(0, 20))

        # Excel export button
        excel_btn = ttk.Button(
            options_frame,
            text="📊 Exportar a Excel (Múltiples Hojas)",
            style='success.TButton',
            command=lambda: self._export_to_excel(dialog)
        )
        excel_btn.pack(fill=X, pady=(0, 10))

        # Clipboard copy button
        clipboard_btn = ttk.Button(
            options_frame,
            text="📋 Copiar Datos al Portapapeles",
            style='info.TButton',
            command=lambda: self._show_clipboard_dialog(dialog)
        )
        clipboard_btn.pack(fill=X, pady=(0, 10))

        # Cancel button
        cancel_btn = ttk.Button(
            options_frame,
            text="❌ Cancelar",
            style='secondary.Outline.TButton',
            command=dialog.destroy
        )
        cancel_btn.pack(fill=X)

        # Info text
        info_label = ttk.Label(
            dialog,
            text="💡 La exportación a Excel incluye todos los datos procesados\n"
                 "en múltiples hojas organizadas por tipo de información.",
            font=("Helvetica", 8),
            style='secondary.TLabel',
            justify=CENTER
        )
        info_label.pack(pady=(10, 0))

    def _export_to_excel(self, parent_dialog):
        """Handle Excel export with enhanced functionality using stored exportable data."""
        parent_dialog.destroy()

        results_data = self.get_last_exportable_data()
        analysis_summary = self.get_last_analysis_summary()

        if not results_data:
            messagebox.showwarning("Advertencia", "No hay datos disponibles para exportar.")
            return

        filename = self.app.file_handler.select_save_file(
            title="Guardar resultados del análisis",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx *.xls")],
        )

        if filename:
            try:
                # Validate output path securely
                validated_output = validate_file_path(filename, allowed_extensions=['.xlsx', '.xls'], create_parent_dirs=True)

                import pandas as pd
                sheets_created = 0

                with pd.ExcelWriter(str(validated_output)) as writer:
                    # Always create summary sheet first using analysis summary
                    sheets_created += self._export_analysis_summary(results_data, analysis_summary, writer, self.app.current_analysis_type)

                    # Export all data from the exportable_data dictionary dynamically
                    for sheet_name, data in results_data.items():
                        try:
                            if isinstance(data, pd.DataFrame) and not data.empty:
                                data.to_excel(writer, sheet_name=sheet_name, index=True)
                                sheets_created += 1
                                self.app.logger.info(f"Exported sheet: {sheet_name}")
                            elif isinstance(data, dict):
                                # Handle nested dictionaries by converting to DataFrame
                                try:
                                    df_from_dict = pd.DataFrame([data]) if not isinstance(list(data.values())[0], (list, dict)) else pd.DataFrame(data)
                                    if not df_from_dict.empty:
                                        df_from_dict.to_excel(writer, sheet_name=sheet_name, index=False)
                                        sheets_created += 1
                                        self.app.logger.info(f"Exported sheet from dict: {sheet_name}")
                                except Exception as e:
                                    self.app.logger.warning(f"Could not export dict data for {sheet_name}: {e}")
                        except Exception as e:
                            self.app.logger.warning(f"Could not export {sheet_name}: {e}")

                if sheets_created > 0:
                    messagebox.showinfo("✅ Exportación Exitosa",
                                      f"Archivo guardado exitosamente:\n{validated_output}\n\n"
                                      f"Se crearon {sheets_created} hojas de Excel con todos los datos del análisis.")
                else:
                    messagebox.showwarning("Advertencia", "No se pudieron exportar datos. Verifica que el análisis se completó correctamente.")

            except SecurityError as e:
                error_msg = handle_file_operation_error(e, filename, "secure export")
                messagebox.showerror("Error de Seguridad", f"Exportación fallida: {error_msg}")
            except Exception as e:
                error_msg = handle_file_operation_error(e, filename, "exporting results")
                messagebox.showerror("Error", f"Error al exportar resultados: {error_msg}")

    def _show_clipboard_dialog(self, parent_dialog):
        """Show dialog to select which data to copy to clipboard using stored exportable data."""
        parent_dialog.destroy()

        results_data = self.get_last_exportable_data()
        if not results_data:
            messagebox.showwarning("Advertencia", "No hay datos disponibles para copiar al portapapeles.")
            return

        # Get available data options from the exportable_data dictionary
        available_data = self._get_available_data_options_from_results(results_data)

        if not available_data:
            messagebox.showwarning("Advertencia", "No hay datos disponibles para copiar al portapapeles.")
            return

        # Create clipboard selection dialog
        dialog = ttk.Toplevel(self.app)
        dialog.title("📋 Copiar al Portapapeles")
        dialog.geometry("400x350")
        dialog.resizable(False, False)
        dialog.transient(self.app)
        dialog.grab_set()

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 200
        y = (dialog.winfo_screenheight() // 2) - 175
        dialog.geometry(f"400x350+{x}+{y}")

        # Title
        title_label = ttk.Label(
            dialog,
            text="Seleccionar Datos para Copiar",
            font=("Helvetica", 12, "bold"),
            style='primary.TLabel'
        )
        title_label.pack(pady=(20, 10))

        # Instructions
        instr_label = ttk.Label(
            dialog,
            text="Elige qué conjunto de datos deseas copiar al portapapeles:",
            font=("Helvetica", 9),
            style='secondary.TLabel',
            wraplength=350,
            justify=CENTER
        )
        instr_label.pack(pady=(0, 15))

        # Data selection frame
        selection_frame = ttk.LabelFrame(dialog, text="Datos Disponibles", padding=10)
        selection_frame.pack(fill=BOTH, expand=YES, padx=20, pady=(0, 20))

        # Listbox for data selection
        listbox_frame = ttk.Frame(selection_frame)
        listbox_frame.pack(fill=BOTH, expand=YES)

        listbox = tk.Listbox(
            listbox_frame,
            selectmode=tk.SINGLE,
            font=("Helvetica", 10),
            height=8
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=YES)

        # Scrollbar
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.config(yscrollcommand=scrollbar.set)

        # Populate listbox
        for data_name, data_desc in available_data.items():
            listbox.insert(tk.END, f"{data_name}: {data_desc}")

        # Buttons frame
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(fill=X, padx=20, pady=(0, 20))

        def copy_selected():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("Advertencia", "Por favor selecciona un conjunto de datos.")
                return

            selected_index = selection[0]
            selected_key = list(available_data.keys())[selected_index]

            try:
                self._copy_data_to_clipboard_from_results(selected_key, results_data)
                dialog.destroy()
                messagebox.showinfo("✅ Copiado",
                                  f"Los datos '{selected_key}' han sido copiados al portapapeles.\n\n"
                                  "Puedes pegarlos en Excel, Google Sheets u otras aplicaciones.")
            except Exception as e:
                messagebox.showerror("Error", f"Error al copiar datos: {str(e)}")

        copy_btn = ttk.Button(
            buttons_frame,
            text="📋 Copiar Seleccionado",
            style='success.TButton',
            command=copy_selected
        )
        copy_btn.pack(side=tk.LEFT, padx=(0, 10), expand=YES)

        cancel_btn = ttk.Button(
            buttons_frame,
            text="❌ Cancelar",
            style='secondary.Outline.TButton',
            command=dialog.destroy
        )
        cancel_btn.pack(side=tk.RIGHT, expand=YES)

    def _get_available_data_options_from_results(self, results_data):
        """Get available data options for clipboard copy from results dictionary."""
        if not results_data:
            return {}

        options = {}

        # Dynamically populate options from results data
        for key, data in results_data.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                # Create user-friendly descriptions
                if 'Originales' in key or 'Datos_Originales' in key:
                    options[key] = 'Datos sin procesar del análisis'
                elif 'Estandarizados' in key or 'Datos_Estandarizados' in key:
                    options[key] = 'Datos normalizados para el análisis'
                elif 'Componentes' in key or 'PCA' in key:
                    options[key] = 'Puntuaciones de componentes principales'
                elif 'Cargas' in key or 'Loadings' in key:
                    options[key] = 'Cargas de las variables en componentes'
                elif 'Varianza' in key:
                    options[key] = 'Explicación de varianza por componente'
                elif 'Correlacion' in key or 'Matriz' in key:
                    options[key] = 'Matriz de similitud/correlación'
                elif 'Clusters' in key or 'Asignacion' in key:
                    options[key] = 'Grupos identificados por clustering'
                elif 'Configuracion' in key:
                    options[key] = 'Parámetros y configuración del análisis'
                elif 'Estadisticas' in key:
                    options[key] = 'Estadísticas del análisis'
                else:
                    options[key] = f'Datos de {key.replace("_", " ").lower()}'

        return options

    def _copy_data_to_clipboard_from_results(self, data_key, results_data):
        """Copy selected data to clipboard from results dictionary."""
        if data_key not in results_data:
            raise ValueError(f"No se encontraron datos para: {data_key}")

        df_to_copy = results_data[data_key]

        if not isinstance(df_to_copy, pd.DataFrame):
            # Try to convert to DataFrame if it's a dict or other format
            try:
                if isinstance(df_to_copy, dict):
                    df_to_copy = pd.DataFrame([df_to_copy]) if not isinstance(list(df_to_copy.values())[0], (list, dict)) else pd.DataFrame(df_to_copy)
                else:
                    df_to_copy = pd.DataFrame(df_to_copy)
            except Exception as e:
                raise ValueError(f"No se pudo convertir los datos a formato tabular: {e}")

        if df_to_copy is None or df_to_copy.empty:
            raise ValueError(f"Los datos para '{data_key}' están vacíos")

        # Copy to clipboard
        df_to_copy.to_clipboard(index=True)
        self.app.logger.info(f"Data '{data_key}' copied to clipboard")

    def _export_cross_section_results(self, writer):
        """Export cross-section analysis results."""
        from backend.constants import MAPEO_INDICADORES
        import numpy as np

        if 'cross_section_data' in self.app.last_results:
            self.app.last_results['cross_section_data'].rename(
                columns=MAPEO_INDICADORES
            ).to_excel(writer, sheet_name="Original", index=True)

        if 'standardized_data' in self.app.last_results:
            self.app.last_results['standardized_data'].rename(columns=MAPEO_INDICADORES).to_excel(
                writer, sheet_name="Estandarizado", index=True
            )

            # Covariance matrix
            df_cov_cs = np.cov(self.app.last_results['standardized_data'].T)
            df_cov_cs = pd.DataFrame(df_cov_cs, index=self.app.last_results['standardized_data'].columns,
                                   columns=self.app.last_results['standardized_data'].columns)
            df_cov_cs.rename(
                index=MAPEO_INDICADORES, columns=MAPEO_INDICADORES
            ).to_excel(writer, sheet_name="Matriz_Covarianza", index=True)

        # PCA components
        if 'components' in self.app.last_results and self.app.last_results['components'] is not None:
            self.app.last_results['components'].to_excel(writer, sheet_name="ComponentesPCA", index=True)

        # Explained variance
        if 'pca_model' in self.app.last_results and self.app.last_results['pca_model'] is not None:
            varianza_explicada = self.app.last_results['pca_model'].explained_variance_ratio_
            df_varianza = pd.DataFrame({
                'Componente': [f'PC{i+1}' for i in range(len(varianza_explicada))],
                'Varianza Explicada': varianza_explicada,
                'Varianza Explicada Acumulada': np.cumsum(varianza_explicada)
            })
            df_varianza.to_excel(writer, sheet_name="PCA_VarianzaExp", index=False)

        # Clusters
        if 'clusters' in self.app.last_results:
            df_clusters = pd.DataFrame({
                'Unidad': self.app.last_results['standardized_data'].index,
                'Cluster': self.app.last_results['clusters']
            })
            df_clusters.to_excel(writer, sheet_name="Clusters", index=False)

    def _export_series_results(self, writer):
        """Export series analysis results."""
        import numpy as np

        if 'original_data' in self.app.last_results:
            self.app.last_results['original_data'].to_excel(writer, sheet_name="Datos_Originales", index=True)

        if 'standardized_data' in self.app.last_results:
            self.app.last_results['standardized_data'].to_excel(writer, sheet_name="Datos_Estandarizados", index=True)

        if 'components' in self.app.last_results and self.app.last_results['components'] is not None:
            self.app.last_results['components'].to_excel(writer, sheet_name="Componentes_PCA", index=True)

        if 'pca_model' in self.app.last_results and self.app.last_results['pca_model'] is not None:
            varianza_explicada = self.app.last_results['pca_model'].explained_variance_ratio_
            df_varianza = pd.DataFrame({
                'Componente': [f'PC{i+1}' for i in range(len(varianza_explicada))],
                'Varianza Explicada': varianza_explicada,
                'Varianza Explicada Acumulada': np.cumsum(varianza_explicada)
            })
            df_varianza.to_excel(writer, sheet_name="Varianza_Explicada", index=False)

    def _export_panel_results(self, writer):
        """Export panel analysis results."""
        import numpy as np

        if 'panel_data' in self.app.last_results:
            self.app.last_results['panel_data'].to_excel(writer, sheet_name="Datos_Panel", index=True)

        if 'visualization_3d' in self.app.last_results:
            viz_3d = self.app.last_results['visualization_3d']
            if 'pc_scores' in viz_3d and viz_3d['pc_scores'] is not None:
                viz_3d['pc_scores'].to_excel(writer, sheet_name="Scores_PCA_3D", index=True)

            if 'pca_model' in viz_3d and viz_3d['pca_model'] is not None:
                varianza_explicada = viz_3d['pca_model'].explained_variance_ratio_
                df_varianza = pd.DataFrame({
                    'Componente': [f'PC{i+1}' for i in range(len(varianza_explicada))],
                    'Varianza Explicada': varianza_explicada,
                    'Varianza Explicada Acumulada': np.cumsum(varianza_explicada)
                })
                df_varianza.to_excel(writer, sheet_name="Varianza_3D", index=False)

    def _export_biplot_results(self, writer):
        """Export biplot analysis results."""
        import numpy as np

        if 'prepared_data' in self.app.last_results:
            self.app.last_results['prepared_data'].to_excel(writer, sheet_name="Datos_Preparados", index=True)

        if 'standardized_data' in self.app.last_results:
            self.app.last_results['standardized_data'].to_excel(writer, sheet_name="Datos_Estandarizados", index=True)

        if 'components' in self.app.last_results and self.app.last_results['components'] is not None:
            self.app.last_results['components'].to_excel(writer, sheet_name="Componentes_PCA", index=True)

        if 'pca_model' in self.app.last_results and self.app.last_results['pca_model'] is not None:
            varianza_explicada = self.app.last_results['pca_model'].explained_variance_ratio_
            df_varianza = pd.DataFrame({
                'Componente': [f'PC{i+1}' for i in range(len(varianza_explicada))],
                'Varianza Explicada': varianza_explicada,
                'Varianza Explicada Acumulada': np.cumsum(varianza_explicada)
            })
            df_varianza.to_excel(writer, sheet_name="Varianza_Explicada", index=False)

    def _export_scatter_results(self, writer):
        """Export scatter plot analysis results."""
        import numpy as np

        if 'prepared_data' in self.app.last_results:
            self.app.last_results['prepared_data'].to_excel(writer, sheet_name="Datos_Preparados", index=True)

        if 'standardized_data' in self.app.last_results:
            self.app.last_results['standardized_data'].to_excel(writer, sheet_name="Datos_Estandarizados", index=True)

        if 'components' in self.app.last_results and self.app.last_results['components'] is not None:
            self.app.last_results['components'].to_excel(writer, sheet_name="Componentes_PCA", index=True)

        if 'pca_model' in self.app.last_results and self.app.last_results['pca_model'] is not None:
            varianza_explicada = self.app.last_results['pca_model'].explained_variance_ratio_
            df_varianza = pd.DataFrame({
                'Componente': [f'PC{i+1}' for i in range(len(varianza_explicada))],
                'Varianza Explicada': varianza_explicada,
                'Varianza Explicada Acumulada': np.cumsum(varianza_explicada)
            })
            df_varianza.to_excel(writer, sheet_name="Varianza_Explicada", index=False)

    def _export_correlation_results(self, writer):
        """Export correlation analysis results."""
        if 'original_data' in self.app.last_results:
            self.app.last_results['original_data'].to_excel(writer, sheet_name="Datos_Originales", index=True)

        if 'similarity_matrix' in self.app.last_results:
            self.app.last_results['similarity_matrix'].to_excel(writer, sheet_name="Matriz_Similitud", index=True)

        if 'statistics' in self.app.last_results:
            stats = self.app.last_results['statistics']
            stats_df = pd.DataFrame(list(stats.items()), columns=['Métrica', 'Valor'])
            stats_df.to_excel(writer, sheet_name="Estadísticas", index=False)

        # Filtering report
        if 'filtering_report' in self.app.last_results and self.app.last_results['filtering_report']:
            filtering = self.app.last_results['filtering_report']
            filtering_df = pd.DataFrame({
                'Unidad': filtering.get('removed_units', []),
                'Razón': ['Baja conectividad'] * len(filtering.get('removed_units', []))
            })
            filtering_df.to_excel(writer, sheet_name="Unidades Filtradas", index=False)

            # Filtering summary
            filtering_summary = pd.DataFrame({
                'Métrica': ['Unidades Originales', 'Unidades Filtradas', 'Unidades Removidas', 'Criterios de Filtrado'],
                'Valor': [
                    filtering.get('original_nodes', 'N/A'),
                    filtering.get('filtered_nodes', 'N/A'),
                    filtering.get('removed_nodes', 'N/A'),
                    str(filtering.get('filtering_criteria', {}))
                ]
            })
            filtering_summary.to_excel(writer, sheet_name="Resumen Filtrado", index=False)

        # Configuration info
        config_info = pd.DataFrame({
            'Parámetro': ['Método de Correlación', 'Datos Agregados', 'Umbral de Similitud', 'Tipo de Visualización'],
            'Valor': [
                self.app.last_results.get('correlation_method', 'N/A'),
                'Sí' if self.app.last_results.get('time_aggregated', True) else 'No',
                str(self.app.last_results.get('similarity_threshold', 0.3)),
                self.app.last_results.get('visualization_type', 'N/A')
            ]
        })
        config_info.to_excel(writer, sheet_name="Configuracion", index=False)

    def _export_hierarchical_results(self, writer):
        """Export hierarchical clustering results."""
        # For hierarchical clustering, results are in the frame configuration
        if hasattr(self.app, 'hierarchical_frame') and self.app.hierarchical_frame.pca_data is not None:
            self.app.hierarchical_frame.pca_data.to_excel(writer, sheet_name="Datos_PCA", index=True)

            # Configuration info
            config_info = pd.DataFrame({
                'Parámetro': ['Método de Enlace', 'Métrica de Distancia'],
                'Valor': [self.app.hierarchical_frame.linkage_var.get(), self.app.hierarchical_frame.metric_var.get()]
            })
            config_info.to_excel(writer, sheet_name="Configuracion", index=False)

    def _run_series_analysis(self, config):
        """Run series analysis with given configuration."""
        from backend.analysis_logic import run_series_analysis
        # Update progress to analysis step
        if self.progress_callback:
            self.progress_callback(2, "PCA analysis")
        result = run_series_analysis(config)
        # Update progress to visualization
        if self.progress_callback:
            self.progress_callback(3, "Visualization")
        return result

    def _run_cross_section_analysis(self, config):
        """Run cross-section analysis with given configuration."""
        from backend.analysis_logic import run_cross_section_analysis
        # Update progress to analysis step
        if self.progress_callback:
            self.progress_callback(2, "PCA analysis")
        result = run_cross_section_analysis(config)
        # Update progress to clustering
        if self.progress_callback:
            self.progress_callback(3, "Clustering")
        # Update progress to visualization
        if self.progress_callback:
            self.progress_callback(4, "Visualization")
        return result

    def _run_panel_analysis(self, config):
        """Run panel analysis with given configuration."""
        from backend.analysis_logic import run_panel_analysis
        # Update progress to analysis step
        if self.progress_callback:
            self.progress_callback(2, "PCA analysis")
        result = run_panel_analysis(config)
        # Update progress to visualization
        if self.progress_callback:
            self.progress_callback(3, "3D visualization")
        return result

    def _run_biplot_analysis(self, config):
        """Run biplot analysis with given configuration."""
        from backend.analysis_logic import run_advanced_biplot_analysis
        # Update progress to analysis step
        if self.progress_callback:
            self.progress_callback(2, "PCA analysis")
        result = run_advanced_biplot_analysis(config)
        # Update progress to visualization
        if self.progress_callback:
            self.progress_callback(3, "Biplot generation")
        return result

    def _run_scatter_analysis(self, config):
        """Run scatter analysis with given configuration."""
        from backend.analysis_logic import run_scatter_plot_analysis
        # Update progress to analysis step
        if self.progress_callback:
            self.progress_callback(2, "PCA analysis")
        result = run_scatter_plot_analysis(config)
        # Update progress to visualization
        if self.progress_callback:
            self.progress_callback(3, "Scatter plot")
        return result

    def _run_correlation_analysis(self, config):
        """Run correlation analysis with given configuration."""
        from backend.analysis_logic import run_correlation_analysis
        # Update progress to analysis step
        if self.progress_callback:
            self.progress_callback(2, "Correlation analysis")
        result = run_correlation_analysis(config)
        # Update progress to visualization
        if self.progress_callback:
            self.progress_callback(3, "Visualization")
        return result

    def _run_hierarchical_analysis(self, config):
        """Run hierarchical clustering analysis with given configuration."""
        from backend.analysis_logic import perform_hierarchical_clustering
        # Update progress to analysis step
        if self.progress_callback:
            self.progress_callback(2, "Hierarchical clustering")
        result = perform_hierarchical_clustering(**config)
        # Update progress to visualization
        if self.progress_callback:
            self.progress_callback(3, "Visualization")
        return result

    def export_results(self):
        """Export analysis results to Excel files."""
        try:
            if not hasattr(self.app, 'last_results') or not self.app.last_results:
                messagebox.showwarning("Warning", "No analysis results available to export")
                return

            results = self.app.last_results
            analysis_type = self.app.current_analysis_type
            
            # Debug: Log available keys in results
            self.app.logger.info(f"Exporting results for {analysis_type}")
            self.app.logger.info(f"Available keys in results: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
            
            # Ask user for save location
            save_path = filedialog.asksaveasfilename(
                title="Save analysis results",
                defaultextension=".xlsx",
                filetypes=[
                    ("Excel files", "*.xlsx"),
                    ("All files", "*.*")
                ]
            )
            
            if not save_path:
                return
                
            base_path = Path(save_path).parent / Path(save_path).stem
            sheets_created = 0
            
            # Create Excel writer
            with pd.ExcelWriter(f"{base_path}_complete_results.xlsx", engine='openpyxl') as writer:
                
                # Always create summary sheet first to ensure at least one sheet exists
                sheets_created += self._export_analysis_summary(results, writer, analysis_type)
                
                # Export based on analysis type
                if analysis_type == "correlation":
                    sheets_created += self._export_correlation_results(results, writer)
                elif analysis_type in ["series", "cross_section", "panel", "biplot", "scatter"]:
                    sheets_created += self._export_pca_results(results, writer)
                elif analysis_type == "hierarchical":
                    sheets_created += self._export_hierarchical_results(results, writer)
                
                # If no sheets were created, create a basic info sheet
                if sheets_created == 0:
                    basic_info = pd.DataFrame({
                        'Info': ['Analysis Type', 'Status', 'Message'],
                        'Value': [analysis_type, 'Exported', 'Results data structure may be different than expected']
                    })
                    basic_info.to_excel(writer, sheet_name='Export_Info', index=False)
                    sheets_created += 1
            
            self.app.logger.info(f"Export completed: {sheets_created} sheets created")
            messagebox.showinfo("Success", f"Results exported successfully to:\n{base_path}_complete_results.xlsx\n({sheets_created} sheets created)")
            
        except Exception as e:
            self.app.logger.error(f"Error exporting results: {e}")
            import traceback
            self.app.logger.error(f"Traceback: {traceback.format_exc()}")
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")

    def _export_correlation_results(self, results, writer):
        """Export correlation analysis specific results."""
        sheets_created = 0
        
        # Get data from nested structure
        data_dict = results.get('data', {}) if isinstance(results, dict) else {}
        
        try:
            # Original data
            original_data = data_dict.get('original_data')
            if original_data is not None and hasattr(original_data, 'to_excel'):
                original_data.to_excel(writer, sheet_name='Original_Data', index=True)
                sheets_created += 1
                self.app.logger.info("Exported Original_Data sheet")
        except Exception as e:
            self.app.logger.warning(f"Could not export Original_Data: {e}")
        
        try:
            # Similarity/Correlation matrix
            similarity_matrix = data_dict.get('similarity_matrix')
            if similarity_matrix is not None and hasattr(similarity_matrix, 'to_excel'):
                similarity_matrix.to_excel(writer, sheet_name='Correlation_Matrix', index=True)
                sheets_created += 1
                self.app.logger.info("Exported Correlation_Matrix sheet")
        except Exception as e:
            self.app.logger.warning(f"Could not export Correlation_Matrix: {e}")
        
        try:
            # Statistics
            statistics = data_dict.get('statistics')
            if statistics is not None:
                stats_df = pd.DataFrame([statistics]) if isinstance(statistics, dict) else pd.DataFrame(statistics)
                if not stats_df.empty:
                    stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                    sheets_created += 1
                    self.app.logger.info("Exported Statistics sheet")
        except Exception as e:
            self.app.logger.warning(f"Could not export Statistics: {e}")
        
        try:
            # Configuration and parameters
            config_data = []
            for key in ['correlation_method', 'time_aggregated', 'similarity_threshold', 'visualization_type']:
                if key in data_dict:
                    config_data.append({'Parameter': key, 'Value': str(data_dict[key])})
            
            if config_data:
                config_df = pd.DataFrame(config_data)
                config_df.to_excel(writer, sheet_name='Configuration', index=False)
                sheets_created += 1
                self.app.logger.info("Exported Configuration sheet")
        except Exception as e:
            self.app.logger.warning(f"Could not export Configuration: {e}")
            
        try:
            # Selected indicators and units
            selected_info = []
            if 'selected_indicators' in data_dict:
                indicators = data_dict['selected_indicators']
                for i, indicator in enumerate(indicators):
                    selected_info.append({'Type': 'Indicator', 'Index': i+1, 'Name': indicator})
            
            if 'selected_units' in data_dict:
                units = data_dict['selected_units']
                for i, unit in enumerate(units):
                    selected_info.append({'Type': 'Unit', 'Index': i+1, 'Name': unit})
            
            if selected_info:
                info_df = pd.DataFrame(selected_info)
                info_df.to_excel(writer, sheet_name='Selection_Info', index=False)
                sheets_created += 1
                self.app.logger.info("Exported Selection_Info sheet")
        except Exception as e:
            self.app.logger.warning(f"Could not export Selection_Info: {e}")
        
        # Export network-specific data if available
        try:
            network_sheets = self._export_network_data(data_dict, writer)
            sheets_created += network_sheets
        except Exception as e:
            self.app.logger.warning(f"Could not export network data: {e}")
        
        return sheets_created

    def _export_network_data(self, data_dict, writer):
        """Export network-specific analysis data."""
        sheets_created = 0
        
        try:
            # Check if network data exists
            network_result = data_dict.get('network_result')
            if network_result is None:
                return 0
            
            # Import NetworkX if available for network operations
            try:
                import networkx as nx
            except ImportError:
                self.app.logger.warning("NetworkX not available, skipping network data export")
                return 0
            
            # Export network edges (connections)
            try:
                if hasattr(network_result, 'edges') and hasattr(network_result, 'nodes'):
                    edges_data = []
                    for u, v, data in network_result.edges(data=True):
                        edge_info = {
                            'Source': u,
                            'Target': v,
                            'Weight': data.get('weight', 0),
                            'Abs_Weight': data.get('abs_weight', abs(data.get('weight', 0)))
                        }
                        # Add any additional edge attributes
                        for key, value in data.items():
                            if key not in ['weight', 'abs_weight']:
                                edge_info[f'Edge_{key}'] = value
                        edges_data.append(edge_info)
                    
                    if edges_data:
                        edges_df = pd.DataFrame(edges_data)
                        edges_df.to_excel(writer, sheet_name='Network_Edges', index=False)
                        sheets_created += 1
                        self.app.logger.info("Exported Network_Edges sheet")
            except Exception as e:
                self.app.logger.warning(f"Could not export network edges: {e}")
            
            # Export network nodes
            try:
                if hasattr(network_result, 'nodes'):
                    nodes_data = []
                    for node, data in network_result.nodes(data=True):
                        node_info = {
                            'Node': node,
                            'Degree': network_result.degree(node) if hasattr(network_result, 'degree') else 0
                        }
                        # Calculate weighted degree if edge weights exist
                        try:
                            weighted_degree = sum(d.get('abs_weight', 0) for u, v, d in network_result.edges(node, data=True))
                            node_info['Weighted_Degree'] = weighted_degree
                        except Exception as e:
                            self.app.logger.debug(f"Could not calculate weighted degree for node {node}: {e}")
                            node_info['Weighted_Degree'] = 0
                        
                        # Add any additional node attributes
                        for key, value in data.items():
                            node_info[f'Node_{key}'] = value
                        nodes_data.append(node_info)
                    
                    if nodes_data:
                        nodes_df = pd.DataFrame(nodes_data)
                        nodes_df.to_excel(writer, sheet_name='Network_Nodes', index=False)
                        sheets_created += 1
                        self.app.logger.info("Exported Network_Nodes sheet")
            except Exception as e:
                self.app.logger.warning(f"Could not export network nodes: {e}")
            
            # Export community detection results
            try:
                if hasattr(network_result, 'nodes') and len(list(network_result.nodes())) > 0:
                    # Try to detect communities for export
                    from network_visualization import detect_network_communities
                    communities = detect_network_communities(network_result, method='auto')
                    
                    if communities:
                        community_data = []
                        for node, community_id in communities.items():
                            community_data.append({
                                'Node': node,
                                'Community_ID': community_id
                            })
                        
                        if community_data:
                            community_df = pd.DataFrame(community_data)
                            community_df.to_excel(writer, sheet_name='Network_Communities', index=False)
                            sheets_created += 1
                            self.app.logger.info("Exported Network_Communities sheet")
                            
                            # Export community summary
                            community_summary = community_df['Community_ID'].value_counts().reset_index()
                            community_summary.columns = ['Community_ID', 'Node_Count']
                            community_summary = community_summary.sort_values('Community_ID')
                            community_summary.to_excel(writer, sheet_name='Community_Summary', index=False)
                            sheets_created += 1
                            self.app.logger.info("Exported Community_Summary sheet")
            except Exception as e:
                self.app.logger.warning(f"Could not export community data: {e}")
            
            # Export network statistics
            try:
                if hasattr(network_result, 'nodes') and hasattr(network_result, 'edges'):
                    stats_data = {
                        'Total_Nodes': len(list(network_result.nodes())),
                        'Total_Edges': len(list(network_result.edges())),
                        'Network_Density': nx.density(network_result) if len(list(network_result.nodes())) > 1 else 0,
                        'Is_Connected': nx.is_connected(network_result) if len(list(network_result.nodes())) > 0 else False,
                        'Number_Connected_Components': nx.number_connected_components(network_result) if len(list(network_result.nodes())) > 0 else 0
                    }
                    
                    # Add degree statistics
                    degrees = [network_result.degree(n) for n in network_result.nodes()]
                    if degrees:
                        stats_data.update({
                            'Average_Degree': np.mean(degrees),
                            'Max_Degree': np.max(degrees),
                            'Min_Degree': np.min(degrees)
                        })
                    
                    # Add weighted degree statistics if weights exist
                    try:
                        weighted_degrees = []
                        for node in network_result.nodes():
                            wd = sum(d.get('abs_weight', 0) for u, v, d in network_result.edges(node, data=True))
                            weighted_degrees.append(wd)
                        
                        if weighted_degrees:
                            stats_data.update({
                                'Average_Weighted_Degree': np.mean(weighted_degrees),
                                'Max_Weighted_Degree': np.max(weighted_degrees),
                                'Min_Weighted_Degree': np.min(weighted_degrees)
                            })
                    except Exception as e:
                        self.app.logger.debug(f"Could not calculate weighted degree statistics: {e}")
                    
                    stats_df = pd.DataFrame([stats_data])
                    stats_df.to_excel(writer, sheet_name='Network_Statistics', index=False)
                    sheets_created += 1
                    self.app.logger.info("Exported Network_Statistics sheet")
            except Exception as e:
                self.app.logger.warning(f"Could not export network statistics: {e}")
            
            # Export filtering report if available
            try:
                filtering_report = data_dict.get('filtering_report')
                if filtering_report and isinstance(filtering_report, dict):
                    report_data = []
                    for key, value in filtering_report.items():
                        report_data.append({
                            'Filter_Parameter': key,
                            'Value': str(value)
                        })
                    
                    if report_data:
                        report_df = pd.DataFrame(report_data)
                        report_df.to_excel(writer, sheet_name='Filtering_Report', index=False)
                        sheets_created += 1
                        self.app.logger.info("Exported Filtering_Report sheet")
            except Exception as e:
                self.app.logger.warning(f"Could not export filtering report: {e}")
            
            # Export network configuration used
            try:
                network_config = data_dict.get('network_config', {})
                if network_config and isinstance(network_config, dict):
                    config_data = []
                    for key, value in network_config.items():
                        config_data.append({
                            'Network_Parameter': key,
                            'Value': str(value)
                        })
                    
                    if config_data:
                        config_df = pd.DataFrame(config_data)
                        config_df.to_excel(writer, sheet_name='Network_Config', index=False)
                        sheets_created += 1
                        self.app.logger.info("Exported Network_Config sheet")
            except Exception as e:
                self.app.logger.warning(f"Could not export network configuration: {e}")
        
        except Exception as e:
            self.app.logger.error(f"Error in network data export: {e}")
        
        return sheets_created

    def _export_pca_results(self, results, writer):
        """Export PCA analysis specific results."""
        sheets_created = 0
        
        try:
            # Original data
            if 'standardized_data' in results and results['standardized_data'] is not None:
                results['standardized_data'].to_excel(writer, sheet_name='Standardized_Data', index=True)
                sheets_created += 1
                self.app.logger.info("Exported Standardized_Data sheet")
        except Exception as e:
            self.app.logger.warning(f"Could not export Standardized_Data: {e}")
        
        try:
            # PCA components
            if 'components' in results and results['components'] is not None:
                results['components'].to_excel(writer, sheet_name='PCA_Components', index=True)
                sheets_created += 1
                self.app.logger.info("Exported PCA_Components sheet")
        except Exception as e:
            self.app.logger.warning(f"Could not export PCA_Components: {e}")
            
        try:
            # Loadings (if available)
            if 'loadings' in results and results['loadings'] is not None:
                results['loadings'].to_excel(writer, sheet_name='PCA_Loadings', index=True)
                sheets_created += 1
                self.app.logger.info("Exported PCA_Loadings sheet")
        except Exception as e:
            self.app.logger.warning(f"Could not export PCA_Loadings: {e}")
            
        try:
            # Variance explained
            if 'variance_explained' in results and results['variance_explained'] is not None:
                variance_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(len(results['variance_explained']))],
                    'Variance_Explained': results['variance_explained'],
                    'Cumulative_Variance': np.cumsum(results['variance_explained'])
                })
                variance_df.to_excel(writer, sheet_name='Variance_Explained', index=False)
                sheets_created += 1
                self.app.logger.info("Exported Variance_Explained sheet")
        except Exception as e:
            self.app.logger.warning(f"Could not export Variance_Explained: {e}")
        
        return sheets_created

    def _export_hierarchical_results(self, results, writer):
        """Export hierarchical clustering specific results."""
        sheets_created = 0
        
        try:
            # Original data
            if 'data' in results and results['data'] is not None:
                results['data'].to_excel(writer, sheet_name='Original_Data', index=True)
                sheets_created += 1
                self.app.logger.info("Exported Original_Data sheet")
        except Exception as e:
            self.app.logger.warning(f"Could not export Original_Data: {e}")
            
        try:
            # Cluster assignments (if available)
            if 'cluster_assignments' in results and results['cluster_assignments'] is not None:
                cluster_df = pd.DataFrame(results['cluster_assignments'])
                if not cluster_df.empty:
                    cluster_df.to_excel(writer, sheet_name='Cluster_Assignments', index=False)
                    sheets_created += 1
                    self.app.logger.info("Exported Cluster_Assignments sheet")
        except Exception as e:
            self.app.logger.warning(f"Could not export Cluster_Assignments: {e}")
        
        return sheets_created

    def _export_analysis_summary(self, results_data, analysis_summary, writer, analysis_type):
        """Export analysis summary and configuration from results data and summary."""
        try:
            import pandas as pd
            from datetime import datetime

            # Start with basic summary data
            summary_data = {
                'Parametro': ['Tipo_Analisis', 'Fecha_Hora_Exportacion', 'Hojas_Exportadas'],
                'Valor': [
                    analysis_type.replace('_', ' ').title(),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    str(len(results_data))
                ]
            }

            # Add information from analysis summary if available
            if analysis_summary and isinstance(analysis_summary, dict):
                for key, value in analysis_summary.items():
                    summary_data['Parametro'].append(key)
                    summary_data['Valor'].append(str(value))

            # Add information about available data
            if isinstance(results_data, dict) and results_data:
                # Count DataFrames vs other data types
                df_count = sum(1 for v in results_data.values() if isinstance(v, pd.DataFrame))
                summary_data['Parametro'].extend(['DataFrames_Disponibles', 'Total_Conjuntos_Datos'])
                summary_data['Valor'].extend([str(df_count), str(len(results_data))])

                # Add analysis-specific information by looking at sheet names
                sheet_names = list(results_data.keys())

                # Check for PCA-related data
                pca_sheets = [s for s in sheet_names if 'PCA' in s or 'Componentes' in s or 'Varianza' in s]
                if pca_sheets:
                    summary_data['Parametro'].append('Datos_PCA_Incluidos')
                    summary_data['Valor'].append('Sí')

                # Check for correlation data
                corr_sheets = [s for s in sheet_names if 'Correlacion' in s or 'Matriz' in s]
                if corr_sheets:
                    summary_data['Parametro'].append('Datos_Correlacion_Incluidos')
                    summary_data['Valor'].append('Sí')

                # Check for clustering data
                cluster_sheets = [s for s in sheet_names if 'Cluster' in s or 'Asignacion' in s]
                if cluster_sheets:
                    summary_data['Parametro'].append('Datos_Clustering_Incluidos')
                    summary_data['Valor'].append('Sí')

                # Check for network data
                network_sheets = [s for s in sheet_names if 'Red' in s or 'Nodos' in s or 'Conexiones' in s]
                if network_sheets:
                    summary_data['Parametro'].append('Datos_Red_Incluidos')
                    summary_data['Valor'].append('Sí')

            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Resumen_Analisis', index=False)
            self.app.logger.info("Exported Resumen_Analisis sheet")
            return 1

        except Exception as e:
            self.app.logger.error(f"Could not create Resumen_Analisis: {e}")
            return 0

    def shutdown(self):
        """Shutdown the thread executor and cancel any running analysis."""
        try:
            # Cancel any running analysis
            if self.is_analysis_running:
                self.cancel_current_analysis()

            # Shutdown executor
            if self.executor:
                self.executor.shutdown(wait=True)
                self.app.logger.info("Thread pool executor closed successfully")
        except Exception as e:
            self.app.logger.warning(f"Error during shutdown: {e}")

    def is_analysis_active(self) -> bool:
        """Check if an analysis is currently running."""
        return self.is_analysis_running

    def get_current_analysis_type(self) -> Optional[str]:
        """Get the type of analysis currently running."""
        return self.app.current_analysis_type if self.is_analysis_running else None