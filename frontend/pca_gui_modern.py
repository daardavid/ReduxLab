# pca_gui_modern.py
"""
Refactored PCA Application with ttkbootstrap

Modern interface with lateral navigation panel and dynamic content.
REFACTORED VERSION - Modular design with separate managers
"""

import os
from pathlib import Path
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import messagebox

# Matplotlib is imported on first use (see _ensure_matplotlib) to speed startup.
_matplotlib_figure_class = None

def _ensure_matplotlib():
    """Initialize matplotlib backend once and return Figure class. Used by plot methods."""
    global _matplotlib_figure_class
    if _matplotlib_figure_class is not None:
        return _matplotlib_figure_class
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    _matplotlib_figure_class = Figure
    return _matplotlib_figure_class

# Importar módulos de seguridad (backend)
from backend.security_utils import validate_directory_path, secure_temp_directory, SecurityError
from backend.secure_error_handler import handle_file_operation_error, safe_exception_handler

# analysis_logic is imported lazily in backend.analysis_manager (_run_*_analysis methods)

# Import new modular components (frames are imported lazily in show_*_content())
from frontend.ui_manager import UIManager
from backend.analysis_manager import AnalysisManager
from backend.file_handler import FileHandler
from backend.project_save_config import ProjectConfig
from backend.group_manager import get_universal_group_manager, show_group_manager_gui

# Import enhanced systems (backend)
try:
    from backend.logging_config import get_logger, setup_application_logging
    setup_application_logging(debug_mode=False)
    ENHANCED_SYSTEMS_AVAILABLE = True
except ImportError:
    ENHANCED_SYSTEMS_AVAILABLE = False
    import logging
    def get_logger(name):
        return logging.getLogger(name)


class PCAApp(ttk.Window):
    """Refactored PCA application with modular design and dependency injection."""

    def __init__(self, ui_manager, analysis_manager, file_handler, project_config=None, group_manager=None):
        # logger not available yet – will be configured below
        # Initialize with modern theme
        super().__init__(themename="darkly")

        self.title("🔬 ReduxLab - Advanced Analysis")
        self.geometry("1400x900")
        self.minsize(1200, 700)

        # Configure logging
        if ENHANCED_SYSTEMS_AVAILABLE:
            self.logger = get_logger("pca_gui_modern")
        else:
            self.logger = logging.getLogger("pca_gui_modern")

        self.logger.info("Starting refactored PCA application")

        # Inject dependencies
        self.ui_manager = ui_manager
        self.analysis_manager = analysis_manager
        self.file_handler = file_handler
        self.project_config = project_config or ProjectConfig()
        self.group_manager = group_manager

        # Initialize variables
        self.current_analysis_type = "welcome"
        self.last_results = None
        self.temp_files = []  # Track temporary files for cleanup
        self.current_network_graph = None  # Store the current network graph for degree distribution analysis

        # Initialize secure temp directory
        try:
            self.secure_temp_dir = secure_temp_directory(prefix='pca_app_')
            self.logger.info(f"Created secure temp directory: {self.secure_temp_dir}")
        except SecurityError as e:
            error_msg = handle_file_operation_error(e, "temp directory creation", "security setup")
            self.logger.error(f"Failed to create secure temp directory: {error_msg}")
            # Fallback to system temp
            import tempfile
            self.secure_temp_dir = Path(tempfile.gettempdir()) / 'pca_app_fallback'
            self.secure_temp_dir.mkdir(exist_ok=True)
            self.logger.warning(f"Using fallback temp directory: {self.secure_temp_dir}")

        # Configure cleanup on close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Sheets and notebook are created in setup_content_area()
        self.sheets = []
        self.current_sheet_index = 0

    def get_active_sheet(self):
        """Return the state dict for the currently selected sheet (tab)."""
        if not getattr(self, "sheets", None) or not self.sheets:
            return None
        idx = getattr(self, "current_sheet_index", 0)
        idx = max(0, min(idx, len(self.sheets) - 1))
        return self.sheets[idx]

    @property
    def dynamic_frame(self):
        """Dynamic content area of the active sheet."""
        sheet = self.get_active_sheet()
        return sheet["dynamic_frame"] if sheet else None

    @property
    def content_title(self):
        """Title label of the active sheet."""
        sheet = self.get_active_sheet()
        return sheet["content_title"] if sheet else None

    @property
    def status(self):
        """Status label of the active sheet."""
        sheet = self.get_active_sheet()
        return sheet["status"] if sheet else None

    def setup_application(self):
        """Setup the application after dependency injection."""
        # Initialize universal group manager if not provided
        if self.group_manager is None:
            self.group_manager = get_universal_group_manager(self)

        # Setup UI using managers (minimal so window appears quickly)
        self.ui_manager.setup_ui()
        self.ui_manager.setup_navigation()
        self.ui_manager.setup_content_area()

        # Show initial view immediately so user sees the window
        self.show_analysis_view("welcome")

        # Defer non-critical setup to next idle so first paint is not blocked
        self.after(50, self._deferred_setup)

    def _deferred_setup(self):
        """Run after first paint: shortcuts and saved geometry."""
        self._setup_keyboard_shortcuts()
        self._restore_window_geometry()

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------
    def _setup_keyboard_shortcuts(self):
        """Bind keyboard shortcuts."""
        # Ctrl+R = Run analysis
        self.bind("<Control-r>", lambda e: self._run_current_analysis())
        # Ctrl+Q = Quit
        self.bind("<Control-q>", lambda e: self.on_closing())

    def _run_current_analysis(self):
        """Trigger the run button if it is enabled."""
        if hasattr(self, "btn_run") and str(self.btn_run.cget("state")) != "disabled":
            self.btn_run.invoke()

    # ------------------------------------------------------------------
    # Non-blocking guidance banner
    # ------------------------------------------------------------------
    def _show_guidance_banner(self, message: str, duration_ms: int = 5000):
        """Show a non-blocking banner at the top of the content area that fades after *duration_ms*."""
        try:
            # Remove previous banner if any
            if hasattr(self, "_guidance_banner") and self._guidance_banner is not None:
                try:
                    self._guidance_banner.destroy()
                except Exception:
                    pass

            banner = ttk.Frame(self.dynamic_frame, style="info.TFrame")
            banner.pack(fill=X, padx=10, pady=(0, 5), before=self.dynamic_frame.winfo_children()[0] if self.dynamic_frame.winfo_children() else None)

            ttk.Label(
                banner,
                text=message,
                font=("Helvetica", 10),
                wraplength=800,
                padding=(10, 6),
            ).pack(side=LEFT, fill=X, expand=YES)

            close_btn = ttk.Button(
                banner, text="x", width=3, style="secondary.Outline.TButton",
                command=lambda: banner.destroy(),
            )
            close_btn.pack(side=RIGHT, padx=(0, 5), pady=2)

            self._guidance_banner = banner

            # Auto-dismiss
            self.after(duration_ms, lambda: banner.destroy() if banner.winfo_exists() else None)
        except Exception:
            pass  # non-critical UI element

    # ------------------------------------------------------------------
    # Window geometry persistence
    # ------------------------------------------------------------------
    def _restore_window_geometry(self):
        """Restore window position & size from settings.json."""
        try:
            import json
            settings_path = Path(__file__).parent.parent / "config" / "settings.json"
            if settings_path.exists():
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                geo = settings.get("window_geometry")
                if geo:
                    self.geometry(geo)
                    self.logger.info(f"Restored window geometry: {geo}")
        except Exception as e:
            self.logger.debug(f"Could not restore window geometry: {e}")

    def _save_window_geometry(self):
        """Persist current window geometry to settings.json."""
        try:
            import json
            settings_path = Path(__file__).parent.parent / "config" / "settings.json"
            settings = {}
            if settings_path.exists():
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
            settings["window_geometry"] = self.geometry()
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            self.logger.debug(f"Could not save window geometry: {e}")

    def plot_degree_distribution(self, graph=None):
        """
        Generate and display the degree distribution of the network graph.

        Args:
            graph: NetworkX graph object. If None, uses self.current_network_graph
        """
        from backend.plotting_utils import (
            validate_graph_and_networkx, calculate_degrees_and_counts,
            create_degree_window, create_bar_plot, add_statistics_text,
            create_plot_ui
        )

        try:
            # Use provided graph or current stored graph
            if graph is None:
                graph = getattr(self, 'current_network_graph', None)

            # Validate graph and NetworkX availability
            if not validate_graph_and_networkx(graph, self.logger):
                return

            self.logger.info("Generating degree distribution analysis")

            # Calculate degree distribution
            degrees_list, counts_list, degrees = calculate_degrees_and_counts(graph)
            if degrees_list is None:
                return

            # Create window for the plot
            degree_window = create_degree_window()

            # Create matplotlib figure (lazy init)
            Figure = _ensure_matplotlib()
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)

            # Create bar chart
            create_bar_plot(ax, degrees_list, counts_list, show_values=True, color='steelblue')

            # Customize the plot
            ax.set_title('Distribución de Grado del Nodo\\nAnálisis de Conectividad de la Red',
                        fontsize=14, fontweight='bold', pad=20)

            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='y')

            # Add statistics text
            add_statistics_text(ax, degrees)

            # Adjust layout
            fig.tight_layout()

            # Create UI elements
            create_plot_ui(degree_window, fig, self._save_degree_plot)

            total_nodes = len(degrees)
            mean_degree = sum(degrees) / len(degrees)
            self.logger.info(f"Degree distribution displayed: {total_nodes} nodes, mean degree {mean_degree:.2f}")

        except Exception as e:
            self.logger.error(f"Error generating degree distribution: {e}")
            messagebox.showerror("Error", f"Error al generar la distribución de grado:\\n{str(e)}")

    def _save_degree_plot(self, fig):
        """Save the degree distribution plot to file."""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")],
                title="Guardar Gráfico de Distribución de Grado"
            )
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Éxito", f"Gráfico guardado como:\\n{filename}")
                self.logger.info(f"Degree distribution plot saved to: {filename}")
        except Exception as e:
            self.logger.error(f"Error saving degree plot: {e}")
            messagebox.showerror("Error", f"Error al guardar el gráfico:\\n{str(e)}")

    def show_degree_distribution(self):
        """Public method to show degree distribution - to be called by button."""
        self.plot_degree_distribution()

    def plot_degree_distribution_advanced(self, graph=None, viz_type='bar', color_scheme='steelblue',
                                        show_stats=True, show_values=True):
        """
        Advanced degree distribution visualization with multiple chart types.

        Args:
            graph: NetworkX graph object
            viz_type: Type of visualization ('bar', 'line', 'scatter', 'histogram', 'loglog')
            color_scheme: Color scheme for the plot
            show_stats: Whether to show network statistics
            show_values: Whether to show values on chart elements
        """
        from backend.plotting_utils import (
            validate_graph_and_networkx, calculate_degrees_and_counts,
            create_degree_window, get_color_map, get_viz_titles, get_viz_descriptions,
            create_bar_plot, create_line_plot, create_scatter_plot,
            create_histogram_plot, create_loglog_plot, add_statistics_text,
            create_plot_ui
        )

        try:
            # Use provided graph or current stored graph
            if graph is None:
                graph = getattr(self, 'current_network_graph', None)

            # Validate graph and NetworkX availability
            if not validate_graph_and_networkx(graph, self.logger):
                return

            self.logger.info(f"Generating {viz_type} degree distribution analysis")

            # Calculate degree distribution
            degrees_list, counts_list, degrees = calculate_degrees_and_counts(graph)
            if degrees_list is None:
                return

            # Create window for the plot
            viz_titles = get_viz_titles()
            degree_window = create_degree_window(
                title=f"📊 Distribución de Grado del Nodo - {viz_type.title()}",
                width=900, height=700
            )

            # Create matplotlib figure (lazy init)
            Figure = _ensure_matplotlib()
            fig = Figure(figsize=(12, 8), dpi=100)
            ax = fig.add_subplot(111)

            # Get color mapping
            color_map = get_color_map()
            plot_color = color_map.get(color_scheme, color_scheme)

            # Create different types of plots
            if viz_type == 'bar':
                create_bar_plot(ax, degrees_list, counts_list, show_values, plot_color)
            elif viz_type == 'line':
                create_line_plot(ax, degrees_list, counts_list, show_values, plot_color)
            elif viz_type == 'scatter':
                create_scatter_plot(ax, degrees_list, counts_list, show_values, plot_color)
            elif viz_type == 'histogram':
                create_histogram_plot(ax, degrees, plot_color)
            elif viz_type == 'loglog':
                create_loglog_plot(ax, degrees_list, counts_list, show_values, plot_color)

            # Customize the plot
            ax.set_title(viz_titles.get(viz_type, 'Distribución de Grado del Nodo'),
                        fontsize=16, fontweight='bold', pad=20)

            # Add grid for better readability
            ax.grid(True, alpha=0.3)

            # Add statistics text if requested
            if show_stats:
                add_statistics_text(ax, degrees, fontsize=11, position=(0.02, 0.98))

            # Adjust layout
            fig.tight_layout()

            # Get descriptions and create UI
            viz_descriptions = get_viz_descriptions()
            info_text = viz_descriptions.get(viz_type, '💡 Análisis de distribución de grado')
            create_plot_ui(degree_window, fig, self._save_degree_plot, info_text)

            total_nodes = len(degrees)
            self.logger.info(f"Advanced degree distribution displayed: {viz_type} chart, {total_nodes} nodes")

        except Exception as e:
            self.logger.error(f"Error generating advanced degree distribution: {e}")
            messagebox.showerror("Error", f"Error al generar la distribución de grado:\\n{str(e)}")

    def update_network_graph_reference(self, graph):
        """Update the stored network graph reference and button state."""
        self.current_network_graph = graph
        # Update button state immediately when graph is available
        self.sync_gui_from_cfg()
        
        # Also update the degree analysis button in correlation frame if it exists
        if (hasattr(self, 'correlation_frame') and self.correlation_frame and 
            hasattr(self.correlation_frame, 'update_degree_analysis_button_state')):
            self.correlation_frame.update_degree_analysis_button_state()

    @safe_exception_handler
    def on_closing(self):
        """Handle application closing with secure cleanup."""
        # Save window geometry before closing
        self._save_window_geometry()

        # Clean up temporary files securely
        for temp_file in self.temp_files:
            try:
                temp_path = Path(temp_file)
                if temp_path.exists() and temp_path.is_file():
                    # Ensure the file is within our secure temp directory
                    if self.secure_temp_dir and temp_path.parent == self.secure_temp_dir:
                        temp_path.unlink()
                        self.logger.info(f"Cleaned up temporary file: {temp_file}")
                    else:
                        self.logger.warning(f"Skipping cleanup of file outside secure temp dir: {temp_file}")
            except Exception as e:
                error_msg = handle_file_operation_error(e, str(temp_file), "cleanup")
                self.logger.warning(error_msg)

        # Clean up secure temp directory
        try:
            if hasattr(self, 'secure_temp_dir') and self.secure_temp_dir and self.secure_temp_dir.exists():
                import shutil
                shutil.rmtree(str(self.secure_temp_dir))
                self.logger.info(f"Cleaned up secure temp directory: {self.secure_temp_dir}")
        except Exception as e:
            error_msg = handle_file_operation_error(e, str(self.secure_temp_dir), "temp directory cleanup")
            self.logger.warning(error_msg)

        self.quit()

    # UI setup methods are now handled by UIManager

    def show_analysis_view(self, analysis_type):
        """Show the view corresponding to the analysis type."""
        # Clear previous content
        self.ui_manager.clear_dynamic_content()

        # Update navigation
        self.ui_manager.update_nav_buttons(analysis_type)

        # Route legacy cross_section to biplot (merged)
        if analysis_type == "cross_section":
            analysis_type = "biplot"

        # Update title
        titles = {
            "welcome": f"{self.ui_manager.icon_map['welcome']} Bienvenido a ReduxLab",
            "series": f"{self.ui_manager.icon_map['series']} Serie de tiempo",
            "panel": f"{self.ui_manager.icon_map['panel']} Análisis 3D",
            "correlation": f"{self.ui_manager.icon_map['correlation']} Correlación y redes",
            "biplot": f"{self.ui_manager.icon_map.get('biplot', '📊')} Corte transversal / Biplot",
            "scatter": f"{self.ui_manager.icon_map['scatter']} Scatterplot",
            "hierarchical": f"{self.ui_manager.icon_map['hierarchical']} Hierarchical Clustering"
        }
        self.content_title.config(text=titles.get(analysis_type, "Analysis"))

        # Set current analysis type and store view in active sheet
        self.current_analysis_type = analysis_type
        sheet = self.get_active_sheet()
        if sheet:
            sheet["view"] = analysis_type

        # Load specific content
        if analysis_type == "welcome":
            self.show_welcome_content()
        elif analysis_type == "series":
            self.show_series_content()
        elif analysis_type == "panel":
            self.show_panel_content()
        elif analysis_type == "correlation":
            self.show_correlation_content()
        elif analysis_type == "biplot":
            self.show_biplot_content()
        elif analysis_type == "scatter":
            self.show_scatter_content()
        elif analysis_type == "hierarchical":
            self.show_hierarchical_content()

        if hasattr(self.ui_manager, "sync_toolbar_view_from_sheet"):
            self.ui_manager.sync_toolbar_view_from_sheet()

    def show_welcome_content(self):
        """Show welcome content with intention-based cards."""
        welcome_frame = ttk.Frame(self.dynamic_frame)
        welcome_frame.pack(fill=BOTH, expand=YES)

        # Main title
        title_label = ttk.Label(
            welcome_frame,
            text="¿Qué pregunta quieres responder hoy?",
            font=("Helvetica", 24, "bold"),
            style='primary.TLabel'
        )
        title_label.pack(pady=(30, 50))

        # Cards container
        cards_container = ttk.Frame(welcome_frame)
        cards_container.pack(fill=BOTH, expand=YES, padx=50)

        # Configure grid layout for cards (2x2)
        cards_container.columnconfigure(0, weight=1)
        cards_container.columnconfigure(1, weight=1)
        cards_container.rowconfigure(0, weight=1)
        cards_container.rowconfigure(1, weight=1)

        # Card 1: Find patterns and groups
        self.create_intention_card(
            cards_container, 0, 0,
            icon="👥",
            title="Encontrar patrones y grupos",
            subtitle="Recomendado: Análisis de Clúster, Componentes principales",
            target_view="hierarchical",
            description="Descubre agrupaciones naturales en tus datos y identifica patrones ocultos."
        )

        # Card 2: Understand important variables
        self.create_intention_card(
            cards_container, 0, 1,
            icon="📊",
            title="Entender variables importantes",
            subtitle="Recomendado: Componentes principales, Correlación/Redes",
            target_view="correlation",
            description="Identifica qué variables explican mejor la variabilidad en tus datos."
        )

        # Card 3: Compare products/stores
        self.create_intention_card(
            cards_container, 1, 0,
            icon="🔗",
            title="Comparar mis unidades de investigación",
            subtitle="Recomendado: Biplot, Redes",
            target_view="biplot",
            description="Visualiza similitudes y diferencias entre tus unidades de análisis."
        )

        # Card 4: See a trend over time
        self.create_intention_card(
            cards_container, 1, 1,
            icon="📈",
            title="Ver una tendencia en el tiempo",
            subtitle="Recomendado: Serie de Tiempo",
            target_view="series",
            description="Analiza cómo cambian tus variables a lo largo del tiempo."
        )

        # Footer with preprocessing option
        footer_frame = ttk.Frame(welcome_frame)
        footer_frame.pack(fill=X, pady=(50, 20), padx=50)

        ttk.Separator(footer_frame, orient=HORIZONTAL).pack(fill=X, pady=(0, 20))

        preprocessing_frame = ttk.LabelFrame(footer_frame, text="🔧 Herramientas Adicionales", padding=20)
        preprocessing_frame.pack(fill=X)

        ttk.Button(
            preprocessing_frame,
            text="📊 Consolidar Datos por Años",
            style='info.TButton',
            command=self.file_handler.consolidate_company_data_gui
        ).pack(pady=10)

        preprocessing_desc = ttk.Label(
            preprocessing_frame,
            text="Transforma datos históricos en perfiles estadísticos únicos\n"
                 "(media y desviación estándar) para análisis posteriores.",
            font=("Helvetica", 9),
            style='secondary.TLabel',
            justify=CENTER
        )
        preprocessing_desc.pack(pady=(0, 10))

    def create_intention_card(self, parent, row, col, icon, title, subtitle, target_view, description):
        """Create an interactive intention card."""
        # Card frame with hover effects
        card_frame = ttk.Frame(parent, style='card.TFrame', padding=20)
        card_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

        # Configure card styling
        card_frame.configure(relief='raised', borderwidth=1)

        # Icon and title section
        header_frame = ttk.Frame(card_frame)
        header_frame.pack(fill=X, pady=(0, 15))

        icon_label = ttk.Label(
            header_frame,
            text=icon,
            font=("Helvetica", 36),
            style='primary.TLabel'
        )
        icon_label.pack(pady=(0, 10))

        title_label = ttk.Label(
            header_frame,
            text=title,
            font=("Helvetica", 16, "bold"),
            style='primary.TLabel',
            wraplength=250,
            justify=CENTER
        )
        title_label.pack(pady=(0, 5))

        subtitle_label = ttk.Label(
            header_frame,
            text=subtitle,
            font=("Helvetica", 10),
            style='secondary.TLabel',
            wraplength=250,
            justify=CENTER
        )
        subtitle_label.pack()

        # Description
        desc_label = ttk.Label(
            card_frame,
            text=description,
            font=("Helvetica", 11),
            style='secondary.TLabel',
            wraplength=280,
            justify=CENTER
        )
        desc_label.pack(pady=(15, 20))

        # Action button
        action_button = ttk.Button(
            card_frame,
            text="Seleccionar →",
            style='success.TButton',
            command=lambda: self.navigate_to_intention(target_view)
        )
        action_button.pack(pady=(10, 0))

        # Hover effects
        def on_enter(event):
            card_frame.configure(relief='solid', borderwidth=2)
            icon_label.configure(style='info.TLabel')

        def on_leave(event):
            card_frame.configure(relief='raised', borderwidth=1)
            icon_label.configure(style='primary.TLabel')

        card_frame.bind("<Enter>", on_enter)
        card_frame.bind("<Leave>", on_leave)

        # Make the whole card clickable
        for child in [icon_label, title_label, subtitle_label, desc_label]:
            child.bind("<Button-1>", lambda e, tv=target_view: self.navigate_to_intention(tv))
            child.configure(cursor="hand2")

    def navigate_to_intention(self, target_view):
        """Navigate to the selected analysis view with guidance."""
        # Navigate to the target view
        self.show_analysis_view(target_view)

        # Add contextual guidance based on the intention
        guidance_messages = {
            "hierarchical": "¡Perfecto! Sube tus datos para encontrar grupos naturales en tus datos.",
            "correlation": "¡Excelente! Vamos a explorar las relaciones entre tus variables.",
            "biplot": "¡Genial! Visualicemos las similitudes entre tus unidades de análisis.",
            "series": "¡Perfecto! Analicemos cómo cambian tus datos a lo largo del tiempo."
        }

        if target_view in guidance_messages:
            # Show non-blocking banner instead of modal dialog
            self._show_guidance_banner(guidance_messages[target_view])

    def show_series_content(self):
        """Show time series analysis content."""
        from frontend.frames.series_frame import SeriesAnalysisFrame
        self.current_series_frame = SeriesAnalysisFrame(self.dynamic_frame, self)
        self.current_series_frame.pack(fill=X, expand=YES, anchor=N)
        self._store_frame_in_sheet("series", self.current_series_frame)

    def show_panel_content(self):
        """Show PCA 3D analysis content."""
        from frontend.frames.panel_frame import PanelAnalysisFrame
        self.panel_frame = PanelAnalysisFrame(self.dynamic_frame, self)
        self.panel_frame.pack(fill=X, expand=YES, anchor=N)
        self._store_frame_in_sheet("panel", self.panel_frame)

    def show_biplot_content(self):
        """Show advanced biplot content."""
        from frontend.frames.biplot_frame import BiplotAnalysisFrame
        self.biplot_frame = BiplotAnalysisFrame(self.dynamic_frame, self)
        self.biplot_frame.pack(fill=X, expand=YES, anchor=N)
        self._store_frame_in_sheet("biplot", self.biplot_frame)

    def show_scatter_content(self):
        """Show PCA scatterplot content."""
        from frontend.frames.scatter_frame import ScatterAnalysisFrame
        self.scatter_frame = ScatterAnalysisFrame(self.dynamic_frame, self)
        self.scatter_frame.pack(fill=X, expand=YES, anchor=N)
        self._store_frame_in_sheet("scatter", self.scatter_frame)

    def show_hierarchical_content(self):
        """Show hierarchical clustering content."""
        from frontend.frames.hierarchical_frame import HierarchicalClusteringFrame
        self.hierarchical_frame = HierarchicalClusteringFrame(self.dynamic_frame, self)
        self.hierarchical_frame.pack(fill=X, expand=YES, anchor=N)
        self._store_frame_in_sheet("hierarchical", self.hierarchical_frame)

    def show_correlation_content(self):
        """Show correlation/network analysis content."""
        from frontend.frames.correlation_frame import CorrelationAnalysisFrame
        self.correlation_frame = CorrelationAnalysisFrame(self.dynamic_frame, self)
        self.correlation_frame.pack(fill=X, expand=YES, anchor=N)
        self._store_frame_in_sheet("correlation", self.correlation_frame)

    def _store_frame_in_sheet(self, view_name: str, frame):
        """Store the current analysis frame in the active sheet for tab switching."""
        sheet = self.get_active_sheet()
        if sheet and "frames" in sheet:
            sheet["frames"][view_name] = frame

    def run_advanced_tool(self, tool_name: str):
        """Run an advanced analytics tool (Bartlett, KMO, t-SNE, etc.) on the active sheet's data."""
        sheet = self.get_active_sheet()
        if not sheet or sheet.get("df") is None or sheet["df"].empty:
            messagebox.showwarning("Sin datos", "Carga datos en la hoja activa primero (Cargar datos).")
            return
        df = sheet["df"]
        # Use only numeric columns
        df_num = df.select_dtypes(include=["number"])
        if df_num.empty or len(df_num.columns) < 2:
            messagebox.showwarning("Datos insuficientes", "Se necesitan al menos 2 columnas numéricas.")
            return
        try:
            from backend import advanced_analytics as adv
            from tkinter import Toplevel
            from tkinter.scrolledtext import ScrolledText
            result_text = []
            if tool_name == "bartlett":
                r = adv.bartlett_test(df_num)
                result_text = [f"Chi² = {r['chi_square']:.4f}", f"p = {r['p_value']:.4f}", r["interpretation"]]
            elif tool_name == "kmo":
                r = adv.kmo_test(df_num)
                result_text = [f"KMO global = {r['overall_kmo']:.4f}", r.get("interpretation", "")]
            elif tool_name == "cronbach":
                r = adv.cronbach_alpha(df_num)
                result_text = [f"Alfa = {r['alpha']:.4f} ({r['quality']})", r["interpretation"]]
            elif tool_name == "mahalanobis":
                s = adv.mahalanobis_distance(df_num)
                result_text = [f"Distancia de Mahalanobis (n={len(s)})", f"Min={s.min():.4f}, Max={s.max():.4f}, Media={s.mean():.4f}", s.describe().to_string()]
            elif tool_name == "tsne":
                r = adv.run_tsne(df_num, n_components=2)
                result_text = [f"t-SNE realizado. Forma: {r['embedding'].shape}"]
            elif tool_name == "umap":
                r = adv.run_umap(df_num, n_components=2)
                result_text = [f"UMAP realizado. Forma: {r['embedding'].shape}"]
            elif tool_name == "fa":
                r = adv.factor_analysis(df_num, n_factors=2)
                result_text = [f"Análisis factorial: {list(r.keys())}", str(r.get("loadings", r))[:600]]
            elif tool_name == "ica":
                r = adv.run_ica(df_num, n_components=2)
                result_text = [f"ICA: fuentes forma {r['sources'].shape}", f"Iteraciones: {r.get('n_iter', 'N/A')}", r["mixing_matrix"].to_string()[:800]]
            elif tool_name == "kmeans":
                r = adv.kmeans_analysis(df_num, max_k=10)
                result_text = [f"K-Means: k óptimo = {r['optimal_k']}", f"Silhouette scores (k=2..10): {[f'{x:.3f}' for x in r['silhouette_scores']]}", f"Etiquetas: {r['labels'].value_counts().to_string()}"]
            elif tool_name == "dbscan":
                r = adv.dbscan_analysis(df_num, eps=0.5)
                result_text = [f"Clusters: {r.get('n_clusters', 'N/A')}", f"Ruido: {r.get('n_noise', 0)}", str(r.get("params", r))[:400]]
            elif tool_name == "gmm":
                r = adv.gmm_analysis(df_num, max_components=10)
                result_text = [f"GMM: componentes óptimas = {r['optimal_n']}", f"BIC scores: {[f'{x:.1f}' for x in r['bic_scores']]}", r["labels"].value_counts().to_string()]
            elif tool_name == "yeo_johnson":
                df_out, lambdas = adv.yeo_johnson_transform(df_num)
                result_text = [f"Transformación aplicada. Lambdas: {lambdas}"]
            elif tool_name == "winsorize":
                df_out = adv.winsorize(df_num, limits=(0.05, 0.05))
                result_text = [f"Winsorización (5%-95%) aplicada. Forma: {df_out.shape}", df_out.describe().to_string()[:1500]]
            elif tool_name == "rank_inverse_normal":
                df_out = adv.rank_inverse_normal(df_num)
                result_text = [f"Rank inverse normal aplicado. Forma: {df_out.shape}", df_out.describe().to_string()[:1500]]
            elif tool_name == "robust_scale":
                df_out = adv.robust_scale(df_num)
                result_text = [f"Robust scaling (mediana/IQR) aplicado. Forma: {df_out.shape}", df_out.describe().to_string()[:1500]]
            else:
                result_text = ["Herramienta no implementada en diálogo."]
            win = Toplevel(self)
            win.title(f"Resultado: {tool_name}")
            win.geometry("500x300")
            st = ScrolledText(win, wrap="word", font=("Consolas", 9))
            st.pack(fill="both", expand=True, padx=10, pady=10)
            st.insert("1.0", "\n".join(result_text))
            st.config(state="disabled")
        except Exception as e:
            self.logger.exception("Advanced tool error")
            messagebox.showerror("Error", f"Error en {tool_name}: {str(e)}")

    def _ask_choose_sheets(self, loaded):
        """Show dialog to choose one or more sheets (variables) from loaded dict.
        Returns list of chosen keys, or None if cancelled. For PCA you can select several sheets."""
        import tkinter as tk
        keys_in_order = list(loaded.keys())
        keys_with_shape = []
        for k in keys_in_order:
            v = loaded[k]
            if hasattr(v, "shape"):
                keys_with_shape.append(f"{k} ({v.shape[0]}×{v.shape[1]})")
            else:
                keys_with_shape.append(k)
        choice = [None]  # None = cancel, [] = no selection, [key, ...] = selected keys

        win = tk.Toplevel(self)
        win.title("Seleccione la hoja o hojas")
        win.transient(self)
        win.grab_set()
        win.geometry("480x440")
        win.minsize(400, 380)
        f = ttk.Frame(win, padding=15)
        f.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            f,
            text="Seleccione una o varias hojas (variables) para la matriz de datos (n×p). Para PCA puede usar varias.",
            style="primary.TLabel",
            wraplength=440,
        ).pack(anchor=tk.W)
        # Listbox with multi-select (fixed height so buttons stay visible)
        list_frame = ttk.Frame(f)
        list_frame.pack(pady=(8, 8), fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        lb = tk.Listbox(
            list_frame,
            height=10,
            selectmode=tk.EXTENDED,
            yscrollcommand=scrollbar.set,
            font=("Segoe UI", 10),
            activestyle="dotbox",
        )
        for item in keys_with_shape:
            lb.insert(tk.END, item)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=lb.yview)
        # Buttons: Select all, Clear selection (always visible below list)
        sel_f = ttk.Frame(f)
        sel_f.pack(fill=tk.X, pady=(0, 8))
        def select_all():
            lb.selection_set(0, tk.END)
        def clear_selection():
            lb.selection_clear(0, tk.END)
        ttk.Button(sel_f, text="Seleccionar todas", command=select_all, style="secondary.Outline.TButton").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(sel_f, text="Quitar selección", command=clear_selection, style="secondary.Outline.TButton").pack(side=tk.LEFT)
        def on_ok():
            sel = lb.curselection()
            if not sel:
                choice[0] = []
                win.destroy()
                return
            chosen_display = [keys_with_shape[i] for i in sel]
            try:
                choice[0] = [keys_in_order[keys_with_shape.index(d)] for d in chosen_display]
            except ValueError:
                choice[0] = [keys_in_order[i] for i in sel if 0 <= i < len(keys_in_order)]
            win.destroy()
        def on_cancel():
            win.destroy()
        btn_f = ttk.Frame(f)
        btn_f.pack(fill=tk.X)
        ttk.Button(btn_f, text="Aceptar", command=on_ok, style="primary.TButton").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_f, text="Cancelar", command=on_cancel).pack(side=tk.LEFT)
        # Enter = Aceptar, Escape = Cancelar
        win.bind("<Return>", lambda e: on_ok())
        win.bind("<Escape>", lambda e: on_cancel())
        lb.bind("<Return>", lambda e: on_ok())
        lb.bind("<Double-Button-1>", lambda e: on_ok())
        win.wait_window()
        if choice[0] is None:
            return None
        if choice[0] == []:
            return None
        return choice[0]

    def show_data_prep_dialog(self):
        """Open Prepare data / Build matrix dialog (merge sheets, choose sheet)."""
        from frontend.data_prep_dialog import show_data_prep_dialog
        show_data_prep_dialog(self)

    def toolbar_load_data(self):
        """Load a dataset into the active sheet (toolbar 'Cargar datos')."""
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Cargar datos",
            filetypes=[
                ("Excel", "*.xlsx *.xls"),
                ("CSV", "*.csv"),
                ("Parquet", "*.parquet"),
                ("SQLite", "*.db *.sqlite *.sqlite3"),
                ("Todos", "*.*")
            ]
        )
        if not file_path:
            return
        # For SQLite, allow even if validate_file_path restricts extensions (check exists + readable)
        if not file_path.lower().endswith(('.db', '.sqlite', '.sqlite3')) and not self.file_handler.validate_file_path(file_path):
            return
        if file_path.lower().endswith(('.db', '.sqlite', '.sqlite3')):
            import os
            if not os.path.isfile(file_path) or not os.access(file_path, os.R_OK):
                messagebox.showerror("Error", "No se puede leer el archivo de base de datos.")
                return
        sheet = self.get_active_sheet()
        if not sheet:
            return
        try:
            import pandas as pd
            from backend import data_loader_module as dl
            if file_path.lower().endswith(('.xlsx', '.xls')):
                loaded = dl.load_excel_file(file_path)
                if loaded:
                    # Contract: sheet["df"] must be n×p matrix (observations × variables)
                    if len(loaded) > 1:
                        chosen_keys = self._ask_choose_sheets(loaded)
                        if not chosen_keys:
                            return
                        if len(chosen_keys) == 1:
                            selected_df = loaded[chosen_keys[0]].copy()
                        else:
                            # Varias hojas: combinar como variables (columnas), alineando por índice
                            dfs = [loaded[k] for k in chosen_keys]
                            selected_df = pd.concat(dfs, axis=1, join="outer", copy=False)
                    else:
                        chosen_key = next(iter(loaded))
                        selected_df = loaded[chosen_key]
                    sheet["df"] = selected_df if isinstance(selected_df, pd.DataFrame) else pd.DataFrame(selected_df)
                else:
                    sheet["df"] = pd.read_excel(file_path)
            elif file_path.lower().endswith('.csv'):
                # Contract: sheet["df"] = n×p matrix (observations × variables)
                sheet["df"] = pd.read_csv(file_path)
            elif file_path.lower().endswith('.parquet'):
                sheet["df"] = pd.read_parquet(file_path)
            elif file_path.lower().endswith(('.db', '.sqlite', '.sqlite3')):
                from backend import data_connectors as dc
                loaded = dc.load_sqlite(file_path)
                if not loaded:
                    messagebox.showerror("Error", "La base de datos no tiene tablas o no pudo leerse.")
                    return
                if len(loaded) > 1:
                    chosen_keys = self._ask_choose_sheets(loaded)
                    if not chosen_keys:
                        return
                    if len(chosen_keys) == 1:
                        selected_df = loaded[chosen_keys[0]].copy()
                    else:
                        dfs = [loaded[k] for k in chosen_keys]
                        selected_df = pd.concat(dfs, axis=1, join="outer", copy=False)
                else:
                    chosen_key = next(iter(loaded))
                    selected_df = loaded[chosen_key]
                sheet["df"] = selected_df if isinstance(selected_df, pd.DataFrame) else pd.DataFrame(selected_df)
            else:
                sheet["df"] = pd.read_excel(file_path)
            sheet["file_path"] = file_path
            name = Path(file_path).stem
            idx = self.current_sheet_index
            if hasattr(self, "sheets_notebook") and self.sheets_notebook and idx < self.sheets_notebook.index("end"):
                self.sheets_notebook.tab(idx, text=name[:20] + ("..." if len(name) > 20 else ""))
            # Optional n×p validation (warnings only)
            from backend.data_validation import validate_matrix_shape
            matrix_warnings = validate_matrix_shape(sheet["df"])
            if matrix_warnings:
                self.status.config(text=f"Datos cargados: {name} — {matrix_warnings[0]}")
            else:
                self.status.config(text=f"Datos cargados: {name}")
        except Exception as e:
            self.logger.exception("Error loading file")
            messagebox.showerror("Error", f"Error al cargar: {str(e)}")

    def run_current_analysis(self):
        """Execute the current analysis."""
        self.analysis_manager.run_current_analysis()

    def export_results(self):
        """Export analysis results."""
        self.analysis_manager.export_results()

    def show_group_manager(self):
        """Show the universal group manager."""
        try:
            # Get available units from current analysis
            available_units = []
            
            if hasattr(self, 'last_results') and self.last_results:
                # Try to extract units from last results
                if 'data' in self.last_results:
                    data = self.last_results['data']
                    if isinstance(data, dict):
                        # Check for similarity matrix index (correlation analysis)
                        similarity_matrix = data.get('similarity_matrix')
                        if similarity_matrix is not None and hasattr(similarity_matrix, 'index'):
                            available_units = list(similarity_matrix.index)
                        
                        # Check for original data
                        if not available_units:
                            original_data = data.get('original_data')
                            if original_data is not None and hasattr(original_data, 'index'):
                                available_units = list(original_data.index)
            
            # If no units from results, try to get from active frame
            if not available_units:
                if self.current_analysis_type == "correlation" and hasattr(self, 'correlation_frame'):
                    # Try to load units from the correlation frame's data
                    try:
                        file_path = self.correlation_frame.file_entry.get().strip()
                        if file_path and os.path.exists(file_path):
                            # Load the data to get available units
                            if file_path.endswith('.csv'):
                                import pandas as pd
                                df = pd.read_csv(file_path)
                                if 'Unit' in df.columns:
                                    available_units = list(df['Unit'].unique())
                                elif 'Empresa' in df.columns:
                                    available_units = list(df['Empresa'].unique())
                                else:
                                    # Use index if no Unit column
                                    available_units = list(df.index)
                    except Exception as e:
                        self.logger.warning(f"Could not load units from file: {e}")
            
            # Show group manager
            show_group_manager_gui(self, available_units)
            
        except Exception as e:
            self.logger.error(f"Error showing group manager: {e}")
            messagebox.showerror("Error", f"Error showing group manager: {e}")

    # ===== Donación / Apoyo =====
    def open_kofi(self):
        """Open Ko‑fi donation page in default browser."""
        try:
            import webbrowser
            webbrowser.open("https://ko-fi.com/daardavid")
        except Exception as e:
            messagebox.showwarning("Navegador", f"No fue posible abrir el navegador: {e}")

    def show_bank_transfer_dialog(self):
        """Show a small dialog with bank transfer details (Mexico only)."""
        dialog = ttk.Toplevel(self)
        dialog.title("Transferencia bancaria")
        dialog.geometry("460x260")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        container = ttk.Frame(dialog, padding=20)
        container.pack(fill=BOTH, expand=YES)

        ttk.Label(
            container,
            text="Gracias por tu apoyo 🙏",
            font=("Helvetica", 14, "bold"),
            bootstyle="secondary",
        ).pack(anchor=W)

        info = (
            "Banco: BBVA\n"
            "Nombre: DAVID ARMANDO ABREU ROSIQUE\n"
            "Cuenta: 0108748743\n"
            "CLABE: 021180065956536300"
        )
        ttk.Label(
            container,
            text=info,
            font=("Consolas", 11),
            justify=LEFT,
            bootstyle="light",
        ).pack(anchor=W, pady=(10, 14))

        # Actions row
        actions = ttk.Frame(container)
        actions.pack(fill=X)

        def copy_clabe():
            try:
                dialog.clipboard_clear()
                dialog.clipboard_append("021180065956536300")
                messagebox.showinfo("Copiado", "CLABE copiada al portapapeles")
            except Exception as e:
                messagebox.showwarning("Error", f"No se pudo copiar la CLABE: {e}")

        ttk.Button(
            actions,
            text="Copiar CLABE",
            style="info.TButton",
            command=copy_clabe,
            width=18,
        ).pack(side=LEFT)

        ttk.Button(
            actions,
            text="Cerrar",
            style="secondary.Outline.TButton",
            command=dialog.destroy,
            width=12,
        ).pack(side=RIGHT)

    def sync_gui_from_cfg(self):
        """Synchronize GUI with current configuration."""
        ready = False

        if self.current_analysis_type == "series":
            if hasattr(self, 'current_series_frame') and self.current_series_frame:
                ready = (
                    hasattr(self.current_series_frame, 'file_entry') and self.current_series_frame.file_entry.get().strip() and
                    self.current_series_frame.selected_indicators and
                    self.current_series_frame.selected_country and
                    self.current_series_frame.selected_years
                )

        elif self.current_analysis_type == "panel":
            if hasattr(self, 'panel_frame') and self.panel_frame:
                ready = (
                    hasattr(self.panel_frame, 'file_entry') and self.panel_frame.file_entry.get().strip() and
                    self.panel_frame.selected_indicators and
                    self.panel_frame.selected_units and
                    self.panel_frame.selected_years
                )

        elif self.current_analysis_type == "biplot":
            if hasattr(self, 'biplot_frame') and self.biplot_frame:
                ready = (
                    hasattr(self.biplot_frame, 'file_entry') and self.biplot_frame.file_entry.get().strip() and
                    self.biplot_frame.selected_indicators and
                    self.biplot_frame.selected_countries and
                    self.biplot_frame.selected_year
                )

        elif self.current_analysis_type == "scatter":
            if hasattr(self, 'scatter_frame') and self.scatter_frame:
                ready = (
                    hasattr(self.scatter_frame, 'file_entry') and self.scatter_frame.file_entry.get().strip() and
                    self.scatter_frame.selected_indicators and
                    self.scatter_frame.selected_countries and
                    self.scatter_frame.selected_year
                )

        elif self.current_analysis_type == "correlation":
            if hasattr(self, 'correlation_frame') and self.correlation_frame:
                ready = (
                    hasattr(self.correlation_frame, 'file_entry') and
                    self.correlation_frame.file_entry.get().strip() and
                    self.correlation_frame.correlation_method.get() and
                    self.correlation_frame.visualization_type.get() and
                    hasattr(self.correlation_frame, 'selected_units') and
                    self.correlation_frame.selected_units and
                    len(self.correlation_frame.selected_units) >= 2  # Minimum 2 units for correlation
                )

        elif self.current_analysis_type == "hierarchical":
            if hasattr(self, 'hierarchical_frame') and self.hierarchical_frame:
                ready = self.hierarchical_frame.pca_data is not None

        # Enable/disable run button
        if hasattr(self, 'btn_run'):
            self.btn_run.config(state=NORMAL if ready else DISABLED)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ReduxLab - Análisis de datos avanzado")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose logging, debug panel in UI)",
    )
    args = parser.parse_args()

    # Initialise logging BEFORE anything else
    from backend.logging_config import setup_application_logging

    setup_application_logging(debug_mode=args.debug)

    # Create managers with dependency injection
    app = PCAApp(
        ui_manager=UIManager(None),  # Will be set to app later
        analysis_manager=AnalysisManager(None),  # Will be set to app later
        file_handler=FileHandler(None)  # Will be set to app later
    )

    # Store debug flag on app for use by other modules
    app.debug_mode = args.debug

    # Set the app reference in managers
    app.ui_manager.app = app
    app.analysis_manager.app = app
    app.file_handler.app = app

    # Setup the application
    app.setup_application()

    app.mainloop()