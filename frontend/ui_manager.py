"""
UI Manager for PCA Application

Handles UI setup, navigation, and content management.
"""

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from ttkbootstrap.scrolled import ScrolledFrame


class UIManager:
    """Manages the user interface components of the PCA application."""

    def __init__(self, app):
        self.app = app
        self.icon_map = {
            "welcome": "🏠",
            "series": "📈",
            "cross_section": "✂️",
            "panel": "🎯",
            "correlation": "🔗",
            "biplot": "✂️",
            "scatter": "🟢",
            "hierarchical": "🌳",
            "file": "📁",
            "settings": "⚙️",
            "run": "▶",
            "export": "💾"
        }

    def setup_ui(self):
        """Configure the main UI structure."""
        # Frame principal que divide la ventana
        self.app.main_container = ttk.Frame(self.app)
        self.app.main_container.pack(fill=BOTH, expand=YES)

        # Panel de navegación (izquierda)
        self.app.nav_panel = ttk.Frame(
            self.app.main_container,
            width=320,
            style='secondary.TFrame'
        )
        self.app.nav_panel.pack(side=LEFT, fill=Y)
        self.app.nav_panel.pack_propagate(False)

        # Área de contenido principal (derecha)
        self.app.content_area = ttk.Frame(self.app.main_container)
        self.app.content_area.pack(side=RIGHT, fill=BOTH, expand=YES)

    def setup_navigation(self):
        """Configure the navigation panel."""
        # Título del panel
        nav_title = ttk.Label(
            self.app.nav_panel,
            text="📊 ReduxLab",
            font=("Helvetica", 16, "bold"),
            style='inverse.primary.TLabel',
            anchor=CENTER
        )
        nav_title.pack(pady=(30, 40), padx=20, fill=X)

        # Botones de navegación
        self.create_nav_buttons()

        # Nueva hoja (pestaña)
        ttk.Button(
            self.app.nav_panel,
            text="➕ Nueva hoja",
            style='secondary.Outline.TButton',
            command=self.add_new_sheet,
            width=25
        ).pack(pady=(8, 0), padx=20, fill=X)

        # Separador
        ttk.Separator(self.app.nav_panel, orient=HORIZONTAL).pack(fill=X, pady=30, padx=20)

        # Botones de acción
        self.app.btn_run = ttk.Button(
            self.app.nav_panel,
            text=f"{self.icon_map['run']} Ejecutar Análisis",
            style='success.TButton',
            command=self.app.analysis_manager.run_current_analysis,
            state=DISABLED,
            width=25
        )
        self.app.btn_run.pack(pady=(0, 10), padx=20, fill=X)

        self.app.btn_export = ttk.Button(
            self.app.nav_panel,
            text=f"{self.icon_map['export']} Exportar Resultados",
            style='info.Outline.TButton',
            command=self.app.export_results,
            state=DISABLED,
            width=25
        )
        self.app.btn_export.pack(pady=(0, 10), padx=20, fill=X)

        # Botón de gestión de grupos
        self.app.btn_groups = ttk.Button(
            self.app.nav_panel,
            text="🏷️ Gestionar Grupos",
            style='warning.Outline.TButton',
            command=self.app.show_group_manager,
            width=25
        )
        self.app.btn_groups.pack(pady=(0, 30), padx=20, fill=X)

        # Sección de apoyo / donaciones
        support_frame = ttk.Labelframe(
            self.app.nav_panel,
            text="❤️ Apoya el proyecto",
            padding=10,
            bootstyle="secondary",
        )
        # Anclar al fondo para que quede separado del resto
        support_frame.pack(side=BOTTOM, fill=X, padx=20, pady=20)

        # Botón Ko‑fi
        ttk.Button(
            support_frame,
            text="☕ Invítame un café (Ko‑fi)",
            style="warning.TButton",
            command=getattr(self.app, "open_kofi", lambda: None),
            width=25,
        ).pack(fill=X)

        # Separador pequeño
        ttk.Separator(support_frame, orient=HORIZONTAL).pack(fill=X, pady=8)

        # Botón Transferencia MX
        ttk.Button(
            support_frame,
            text="🏦 Transferencia bancaria",
            style="secondary.Outline.TButton",
            command=getattr(self.app, "show_bank_transfer_dialog", lambda: None),
            width=25,
        ).pack(fill=X)

    def create_nav_buttons(self):
        """Create navigation buttons (reduced: Inicio only; view selection is in toolbar)."""
        button_style = {
            'style': 'primary.Outline.TButton',
            'width': 25,
            'padding': 10
        }

        # Solo Inicio en sidebar; el resto se elige en la barra de herramientas
        analyses = [("welcome", "Inicio")]

        self.app.nav_buttons = {}

        for analysis_type, text in analyses:
            icon = self.icon_map.get(analysis_type, "•")
            btn = ttk.Button(
                self.app.nav_panel,
                text=f"{icon} {text}",
                command=lambda t=analysis_type: self.app.show_analysis_view(t),
                **button_style
            )
            btn.pack(pady=(0, 8), padx=20, fill=X)
            self.app.nav_buttons[analysis_type] = btn

    def setup_content_area(self):
        """Configure the main content area with toolbar and Notebook (one tab per sheet)."""
        self.app.content_container = ttk.Frame(self.app.content_area)
        self.app.content_container.pack(fill=BOTH, expand=YES, padx=10, pady=10)

        # Toolbar above the notebook
        self.setup_toolbar()

        # Notebook for multiple sheets (tabs)
        self.app.sheets_notebook = ttk.Notebook(self.app.content_container)
        self.app.sheets_notebook.pack(fill=BOTH, expand=YES)

        # Sheet state: list of dicts per tab
        self.app.sheets = []
        self.app.current_sheet_index = 0

        # Create first default sheet
        self._add_sheet("Hoja 1")

        # Bind tab change to update active sheet
        self.app.sheets_notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def setup_toolbar(self):
        """Create toolbar with Load data, view selection, Run, Export, Manage groups."""
        toolbar = ttk.Frame(self.app.content_container)
        toolbar.pack(fill=X, pady=(0, 10))

        # Cargar datos
        ttk.Button(
            toolbar,
            text=f"{self.icon_map['file']} Cargar datos",
            style='info.Outline.TButton',
            command=getattr(self.app, 'toolbar_load_data', lambda: None),
            width=14
        ).pack(side=LEFT, padx=(0, 8))

        # Preparar datos / Construir matriz (merge hojas, n×p)
        ttk.Button(
            toolbar,
            text="Preparar datos",
            style='secondary.Outline.TButton',
            command=getattr(self.app, 'show_data_prep_dialog', lambda: None),
            width=12
        ).pack(side=LEFT, padx=(0, 8))

        ttk.Separator(toolbar, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=8, pady=4)

        # Vista / tipo de análisis (menú o botones compactos)
        view_label = ttk.Label(toolbar, text="Vista:", style='secondary.TLabel')
        view_label.pack(side=LEFT, padx=(0, 4))
        self.app.toolbar_view_var = tk.StringVar(value="welcome")
        view_combo = ttk.Combobox(
            toolbar,
            textvariable=self.app.toolbar_view_var,
            values=[
                "Inicio", "Serie de Tiempo", "Corte Transversal", "Análisis 3D",
                "Scatterplot", "Clustering Jerárquico", "Correlación/Redes"
            ],
            state="readonly",
            width=22
        )
        view_combo.pack(side=LEFT, padx=(0, 4))
        self._view_map = {
            "Inicio": "welcome",
            "Serie de Tiempo": "series",
            "Corte Transversal": "biplot",
            "Análisis 3D": "panel",
            "Scatterplot": "scatter",
            "Clustering Jerárquico": "hierarchical",
            "Correlación/Redes": "correlation",
        }
        self._view_to_display = {v: k for k, v in self._view_map.items()}
        view_combo.bind("<<ComboboxSelected>>", self._on_toolbar_view_selected)

        ttk.Separator(toolbar, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=8, pady=4)

        # Ejecutar
        self.app.toolbar_btn_run = ttk.Button(
            toolbar,
            text=f"{self.icon_map['run']} Ejecutar",
            style='success.TButton',
            command=self.app.analysis_manager.run_current_analysis,
            state=DISABLED,
            width=10
        )
        self.app.toolbar_btn_run.pack(side=LEFT, padx=(0, 4))

        # Exportar
        self.app.toolbar_btn_export = ttk.Button(
            toolbar,
            text=f"{self.icon_map['export']} Exportar",
            style='info.Outline.TButton',
            command=self.app.export_results,
            state=DISABLED,
            width=10
        )
        self.app.toolbar_btn_export.pack(side=LEFT, padx=(0, 4))

        # Gestionar Grupos
        ttk.Button(
            toolbar,
            text="🏷️ Grupos",
            style='warning.Outline.TButton',
            command=self.app.show_group_manager,
            width=8
        ).pack(side=LEFT, padx=(0, 4))

        ttk.Separator(toolbar, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=8, pady=4)

        # Herramientas / Análisis avanzado (Phase D - advanced_analytics.py)
        tools_btn = ttk.Menubutton(toolbar, text="Herramientas", style='secondary.Outline.TButton', width=12)
        tools_btn.pack(side=LEFT, padx=(0, 4))
        tools_menu = tk.Menu(tools_btn, tearoff=0)
        tools_btn["menu"] = tools_menu
        # Tests estadísticos (pre-PCA)
        tools_menu.add_command(label="Bartlett (esfericidad)", command=lambda: self._run_tool("bartlett"))
        tools_menu.add_command(label="KMO (adecuación muestral)", command=lambda: self._run_tool("kmo"))
        tools_menu.add_command(label="Alfa de Cronbach", command=lambda: self._run_tool("cronbach"))
        tools_menu.add_command(label="Distancia de Mahalanobis", command=lambda: self._run_tool("mahalanobis"))
        tools_menu.add_separator()
        # Reducción de dimensionalidad
        tools_menu.add_command(label="t-SNE", command=lambda: self._run_tool("tsne"))
        tools_menu.add_command(label="UMAP", command=lambda: self._run_tool("umap"))
        tools_menu.add_command(label="Análisis factorial (FA)", command=lambda: self._run_tool("fa"))
        tools_menu.add_command(label="ICA", command=lambda: self._run_tool("ica"))
        tools_menu.add_separator()
        # Clustering
        tools_menu.add_command(label="K-Means (codo/silueta)", command=lambda: self._run_tool("kmeans"))
        tools_menu.add_command(label="DBSCAN", command=lambda: self._run_tool("dbscan"))
        tools_menu.add_command(label="GMM (mezcla gaussianas)", command=lambda: self._run_tool("gmm"))
        tools_menu.add_separator()
        # Transformaciones de datos
        tools_menu.add_command(label="Yeo-Johnson", command=lambda: self._run_tool("yeo_johnson"))
        tools_menu.add_command(label="Winsorización", command=lambda: self._run_tool("winsorize"))
        tools_menu.add_command(label="Rank inverse normal", command=lambda: self._run_tool("rank_inverse_normal"))
        tools_menu.add_command(label="Robust scaling", command=lambda: self._run_tool("robust_scale"))

        self.app.toolbar_frame = toolbar

    def _on_toolbar_view_selected(self, event=None):
        """Switch the active sheet's view when toolbar combobox selection changes."""
        try:
            display = self.app.toolbar_view_var.get()
            view = self._view_map.get(display, "welcome")
            self.app.show_analysis_view(view)
        except Exception:
            pass

    def _run_tool(self, tool_name: str):
        """Run an advanced math tool on the active sheet's data."""
        if hasattr(self.app, "run_advanced_tool"):
            self.app.run_advanced_tool(tool_name)
        else:
            from tkinter import messagebox
            messagebox.showinfo("Herramientas", "Carga datos en la hoja activa y usa la barra de herramientas.")

    def _on_tab_changed(self, event=None):
        """Update current_sheet_index when user switches tab."""
        try:
            sel = self.app.sheets_notebook.select()
            idx = self.app.sheets_notebook.index(sel)
            if 0 <= idx < len(self.app.sheets):
                self.app.current_sheet_index = idx
                sheet = self.app.sheets[idx]
                self.app.current_analysis_type = sheet.get("view", "welcome")
                # Restore frame references for this sheet so run/export use the correct tab
                frames = sheet.get("frames") or {}
                self.app.current_series_frame = frames.get("series")
                self.app.panel_frame = frames.get("panel")
                self.app.biplot_frame = frames.get("biplot")
                self.app.scatter_frame = frames.get("scatter")
                self.app.hierarchical_frame = frames.get("hierarchical")
                self.app.correlation_frame = frames.get("correlation")
                if hasattr(self.app, "nav_buttons") and self.app.nav_buttons:
                    self.app.ui_manager.update_nav_buttons(self.app.current_analysis_type)
                self.sync_toolbar_view_from_sheet()
        except Exception:
            pass

    def sync_toolbar_view_from_sheet(self):
        """Update toolbar view combobox to match current sheet view."""
        if hasattr(self.app, "toolbar_view_var") and self.app.toolbar_view_var:
            display = self._view_to_display.get(self.app.current_analysis_type, "Inicio")
            self.app.toolbar_view_var.set(display)

    def _add_sheet(self, title="Nueva hoja"):
        """Add a new sheet (tab) with its own title, status, and dynamic frame."""
        sheet_container = ttk.Frame(self.app.sheets_notebook, padding=20)
        self.app.sheets_notebook.add(sheet_container, text=title)

        content_title = ttk.Label(
            sheet_container,
            text="Selecciona un tipo de análisis",
            font=("Helvetica", 24, "bold"),
            style='primary.TLabel'
        )
        content_title.pack(pady=(0, 30), anchor=W)

        status = ttk.Label(
            sheet_container,
            text="Listo para analizar",
            font=("Helvetica", 10),
            style='secondary.TLabel'
            )
        status.pack(pady=(0, 20), anchor=W)

        dynamic_frame = ScrolledFrame(sheet_container, autohide=True)
        dynamic_frame.pack(fill=BOTH, expand=YES)

        sheet_state = {
            "df": None,
            "file_path": "",
            "view": "welcome",
            "content_title": content_title,
            "status": status,
            "dynamic_frame": dynamic_frame,
            "container": sheet_container,
            "frames": {},  # view_name -> frame widget (series, panel, biplot, scatter, hierarchical, correlation)
        }
        self.app.sheets.append(sheet_state)
        return sheet_state

    def add_new_sheet(self):
        """Add a new sheet and switch to it. Returns the new sheet index."""
        idx = len(self.app.sheets)
        title = f"Hoja {idx + 1}"
        self._add_sheet(title)
        self.app.sheets_notebook.select(idx)
        self.app.current_sheet_index = idx
        self.app.show_analysis_view("welcome")
        return idx

    def clear_dynamic_content(self):
        """Clear the dynamic content area of the active sheet."""
        sheet = self.app.get_active_sheet()
        if sheet and sheet.get("dynamic_frame"):
            for widget in sheet["dynamic_frame"].winfo_children():
                widget.destroy()
            sheet["frames"] = {}

    def update_nav_buttons(self, active_type):
        """Update navigation button states."""
        for btn_type, button in self.app.nav_buttons.items():
            if btn_type == active_type:
                button.config(style='primary.TButton')  # Active button
            else:
                button.config(style='primary.Outline.TButton')  # Inactive buttons

    def show_loading(self, message="Cargando..."):
        """Show loading indicator."""
        try:
            # Create loading window if it doesn't exist
            if not hasattr(self.app, 'loading_window') or self.app.loading_window is None:
                self.app.loading_window = tk.Toplevel(self.app)
                self.app.loading_window.title("Cargando")
                self.app.loading_window.geometry("300x100")
                self.app.loading_window.resizable(False, False)
                self.app.loading_window.transient(self.app)
                self.app.loading_window.grab_set()

                # Center window
                self.app.loading_window.update_idletasks()
                x = (self.app.loading_window.winfo_screenwidth() // 2) - 150
                y = (self.app.loading_window.winfo_screenheight() // 2) - 50
                self.app.loading_window.geometry(f"300x100+{x}+{y}")

                # Content
                frame = ttk.Frame(self.app.loading_window, padding=20)
                frame.pack(fill=BOTH, expand=YES)

                self.app.loading_label = ttk.Label(
                    frame,
                    text=message,
                    font=("Helvetica", 10)
                )
                self.app.loading_label.pack(pady=(0, 10))

                # Progress bar
                self.app.progress_bar = ttk.Progressbar(
                    frame,
                    mode='indeterminate',
                    length=200
                )
                self.app.progress_bar.pack()
                self.app.progress_bar.start()

            else:
                # Update message if already exists
                if hasattr(self.app, 'loading_label'):
                    self.app.loading_label.config(text=message)

        except Exception as e:
            self.app.logger.warning(f"Error showing loading: {e}")

    def show_loading_with_progress(self, message="Procesando...", steps=None):
        """Show enhanced loading indicator with progress steps."""
        try:
            # Create loading window if it doesn't exist
            if not hasattr(self.app, 'loading_window') or self.app.loading_window is None:
                self.app.loading_window = tk.Toplevel(self.app)
                self.app.loading_window.title("Procesando Análisis")
                self.app.loading_window.geometry("400x180")
                self.app.loading_window.resizable(False, False)
                self.app.loading_window.transient(self.app)
                self.app.loading_window.grab_set()

                # Center window
                self.app.loading_window.update_idletasks()
                x = (self.app.loading_window.winfo_screenwidth() // 2) - 200
                y = (self.app.loading_window.winfo_screenheight() // 2) - 90
                self.app.loading_window.geometry(f"400x180+{x}+{y}")

                # Content
                frame = ttk.Frame(self.app.loading_window, padding=20)
                frame.pack(fill=BOTH, expand=YES)

                # Main message
                self.app.loading_label = ttk.Label(
                    frame,
                    text=message,
                    font=("Helvetica", 11, "bold")
                )
                self.app.loading_label.pack(pady=(0, 10))

                # Current step indicator
                self.app.step_label = ttk.Label(
                    frame,
                    text="Iniciando...",
                    font=("Helvetica", 9)
                )
                self.app.step_label.pack(pady=(0, 10))

                # Progress bar (determinate mode for steps)
                self.app.progress_bar = ttk.Progressbar(
                    frame,
                    mode='determinate',
                    length=300,
                    maximum=100
                )
                self.app.progress_bar.pack()

                # Cancel button
                self.app.cancel_button = ttk.Button(
                    frame,
                    text="Cancelar",
                    command=self._cancel_analysis,
                    style='danger.Outline.TButton'
                )
                self.app.cancel_button.pack(pady=(10, 0))

                # Initialize progress tracking
                self.app.current_step = 0
                self.app.total_steps = len(steps) if steps else 1
                self.app.progress_steps = steps or ["Procesando"]

            else:
                # Update message and reset progress if already exists
                if hasattr(self.app, 'loading_label'):
                    self.app.loading_label.config(text=message)
                if hasattr(self.app, 'step_label'):
                    self.app.step_label.config(text="Iniciando...")
                if hasattr(self.app, 'progress_bar'):
                    self.app.progress_bar.config(value=0)
                if hasattr(self.app, 'cancel_button'):
                    self.app.cancel_button.config(state=NORMAL)

                # Reset progress tracking
                self.app.current_step = 0
                self.app.total_steps = len(steps) if steps else 1
                self.app.progress_steps = steps or ["Procesando"]

        except Exception as e:
            self.app.logger.warning(f"Error showing loading with progress: {e}")
            # Fallback to simple loading
            self.show_loading(message)

    def update_progress(self, step_index: int, step_name: str):
        """Update progress indicator."""
        try:
            if hasattr(self.app, 'step_label') and hasattr(self.app, 'progress_bar'):
                self.app.step_label.config(text=f"Paso {step_index + 1}: {step_name}")

                # Calculate progress percentage
                progress = ((step_index + 1) / self.app.total_steps) * 100
                self.app.progress_bar.config(value=progress)

                self.app.current_step = step_index

        except Exception as e:
            self.app.logger.warning(f"Error updating progress: {e}")

    def _cancel_analysis(self):
        """Cancel the current analysis."""
        try:
            if hasattr(self.app, 'analysis_manager'):
                self.app.analysis_manager.cancel_current_analysis()
                if hasattr(self.app, 'cancel_button'):
                    self.app.cancel_button.config(state=DISABLED, text="Cancelando...")
        except Exception as e:
            self.app.logger.warning(f"Error cancelling analysis: {e}")

    def hide_loading(self):
        """Hide loading indicator."""
        try:
            if hasattr(self.app, 'loading_window') and self.app.loading_window is not None:
                self.app.loading_window.destroy()
                self.app.loading_window = None

                # Clean up progress-related attributes
                for attr in ['loading_label', 'step_label', 'progress_bar', 'cancel_button',
                           'current_step', 'total_steps', 'progress_steps']:
                    if hasattr(self.app, attr):
                        delattr(self.app, attr)

        except Exception as e:
            self.app.logger.warning(f"Error hiding loading: {e}")

    def display_figure(self, fig):
        """Display a matplotlib figure in the UI."""
        try:
            # Clear any existing figure
            self.clear_dynamic_content()

            # Create canvas for the figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import matplotlib.pyplot as plt

            # Create frame for the plot
            plot_frame = ttk.Frame(self.app.dynamic_frame)
            plot_frame.pack(fill=BOTH, expand=YES, padx=10, pady=10)

            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()

            # Pack canvas
            canvas.get_tk_widget().pack(fill=BOTH, expand=YES)

            # Add toolbar if available
            try:
                from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
                toolbar = NavigationToolbar2Tk(canvas, plot_frame)
                toolbar.update()
                toolbar.pack(side=BOTTOM, fill=X)
            except Exception as e:
                self.app.logger.debug(f"Could not add matplotlib toolbar: {e}")

            # Store reference to prevent garbage collection
            if not hasattr(self.app, 'current_canvas'):
                self.app.current_canvas = []
            self.app.current_canvas.append(canvas)

            self.app.logger.info("Figure displayed successfully in UI")

        except Exception as e:
            self.app.logger.error(f"Error displaying figure: {e}")
            import traceback
            self.app.logger.error(traceback.format_exc())

            # Show error message in UI
            error_frame = ttk.Frame(self.app.dynamic_frame)
            error_frame.pack(fill=BOTH, expand=YES, padx=20, pady=20)

            ttk.Label(
                error_frame,
                text="❌ Error al mostrar el gráfico",
                font=("Helvetica", 14, "bold"),
                style='danger.TLabel'
            ).pack(pady=(0, 10))

            ttk.Label(
                error_frame,
                text=f"No se pudo mostrar el gráfico:\n{str(e)}",
                font=("Helvetica", 10),
                style='secondary.TLabel',
                justify=CENTER
            ).pack()