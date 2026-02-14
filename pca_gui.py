import os
import json
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, simpledialog, colorchooser, ttk
import data_loader_module as dl
import preprocessing_module as dl_prep
import visualization_module as dl_viz
import pca_module as pca_mod
from constants import MAPEO_INDICADORES, CODE_TO_NAME
from pca_panel3d_logic import PCAPanel3DLogic
from project_save_config import ProjectConfig
import platform
import subprocess
import sys
import webbrowser
import functools

# Importar módulos refactorizados
from ui_components import UIComponents
from event_handlers import EventHandlers
from dialogs import DialogManager

# Importar sistemas mejorados
try:
    from logging_config import get_logger, setup_application_logging
    from config_manager import get_config, update_config, save_config
    from performance_optimizer import profiled

    # Configurar logging mejorado
    setup_application_logging(debug_mode=False)
    ENHANCED_SYSTEMS_AVAILABLE = True
except ImportError:
    ENHANCED_SYSTEMS_AVAILABLE = False

    def get_logger(name):
        return logging.getLogger(name)


DEPENDENCY_MANAGER_AVAILABLE = False


def safe_import(module_name, alias=None):
    import importlib

    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


# Verificación de dependencias críticas al inicio
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Warning: matplotlib no disponible. Las visualizaciones no funcionarán.")


# === i18n infrastructure ===


# --- Inicialización robusta del idioma ---
def get_saved_lang():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            settings = json.load(f)
            return settings.get("lang", "es")
    except Exception:
        return "es"


_LANG = get_saved_lang()
_TRANSLATIONS = None


def set_language(lang):
    global _LANG, _TRANSLATIONS
    _LANG = lang
    if lang == "en":
        from i18n_en import TRANSLATIONS as T
    else:
        from i18n_es import TRANSLATIONS as T
    _TRANSLATIONS = T


set_language(_LANG)


def tr(key):
    if _TRANSLATIONS is None:
        set_language(_LANG)
    return _TRANSLATIONS.get(key, key)


# === Configuración de rutas y logging ===
PROJECTS_DIR = r"C:\Users\messi\OneDrive\Escritorio\escuela\Servicio Social\Python\PCA\Proyectos save"
LOG_PATH = os.path.join(os.path.dirname(__file__), "pca_gui.log")

# Usar sistema de logging mejorado si está disponible
if ENHANCED_SYSTEMS_AVAILABLE:
    logger = get_logger("pca_gui")
else:
    # Fallback al sistema de logging original
    logger = logging.getLogger("pca_gui")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(
            LOG_PATH, maxBytes=200_000, backupCount=3, encoding="utf-8"
        )
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)


# === Decorador para callbacks seguros ===
def safe_gui_callback(func):
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {func.__name__}")
            tk = args[0] if args else None
            from tkinter import messagebox

            msg = str(e)
            if hasattr(e, "args") and e.args and isinstance(e.args[0], str):
                msg = e.args[0]
            messagebox.showerror(
                tr("error"),
                f"{tr('unexpected_error') if 'unexpected_error' in _TRANSLATIONS else 'Unexpected error'}:\n{msg}",
            )

    return wrapper


# === Definición de la clase principal de la app ===

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.json")


class PCAApp(tk.Tk):
    """Aplicación PCA GUI principal."""

    def __init__(self):
        super().__init__()
        self.title("🔬 ReduxLab - Análisis Avanzado")

        # Configurar tamaño mínimo y inicial
        self.geometry("900x700")
        self.minsize(800, 600)

        logger.info("Iniciando aplicación PCA GUI")

        # Inicializar componentes refactorizados
        self.ui = UIComponents(self)
        self.events = EventHandlers(self)
        self.dialogs = DialogManager(self)

        # Initialize analysis manager
        from analysis_manager import AnalysisManager
        self.analysis_manager = AnalysisManager(self)

        self._setup_config()
        self._setup_ui()
        self._bind_events()

    @safe_gui_callback
    @profiled if ENHANCED_SYSTEMS_AVAILABLE else lambda x: x
    def start_cross_section_analysis(self):
        """Inicia el flujo de análisis de corte transversal (varias unidades, un año o varios años)."""
        try:
            self._last_analysis_type = "cross_section_config"
            self.status.config(text="Flujo: Corte transversal (varias unidades, años)")
            self.cross_section_wizard()
        except tk.TclError:
            pass

    def cross_section_wizard(self):
        self.step_select_file(
            lambda: self.step_select_indicators(
                lambda: self.step_select_units(
                    lambda: self.step_select_year(
                        lambda: self.run_cross_section_analysis(), multi=True
                    ),
                    allow_multiple=True,
                ),
                multi=True,
            )
        )

    @safe_gui_callback
    @profiled if ENHANCED_SYSTEMS_AVAILABLE else lambda x: x
    def run_series_analysis(self):
        """Ejecuta el análisis de serie de tiempo para la unidad y años seleccionados (pueden ser varios años)."""
        from pca_logic import PCAAnalysisLogic

        cfg = self.project_config.series_config
        estrategia, params = None, None

        # Validaciones iniciales
        if not cfg.get("data_file"):
            messagebox.showerror("Error", "No se ha seleccionado un archivo de datos.")
            return
        if not cfg.get("selected_indicators"):
            messagebox.showerror("Error", "No se han seleccionado indicadores.")
            return
        if not cfg.get("selected_units") or len(cfg["selected_units"]) == 0:
            messagebox.showerror("Error", "No se ha seleccionado una unidad de investigación.")
            return

        # Obtener los años seleccionados (pueden ser varios)
        selected_years = []
        if cfg.get("selected_years"):
            selected_years = (
                cfg["selected_years"]
                if isinstance(cfg["selected_years"], list)
                else [cfg["selected_years"]]
            )

        print(f"Ejecutando análisis de serie de tiempo:")
        print(f"  - Archivo: {cfg['data_file']}")
        print(f"  - Indicadores: {cfg['selected_indicators']}")
        print(f"  - Unidad: {cfg['selected_units'][0]}")
        print(f"  - Años: {selected_years if selected_years else 'Todos disponibles'}")

        # Primero ejecuta una validación para detectar datos faltantes
        temp_results = PCAAnalysisLogic.run_series_analysis_logic(
            cfg,
            imputation_strategy=None,
            imputation_params=None,
            selected_years=selected_years,
        )
        if "warning" in temp_results and (
            "faltantes" in temp_results["warning"]
            or "datos faltantes" in temp_results["warning"]
        ):
            respuesta = messagebox.askyesno(
                "Datos faltantes detectados",
                f"Se encontraron datos faltantes en la serie de tiempo.\n¿Quieres imputar los valores faltantes?\n\nDetalle: {temp_results['warning']}",
            )
            if respuesta:
                estrategia, params = self.gui_select_imputation_strategy()
            else:
                # Si el usuario no quiere imputar, mostrar el warning y salir
                messagebox.showwarning("Advertencia", temp_results["warning"])
                return

        # Ejecuta la lógica real con la estrategia de imputación seleccionada
        results = PCAAnalysisLogic.run_series_analysis_logic(
            cfg,
            imputation_strategy=estrategia,
            imputation_params=params,
            selected_years=selected_years,
        )
        if "warning" in results:
            messagebox.showwarning("Atención", results["warning"])
            # Continuar con el procesamiento aunque haya warnings
        if "error" in results:
            messagebox.showerror("Error", results["error"])
            return

        # Verificar que tenemos resultados válidos para graficar
        valid_results = False
        for key in ["df_consolidado", "df_imputado", "df_estandarizado"]:
            if results.get(key) is not None and not results[key].empty:
                valid_results = True
                break

        if not valid_results:
            messagebox.showerror(
                "Error", "No se generaron datos válidos para visualizar."
            )
            return

        # Graficar resultados de serie de tiempo
        dfs_dict = {
            "Consolidado": results.get("df_consolidado"),
            "Imputado": results.get("df_imputado"),
            "Estandarizado": results.get("df_estandarizado"),
        }

        # NUEVO: Agregar componentes principales si están disponibles
        if results.get("df_componentes_principales") is not None:
            dfs_dict["Componentes Principales (PCA)"] = results[
                "df_componentes_principales"
            ]
            print("✅ Agregando visualización de componentes principales en el tiempo")

        try:
            dl_viz.graficar_series_de_tiempo(
                dfs_dict, titulo_general="Serie de Tiempo - Análisis PCA"
            )

            # NUEVO: Crear gráfico especializado para componentes principales si están disponibles
            if results.get("df_componentes_principales") is not None:
                varianza_info = None
                if (
                    results.get("pca_sugerencias")
                    and results["pca_sugerencias"].get("evr") is not None
                ):
                    varianza_info = results["pca_sugerencias"]["evr"]

                print("🎨 Creando gráfico especializado de componentes principales...")
                dl_viz.graficar_componentes_principales_tiempo(
                    results["df_componentes_principales"],
                    varianza_explicada=varianza_info,
                    titulo=f"Evolución de Componentes Principales - {cfg['selected_units'][0]}",
                )
        except Exception as e:
            print(f"Error al generar gráficos: {e}")
            messagebox.showwarning(
                "Advertencia",
                f"Se completó el análisis pero hubo problemas al generar los gráficos: {e}",
            )

        # Preguntar si desea exportar los resultados
        if messagebox.askyesno(
            tr("export_title") if "export_title" in _TRANSLATIONS else "¿Exportar?",
            (
                tr("export_msg")
                if "export_msg" in _TRANSLATIONS
                else "¿Quieres guardar los resultados del análisis de serie de tiempo?"
            ),
        ):
            filename = filedialog.asksaveasfilename(
                title=(
                    tr("save_results_title")
                    if "save_results_title" in _TRANSLATIONS
                    else "Guardar resultados serie de tiempo"
                ),
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx *.xls")],
            )
            if filename:
                try:
                    import pandas as pd

                    with pd.ExcelWriter(filename) as writer:
                        if (
                            results.get("df_consolidado") is not None
                            and not results["df_consolidado"].empty
                        ):
                            results["df_consolidado"].to_excel(
                                writer, sheet_name="Consolidado", index=True
                            )
                        if (
                            results.get("df_imputado") is not None
                            and not results["df_imputado"].empty
                        ):
                            results["df_imputado"].to_excel(
                                writer, sheet_name="Imputado", index=True
                            )
                        if (
                            results.get("df_estandarizado") is not None
                            and not results["df_estandarizado"].empty
                        ):
                            results["df_estandarizado"].to_excel(
                                writer, sheet_name="Estandarizado", index=True
                            )
                        # NUEVO: Exportar componentes principales
                        if (
                            results.get("df_componentes_principales") is not None
                            and not results["df_componentes_principales"].empty
                        ):
                            results["df_componentes_principales"].to_excel(
                                writer, sheet_name="Componentes_PCA", index=True
                            )
                        if (
                            results.get("df_covarianza") is not None
                            and not results["df_covarianza"].empty
                        ):
                            results["df_covarianza"].to_excel(
                                writer, sheet_name="Matriz_Covarianza", index=True
                            )
                        if (
                            results.get("pca_sugerencias")
                            and results["pca_sugerencias"].get("df_varianza_explicada")
                            is not None
                        ):
                            results["pca_sugerencias"][
                                "df_varianza_explicada"
                            ].to_excel(writer, sheet_name="PCA_VarianzaExp", index=True)
                    messagebox.showinfo(
                        tr("done") if "done" in _TRANSLATIONS else "Listo",
                        (
                            tr("file_saved").format(filename=filename)
                            if "file_saved" in _TRANSLATIONS
                            else f"Archivo guardado:\n{filename}"
                        ),
                    )
                except Exception as e:
                    messagebox.showerror("Error", f"No se pudo guardar el archivo: {e}")

        messagebox.showinfo(
            "Info", "Análisis de serie de tiempo completado correctamente."
        )

    def start_series_analysis(self):
        """Inicia el flujo de análisis de serie de tiempo (1 unidad, varios años)."""
        try:
            self.status.config(text="Flujo: Serie de tiempo (1 unidad)")
            self.series_wizard()
        except tk.TclError:
            pass

    def series_wizard(self):
        # Limpia selección de años al iniciar el flujo
        self.project_config.series_config["selected_years"] = []
        self.step_select_file(lambda: self._series_select_indicators())

    def _series_select_indicators(self):
        # Limpia selección de años al cambiar indicadores
        self.project_config.series_config["selected_years"] = []
        self.step_select_indicators(lambda: self._series_select_units(), multi=True)

    def _series_select_units(self):
        # Limpia selección de años al cambiar unidad
        self.project_config.series_config["selected_years"] = []
        self.step_select_units(
            lambda: self._series_select_years(), allow_multiple=False
        )

    def _series_select_years(self):
        self.step_select_year(lambda: self.sync_gui_from_cfg(), multi=True)

    def _create_settings_section(self, parent, title, pady_top=0):
        """Crea una sección con título en la ventana de configuración."""
        section_label = tk.Label(
            parent,
            text=title,
            font=("Segoe UI", 12, "bold"),
            bg=getattr(self, "bg_primary", "#ffffff"),
            fg=getattr(self, "accent_color", "#3b82f6"),
        )
        section_label.pack(anchor="w", pady=(pady_top, 5))
        return section_label

    def _create_modern_entry(self, parent, textvariable, placeholder="", width=25):
        """Crea un Entry moderno con placeholder."""
        entry_frame = tk.Frame(parent, bg=getattr(self, "bg_primary", "#ffffff"))
        entry_frame.pack(fill="x", pady=(5, 10))

        entry = tk.Entry(
            entry_frame,
            textvariable=textvariable,
            font=("Segoe UI", 10),
            bg=getattr(self, "bg_secondary", "#f8fafc"),
            fg=getattr(self, "fg_primary", "#1e293b"),
            relief="flat",
            bd=1,
            width=width,
            insertbackground=getattr(self, "fg_primary", "#1e293b"),
        )
        entry.pack(padx=(20, 0), anchor="w")

        # Añadir placeholder como etiqueta si está vacío
        if placeholder:
            placeholder_label = tk.Label(
                entry_frame,
                text=f"💡 {placeholder}",
                font=("Segoe UI", 8),
                fg=getattr(self, "fg_secondary", "#64748b"),
                bg=getattr(self, "bg_primary", "#ffffff"),
            )
            placeholder_label.pack(padx=(25, 0), anchor="w")

    def create_modern_window(self, title, width=400, height=500, resizable=True):
        """Crea una ventana moderna con el tema aplicado."""
        win = Toplevel(self)
        win.title(title)
        win.geometry(f"{width}x{height}")
        win.resizable(resizable, resizable)

        # Usar colores seguros por defecto
        bg_color = getattr(self, "bg_primary", "#ffffff")
        win.configure(bg=bg_color)

        # Centrar ventana
        win.update_idletasks()
        x = (win.winfo_screenwidth() // 2) - (width // 2)
        y = (win.winfo_screenheight() // 2) - (height // 2)
        win.geometry(f"{width}x{height}+{x}+{y}")

        return win

    def create_modern_listbox(self, parent, selectmode=tk.MULTIPLE, height=15):
        """Crea un Listbox moderno con scrollbar."""
        # Frame contenedor
        listbox_frame = tk.Frame(parent, bg=getattr(self, "bg_primary", "#ffffff"))

        # Listbox con estilo moderno
        listbox = tk.Listbox(
            listbox_frame,
            selectmode=selectmode,
            font=("Segoe UI", 10),
            bg=getattr(self, "bg_secondary", "#f8fafc"),
            fg=getattr(self, "fg_primary", "#1e293b"),
            selectbackground=getattr(self, "accent_color", "#3b82f6"),
            selectforeground="white",
            relief="flat",
            bd=1,
            height=height,
            activestyle="none",
        )

        # Scrollbar moderna
        scrollbar = tk.Scrollbar(
            listbox_frame,
            orient="vertical",
            command=listbox.yview,
            bg=getattr(self, "bg_secondary", "#f8fafc"),
            troughcolor=getattr(self, "bg_primary", "#ffffff"),
            relief="flat",
            bd=0,
        )

        listbox.configure(yscrollcommand=scrollbar.set)

        # Empaquetar
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        return listbox_frame, listbox

    def apply_matplotlib_style(self):
        import matplotlib.pyplot as plt

        if getattr(self, "theme", "light") == "dark":
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

    def apply_theme(self):
        """Aplica tema visual moderno con colores mejorados y efectos."""
        # Colores modernos mejorados
        if getattr(self, "theme", "light") == "dark":
            # Tema oscuro moderno
            self.bg_primary = "#1e1e2e"
            self.bg_secondary = "#313244"
            self.fg_primary = "#cdd6f4"
            self.fg_secondary = "#a6adc8"
            self.accent_color = "#89b4fa"
            self.success_color = "#a6e3a1"
            self.warning_color = "#f9e2af"
            self.error_color = "#f38ba8"
            self.btn_primary = "#89b4fa"
            self.btn_secondary = "#585b70"
            self.btn_success = "#a6e3a1"
            self.btn_hover = "#74c7ec"
        else:
            # Tema claro moderno
            self.bg_primary = "#ffffff"
            self.bg_secondary = "#f8fafc"
            self.fg_primary = "#1e293b"
            self.fg_secondary = "#475569"
            self.accent_color = "#3b82f6"
            self.success_color = "#10b981"
            self.warning_color = "#f59e0b"
            self.error_color = "#ef4444"
            self.btn_primary = "#3b82f6"
            self.btn_secondary = "#64748b"
            self.btn_success = "#10b981"
            self.btn_hover = "#2563eb"

        # Aplicar colores base
        self.configure(bg=self.bg_primary)

        # Actualizar widgets recursivamente
        self._update_widget_theme(self)

        # Menú con colores modernos
        if hasattr(self, "menu_bar"):
            self.menu_bar.configure(
                bg=self.bg_secondary,
                fg=self.fg_primary,
                activebackground=self.accent_color,
                activeforeground=self.bg_primary,
            )

    def _update_widget_theme(self, parent):
        """Actualiza recursivamente el tema de todos los widgets."""
        for widget in parent.winfo_children():
            widget_class = widget.winfo_class()

            if widget_class == "Label":
                widget.configure(bg=self.bg_primary, fg=self.fg_primary)
            elif widget_class == "Frame":
                widget.configure(bg=self.bg_primary)
                self._update_widget_theme(widget)  # Recursivo para frames
            elif widget_class == "Button":
                # No aplicar tema automático a botones, se manejan individualmente
                pass
            elif widget_class == "Entry":
                widget.configure(
                    bg=self.bg_secondary,
                    fg=self.fg_primary,
                    insertbackground=self.fg_primary,
                    relief="flat",
                    bd=1,
                )
            elif widget_class == "Listbox":
                widget.configure(
                    bg=self.bg_secondary,
                    fg=self.fg_primary,
                    selectbackground=self.accent_color,
                    relief="flat",
                    bd=1,
                )

    def apply_font_settings(self):
        """Aplica configuración de fuente moderna."""
        font = getattr(self, "custom_font", "Segoe UI")
        fontsize = getattr(self, "custom_fontsize", 10)

        # Fuentes diferenciadas para jerarquía visual
        self.font_title = (font, fontsize + 4, "bold")
        self.font_button = (font, fontsize, "normal")
        self.font_label = (font, fontsize, "normal")
        self.font_small = (font, fontsize - 1, "normal")

        # Aplicar a widgets principales
        for widget in self.winfo_children():
            if isinstance(widget, tk.Label):
                if hasattr(widget, "_is_title"):
                    widget.configure(font=self.font_title)
                else:
                    widget.configure(font=self.font_label)

    def create_modern_button(
        self, parent, text, command=None, style="primary", width=None, height=2
    ):
        """Crea un botón moderno con efectos hover y colores mejorados."""
        # Asegurar que las fuentes estén definidas
        if not hasattr(self, "font_button"):
            font = getattr(self, "custom_font", "Segoe UI")
            fontsize = getattr(self, "custom_fontsize", 10)
            self.font_button = (font, fontsize, "normal")

        # Configurar colores según el estilo
        if style == "primary":
            bg_normal = getattr(self, "btn_primary", "#3b82f6")
            bg_hover = getattr(self, "btn_hover", "#2563eb")
            fg_color = "#ffffff"
        elif style == "success":
            bg_normal = getattr(self, "btn_success", "#10b981")
            bg_hover = "#059669"
            fg_color = "#ffffff"
        elif style == "secondary":
            bg_normal = getattr(self, "btn_secondary", "#64748b")
            bg_hover = "#475569"
            fg_color = "#ffffff"
        else:
            bg_normal = getattr(self, "btn_primary", "#3b82f6")
            bg_hover = getattr(self, "btn_hover", "#2563eb")
            fg_color = "#ffffff"

        # Crear botón con configuración moderna
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg_normal,
            fg=fg_color,
            font=self.font_button,
            relief="flat",
            bd=0,
            padx=20,
            pady=8,
            cursor="hand2",
            width=width,
            height=height,
        )

        # Añadir efectos hover
        btn.bind("<Enter>", lambda e: btn.configure(bg=bg_hover))
        btn.bind("<Leave>", lambda e: btn.configure(bg=bg_normal))

        return btn

    def load_settings(self):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {
                "last_dir": "",
                "theme": "light",
                "window_size": "900x700",
                "custom_font": "Segoe UI",
                "custom_fontsize": 10,
                "lang": "es",
            }

    def save_settings(self):
        settings = {
            "last_dir": getattr(self, "last_dir", ""),
            "theme": getattr(self, "theme", "light"),
            "window_size": self.geometry(),
            "custom_font": getattr(self, "custom_font", "Segoe UI"),
            "custom_fontsize": getattr(self, "custom_fontsize", 10),
            "lang": getattr(self, "lang", "es"),
        }
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"No se pudo guardar settings.json: {e}")

    def apply_settings(self, settings):
        # Tamaño de ventana
        if "window_size" in settings:
            try:
                self.geometry(settings["window_size"])
            except Exception:
                pass
        self.theme = settings.get("theme", "light")
        self.last_dir = settings.get("last_dir", "")
        self.custom_font = settings.get("custom_font", "Arial")
        self.custom_fontsize = settings.get("custom_fontsize", 12)
        self.lang = settings.get("lang", "es")
        self.apply_matplotlib_style()

    def _setup_config(self):
        """Carga la configuración inicial y crea directorios necesarios."""
        settings = self.load_settings()
        self.apply_settings(settings)

        # Inicializar colores de tema y fuentes antes de crear UI
        self.apply_theme()
        self.apply_font_settings()

        self.project_config = ProjectConfig()
        if not os.path.exists(PROJECTS_DIR):
            os.makedirs(PROJECTS_DIR)

    def _bind_events(self):
        """Enlaza eventos de la aplicación (por ejemplo cerrar ventana)."""
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_ui(self):
        """Crea la barra de menús y widgets principales de la aplicación."""
        # ========== Menús ==========
        menu_bar = tk.Menu(self)

        # Configuración
        mnu_settings = tk.Menu(menu_bar, tearoff=0)
        mnu_settings.add_command(
            label="Preferencias... ⚙️", command=self.dialogs.show_settings_window
        )
        menu_bar.add_cascade(
            label=(
                tr("settings_menu")
                if "settings_menu" in _TRANSLATIONS
                else "Configuración"
            ),
            menu=mnu_settings,
        )

        # Proyecto
        mnu_project = tk.Menu(menu_bar, tearoff=0)
        mnu_project.add_command(label=tr("new_project"), command=self.new_project)
        mnu_project.add_command(label=tr("open_project"), command=self.load_project)
        mnu_project.add_command(label=tr("save_project"), command=self.save_project)
        menu_bar.add_cascade(label=tr("project"), menu=mnu_project)

        # Editar
        mnu_edit = tk.Menu(menu_bar, tearoff=0)
        # Serie de Tiempo
        mnu_edit_series = tk.Menu(mnu_edit, tearoff=0)
        for lbl, cmd in [
            ("Editar título...", lambda: self.edit_title_dialog("series_config")),
            ("Editar leyenda...", lambda: self.edit_legend_dialog("series_config")),
            ("Asignar colores...", lambda: self.edit_colors_dialog("series_config")),
            ("Modificar unidades", lambda: self.edit_units_dialog("series_config")),
            (
                "Editar pie de página...",
                lambda: self.edit_footer_dialog("series_config"),
            ),
        ]:
            mnu_edit_series.add_command(label=lbl, command=cmd)
        mnu_edit.add_cascade(
            label=(
                tr("series_menu")
                if "series_menu" in _TRANSLATIONS
                else "Serie de Tiempo"
            ),
            menu=mnu_edit_series,
        )

        # Biplot 2D
        mnu_edit_biplot = tk.Menu(mnu_edit, tearoff=0)
        for lbl, cmd in [
            (
                "Editar título...",
                lambda: self.edit_title_dialog("cross_section_config"),
            ),
            (
                "Editar leyenda...",
                lambda: self.edit_legend_dialog("cross_section_config"),
            ),
            (
                "Asignar colores...",
                lambda: self.edit_colors_dialog("cross_section_config"),
            ),
            (
                "Modificar unidades",
                lambda: self.edit_units_dialog("cross_section_config"),
            ),
            (
                "Editar pie de página...",
                lambda: self.edit_footer_dialog("cross_section_config"),
            ),
        ]:
            mnu_edit_biplot.add_command(label=lbl, command=cmd)
        mnu_edit.add_cascade(
            label=tr("biplot_menu") if "biplot_menu" in _TRANSLATIONS else "Biplot 2D",
            menu=mnu_edit_biplot,
        )

        # Scatterplot PCA (nuevo)
        mnu_edit_scatter = tk.Menu(mnu_edit, tearoff=0)
        for lbl, cmd in [
            (
                "Editar título...",
                lambda: self.edit_title_dialog("cross_section_config"),
            ),
            (
                "Editar leyenda...",
                lambda: self.edit_legend_dialog("cross_section_config"),
            ),
            (
                "Asignar colores...",
                lambda: self.edit_colors_dialog("cross_section_config"),
            ),
            (
                "Modificar unidades",
                lambda: self.edit_units_dialog("cross_section_config"),
            ),
            (
                "Editar pie de página...",
                lambda: self.edit_footer_dialog("cross_section_config"),
            ),
        ]:
            mnu_edit_scatter.add_command(label=lbl, command=cmd)
        mnu_edit.add_cascade(label="Scatterplot (PCA)", menu=mnu_edit_scatter)

        # Biplot Avanzado
        mnu_edit_biplot_advanced = tk.Menu(mnu_edit, tearoff=0)
        mnu_edit_biplot_advanced.add_command(
            label="Crear Biplot Avanzado...", command=self.create_advanced_biplot_dialog
        )
        # Temporarily commented out - methods not implemented yet
        # mnu_edit_biplot_advanced.add_command(
        #     label="Configurar Categorías...", command=self.configure_categories_dialog
        # )
        # mnu_edit_biplot_advanced.add_command(
        #     label="Configurar Marcadores...", command=self.configure_markers_dialog
        # )
        # mnu_edit_biplot_advanced.add_command(
        #     label="Configurar Colores...", command=self.configure_colors_dialog
        # )
        mnu_edit.add_cascade(label="Biplot Avanzado 🎨", menu=mnu_edit_biplot_advanced)

        # Panel 3D
        mnu_edit_panel = tk.Menu(mnu_edit, tearoff=0)
        for lbl, cmd in [
            ("Editar título...", lambda: self.edit_title_dialog("panel_config")),
            ("Editar leyenda...", lambda: self.edit_legend_dialog("panel_config")),
            ("Asignar colores...", lambda: self.edit_colors_dialog("panel_config")),
            ("Modificar unidades", lambda: self.edit_units_dialog("panel_config")),
            (
                "Editar pie de página...",
                lambda: self.edit_footer_dialog("panel_config"),
            ),
        ]:
            mnu_edit_panel.add_command(label=lbl, command=cmd)
        mnu_edit.add_cascade(
            label=tr("panel_menu") if "panel_menu" in _TRANSLATIONS else "PCA 3D",
            menu=mnu_edit_panel,
        )

        menu_bar.add_cascade(
            label=tr("edit_menu") if "edit_menu" in _TRANSLATIONS else "Editar",
            menu=mnu_edit,
        )

        # Ayuda
        mnu_help = tk.Menu(menu_bar, tearoff=0)
        mnu_help.add_command(
            label=tr("manual_menu") if "manual_menu" in _TRANSLATIONS else "Manual",
            command=self.dialogs.show_manual_window,
        )
        mnu_help.add_command(
            label=(
                tr("about_menu")
                if "about_menu" in _TRANSLATIONS
                else "Acerca de nosotros"
            ),
            command=self.dialogs.show_about_window,
        )
        menu_bar.add_cascade(
            label=tr("help_menu") if "help_menu" in _TRANSLATIONS else "Ayuda",
            menu=mnu_help,
        )

        self.config(menu=menu_bar)
        # Referencias
        self.menu_bar = menu_bar
        self.mnu_project = mnu_project
        self.mnu_edit = mnu_edit
        self.mnu_help = mnu_help
        self.mnu_settings = mnu_settings
        self.mnu_edit_series = mnu_edit_series
        self.mnu_edit_biplot = mnu_edit_biplot
        self.mnu_edit_biplot_advanced = mnu_edit_biplot_advanced
        self.mnu_edit_panel = mnu_edit_panel

        # ========== Interfaz principal ==========
        title_frame = tk.Frame(
            self, bg=self.bg_primary if hasattr(self, "bg_primary") else "#ffffff"
        )
        title_frame.pack(pady=(30, 20), fill="x")
        self.lbl_analysis_type = tk.Label(
            title_frame,
            text="🔬 " + tr("select_analysis_type"),
            font=("Segoe UI", 18, "bold"),
            bg=self.bg_primary if hasattr(self, "bg_primary") else "#ffffff",
            fg=self.fg_primary if hasattr(self, "fg_primary") else "#1e293b",
        )
        self.lbl_analysis_type._is_title = True
        self.lbl_analysis_type.pack()

        main_container = tk.Frame(
            self, bg=self.bg_primary if hasattr(self, "bg_primary") else "#ffffff"
        )
        main_container.pack(pady=20, padx=40, fill="both", expand=True)

        def add_section(
            parent, emoji, title, btn_text, start_cmd, run_cmd, attr_prefix
        ):
            frame = tk.Frame(
                parent, bg=self.bg_primary if hasattr(self, "bg_primary") else "#ffffff"
            )
            frame.pack(pady=10, fill="x")
            tk.Label(
                frame,
                text=f"{emoji} {title}",
                font=("Segoe UI", 11, "bold"),
                bg=self.bg_primary if hasattr(self, "bg_primary") else "#ffffff",
                fg=self.fg_primary if hasattr(self, "fg_primary") else "#1e293b",
            ).pack(anchor="w", pady=(0, 5))
            inner = tk.Frame(
                frame, bg=self.bg_primary if hasattr(self, "bg_primary") else "#ffffff"
            )
            inner.pack(fill="x")
            btn_cfg = self.create_modern_button(
                inner, text=btn_text, command=start_cmd, style="primary", width=35
            )
            btn_cfg.pack(side="left", padx=(0, 10))
            btn_run = self.create_modern_button(
                inner, text="▶ Ejecutar", command=run_cmd, style="success", width=12
            )
            btn_run.pack(side="left")
            setattr(self, f"btn_{attr_prefix}", btn_cfg)
            setattr(self, f"btn_run_{attr_prefix}", btn_run)
            btn_run.config(state=tk.DISABLED)

        add_section(
            main_container,
            "📈",
            "Serie de Tiempo",
            tr("series_analysis"),
            self.start_series_analysis,
            self.events.run_series_analysis,
            "series",
        )
        add_section(
            main_container,
            "📊",
            "Corte Transversal",
            tr("cross_section_analysis"),
            self.events.start_cross_section_analysis,
            self.events.run_cross_section_analysis,
            "cross",
        )
        add_section(
            main_container,
            "🎯",
            "PCA 3D",
            tr("panel_analysis"),
            self.start_panel_analysis,
            self.events.run_panel_analysis,
            "panel",
        )
        add_section(
            main_container,
            "🎨",
            "Biplot Avanzado",
            "🎨 Configurar Biplot Avanzado",
            self.start_advanced_biplot_analysis,
            self.events.run_advanced_biplot_analysis,
            "biplot_adv",
        )
        add_section(
            main_container,
            "🟢",
            "Scatterplot (PCA)",
            "⚙ Configurar Scatterplot (PCA)",
            self.start_scatter_plot,
            self.events.run_scatter_plot,
            "scatter",
        )

        # Barra de estado
        status_frame = tk.Frame(
            self,
            bg=self.bg_secondary if hasattr(self, "bg_secondary") else "#f8fafc",
            height=60,
        )
        status_frame.pack(side="bottom", fill="x", pady=(20, 0))
        status_frame.pack_propagate(False)
        self.status = tk.Label(
            status_frame,
            text="⏳ " + tr("status_waiting"),
            fg=self.accent_color if hasattr(self, "accent_color") else "#3b82f6",
            bg=self.bg_secondary if hasattr(self, "bg_secondary") else "#f8fafc",
            font=("Segoe UI", 10, "italic"),
        )
        self.status.pack(pady=15)
        self.lbl_project = tk.Label(
            status_frame,
            text=f"📁 {tr('project')}: Ninguno",
            fg=self.fg_secondary if hasattr(self, "fg_secondary") else "#475569",
            bg=self.bg_secondary if hasattr(self, "bg_secondary") else "#f8fafc",
            font=("Segoe UI", 9),
        )
        self.lbl_project.pack()

        self.apply_matplotlib_style()
        self.change_language(self.lang)

        # Menú Donar
        import webbrowser

        mnu_donate = tk.Menu(menu_bar, tearoff=0)
        mnu_donate.add_command(
            label="☕ Invítame un café",
            command=lambda: webbrowser.open("https://ko-fi.com/daardavid"),
        )

        def show_bank_transfer():
            win = Toplevel(self)
            win.title("Transferencia bancaria")
            win.geometry("420x220")
            msg = "Banco: BBVA\nNombre: DAVID ARMANDO ABREU ROSIQUE\nCuenta: 0108748743\nCLABE: 021180065956536300"
            lbl = tk.Label(win, text=msg, justify="left")
            lbl.pack(pady=10)

            def copy_clabe():
                win.clipboard_clear()
                win.clipboard_append("021180065956536300")
                lbl.config(text=msg + "\n¡CLABE copiada al portapapeles!")

            tk.Button(
                win,
                text="Copiar CLABE",
                command=copy_clabe,
                bg="#988bfd",
                font=("Arial", 11, "bold"),
            ).pack(pady=6)
            tk.Button(win, text="Cerrar", command=win.destroy).pack(pady=4)
            win.grab_set()
            win.focus_set()

        mnu_donate.add_command(
            label="Transferencia bancaria", command=show_bank_transfer
        )
        menu_bar.add_cascade(label="Donar", menu=mnu_donate)

        # (Eliminado bloque duplicado de secciones manuales para evitar repetición visual)

        # Barra de estado moderna
        status_frame = tk.Frame(
            self,
            bg=self.bg_secondary if hasattr(self, "bg_secondary") else "#f8fafc",
            height=60,
        )
        status_frame.pack(side="bottom", fill="x", pady=(20, 0))
        status_frame.pack_propagate(False)

        self.status = tk.Label(
            status_frame,
            text="⏳ " + tr("status_waiting"),
            fg=self.accent_color if hasattr(self, "accent_color") else "#3b82f6",
            bg=self.bg_secondary if hasattr(self, "bg_secondary") else "#f8fafc",
            font=("Segoe UI", 10, "italic"),
        )
        self.status.pack(pady=15)

        self.lbl_project = tk.Label(
            status_frame,
            text=f"📁 {tr('project')}: Ninguno",
            fg=self.fg_secondary if hasattr(self, "fg_secondary") else "#475569",
            bg=self.bg_secondary if hasattr(self, "bg_secondary") else "#f8fafc",
            font=("Segoe UI", 9),
        )
        self.lbl_project.pack()

        self.apply_matplotlib_style()
        self.change_language(self.lang)

        # Menú Donar
        import webbrowser

        mnu_donate = tk.Menu(menu_bar, tearoff=0)
        mnu_donate.add_command(
            label="☕ Invítame un café",
            command=lambda: webbrowser.open("https://ko-fi.com/daardavid"),
        )

        def show_bank_transfer():
            win = Toplevel(self)
            win.title("Transferencia bancaria")
            win.geometry("420x220")
            msg = (
                "Gracias por tu apoyo.\n\n"
                "Banco: HSBC\n"
                "CLABE: 021180065956536300\n"
                "A nombre de: David Abreu Rosique\n\n"
                "Puedes copiar estos datos para transferir desde tu app bancaria.\n"
            )
            lbl = tk.Label(win, text=msg, font=("Arial", 12), justify="left")
            lbl.pack(padx=18, pady=18)

            def copy_clabe():
                win.clipboard_clear()
                win.clipboard_append("021180065956536300")
                lbl.config(text=msg + "\n¡CLABE copiada al portapapeles!")

            btn_copy = tk.Button(
                win,
                text="Copiar CLABE",
                command=copy_clabe,
                bg="#988bfd",
                font=("Arial", 11, "bold"),
            )
            btn_copy.pack(pady=6)
            btn_close = tk.Button(win, text="Cerrar", command=win.destroy)
            btn_close.pack(pady=4)
            win.grab_set()
            win.focus_set()

        mnu_donate.add_command(
            label="Transferencia bancaria", command=show_bank_transfer
        )
        menu_bar.add_cascade(label="Donar", menu=mnu_donate)

    def on_close(self):
        # Shutdown analysis manager before closing
        if hasattr(self, 'analysis_manager'):
            self.analysis_manager.shutdown()

        self.save_settings()
        self.destroy()

    def show_about_window(self):
        import webbrowser

        about_text = (
            "# Acerca de nosotros\n\n"
            "Este programa fue desarrollado por David Armando Abreu Rosique.\n\n"
            "Historia: Esta aplicación nació para facilitar el análisis de datos (indicadores y variables) y el uso de técnicas como componentes principales para usuarios no expertos.\n\n"
            "Agradezco a todo el equipo del Instituto de Investigaciones Económicas de la UNAM.\n\n"
            "Contacto: davidabreu1110@gmail.com.\n\n"
            "¿Te gusta el programa? Puedes apoyarme invitándome un café en Ko-fi.\n"
        )
        win = Toplevel(self)
        win.title("Acerca de nosotros")
        win.geometry("600x400")
        frame = tk.Frame(win)
        frame.pack(fill="both", expand=True)
        txt = tk.Text(frame, wrap="word", font=("Arial", 12))
        txt.insert("1.0", about_text)
        txt.config(state="disabled")
        txt.pack(side="top", fill="both", expand=True)
        scroll = tk.Scrollbar(frame, command=txt.yview)
        txt.config(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        btn_kofi = tk.Button(
            frame,
            text="Visitar Ko-fi",
            bg="#ffdd57",
            font=("Arial", 11, "bold"),
            command=lambda: webbrowser.open("https://ko-fi.com/daardavid"),
        )
        btn_kofi.pack(side="bottom", pady=10)

    # El resto de los métodos de la clase PCAApp deben ir aquí
    # ...

    def build_menu_bar(self):
        # Reconstruye la barra de menús y actualiza referencias
        import webbrowser

        menu_bar = tk.Menu(self)

        # --- Configuración (ícono de engranaje) ---
        mnu_settings = tk.Menu(menu_bar, tearoff=0)
        mnu_settings.add_command(
            label=(
                tr("settings_menu")
                if "settings_menu" in _TRANSLATIONS
                else "Preferencias... ⚙️"
            ),
            command=self.show_settings_window,
        )
        menu_bar.add_cascade(
            label=(
                tr("settings_menu")
                if "settings_menu" in _TRANSLATIONS
                else "Configuración"
            ),
            menu=mnu_settings,
        )

        # --- Proyecto ---
        mnu_project = tk.Menu(menu_bar, tearoff=0)
        mnu_project.add_command(label=tr("new_project"), command=self.new_project)
        mnu_project.add_command(label=tr("open_project"), command=self.load_project)
        mnu_project.add_command(label=tr("save_project"), command=self.save_project)
        menu_bar.add_cascade(label=tr("project"), menu=mnu_project)

        # --- Editar ---
        mnu_edit = tk.Menu(menu_bar, tearoff=0)
        # Submenú Serie de Tiempo
        mnu_edit_series = tk.Menu(mnu_edit, tearoff=0)
        mnu_edit_series.add_command(
            label=tr("edit_title"),
            command=lambda: self.edit_title_dialog("series_config"),
        )
        mnu_edit_series.add_command(
            label=tr("edit_legend"),
            command=lambda: self.edit_legend_dialog("series_config"),
        )
        mnu_edit_series.add_command(
            label=tr("assign_colors"),
            command=lambda: self.edit_colors_dialog("series_config"),
        )
        mnu_edit_series.add_command(
            label=tr("edit_units"),
            command=lambda: self.edit_units_dialog("series_config"),
        )
        mnu_edit_series.add_command(
            label=tr("edit_footer"),
            command=lambda: self.edit_footer_dialog("series_config"),
        )
        mnu_edit.add_cascade(
            label=(
                tr("series_menu")
                if "series_menu" in _TRANSLATIONS
                else "Serie de Tiempo"
            ),
            menu=mnu_edit_series,
        )

        # Submenú Biplot 2D
        mnu_edit_biplot = tk.Menu(mnu_edit, tearoff=0)
        mnu_edit_biplot.add_command(
            label=tr("edit_title"),
            command=lambda: self.edit_title_dialog("cross_section_config"),
        )
        mnu_edit_biplot.add_command(
            label=tr("edit_legend"),
            command=lambda: self.edit_legend_dialog("cross_section_config"),
        )
        mnu_edit_biplot.add_command(
            label=tr("assign_colors"),
            command=lambda: self.edit_colors_dialog("cross_section_config"),
        )
        mnu_edit_biplot.add_command(
            label=tr("edit_units"),
            command=lambda: self.edit_units_dialog("cross_section_config"),
        )
        mnu_edit_biplot.add_command(
            label=tr("edit_footer"),
            command=lambda: self.edit_footer_dialog("cross_section_config"),
        )
        mnu_edit.add_cascade(
            label=tr("biplot_menu") if "biplot_menu" in _TRANSLATIONS else "Biplot 2D",
            menu=mnu_edit_biplot,
        )

        # Submenú PCA 3D
        mnu_edit_panel = tk.Menu(mnu_edit, tearoff=0)
        mnu_edit_panel.add_command(
            label=tr("edit_title"),
            command=lambda: self.edit_title_dialog("panel_config"),
        )
        mnu_edit_panel.add_command(
            label=tr("edit_legend"),
            command=lambda: self.edit_legend_dialog("panel_config"),
        )
        mnu_edit_panel.add_command(
            label=tr("assign_colors"),
            command=lambda: self.edit_colors_dialog("panel_config"),
        )
        mnu_edit_panel.add_command(
            label=tr("edit_units"),
            command=lambda: self.edit_units_dialog("panel_config"),
        )
        mnu_edit_panel.add_command(
            label=tr("edit_footer"),
            command=lambda: self.edit_footer_dialog("panel_config"),
        )
        mnu_edit.add_cascade(
            label=tr("panel_menu") if "panel_menu" in _TRANSLATIONS else "PCA 3D",
            menu=mnu_edit_panel,
        )

        menu_bar.add_cascade(
            label=tr("edit_menu") if "edit_menu" in _TRANSLATIONS else "Editar",
            menu=mnu_edit,
        )

        # --- Ayuda ---
        mnu_help = tk.Menu(menu_bar, tearoff=0)
        mnu_help.add_command(
            label=tr("manual_menu") if "manual_menu" in _TRANSLATIONS else "Manual",
            command=self.dialogs.show_manual_window,
        )
        mnu_help.add_command(
            label=(
                tr("about_menu")
                if "about_menu" in _TRANSLATIONS
                else "Acerca de nosotros"
            ),
            command=self.dialogs.show_about_window,
        )
        menu_bar.add_cascade(
            label=tr("help_menu") if "help_menu" in _TRANSLATIONS else "Ayuda",
            menu=mnu_help,
        )

        # --- Donar ---
        mnu_donate = tk.Menu(menu_bar, tearoff=0)
        mnu_donate.add_command(
            label=(
                tr("donate_menu")
                if "donate_menu" in _TRANSLATIONS
                else "☕ Invítame un café"
            ),
            command=lambda: webbrowser.open("https://ko-fi.com/daardavid"),
        )

        def show_bank_transfer():
            win = Toplevel(self)
            win.title("Transferencia bancaria")
            win.geometry("420x220")
            msg = (
                "Gracias por tu apoyo.\n\n"
                "Banco: HSBC\n"
                "CLABE: 021180065956536300\n"
                "A nombre de: David Abreu Rosique\n\n"
                "Puedes copiar estos datos para transferir desde tu app bancaria.\n"
            )
            lbl = tk.Label(win, text=msg, font=("Arial", 12), justify="left")
            lbl.pack(padx=18, pady=18)

            def copy_clabe():
                win.clipboard_clear()
                win.clipboard_append("021180065956536300")
                lbl.config(text=msg + "\n¡CLABE copiada al portapapeles!")

            btn_copy = tk.Button(
                win,
                text="Copiar CLABE",
                command=copy_clabe,
                bg="#988bfd",
                font=("Arial", 11, "bold"),
            )
            btn_copy.pack(pady=6)
            btn_close = tk.Button(win, text="Cerrar", command=win.destroy)
            btn_close.pack(pady=4)
            win.grab_set()
            win.focus_set()

        mnu_donate.add_command(
            label=(
                tr("donate_bank")
                if "donate_bank" in _TRANSLATIONS
                else "Transferencia bancaria"
            ),
            command=show_bank_transfer,
        )
        menu_bar.add_cascade(
            label=tr("donate_menu") if "donate_menu" in _TRANSLATIONS else "Donar",
            menu=mnu_donate,
        )

        # Actualiza referencias
        self.menu_bar = menu_bar
        self.mnu_project = mnu_project
        self.mnu_edit = mnu_edit
        self.mnu_help = mnu_help
        self.mnu_settings = mnu_settings
        self.mnu_edit_series = mnu_edit_series
        self.mnu_edit_biplot = mnu_edit_biplot
        self.mnu_edit_panel = mnu_edit_panel
        self.mnu_donate = mnu_donate
        self.menu_indices = {
            "settings": 0,
            "project": 1,
            "edit": 2,
            "help": 3,
            "donate": 4,
        }
        self.config(menu=menu_bar)

    def change_language(self, lang):
        if lang == getattr(self, "lang", "es"):
            return  # No hay cambio
        answer = messagebox.askyesno(
            "Reiniciar aplicación",
            "Para aplicar el cambio de idioma, la aplicación debe reiniciarse. ¿Deseas continuar?\n\nTo apply the language change, the app must restart. Continue?",
        )
        if answer:
            self.lang = lang
            self.save_settings()
            python = sys.executable
            import os

            os.execl(python, python, *sys.argv)
        # Si no, no hace nada

    if not os.path.exists(PROJECTS_DIR):
        os.makedirs(PROJECTS_DIR)

    @safe_gui_callback
    def new_project(self):
        while True:
            nombre = simpledialog.askstring(
                tr("new_project"),
                (
                    tr("ask_project_name")
                    if "ask_project_name" in _TRANSLATIONS
                    else "¿Cómo se va a llamar tu proyecto?"
                ),
            )
            if nombre is None:
                self.status.config(
                    text=(
                        tr("cancelled_project_creation")
                        if "cancelled_project_creation" in _TRANSLATIONS
                        else "Creación de proyecto cancelada."
                    )
                )
                return
            nombre = nombre.strip()
            if not nombre:
                messagebox.showwarning(
                    tr("warning"),
                    (
                        tr("empty_project_name")
                        if "empty_project_name" in _TRANSLATIONS
                        else "El nombre no puede estar vacío."
                    ),
                )
                continue
            if any(c in nombre for c in r'<>:"/\\|?*'):
                messagebox.showwarning(
                    tr("warning"),
                    (
                        tr("invalid_project_name")
                        if "invalid_project_name" in _TRANSLATIONS
                        else "El nombre contiene caracteres no permitidos."
                    ),
                )
                continue
            break
        self.project_config = ProjectConfig()  # Reinicia la configuración
        self.project_config.project_name = nombre
        self.status.config(text=f"{tr('new_project')}: {nombre}")
        self.sync_gui_from_cfg()

    @safe_gui_callback
    def save_project(self):
        project_name = self.project_config.project_name or "mi_proyecto"
        save_path = os.path.join(PROJECTS_DIR, f"{project_name}.json")
        self.project_config.save_to_file(save_path)
        messagebox.showinfo(
            tr("info"), f"{tr('project')} {tr('save_project').lower()}\n{save_path}"
        )
        self.sync_gui_from_cfg()

    @safe_gui_callback
    def load_project(self):
        initial_dir = getattr(self, "last_dir", PROJECTS_DIR) or PROJECTS_DIR
        file_path = filedialog.askopenfilename(
            title=tr("open_project"),
            initialdir=initial_dir,
            filetypes=[(tr("project") + " JSON", "*.json")],
        )
        if file_path:
            self.last_dir = os.path.dirname(file_path)
            self.project_config = ProjectConfig.load_from_file(file_path)
            self.sync_gui_from_cfg()

    @safe_gui_callback
    def open_projects_folder(self):
        folder = PROJECTS_DIR
        if platform.system() == "Windows":
            os.startfile(folder)
        elif platform.system() == "Darwin":
            subprocess.call(["open", folder])
        else:
            subprocess.call(["xdg-open", folder])

    def populate_from_project_config(self):
        pass  # Ya no es necesario, todo vive en self.project_config

    def sync_gui_from_cfg(self):
        """Refresca etiquetas y habilita el botón 'Ejecutar' según la configuración."""
        cfg = self.project_config
        self.lbl_project.config(
            text=f"{tr('project')}: {cfg.project_name or tr('unnamed_project') if 'unnamed_project' in _TRANSLATIONS else 'Sin nombre'}"
        )
        # Habilita o deshabilita los botones 'Ejecutar' según la configuración de cada análisis
        # Serie de tiempo
        series_cfg = getattr(cfg, "series_config", {})
        ready_series = (
            bool(series_cfg.get("data_file"))
            and bool(series_cfg.get("selected_indicators"))
            and bool(series_cfg.get("selected_units"))
            and bool(series_cfg.get("selected_years"))
        )
        self.btn_run_series.config(state=tk.NORMAL if ready_series else tk.DISABLED)
        # Corte transversal
        cross_cfg = getattr(cfg, "cross_section_config", {})
        ready_cross = (
            bool(cross_cfg.get("data_file"))
            and bool(cross_cfg.get("selected_indicators"))
            and bool(cross_cfg.get("selected_units"))
            and bool(cross_cfg.get("selected_years"))
        )
        self.btn_run_cross.config(state=tk.NORMAL if ready_cross else tk.DISABLED)
        # Panel 3D
        panel_cfg = getattr(cfg, "panel_config", {})
        ready_panel = (
            bool(panel_cfg.get("data_file"))
            and bool(panel_cfg.get("selected_indicators"))
            and bool(panel_cfg.get("selected_units"))
            and bool(panel_cfg.get("selected_years"))
        )
        self.btn_run_panel.config(state=tk.NORMAL if ready_panel else tk.DISABLED)

        # Biplot Avanzado
        biplot_adv_cfg = getattr(cfg, "biplot_advanced_config", {})
        ready_biplot_adv = (
            bool(biplot_adv_cfg.get("data_file"))
            and bool(biplot_adv_cfg.get("selected_indicators"))
            and bool(biplot_adv_cfg.get("selected_units"))
            and bool(biplot_adv_cfg.get("selected_years"))
        )
        self.btn_run_biplot_adv.config(
            state=tk.NORMAL if ready_biplot_adv else tk.DISABLED
        )
        # Scatterplot habilitación independiente
        scatter_cfg = getattr(cfg, "scatter_plot_config", {})
        ready_scatter = self._scatter_is_ready(scatter_cfg)
        # Debug (opcional): imprimir razones si no está listo
        if not ready_scatter:
            missing = [
                k
                for k in [
                    "data_file",
                    "selected_indicators",
                    "selected_units",
                    "selected_years",
                ]
                if not scatter_cfg.get(k)
            ]
            print(
                f"[Scatter DEBUG] Falta para habilitar ejecutar: {missing} -> cfg actual: {scatter_cfg}"
            )
        self._set_button_state(
            self.btn_run_scatter,
            tk.NORMAL if ready_scatter else tk.DISABLED,
            debug_tag="scatter",
        )

    def _scatter_is_ready(self, scatter_cfg):
        required = [
            "data_file",
            "selected_indicators",
            "selected_units",
            "selected_years",
        ]
        return all(bool(scatter_cfg.get(k)) for k in required)

    def _set_button_state(self, widget, state, debug_tag=""):
        """Intenta establecer el estado de un botón aún si está envuelto en un Frame personalizado."""
        target = None
        import tkinter as tk

        if isinstance(widget, (tk.Button,)):
            target = widget
        else:
            # buscar descendiente Button
            try:
                for child in widget.winfo_children():
                    if isinstance(child, tk.Button):
                        target = child
                        break
            except Exception:
                pass
        if target is not None:
            try:
                target.config(state=state)
            except Exception:
                pass
            print(
                f"[DEBUG BUTTON] {debug_tag} -> state set to {state}; widget={target}"
            )
        else:
            print(
                f"[DEBUG BUTTON] {debug_tag} -> no tk.Button found in widget {widget}"
            )

    # ===== Scatterplot PCA =====
    def start_scatter_plot(self):
        """Inicia (o continúa) el flujo de selección para el Scatterplot PCA independiente."""
        # Aseguramos que todas las funciones de selección escriban en la sub-config correcta
        self._last_analysis_type = "scatter_plot_config"
        cfg = self.project_config.scatter_plot_config

        # Definimos callbacks encadenados para continuar automáticamente el flujo
        def after_year():
            self._show_scatter_config_dialog()

        def after_units():
            # Elegimos solo 1 año para scatter (multi=False)
            self.step_select_year(after_year, multi=False)

        def after_indicators():
            self.step_select_units(after_units, allow_multiple=True)

        def after_file():
            self.step_select_indicators(after_indicators, multi=True)

        # Decidir el siguiente paso según lo ya seleccionado
        if not cfg.get("data_file"):
            self.step_select_file(after_file)
        elif not cfg.get("selected_indicators"):
            self.step_select_indicators(after_indicators, multi=True)
        elif not cfg.get("selected_units"):
            self.step_select_units(after_units, allow_multiple=True)
        elif not cfg.get("selected_years"):
            self.step_select_year(after_year, multi=False)
        else:
            self._show_scatter_config_dialog()

    def _show_scatter_config_dialog(self):
        cfg = self.project_config.scatter_plot_config
        win = tk.Toplevel(self)
        win.title("Scatterplot (PCA)")
        win.geometry("420x600")
        win.configure(bg="#f8fafc")

        try:
            self.sync_gui_from_cfg()
        except Exception:
            pass

        frame = tk.Frame(win, bg="#f8fafc")
        frame.pack(fill="both", expand=True, padx=15, pady=15)

        tk.Label(
            frame,
            text="Configuración Scatterplot PCA",
            font=("Segoe UI", 14, "bold"),
            bg="#f8fafc",
            fg="#1e293b",
        ).pack(pady=(0, 10))

        # Componentes principales
        pcs_frame = tk.LabelFrame(
            frame,
            text="Componentes",
            bg="#f8fafc",
            fg="#374151",
            font=("Segoe UI", 10, "bold"),
        )
        pcs_frame.pack(fill="x", pady=5)
        self.var_pc_x = tk.IntVar(value=cfg.get("pc_x", 1))
        self.var_pc_y = tk.IntVar(value=cfg.get("pc_y", 2))
        tk.Label(pcs_frame, text="PC X:", bg="#f8fafc").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        tk.Spinbox(pcs_frame, from_=1, to=10, width=5, textvariable=self.var_pc_x).grid(
            row=0, column=1, padx=5, pady=5
        )
        tk.Label(pcs_frame, text="PC Y:", bg="#f8fafc").grid(
            row=0, column=2, padx=5, pady=5, sticky="w"
        )
        tk.Spinbox(pcs_frame, from_=1, to=10, width=5, textvariable=self.var_pc_y).grid(
            row=0, column=3, padx=5, pady=5
        )

        # Apariencia
        appear_frame = tk.LabelFrame(
            frame,
            text="Apariencia",
            bg="#f8fafc",
            fg="#374151",
            font=("Segoe UI", 10, "bold"),
        )
        appear_frame.pack(fill="x", pady=5)
        self.var_alpha = tk.DoubleVar(value=cfg.get("alpha", 0.7))
        self.var_size = tk.IntVar(value=cfg.get("point_size", 30))
        tk.Label(appear_frame, text="Alpha:", bg="#f8fafc").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        tk.Scale(
            appear_frame,
            from_=0.1,
            to=1.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.var_alpha,
            bg="#f8fafc",
        ).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        tk.Label(appear_frame, text="Tamaño:", bg="#f8fafc").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        tk.Scale(
            appear_frame,
            from_=10,
            to=120,
            resolution=5,
            orient="horizontal",
            variable=self.var_size,
            bg="#f8fafc",
        ).grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # Opciones avanzadas
        adv_frame = tk.LabelFrame(
            frame,
            text="Opciones",
            bg="#f8fafc",
            fg="#374151",
            font=("Segoe UI", 10, "bold"),
        )
        adv_frame.pack(fill="x", pady=5)
        self.var_use_cmap = tk.BooleanVar(value=cfg.get("use_cmap", False))
        self.var_cmap = tk.StringVar(value=cfg.get("cmap", "viridis"))
        self.var_density = tk.BooleanVar(value=cfg.get("density", False))
        self.var_gradient = tk.StringVar(value=cfg.get("gradient", ""))
        self.var_edgecolor = tk.StringVar(value=cfg.get("edgecolor", "None"))
        self.var_ht2 = tk.BooleanVar(value=cfg.get("HT2", False))
        self.var_spe = tk.BooleanVar(value=cfg.get("SPE", False))
        self.var_show_labels = tk.BooleanVar(value=cfg.get("show_labels", False))

        tk.Checkbutton(
            adv_frame, text="Usar Colormap", variable=self.var_use_cmap, bg="#f8fafc"
        ).grid(row=0, column=0, sticky="w", padx=5, pady=2)
        tk.Label(adv_frame, text="Cmap:", bg="#f8fafc").grid(
            row=0, column=1, sticky="e"
        )
        tk.OptionMenu(
            adv_frame, self.var_cmap, "viridis", "plasma", "tab10", "Set2", "Set3"
        ).grid(row=0, column=2, sticky="w", padx=5)
        tk.Checkbutton(
            adv_frame, text="Densidad", variable=self.var_density, bg="#f8fafc"
        ).grid(row=1, column=0, sticky="w", padx=5, pady=2)
        tk.Checkbutton(
            adv_frame, text="Hotelling T²", variable=self.var_ht2, bg="#f8fafc"
        ).grid(row=1, column=1, sticky="w", padx=5, pady=2)
        tk.Checkbutton(
            adv_frame, text="Outliers SPE", variable=self.var_spe, bg="#f8fafc"
        ).grid(row=1, column=2, sticky="w", padx=5, pady=2)
        tk.Checkbutton(
            adv_frame,
            text="Mostrar etiquetas",
            variable=self.var_show_labels,
            bg="#f8fafc",
        ).grid(row=2, column=0, sticky="w", padx=5, pady=2)
        tk.Label(adv_frame, text="Gradiente (hex):", bg="#f8fafc").grid(
            row=3, column=0, sticky="w", padx=5, pady=2
        )
        tk.Entry(adv_frame, textvariable=self.var_gradient, width=10).grid(
            row=3, column=1, sticky="w", padx=5
        )
        tk.Label(adv_frame, text="Edgecolor:", bg="#f8fafc").grid(
            row=3, column=2, sticky="e", padx=5
        )
        tk.OptionMenu(
            adv_frame, self.var_edgecolor, "None", "black", "white", "#333333"
        ).grid(row=3, column=3, sticky="w", padx=5)

        # Botones
        btn_frame = tk.Frame(frame, bg="#f8fafc")
        btn_frame.pack(fill="x", pady=15)
        tk.Button(
            btn_frame,
            text="Aplicar",
            command=lambda w=win: self._apply_scatter_config(w),
            bg="#3b82f6",
            fg="white",
        ).pack(side="left", padx=5)
        tk.Button(
            btn_frame, text="Cerrar", command=win.destroy, bg="#64748b", fg="white"
        ).pack(side="right", padx=5)

    def _apply_scatter_config(self, dialog):
        self.project_config.scatter_plot_config.update(
            {
                "pc_x": self.var_pc_x.get(),
                "pc_y": self.var_pc_y.get(),
                "alpha": self.var_alpha.get(),
                "point_size": self.var_size.get(),
                "use_cmap": self.var_use_cmap.get(),
                "cmap": self.var_cmap.get(),
                "density": self.var_density.get(),
                "gradient": self.var_gradient.get(),
                "edgecolor": self.var_edgecolor.get(),
                "HT2": self.var_ht2.get(),
                "SPE": self.var_spe.get(),
                "show_labels": self.var_show_labels.get(),
            }
        )
        # Actualiza GUI y habilita botón ejecutar
        self.sync_gui_from_cfg()
        try:
            dialog.destroy()
        except Exception:
            pass
        # Ejecutar automáticamente sin depender del botón
        try:
            self.status.config(text="Generando Scatterplot PCA (auto)...")
        except Exception:
            pass
        # Mensaje informativo breve
        self.run_scatter_plot()

    def run_scatter_plot(self):
        """Ejecuta el Scatterplot PCA usando únicamente su propia configuración."""
        cfg = self.project_config.scatter_plot_config
        # Validaciones básicas
        missing = [
            k
            for k in [
                "data_file",
                "selected_indicators",
                "selected_units",
                "selected_years",
            ]
            if not cfg.get(k)
        ]
        if missing:
            messagebox.showwarning(
                "Configuración incompleta",
                f"Faltan: {', '.join(missing)}. Pulsa Configurar Scatterplot.",
            )
            return
        try:
            self.status.config(text="Generando Scatterplot PCA ...")
        except Exception:
            pass
        try:
            year = str(cfg["selected_years"][0])  # Solo 1 año
            data_file = cfg["data_file"]
            indicators = cfg["selected_indicators"]
            units = cfg["selected_units"]

            # Cargar todas las hojas requeridas solo una vez
            try:
                all_sheets = dl.load_excel_file(data_file)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")
                return
            if not all_sheets:
                messagebox.showerror("Error", "El archivo no contiene hojas legibles.")
                return

            import pandas as pd
            from sklearn.preprocessing import StandardScaler
            from scatter_plot import generate_scatter_plot

            # Construir DataFrame combinado: cada indicador -> columna
            merged = None
            for ind in indicators:
                if ind not in all_sheets:
                    messagebox.showwarning(
                        "Indicador faltante", f"La hoja '{ind}' no se encontró."
                    )
                    return
                df_ind = all_sheets[ind]
                if "Unnamed: 0" not in df_ind.columns:
                    messagebox.showerror(
                        "Error", f"Hoja '{ind}' sin columna 'Unnamed: 0'."
                    )
                    return
                if year not in df_ind.columns.map(str):
                    messagebox.showerror(
                        "Error", f"Año {year} no encontrado en '{ind}'."
                    )
                    return
                # Aseguramos nombre exacto del año como string
                # Renombrar la columna del año al nombre del indicador
                # Convertimos a string para evitar mezcla tipo int
                cols_map = {c: c for c in df_ind.columns}
                # Encontrar la columna que represente el año (puede ser int o str)
                for c in df_ind.columns:
                    if str(c) == year:
                        cols_map[c] = ind
                df_sel = df_ind[
                    ["Unnamed: 0", next(c for c in df_ind.columns if str(c) == year)]
                ].rename(columns=cols_map)
                if merged is None:
                    merged = df_sel
                else:
                    merged = merged.merge(df_sel, on="Unnamed: 0", how="inner")

            if merged is None or merged.empty:
                messagebox.showerror(
                    "Error", "No se pudo construir la tabla combinada."
                )
                return

            # Filtrar unidades
            merged = merged[merged["Unnamed: 0"].isin(units)]
            if merged.empty:
                messagebox.showerror(
                    "Error", "Después de filtrar unidades no quedan datos."
                )
                return
            merged.set_index("Unnamed: 0", inplace=True)

            # Estandarizar
            X = merged.values.astype(float)
            scaler = StandardScaler()
            X_std = scaler.fit_transform(X)
            df_std = pd.DataFrame(X_std, index=merged.index, columns=merged.columns)

            # Etiquetas (usar código; podría mapear a nombre si existe diccionario)
            labels = merged.index.tolist()

            # Generar scatterplot (modelo PCA interno se creará dentro si no se pasa)
            generate_scatter_plot(df_std, labels, cfg, existing_model=None)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar scatterplot: {e}")

    @safe_gui_callback
    def run_cross_section_analysis(self):
        from pca_cross_logic import PCAAnalysisLogic

        cfg = self.project_config.cross_section_config
        selected_years = [int(y) for y in cfg["selected_years"]]
        for year_to_analyze in selected_years:
            estrategia, params = None, None
            temp_results = PCAAnalysisLogic.run_cross_section_analysis_logic(
                cfg, year_to_analyze
            )
            if "warning" in temp_results and "faltantes" in temp_results["warning"]:
                respuesta = messagebox.askyesno(
                    f"Imputar año {year_to_analyze}",
                    f"Se encontraron datos faltantes para el año {year_to_analyze}.\n¿Quieres imputar los valores faltantes?",
                )
                if respuesta:
                    estrategia, params = self.gui_select_imputation_strategy()
            results = PCAAnalysisLogic.run_cross_section_analysis_logic(
                cfg,
                year_to_analyze,
                imputation_strategy=estrategia,
                imputation_params=params,
            )
            if "warning" in results:
                messagebox.showwarning("Atención", results["warning"])
                continue
            df_year_cross_section = results["df_year_cross_section"]
            df_year_processed = results["df_year_processed"]
            df_year_estandarizado = results["df_year_estandarizado"]
            scaler = results["scaler"]
            df_cov_cs = results["df_cov_cs"]
            pca_model_cs = results["pca_model_cs"]
            df_pc_scores_cs = results["df_pc_scores_cs"]
            df_varianza_explicada_cs = results["df_varianza_explicada_cs"]
            evr_cs = results["evr_cs"]
            cum_evr_cs = results["cum_evr_cs"]

            if len(evr_cs) >= 2:
                msg = (
                    tr("explained_variance_2").format(
                        year=year_to_analyze,
                        pc1=evr_cs[0],
                        pc2=evr_cs[1],
                        total=cum_evr_cs[1],
                    )
                    if "explained_variance_2" in _TRANSLATIONS
                    else (
                        f"Para el año {year_to_analyze}, los 2 primeros componentes explican:\n"
                        f"PC1: {evr_cs[0]:.2%}\n"
                        f"PC2: {evr_cs[1]:.2%}\n"
                        f"Total: {cum_evr_cs[1]:.2%} de la varianza"
                    )
                )
                title = (
                    tr("explained_variance_title")
                    if "explained_variance_title" in _TRANSLATIONS
                    else "Varianza explicada por los 2 componentes"
                )
            elif len(evr_cs) == 1:
                msg = (
                    tr("explained_variance_1").format(
                        year=year_to_analyze, pc1=evr_cs[0]
                    )
                    if "explained_variance_1" in _TRANSLATIONS
                    else (
                        f"Solo se pudo calcular un componente principal para el año {year_to_analyze}.\n"
                        f"PC1 explica: {evr_cs[0]:.2%} de la varianza"
                    )
                )
                title = (
                    tr("explained_variance_title")
                    if "explained_variance_title" in _TRANSLATIONS
                    else "Varianza explicada por los 2 componentes"
                )
            else:
                msg = (
                    tr("explained_variance_none")
                    if "explained_variance_none" in _TRANSLATIONS
                    else "No se pudo calcular componentes principales para este año."
                )
                title = (
                    tr("explained_variance_title")
                    if "explained_variance_title" in _TRANSLATIONS
                    else "Varianza explicada por los 2 componentes"
                )
            messagebox.showinfo(title, msg)

            try:
                custom_colors = cfg.get("color_groups", {}) or {}
                unit_order = df_year_estandarizado.index.tolist()
                grupos_individuos = [
                    unit if unit in custom_colors else "Otros" for unit in unit_order
                ]
                mapa_de_colores = {"Otros": "#808080"}
                mapa_de_colores.update(custom_colors)
                dl_viz.graficar_biplot_corte_transversal(
                    pca_model=pca_model_cs,
                    df_pc_scores=df_pc_scores_cs,
                    nombres_indicadores_originales=df_year_estandarizado.columns.tolist(),
                    nombres_indicadores_etiquetas=[
                        MAPEO_INDICADORES.get(code, code)
                        for code in df_year_estandarizado.columns.tolist()
                    ],
                    nombres_individuos_etiquetas=unit_order,
                    grupos_individuos=grupos_individuos,
                    mapa_de_colores=mapa_de_colores,
                    titulo=cfg.get("custom_titles", {}).get(
                        "biplot", f"Biplot {year_to_analyze}"
                    ),
                    legend_title=cfg.get("custom_titles", {}).get(
                        "legend", "Grupos de Unidades de Investigación"
                    ),
                    ruta_guardado=None,
                    footer_note=cfg.get("custom_titles", {}).get("footer", ""),
                )
            except Exception as e:
                messagebox.showwarning(
                    "Error Biplot", f"No se pudo generar el biplot: {e}"
                )

            export_title = (
                tr("export_title") if "export_title" in _TRANSLATIONS else "¿Exportar?"
            )
            export_msg = (
                tr("export_msg").format(year=year_to_analyze)
                if "export_msg" in _TRANSLATIONS
                else f"¿Quieres guardar los resultados para el año {year_to_analyze}?"
            )
            if messagebox.askyesno(export_title, export_msg):
                filename = filedialog.asksaveasfilename(
                    title=(
                        tr("save_results_title").format(year=year_to_analyze)
                        if "save_results_title" in _TRANSLATIONS
                        else f"Guardar resultados {year_to_analyze}"
                    ),
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx *.xls")],
                )
                if filename:
                    with pd.ExcelWriter(filename) as writer:
                        df_year_cross_section.rename(
                            columns=MAPEO_INDICADORES
                        ).to_excel(writer, sheet_name="Original", index=True)
                        df_year_processed.rename(columns=MAPEO_INDICADORES).to_excel(
                            writer, sheet_name="Procesado", index=True
                        )
                        df_year_estandarizado.rename(
                            columns=MAPEO_INDICADORES
                        ).to_excel(writer, sheet_name="Estandarizado", index=True)
                        df_cov_cs.rename(
                            index=MAPEO_INDICADORES, columns=MAPEO_INDICADORES
                        ).to_excel(writer, sheet_name="Matriz_Covarianza", index=True)
                        if df_pc_scores_cs is not None:
                            df_pc_scores_cs.to_excel(
                                writer, sheet_name="ComponentesPCA", index=True
                            )
                        if df_varianza_explicada_cs is not None:
                            df_varianza_explicada_cs.to_excel(
                                writer, sheet_name="PCA_VarianzaExp", index=True
                            )
                    messagebox.showinfo(
                        tr("done") if "done" in _TRANSLATIONS else "Listo",
                        (
                            tr("file_saved").format(filename=filename)
                            if "file_saved" in _TRANSLATIONS
                            else f"Archivo guardado:\n{filename}"
                        ),
                    )
        try:
            self.status.config(text="Análisis de corte transversal completado.")
        except tk.TclError:
            pass

    # --- FLUJO 3: Panel 3D ---
    @safe_gui_callback
    def start_panel_analysis(self):
        try:
            self._last_analysis_type = "panel_config"
            self.status.config(text="Flujo: Trayectorias panel (PCA 3D)")
            self.panel_wizard()
        except tk.TclError:
            pass

    def panel_wizard(self):
        self.step_select_file(
            lambda: self.step_select_indicators(
                lambda: self.step_select_units(
                    lambda: self.step_select_year(lambda: self.run_panel_analysis())
                )
            )
        )

    @safe_gui_callback
    def run_panel_analysis(self):
        try:
            cfg = self.project_config.panel_config
            self.status.config(
                text=f"Panel 3D para años {cfg.get('selected_years', [])} y unidades: {cfg.get('selected_units', [])}"
            )
            if (
                not cfg.get("data_file")
                or not cfg.get("selected_indicators")
                or not cfg.get("selected_units")
            ):
                messagebox.showerror(
                    "Error",
                    "Faltan datos para el análisis 3D. Selecciona archivo, indicadores y unidades de investigación.",
                )
                return
            all_sheets_data = dl.load_excel_file(cfg["data_file"])
            if not all_sheets_data:
                messagebox.showerror(
                    "Error", "No se pudieron cargar los datos del archivo seleccionado."
                )
                return
            results = PCAPanel3DLogic.run_panel3d_analysis_logic(
                all_sheets_data,
                list(all_sheets_data.keys()),
                cfg["selected_indicators"],
                cfg["selected_units"],
            )
            if "error" in results:
                messagebox.showerror("Error", results["error"])
                return
            try:
                dl_viz.graficar_trayectorias_3d(
                    results["df_pc_scores_panel"],
                    results["pca_model_panel"],
                    results["country_groups"],
                    results["group_colors"],
                    titulo="Trayectorias en el espacio de componentes (Panel 3D)",
                )
            except Exception as e:
                messagebox.showerror("Error", f"Error al graficar el análisis 3D: {e}")

            # Opción de exportar resultados
            if messagebox.askyesno(
                tr("export_title") if "export_title" in _TRANSLATIONS else "¿Exportar?",
                (
                    tr("export_msg")
                    if "export_msg" in _TRANSLATIONS
                    else "¿Quieres guardar los resultados del análisis Panel 3D?"
                ),
            ):
                filename = filedialog.asksaveasfilename(
                    title=(
                        tr("save_results_title")
                        if "save_results_title" in _TRANSLATIONS
                        else "Guardar resultados Panel 3D"
                    ),
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx *.xls")],
                )
                if filename:
                    import pandas as pd

                    with pd.ExcelWriter(filename) as writer:
                        if results.get("df_pc_scores_panel") is not None:
                            results["df_pc_scores_panel"].to_excel(
                                writer, sheet_name="PC_Scores_Panel", index=True
                            )
                        if results.get("country_groups") is not None:
                            pd.DataFrame(results["country_groups"]).to_excel(
                                writer, sheet_name="Country_Groups"
                            )
                        if results.get("group_colors") is not None:
                            pd.DataFrame(
                                list(results["group_colors"].items()),
                                columns=["Group", "Color"],
                            ).to_excel(writer, sheet_name="Group_Colors", index=False)
                    messagebox.showinfo(
                        tr("done") if "done" in _TRANSLATIONS else "Listo",
                        (
                            tr("file_saved").format(filename=filename)
                            if "file_saved" in _TRANSLATIONS
                            else f"Archivo guardado:\n{filename}"
                        ),
                    )
        except tk.TclError:
            pass

    # --- FLUJO 4: Biplot Avanzado ---
    @safe_gui_callback
    def start_advanced_biplot_analysis(self):
        """Inicia el flujo de configuración del biplot avanzado."""
        try:
            self._last_analysis_type = "biplot_advanced_config"
            self.status.config(
                text="Flujo: Biplot Avanzado (marcadores personalizados)"
            )
            self.advanced_biplot_wizard()
        except tk.TclError:
            pass

    def advanced_biplot_wizard(self):
        """Wizard para configurar el biplot avanzado."""
        self.step_select_file(
            lambda: self.step_select_indicators(
                lambda: self.step_select_units(
                    lambda: self.step_select_year(
                        lambda: self._configure_advanced_biplot(), multi=False
                    ),  # Solo un año para biplot
                    allow_multiple=True,
                ),  # Múltiples países
                multi=True,
            )  # Múltiples indicadores
        )

    def _configure_advanced_biplot(self):
        """Abre el diálogo de configuración avanzada del biplot."""
        # Verificar que tenemos la configuración necesaria
        cfg = getattr(self.project_config, "biplot_advanced_config", {})

        if (
            not cfg.get("data_file")
            or not cfg.get("selected_indicators")
            or not cfg.get("selected_units")
            or not cfg.get("selected_years")
        ):
            messagebox.showwarning(
                "Configuración incompleta",
                "Primero debes completar la selección de archivo, indicadores, unidades de investigación y año.",
            )
            return

        # Cargar los datos para el biplot avanzado
        try:
            import data_loader_module as dl

            all_sheets_data = dl.load_excel_file(cfg["data_file"])
            if not all_sheets_data:
                messagebox.showerror(
                    "Error", "No se pudieron cargar los datos del archivo seleccionado."
                )
                return

            # Preparar datos para biplot avanzado
            from pca_cross_logic import PCAAnalysisLogic

            year_to_analyze = int(
                cfg["selected_years"][0]
            )  # Usar el primer año seleccionado

            temp_results = PCAAnalysisLogic.run_cross_section_analysis_logic(
                cfg, year_to_analyze
            )
            if "error" in temp_results:
                messagebox.showerror("Error", temp_results["error"])
                return

            # Guardar los datos para uso del biplot avanzado
            self.biplot_data = {
                "df_year_cross_section": temp_results["df_year_cross_section"],
                "df_year_processed": temp_results[
                    "df_year_processed"
                ],  # original no estandarizado
                "df_year_standardized": temp_results.get(
                    "df_year_estandarizado", temp_results["df_year_processed"]
                ),
                "year": year_to_analyze,
                "config": cfg,
            }

            # Mostrar diálogo de configuración avanzada
            self.create_advanced_biplot_dialog()

        except Exception as e:
            messagebox.showerror(
                "Error", f"Error al cargar datos para biplot avanzado:\n{str(e)}"
            )
            return

    @safe_gui_callback
    def run_advanced_biplot_analysis(self):
        """Ejecuta el análisis de biplot avanzado."""
        try:
            if not hasattr(self, "biplot_data") or not self.biplot_data:
                messagebox.showerror(
                    "Error",
                    "No hay datos cargados para el biplot avanzado. Primero configura el análisis.",
                )
                return

            if not hasattr(self, "biplot_config") or not self.biplot_config:
                messagebox.showwarning(
                    "Configuración incompleta",
                    "Primero debes configurar el biplot avanzado usando '🎨 Configurar Biplot Avanzado'.",
                )
                return

            self.status.config(text="Creando biplot avanzado...")

            # Obtener configuración
            config = {
                "year": str(self.biplot_data["year"]),
                "categorization_scheme": self.biplot_config["categorization"].get(),
                "marker_scheme": self.biplot_config["marker_scheme"].get(),
                "color_scheme": self.biplot_config["color_scheme"].get(),
                "show_arrows": self.biplot_config["show_arrows"].get(),
                "show_labels": self.biplot_config["show_labels"].get(),
                "alpha": self.biplot_config["alpha"].get(),
            }

            # Ejecutar análisis
            self.show_loading("Creando biplot avanzado...")

            # Usar los datos procesados
            # Usar versión estandarizada si existe
            df_for_biplot = self.biplot_data.get(
                "df_year_standardized", self.biplot_data["df_year_processed"]
            )

            # Intentar con biplot avanzado, si falla usar simplificado
            try:
                from biplot_advanced import create_advanced_biplot

                # Añadir categorías personalizadas si existen en project_config
                if (
                    "biplot_advanced_config" in self.project_config
                    and "custom_categories"
                    in self.project_config["biplot_advanced_config"]
                ):
                    config["custom_categories"] = self.project_config[
                        "biplot_advanced_config"
                    ]["custom_categories"]
                success = create_advanced_biplot(df_for_biplot, config)
            except Exception as e:
                print(f"⚠️ Error con biplot_advanced: {e}")
                print("🔄 Intentando con versión simplificada...")
                try:
                    from biplot_simple import create_advanced_biplot_simple

                    success = create_advanced_biplot_simple(df_for_biplot, config)
                except Exception as e2:
                    print(f"❌ Error con biplot simplificado: {e2}")
                    success = False

            if success:
                self.status.config(text="✅ Biplot avanzado creado exitosamente")
                messagebox.showinfo(
                    "Éxito",
                    f"Biplot avanzado creado exitosamente.\n\n"
                    f"Configuración:\n"
                    f"• Año: {config['year']}\n"
                    f"• Categorización: {config['categorization_scheme']}\n"
                    f"• Marcadores: {config['marker_scheme']}\n"
                    f"• Colores: {config['color_scheme']}",
                )
            else:
                self.status.config(text="❌ Error al crear biplot avanzado")
                messagebox.showerror(
                    "Error", "Hubo un problema al crear el biplot avanzado."
                )

        except Exception as e:
            self.status.config(text=f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Error al crear biplot avanzado:\n{str(e)}")
        finally:
            self.hide_loading()

    def _copy_current_to_config(self, config_name):
        """Copia la selección actual de la GUI al sub-config correspondiente."""
        cfg = getattr(self.project_config, config_name)
        # Copia los campos principales de la GUI a la subconfig
        # (esto asume que la GUI mantiene los campos temporales en self.project_config)
        for key in [
            "data_file",
            "selected_indicators",
            "selected_units",
            "selected_years",
            "color_groups",
            "group_labels",
            "custom_titles",
            "analysis_results",
            "footer_note",
        ]:
            if hasattr(self.project_config, key):
                cfg[key] = getattr(self.project_config, key, cfg.get(key, None))

    # ========== REUTILIZABLES ==========

    def step_select_file(self, callback):
        file = filedialog.askopenfilename(
            title=tr("select_file"),
            initialdir=getattr(self, "last_dir", PROJECTS_DIR) or PROJECTS_DIR,
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("Parquet files", "*.parquet"),
                ("All files", "*.*")
            ],
        )
        if not file:
            messagebox.showwarning(
                tr("warning"),
                (
                    tr("no_file_selected")
                    if "no_file_selected" in _TRANSLATIONS
                    else "No se seleccionó ningún archivo."
                ),
            )
            return
        if not os.path.isfile(file):
            messagebox.showerror(
                tr("error"),
                (
                    tr("file_not_found")
                    if "file_not_found" in _TRANSLATIONS
                    else "El archivo no existe."
                ),
            )
            return
        config_name = getattr(self, "_last_analysis_type", "series_config")
        getattr(self.project_config, config_name)["data_file"] = file
        self.last_dir = os.path.dirname(file)
        # Limpia selección de años al cambiar archivo en serie de tiempo
        if config_name == "series_config":
            getattr(self.project_config, config_name)["selected_years"] = []
        import data_loader_module as dl

        try:
            all_sheets_data = dl.load_excel_file(file)
        except Exception as e:
            logger.exception("Error loading Excel file")
            messagebox.showerror(
                tr("error"),
                (
                    tr("file_load_error") + f"\n{e}"
                    if "file_load_error" in _TRANSLATIONS
                    else f"No se pudo cargar el archivo seleccionado.\n{e}"
                ),
            )
            self.sheet_names = []
            self.sync_gui_from_cfg()
            return
        if all_sheets_data:
            self.sheet_names = list(all_sheets_data.keys())
            # Si es scatter, primero guarda el archivo
            if getattr(self, "_last_analysis_type", "") == "scatter_plot_config":
                self.project_config.scatter_plot_config["data_file"] = file
            callback()
            self.sync_gui_from_cfg()  # refresca tras selección
        else:
            self.sheet_names = []
            messagebox.showerror(
                tr("error"),
                (
                    tr("file_load_error")
                    if "file_load_error" in _TRANSLATIONS
                    else "No se pudo cargar el archivo seleccionado."
                ),
            )
            self.sync_gui_from_cfg()  # ← nuevo

    def step_select_indicators(self, callback, multi=True):
        """Ventana moderna para seleccionar indicadores/hojas del Excel."""
        if not self.sheet_names:
            messagebox.showerror(
                tr("error"),
                (
                    tr("select_file_first")
                    if "select_file_first" in _TRANSLATIONS
                    else "Primero selecciona un archivo."
                ),
            )
            return
        # Crear ventana moderna (ajustada más grande)
        win = self.create_modern_window("📊 " + tr("select_indicators"), 640, 720)

        # Título y descripción
        title_frame = tk.Frame(win, bg=getattr(self, "bg_primary", "#ffffff"))
        title_frame.pack(fill="x", pady=(20, 10), padx=20)

        title_label = tk.Label(
            title_frame,
            text="📊 Seleccionar Indicadores",
            font=("Segoe UI", 16, "bold"),
            bg=getattr(self, "bg_primary", "#ffffff"),
            fg=getattr(self, "fg_primary", "#1e293b"),
        )
        title_label.pack()

        # Descripción
        n_disp = len(self.sheet_names)
        desc_text = f"Selecciona {'uno o más' if multi else 'un'} indicador{'es' if multi else ''} para el análisis\n({n_disp} disponibles)"

        desc_label = tk.Label(
            title_frame,
            text=desc_text,
            font=("Segoe UI", 10),
            bg=getattr(self, "bg_primary", "#ffffff"),
            fg=getattr(self, "fg_secondary", "#64748b"),
            wraplength=450,
        )
        desc_label.pack(pady=(5, 0))

        # Contenedor del listbox
        content_frame = tk.Frame(win, bg=getattr(self, "bg_primary", "#ffffff"))
        content_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Listbox moderno
        listbox_frame, listbox = self.create_modern_listbox(
            content_frame, selectmode=tk.MULTIPLE if multi else tk.SINGLE, height=18
        )
        listbox_frame.pack(fill="both", expand=True, pady=(0, 20))

        # Llenar listbox con indicadores
        for ind in self.sheet_names:
            listbox.insert(tk.END, f"📈 {ind}")

        # Frame de botones
        buttons_frame = tk.Frame(
            content_frame, bg=getattr(self, "bg_primary", "#ffffff")
        )
        buttons_frame.pack(fill="x")

        # Botones de selección (solo para múltiple)
        if multi:
            selection_frame = tk.Frame(
                buttons_frame, bg=getattr(self, "bg_primary", "#ffffff")
            )
            selection_frame.pack(fill="x", pady=(0, 15))

            select_all_btn = self.create_modern_button(
                selection_frame,
                text="✅ Seleccionar Todo",
                command=lambda: listbox.select_set(0, tk.END),
                style="secondary",
                width=20,
            )
            select_all_btn.pack(side="left", padx=(0, 10))

            unselect_all_btn = self.create_modern_button(
                selection_frame,
                text="❌ Deseleccionar Todo",
                command=lambda: listbox.select_clear(0, tk.END),
                style="secondary",
                width=20,
            )
            unselect_all_btn.pack(side="left")

        # Botones de acción
        action_frame = tk.Frame(
            buttons_frame, bg=getattr(self, "bg_primary", "#ffffff")
        )
        action_frame.pack(fill="x")

        def confirm_selection():
            selected_indices = listbox.curselection()
            if not selected_indices:
                messagebox.showerror(tr("error"), tr("select_at_least_one_indicator"))
                return

            config_name = getattr(self, "_last_analysis_type", "series_config")
            getattr(self.project_config, config_name)["selected_indicators"] = [
                self.sheet_names[i] for i in selected_indices
            ]

            # Limpia selección de años al cambiar indicadores en serie de tiempo
            if config_name == "series_config":
                getattr(self.project_config, config_name)["selected_years"] = []

            win.destroy()
            callback()
            self.sync_gui_from_cfg()

        # Botón confirmar
        confirm_btn = self.create_modern_button(
            action_frame,
            text="✅ Confirmar Selección",
            command=confirm_selection,
            style="success",
            width=20,
        )
        confirm_btn.pack(side="right", padx=(10, 0))

        # Botón cancelar
        cancel_btn = self.create_modern_button(
            action_frame,
            text="❌ Cancelar",
            command=win.destroy,
            style="secondary",
            width=15,
        )
        cancel_btn.pack(side="right")

        # Eventos
        win.bind("<Escape>", lambda event: win.destroy())
        win.grab_set()
        win.focus_set()

    def step_select_units(self, callback, allow_multiple=True):
        config_name = getattr(self, "_last_analysis_type", "series_config")
        cfg = getattr(self.project_config, config_name)
        if not cfg.get("data_file") or not cfg.get("selected_indicators"):
            messagebox.showerror(
                tr("error"),
                (
                    tr("select_file_and_indicators_first")
                    if "select_file_and_indicators_first" in _TRANSLATIONS
                    else "Primero selecciona archivo e indicadores."
                ),
            )
            return
        try:
            excel_data = pd.read_excel(
                cfg["data_file"], sheet_name=cfg["selected_indicators"][0]
            )
        except Exception as e:
            logger.exception("Error reading Excel for units")
            messagebox.showerror(
                tr("error"),
                (
                    tr("file_load_error") + f"\n{e}"
                    if "file_load_error" in _TRANSLATIONS
                    else f"No se pudo cargar el archivo seleccionado.\n{e}"
                ),
            )
            return
        if "Unnamed: 0" not in excel_data.columns:
            messagebox.showerror(
                tr("error"),
                (
                    tr("unnamed_col_missing")
                    if "unnamed_col_missing" in _TRANSLATIONS
                    else "No se encontró la columna 'Unnamed: 0' en la hoja seleccionada."
                ),
            )
            return
        all_units = sorted(excel_data["Unnamed: 0"].dropna().unique())
        if not all_units:
            messagebox.showwarning(
                tr("warning"),
                (
                    tr("no_units_found")
                    if "no_units_found" in _TRANSLATIONS
                    else "No se encontraron unidades en la hoja seleccionada."
                ),
            )
            return
        display_names = [
            f"{CODE_TO_NAME.get(unit, str(unit))} ({unit})" for unit in all_units
        ]

        win = Toplevel(self)
        win.title(tr("select_units"))
        win.geometry("400x430")
        lbl = tk.Label(
            win,
            text=(
                tr("select_units_label")
                if "select_units_label" in _TRANSLATIONS
                else "Selecciona unidades para el análisis:"
            ),
        )
        lbl.pack(pady=10)
        listbox = tk.Listbox(
            win,
            selectmode=tk.MULTIPLE if allow_multiple else tk.SINGLE,
            width=50,
            height=20,
        )
        for name in display_names:
            listbox.insert(tk.END, name)
        listbox.pack()
        button_frame = tk.Frame(win)
        button_frame.pack(pady=10)

        def select_all():
            listbox.select_set(0, tk.END)

        def unselect_all():
            listbox.select_clear(0, tk.END)

        def confirm_selection():
            selected_indices = listbox.curselection()
            if not selected_indices:
                messagebox.showerror(
                    tr("error"),
                    (
                        tr("select_at_least_one_unit")
                        if "select_at_least_one_unit" in _TRANSLATIONS
                        else "Selecciona al menos una unidad."
                    ),
                )
                return
            config_name = getattr(self, "_last_analysis_type", "series_config")
            selected_units = [all_units[i] for i in selected_indices]
            getattr(self.project_config, config_name)["selected_units"] = selected_units
            # Limpia selección de años al cambiar unidad en serie de tiempo
            if config_name == "series_config":
                getattr(self.project_config, config_name)["selected_years"] = []
            win.destroy()
            callback()
            self.sync_gui_from_cfg()  # ← nuevo

        tk.Button(
            button_frame,
            text=tr("select_all"),
            command=select_all,
            bg="lightgreen",
            fg="black",
        ).grid(row=0, column=0, padx=5)
        tk.Button(
            button_frame,
            text=tr("unselect_all"),
            command=unselect_all,
            bg="lightcoral",
            fg="black",
        ).grid(row=0, column=1, padx=5)
        tk.Button(
            button_frame,
            text=tr("ok"),
            command=confirm_selection,
            bg="lightblue",
            fg="black",
        ).grid(row=0, column=2, padx=5)

    def step_select_year(self, callback=None, multi=True):
        config_name = getattr(self, "_last_analysis_type", "series_config")
        cfg = getattr(self.project_config, config_name)
        try:
            df = pd.read_excel(
                cfg["data_file"], sheet_name=cfg["selected_indicators"][0]
            )
        except Exception as e:
            logger.exception("Error reading Excel for years")
            messagebox.showerror(
                tr("error"),
                (
                    tr("file_load_error") + f"\n{e}"
                    if "file_load_error" in _TRANSLATIONS
                    else f"No se pudo cargar el archivo seleccionado.\n{e}"
                ),
            )
            return
        year_columns = [
            col for col in df.columns if col != "Unnamed: 0" and str(col).isdigit()
        ]
        year_columns = sorted(year_columns, key=lambda x: int(x))  # ordena por año
        if not year_columns:
            messagebox.showwarning(
                tr("warning"),
                (
                    tr("no_years_found")
                    if "no_years_found" in _TRANSLATIONS
                    else "No se encontraron años válidos en la hoja seleccionada."
                ),
            )
            return

        win = Toplevel(self)
        win.title(tr("select_years"))
        win.geometry("300x400")
        tk.Label(
            win,
            text=(
                tr("select_years_label")
                if "select_years_label" in _TRANSLATIONS
                else "Select year(s) for analysis:"
            ),
        ).pack(pady=10)
        listbox = tk.Listbox(
            win, selectmode=tk.MULTIPLE if multi else tk.SINGLE, height=20
        )
        for year in year_columns:
            listbox.insert(tk.END, year)
        listbox.pack()

        def select_all():
            listbox.select_set(0, tk.END)

        def unselect_all():
            listbox.select_clear(0, tk.END)

        def confirm():
            idxs = listbox.curselection()
            if not idxs:
                messagebox.showerror(
                    tr("error"),
                    (
                        tr("select_at_least_one_year")
                        if "select_at_least_one_year" in _TRANSLATIONS
                        else "Selecciona al menos un año."
                    ),
                )
                return
            config_name = getattr(self, "_last_analysis_type", "series_config")
            selected_years = [year_columns[i] for i in idxs]
            getattr(self.project_config, config_name)["selected_years"] = selected_years
            win.destroy()
            self.status.config(
                text=f"{tr('select_years') if 'select_years' in _TRANSLATIONS else 'Años seleccionados'}: {selected_years}"
            )
            if callback:
                callback()
            # Forzar enable inmediato para scatter
            if config_name == "scatter_plot_config":
                print(f"[Scatter DEBUG] Años guardados: {selected_years}")
                self._set_button_state(
                    self.btn_run_scatter, tk.NORMAL, debug_tag="scatter-after-years"
                )
            self.sync_gui_from_cfg()

        button_frame = tk.Frame(win)
        button_frame.pack(pady=10)
        tk.Button(
            button_frame, text=tr("select_all"), command=select_all, bg="lightgreen"
        ).grid(row=0, column=0, padx=5)
        tk.Button(
            button_frame, text=tr("unselect_all"), command=unselect_all, bg="lightcoral"
        ).grid(row=0, column=1, padx=5)
        tk.Button(button_frame, text=tr("ok"), command=confirm, bg="lightblue").grid(
            row=0, column=2, padx=5
        )

    def gui_select_imputation_strategy(self):
        estrategia = None
        params = {}

        STRATEGIAS = [
            (
                "interpolacion",
                (
                    tr("impute_interpolation")
                    if "impute_interpolation" in _TRANSLATIONS
                    else "Interpolación (lineal por defecto, o especificar método)"
                ),
            ),
            (
                "mean",
                (
                    tr("impute_mean")
                    if "impute_mean" in _TRANSLATIONS
                    else "Rellenar con la Media"
                ),
            ),
            (
                "median",
                (
                    tr("impute_median")
                    if "impute_median" in _TRANSLATIONS
                    else "Rellenar con la Mediana"
                ),
            ),
            (
                "most_frequent",
                (
                    tr("impute_most_frequent")
                    if "impute_most_frequent" in _TRANSLATIONS
                    else "Rellenar con el Valor Más Frecuente (moda)"
                ),
            ),
            (
                "ffill",
                (
                    tr("impute_ffill")
                    if "impute_ffill" in _TRANSLATIONS
                    else "Rellenar con valor anterior (Forward Fill)"
                ),
            ),
            (
                "bfill",
                (
                    tr("impute_bfill")
                    if "impute_bfill" in _TRANSLATIONS
                    else "Rellenar con valor siguiente (Backward Fill)"
                ),
            ),
            (
                "iterative",
                (
                    tr("impute_iterative")
                    if "impute_iterative" in _TRANSLATIONS
                    else "Imputación Iterativa (multivariada)"
                ),
            ),
            (
                "knn",
                (
                    tr("impute_knn")
                    if "impute_knn" in _TRANSLATIONS
                    else "Imputación KNN (basada en vecinos)"
                ),
            ),
            (
                "valor_constante",
                (
                    tr("impute_constant")
                    if "impute_constant" in _TRANSLATIONS
                    else "Rellenar con un Valor Constante específico"
                ),
            ),
            (
                "eliminar_filas",
                (
                    tr("impute_drop_rows")
                    if "impute_drop_rows" in _TRANSLATIONS
                    else "Eliminar filas con datos faltantes"
                ),
            ),
            (
                "ninguna",
                (
                    tr("impute_none")
                    if "impute_none" in _TRANSLATIONS
                    else "No aplicar ninguna imputación (mantener NaNs)"
                ),
            ),
        ]

        win = Toplevel(self)
        win.title(
            tr("select_imputation_strategy")
            if "select_imputation_strategy" in _TRANSLATIONS
            else "Selecciona Estrategia de Imputación"
        )
        win.geometry("480x420")
        tk.Label(
            win,
            text=(
                tr("select_how_to_impute")
                if "select_how_to_impute" in _TRANSLATIONS
                else "Selecciona cómo quieres imputar los datos faltantes:"
            ),
            font=("Arial", 11, "bold"),
        ).pack(pady=10)

        estrategia_var = tk.StringVar(value="interpolacion")
        for key, txt in STRATEGIAS:
            tk.Radiobutton(
                win,
                text=txt,
                variable=estrategia_var,
                value=key,
                anchor="w",
                justify="left",
            ).pack(fill="x", padx=25)

        valor_entry = tk.Entry(win)

        def on_radio_change(*a):
            if estrategia_var.get() == "valor_constante":
                valor_entry.pack(pady=8)
                valor_entry.delete(0, tk.END)
                valor_entry.insert(0, "0")
            else:
                valor_entry.pack_forget()

        estrategia_var.trace_add("write", on_radio_change)

        def on_ok():
            nonlocal estrategia, params
            estrategia = estrategia_var.get()
            if estrategia == "valor_constante":
                try:
                    params["valor_constante"] = float(valor_entry.get())
                except Exception:
                    params["valor_constante"] = valor_entry.get()
            win.destroy()

        tk.Button(win, text=tr("ok"), command=on_ok, bg="lightblue").pack(pady=14)
        win.transient(self)
        win.grab_set()
        self.wait_window(win)
        return estrategia, params

    def gui_select_n_components(
        self, max_components, suggested_n_90=None, suggested_n_95=None
    ):
        """
        Abre un diálogo para que el usuario seleccione el número de componentes principales a retener.
        """
        selected_n = [
            max_components
        ]  # Usamos lista para tener referencia mutable en closure

        win = Toplevel(self)
        win.title(
            tr("select_n_components_title")
            if "select_n_components_title" in _TRANSLATIONS
            else "Seleccionar número de componentes principales"
        )
        win.geometry("420x250")
        mensaje = (
            tr("select_n_components_msg").format(
                max_components=max_components, n90=suggested_n_90, n95=suggested_n_95
            )
            if "select_n_components_msg" in _TRANSLATIONS
            else f"Ingrese cuántos componentes principales deseas retener (1-{max_components}).\n"
        )
        if suggested_n_90:
            mensaje += f"{tr('suggestion_80') if 'suggestion_80' in _TRANSLATIONS else 'Sugerencia:'} {suggested_n_90} componentes ≈ 80% varianza.\n"
        if suggested_n_95:
            mensaje += f"{tr('suggestion_90') if 'suggestion_90' in _TRANSLATIONS else 'Sugerencia:'} {suggested_n_95} componentes ≈ 90% varianza.\n"
        mensaje += f"{tr('leave_empty_for_all') if 'leave_empty_for_all' in _TRANSLATIONS else f'Deja vacío para usar todos ({max_components}).'}"

        tk.Label(win, text=mensaje, justify="left", wraplength=400).pack(pady=16)

        entry = tk.Entry(win)
        entry.pack(pady=6)
        entry.focus_set()

        def on_ok():
            value = entry.get().strip()
            if not value:
                selected_n[0] = max_components
            else:
                try:
                    n = int(value)
                    if 1 <= n <= max_components:
                        selected_n[0] = n
                    else:
                        messagebox.showerror(
                            tr("error"),
                            (
                                tr("n_components_range").format(
                                    max_components=max_components
                                )
                                if "n_components_range" in _TRANSLATIONS
                                else f"El número debe estar entre 1 y {max_components}."
                            ),
                        )
                        return
                except Exception:
                    messagebox.showerror(
                        tr("error"),
                        (
                            tr("must_be_integer")
                            if "must_be_integer" in _TRANSLATIONS
                            else "Debes ingresar un número entero válido."
                        ),
                    )
                    return
            win.destroy()

        tk.Button(win, text=tr("ok"), command=on_ok, bg="lightblue", width=12).pack(
            pady=16
        )
        win.grab_set()
        self.wait_window(win)
        return selected_n[0]

    def run_project_from_cfg(self):
        cfg = self.project_config
        ready = {
            "series": all(
                [
                    cfg.series_config.get("data_file"),
                    cfg.series_config.get("selected_indicators"),
                    cfg.series_config.get("selected_units"),
                    cfg.series_config.get("selected_years"),
                ]
            ),
            "cross_section": all(
                [
                    cfg.cross_section_config.get("data_file"),
                    cfg.cross_section_config.get("selected_indicators"),
                    cfg.cross_section_config.get("selected_units"),
                    cfg.cross_section_config.get("selected_years"),
                ]
            ),
            "panel": all(
                [
                    cfg.panel_config.get("data_file"),
                    cfg.panel_config.get("selected_indicators"),
                    cfg.panel_config.get("selected_units"),
                    cfg.panel_config.get("selected_years"),
                ]
            ),
        }
        available = [k for k, v in ready.items() if v]
        if not available:
            messagebox.showinfo(
                "Selector de flujo", "No hay ningún análisis configurado para ejecutar."
            )
            return
        if len(available) == 1:
            if available[0] == "series":
                self.run_series_analysis()
            elif available[0] == "cross_section":
                self.run_cross_section_analysis()
            elif available[0] == "panel":
                self.run_panel_analysis()
        else:
            # Preguntar al usuario cuál ejecutar
            win = Toplevel(self)
            win.title("¿Qué análisis quieres ejecutar?")
            win.geometry("340x200")
            tk.Label(
                win, text="Selecciona el análisis a ejecutar:", font=("Arial", 12)
            ).pack(pady=18)
            for tipo, label, func in [
                ("series", "Serie de tiempo", self.run_series_analysis),
                ("cross_section", "Corte transversal", self.run_cross_section_analysis),
                ("panel", "Panel 3D", self.run_panel_analysis),
            ]:
                if tipo in available:
                    tk.Button(
                        win,
                        text=label,
                        width=22,
                        height=2,
                        command=lambda f=func, w=win: (w.destroy(), f()),
                    ).pack(pady=6)
            tk.Button(win, text="Cancelar", command=win.destroy).pack(pady=8)
            win.grab_set()
            win.focus_set()

    @safe_gui_callback
    def edit_title_dialog(self, config_name=None):
        config_name = config_name or "series_config"
        cfg = getattr(self.project_config, config_name)
        new_title = simpledialog.askstring(
            tr("edit_title"),
            (
                tr("enter_new_title")
                if "enter_new_title" in _TRANSLATIONS
                else "Escribe el nuevo título:"
            ),
            initialvalue=cfg["custom_titles"].get("biplot", ""),
        )
        if new_title is not None:
            cfg["custom_titles"]["biplot"] = new_title.strip()
            self.status.config(
                text=(
                    tr("title_updated")
                    if "title_updated" in _TRANSLATIONS
                    else "Título actualizado."
                )
            )
            self.sync_gui_from_cfg()

    @safe_gui_callback
    def edit_legend_dialog(self, config_name=None):
        config_name = config_name or "series_config"
        cfg = getattr(self.project_config, config_name)
        new_txt = simpledialog.askstring(
            tr("edit_legend"),
            (
                tr("enter_legend_title")
                if "enter_legend_title" in _TRANSLATIONS
                else "Escribe cómo quieres que aparezca el encabezado de la leyenda:"
            ),
            initialvalue=cfg["custom_titles"].get("legend", ""),
        )
        if new_txt is not None:
            cfg["custom_titles"]["legend"] = new_txt.strip()
            self.status.config(
                text=(
                    tr("legend_title_updated")
                    if "legend_title_updated" in _TRANSLATIONS
                    else "Título de leyenda actualizado."
                )
            )
            self.sync_gui_from_cfg()

    @safe_gui_callback
    def edit_colors_dialog(self, config_name=None):
        config_name = config_name or "series_config"
        cfg = getattr(self.project_config, config_name)
        win = Toplevel(self)
        win.title(tr("assign_colors"))
        win.geometry("360x420")
        win.resizable(False, False)
        tk.Label(
            win,
            text=(
                tr("choose_units")
                if "choose_units" in _TRANSLATIONS
                else "Elige individuos / unidades:"
            ),
        ).pack(pady=(12, 2))
        lst = tk.Listbox(win, selectmode=tk.EXTENDED, height=14, width=34)
        for name in sorted(cfg["selected_units"]):
            lst.insert(tk.END, name)
        lst.pack()
        preview = tk.Label(win, width=4, relief="groove", bg="gray")
        preview.pack(pady=6)

        def refresh_preview(*_):
            sels = lst.curselection()
            if sels:
                unit = lst.get(sels[0])
                preview.config(bg=cfg["color_groups"].get(unit, "gray"))

        lst.bind("<<ListboxSelect>>", refresh_preview)

        def set_color():
            rgb, hex_ = colorchooser.askcolor()
            if not hex_:
                return
            for i in lst.curselection():
                cfg["color_groups"][lst.get(i)] = hex_
            refresh_preview()

        def unset_color():
            for i in lst.curselection():
                cfg["color_groups"].pop(lst.get(i), None)
            refresh_preview()

        frm_btn = tk.Frame(win)
        frm_btn.pack(pady=4)
        tk.Button(
            frm_btn,
            text=(
                tr("choose_color")
                if "choose_color" in _TRANSLATIONS
                else "Elegir color"
            ),
            width=12,
            command=set_color,
        ).grid(row=0, column=0, padx=4)
        tk.Button(
            frm_btn,
            text=(
                tr("remove_color")
                if "remove_color" in _TRANSLATIONS
                else "Quitar color"
            ),
            width=12,
            command=unset_color,
        ).grid(row=0, column=1, padx=4)

        def accept():
            self.sync_gui_from_cfg()
            win.destroy()

        tk.Button(win, text=tr("ok"), width=12, bg="#b7e0ee", command=accept).pack(
            pady=(12, 8)
        )
        win.grab_set()
        win.focus_set()

    @safe_gui_callback
    def edit_units_dialog(self, config_name=None):
        """
        Permite borrar o agregar unidades en caliente.
        • Muestra dos list-box:
          -  «Disponibles»  (todas las que existen en el Excel)
          -  «Seleccionadas» (las que ya están en cfg.selected_units)
        • Botones  >>  y  <<  para mover.
        """
        config_name = config_name or "series_config"
        cfg = getattr(self.project_config, config_name)
        if not cfg["data_file"] or not cfg["selected_indicators"]:
            messagebox.showwarning(
                tr("warning"),
                (
                    tr("select_file_and_indicators_first")
                    if "select_file_and_indicators_first" in _TRANSLATIONS
                    else "Primero elige archivo/indicadores"
                ),
            )
            return
        all_units = sorted(
            pd.read_excel(cfg["data_file"], sheet_name=cfg["selected_indicators"][0])[
                "Unnamed: 0"
            ]
            .dropna()
            .unique()
        )
        win = Toplevel(self)
        win.title(tr("edit_units"))
        win.resizable(False, False)
        tk.Label(
            win, text=tr("available") if "available" in _TRANSLATIONS else "Disponibles"
        ).grid(row=0, column=0, padx=8, pady=6)
        tk.Label(
            win, text=tr("selected") if "selected" in _TRANSLATIONS else "Seleccionadas"
        ).grid(row=0, column=2, padx=8, pady=6)
        lst_all = tk.Listbox(
            win, height=18, selectmode=tk.EXTENDED, exportselection=False
        )
        lst_sel = tk.Listbox(
            win, height=18, selectmode=tk.EXTENDED, exportselection=False
        )
        for u in all_units:
            lst_all.insert(tk.END, u)
        for u in cfg["selected_units"]:
            lst_sel.insert(tk.END, u)
        lst_all.grid(row=1, column=0, padx=8)
        lst_sel.grid(row=1, column=2, padx=8)
        frm_btn = tk.Frame(win)
        frm_btn.grid(row=1, column=1, padx=4)

        def to_sel():
            for i in lst_all.curselection():
                unit = lst_all.get(i)
                if unit not in lst_sel.get(0, tk.END):
                    lst_sel.insert(tk.END, unit)

        def to_all():
            sel_items = [lst_sel.get(i) for i in lst_sel.curselection()]
            for item in sel_items:
                idx = lst_sel.get(0, tk.END).index(item)
                lst_sel.delete(idx)
                cfg["color_groups"].pop(item, None)

        tk.Button(
            frm_btn,
            text=tr("move_right") if "move_right" in _TRANSLATIONS else ">>",
            command=to_sel,
        ).pack(pady=10)
        tk.Button(
            frm_btn,
            text=tr("move_left") if "move_left" in _TRANSLATIONS else "<<",
            command=to_all,
        ).pack(pady=10)

        def accept():
            cfg["selected_units"] = list(lst_sel.get(0, tk.END))
            self.sync_gui_from_cfg()
            win.destroy()

        tk.Button(win, text=tr("ok"), width=12, command=accept).grid(
            row=2, column=0, columnspan=3, pady=12
        )
        win.grab_set()
        win.focus_set()

    @safe_gui_callback
    def edit_footer_dialog(self, config_name=None):
        config_name = config_name or "series_config"
        cfg = getattr(self.project_config, config_name)
        new_note = simpledialog.askstring(
            tr("edit_footer"),
            (
                tr("enter_footer")
                if "enter_footer" in _TRANSLATIONS
                else "Texto que aparecerá debajo del gráfico:"
            ),
            initialvalue=cfg["custom_titles"].get("footer", ""),
        )
        if new_note is not None:
            cfg["custom_titles"]["footer"] = new_note.strip()
            self.status.config(
                text=(
                    tr("footer_updated")
                    if "footer_updated" in _TRANSLATIONS
                    else "Fuente/leyenda actualizada."
                )
            )
            self.sync_gui_from_cfg()

    # =============================================
    # FUNCIONES PARA BIPLOT AVANZADO
    # =============================================

    @safe_gui_callback
    def create_advanced_biplot_dialog(self):
        """Diálogo principal para crear biplots avanzados."""
        # Verificar que tenemos datos del biplot
        if not hasattr(self, "biplot_data") or not self.biplot_data:
            messagebox.showwarning(
                "Datos no disponibles",
                "Primero debes completar el wizard de configuración.",
            )
            return

        win = Toplevel(self)
        win.title("🎨 Crear Biplot Avanzado")
        win.geometry("450x600")
        win.resizable(False, False)
        win.configure(bg="#f8fafc")

        # Centrar ventana
        win.update_idletasks()
        x = (win.winfo_screenwidth() // 2) - 225
        y = (win.winfo_screenheight() // 2) - 300
        win.geometry(f"450x600+{x}+{y}")

        # Título
        title_frame = tk.Frame(win, bg="#f8fafc")
        title_frame.pack(fill="x", padx=20, pady=(20, 10))

        tk.Label(
            title_frame,
            text="🎨 Configuración de Biplot Avanzado",
            font=("Segoe UI", 16, "bold"),
            bg="#f8fafc",
            fg="#1e293b",
        ).pack()

        # Frame principal con scroll
        main_frame = tk.Frame(win, bg="#f8fafc")
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Variables para almacenar selecciones
        self.biplot_config = {
            "categorization": tk.StringVar(value="continents"),
            "marker_scheme": tk.StringVar(value="classic"),
            "color_scheme": tk.StringVar(value="viridis"),
            "show_arrows": tk.BooleanVar(value=True),
            "show_labels": tk.BooleanVar(value=True),
            "alpha": tk.DoubleVar(value=0.7),
        }

        # Información del año seleccionado
        info_frame = tk.LabelFrame(
            main_frame,
            text="📅 Información del Análisis",
            bg="#f8fafc",
            fg="#374151",
            font=("Segoe UI", 10, "bold"),
        )
        info_frame.pack(fill="x", pady=(0, 15))

        info_inner = tk.Frame(info_frame, bg="#f8fafc")
        info_inner.pack(fill="x", padx=10, pady=5)

        tk.Label(
            info_inner,
            text=f"Año seleccionado: {self.biplot_data['year']}",
            bg="#f8fafc",
            fg="#1e293b",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        # Mostrar información de unidades y indicadores
        cfg = self.biplot_data["config"]
        tk.Label(
            info_inner,
            text=f"Unidades de investigación: {len(cfg.get('selected_units', []))}",
            bg="#f8fafc",
            fg="#475569",
            font=("Segoe UI", 9),
        ).pack(anchor="w")
        tk.Label(
            info_inner,
            text=f"Indicadores: {len(cfg.get('selected_indicators', []))}",
            bg="#f8fafc",
            fg="#475569",
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        # Sección: Categorización
        cat_frame = tk.LabelFrame(
            main_frame,
            text="🌍 Categorización de Unidades de Investigación",
            bg="#f8fafc",
            fg="#374151",
            font=("Segoe UI", 10, "bold"),
        )
        cat_frame.pack(fill="x", pady=(0, 15))

        cat_inner = tk.Frame(cat_frame, bg="#f8fafc")
        cat_inner.pack(fill="x", padx=10, pady=5)

        categorizations = [
            ("continents", "🌎 Por Continentes"),
            ("development", "📈 Por Desarrollo"),
            ("income", "💰 Por Nivel de Ingresos"),
        ]

        for value, text in categorizations:
            tk.Radiobutton(
                cat_inner,
                text=text,
                variable=self.biplot_config["categorization"],
                value=value,
                bg="#f8fafc",
                font=("Segoe UI", 9),
            ).pack(anchor="w", pady=2)

        # Sección: Esquema de Marcadores
        marker_frame = tk.LabelFrame(
            main_frame,
            text="🔵 Esquema de Marcadores",
            bg="#f8fafc",
            fg="#374151",
            font=("Segoe UI", 10, "bold"),
        )
        marker_frame.pack(fill="x", pady=(0, 15))

        marker_inner = tk.Frame(marker_frame, bg="#f8fafc")
        marker_inner.pack(fill="x", padx=10, pady=5)

        marker_schemes = [
            ("classic", "🔵 Clásico (círculos, cuadrados, triángulos)"),
            ("geometric", "🔷 Geométrico (diamantes, pentágonos, estrellas)"),
            ("varied", "🎭 Variado (mix de formas)"),
        ]

        for value, text in marker_schemes:
            tk.Radiobutton(
                marker_inner,
                text=text,
                variable=self.biplot_config["marker_scheme"],
                value=value,
                bg="#f8fafc",
                font=("Segoe UI", 9),
            ).pack(anchor="w", pady=2)

        # Sección: Esquema de Colores
        color_frame = tk.LabelFrame(
            main_frame,
            text="🎨 Esquema de Colores",
            bg="#f8fafc",
            fg="#374151",
            font=("Segoe UI", 10, "bold"),
        )
        color_frame.pack(fill="x", pady=(0, 15))

        color_inner = tk.Frame(color_frame, bg="#f8fafc")
        color_inner.pack(fill="x", padx=10, pady=5)

        color_schemes = [
            ("viridis", "🌊 Viridis (azul-verde-amarillo)"),
            ("plasma", "🔥 Plasma (púrpura-rosa-amarillo)"),
            ("tab10", "🌈 Tab10 (colores distintivos)"),
            ("set3", "🎨 Set3 (colores suaves)"),
        ]

        for value, text in color_schemes:
            tk.Radiobutton(
                color_inner,
                text=text,
                variable=self.biplot_config["color_scheme"],
                value=value,
                bg="#f8fafc",
                font=("Segoe UI", 9),
            ).pack(anchor="w", pady=2)

        # Sección: Opciones Adicionales
        options_frame = tk.LabelFrame(
            main_frame,
            text="⚙️ Opciones Adicionales",
            bg="#f8fafc",
            fg="#374151",
            font=("Segoe UI", 10, "bold"),
        )
        options_frame.pack(fill="x", pady=(0, 15))

        options_inner = tk.Frame(options_frame, bg="#f8fafc")
        options_inner.pack(fill="x", padx=10, pady=5)

        tk.Checkbutton(
            options_inner,
            text="Mostrar flechas de variables",
            variable=self.biplot_config["show_arrows"],
            bg="#f8fafc",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=2)

        tk.Checkbutton(
            options_inner,
            text="Mostrar etiquetas de unidades",
            variable=self.biplot_config["show_labels"],
            bg="#f8fafc",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=2)

        # Control de transparencia
        alpha_frame = tk.Frame(options_inner, bg="#f8fafc")
        alpha_frame.pack(fill="x", pady=5)

        tk.Label(
            alpha_frame, text="Transparencia:", bg="#f8fafc", font=("Segoe UI", 9)
        ).pack(side="left")
        alpha_scale = tk.Scale(
            alpha_frame,
            from_=0.1,
            to=1.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.biplot_config["alpha"],
            bg="#f8fafc",
            font=("Segoe UI", 8),
        )
        alpha_scale.pack(side="right", fill="x", expand=True, padx=(10, 0))

        # Botones
        button_frame = tk.Frame(win, bg="#f8fafc")
        button_frame.pack(fill="x", padx=20, pady=(10, 20))

        tk.Button(
            button_frame,
            text="🎨 Crear Biplot",
            command=lambda: self.execute_advanced_biplot(win),
            bg="#3b82f6",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            width=15,
            height=2,
        ).pack(side="left", padx=(0, 10))

        tk.Button(
            button_frame,
            text="👁️ Vista Previa",
            command=lambda: self.preview_advanced_biplot(win),
            bg="#10b981",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            width=15,
            height=2,
        ).pack(side="left", padx=(0, 10))

        tk.Button(
            button_frame,
            text="💾 Guardar Config",
            command=lambda: self.save_biplot_main_config(),
            bg="#8b5cf6",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            width=15,
            height=2,
        ).pack(side="left", padx=(0, 10))

        tk.Button(
            button_frame,
            text="❌ Cancelar",
            command=win.destroy,
            bg="#ef4444",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            width=15,
            height=2,
        ).pack(side="right")

        win.grab_set()
        win.focus_set()

    def execute_advanced_biplot(self, dialog_window):
        """Ejecuta la creación del biplot avanzado."""
        try:
            # Guardar configuración automáticamente antes de ejecutar
            if hasattr(self, "biplot_config"):
                config_to_save = {
                    "categorization": self.biplot_config["categorization"].get(),
                    "marker_scheme": self.biplot_config["marker_scheme"].get(),
                    "color_scheme": self.biplot_config["color_scheme"].get(),
                    "show_arrows": self.biplot_config["show_arrows"].get(),
                    "show_labels": self.biplot_config["show_labels"].get(),
                    "alpha": self.biplot_config["alpha"].get(),
                }

                # Inicializar project_config si no existe
                if not hasattr(self, "project_config"):
                    self.project_config = {}

                if "biplot_advanced_config" not in self.project_config:
                    self.project_config["biplot_advanced_config"] = {}

                self.project_config["biplot_advanced_config"][
                    "main_config"
                ] = config_to_save
                print(f"✅ Configuración guardada: {config_to_save}")

            # Cerrar diálogo
            dialog_window.destroy()

            # Mostrar loading
            self.show_loading("Creando biplot avanzado...")

            # Obtener configuración
            config = {
                "year": self.biplot_config["year"].get(),
                "categorization_scheme": self.biplot_config["categorization"].get(),
                "marker_scheme": self.biplot_config["marker_scheme"].get(),
                "color_scheme": self.biplot_config["color_scheme"].get(),
                "show_arrows": self.biplot_config["show_arrows"].get(),
                "show_labels": self.biplot_config["show_labels"].get(),
                "alpha": self.biplot_config["alpha"].get(),
            }

            # Usar los datos procesados
            df_for_biplot = self.biplot_data.get(
                "df_year_standardized", self.biplot_data["df_year_processed"]
            )

            # Intentar con biplot avanzado, si falla usar simplificado
            try:
                from biplot_advanced import create_advanced_biplot

                if (
                    "biplot_advanced_config" in self.project_config
                    and "custom_categories"
                    in self.project_config["biplot_advanced_config"]
                ):
                    config["custom_categories"] = self.project_config[
                        "biplot_advanced_config"
                    ]["custom_categories"]
                success = create_advanced_biplot(df_for_biplot, config)
            except Exception as e:
                print(f"⚠️ Error con biplot_advanced: {e}")
                print("🔄 Intentando con versión simplificada...")
                try:
                    from biplot_simple import create_advanced_biplot_simple

                    success = create_advanced_biplot_simple(df_for_biplot, config)
                except Exception as e2:
                    print(f"❌ Error con biplot simplificado: {e2}")
                    success = False

            if success:
                self.status.config(text="✅ Biplot avanzado creado exitosamente")
                messagebox.showinfo(
                    "Éxito", "Biplot avanzado creado y guardado exitosamente."
                )
            else:
                self.status.config(text="❌ Error al crear biplot avanzado")
                messagebox.showerror(
                    "Error", "Hubo un problema al crear el biplot avanzado."
                )

        except Exception as e:
            self.status.config(text=f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Error al crear biplot avanzado:\n{str(e)}")
        finally:
            self.hide_loading()

    def preview_advanced_biplot(self, dialog_window):
        """Muestra una vista previa del biplot avanzado."""
        try:
            from biplot_advanced import get_categorization_preview

            config = {
                "categorization_scheme": self.biplot_config["categorization"].get(),
                "marker_scheme": self.biplot_config["marker_scheme"].get(),
                "color_scheme": self.biplot_config["color_scheme"].get(),
            }

            preview_info = get_categorization_preview(self.df, config)

            # Crear ventana de vista previa
            preview_win = Toplevel(dialog_window)
            preview_win.title("👁️ Vista Previa de Configuración")
            preview_win.geometry("400x300")
            preview_win.configure(bg="#f8fafc")

            # Centrar
            preview_win.update_idletasks()
            x = dialog_window.winfo_x() + 50
            y = dialog_window.winfo_y() + 50
            preview_win.geometry(f"400x300+{x}+{y}")

            # Contenido
            text_widget = tk.Text(
                preview_win,
                wrap="word",
                bg="white",
                fg="#1e293b",
                font=("Consolas", 9),
                padx=10,
                pady=10,
            )
            text_widget.pack(fill="both", expand=True, padx=10, pady=10)

            text_widget.insert("1.0", preview_info)
            text_widget.config(state="disabled")

            tk.Button(
                preview_win,
                text="Cerrar",
                command=preview_win.destroy,
                bg="#6b7280",
                fg="white",
                font=("Segoe UI", 9),
            ).pack(pady=5)

        except Exception as e:
            messagebox.showerror("Error", f"Error en vista previa:\n{str(e)}")


# ------------- FIN CLASE --------------

if __name__ == "__main__":
    app = PCAApp()
    app.mainloop()
