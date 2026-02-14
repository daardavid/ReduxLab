# event_handlers.py
"""
Event Handlers Module for PCA GUI Application.

This module contains event handling methods for analysis execution
and user interactions in the PCA application.

Author: David Armando Abreu Rosique
Fecha: 2025
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
from backend.pca_logic import PCAAnalysisLogic
from backend.pca_cross_logic import PCAAnalysisLogic as CrossPCAAnalysisLogic
from backend.pca_panel3d_logic import PCAPanel3DLogic
from backend import data_loader_module as dl
from backend import visualization_module as dl_viz
from backend.constants import MAPEO_INDICADORES
import logging

logger = logging.getLogger("pca_gui.event_handlers")


class EventHandlers:
    """Clase para manejar eventos de la aplicación PCA."""

    def __init__(self, parent_app):
        """Inicializar con referencia a la app principal."""
        self.parent_app = parent_app

    def start_cross_section_analysis(self):
        """Inicia el flujo de análisis de corte transversal (varias unidades, un año o varios años)."""
        try:
            self.parent_app._last_analysis_type = "cross_section_config"
            self.parent_app.status.config(
                text="Flujo: Corte transversal (varias unidades, años)"
            )
            self.parent_app.cross_section_wizard()
        except tk.TclError:
            pass

    def run_series_analysis(self):
        """Ejecuta el análisis de serie de tiempo para la unidad y años seleccionados (pueden ser varios años)."""
        from pca_logic import PCAAnalysisLogic

        cfg = self.parent_app.project_config.series_config
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

        logger.info(
            f"Ejecutando análisis de serie de tiempo: Archivo: {cfg['data_file']}, Indicadores: {cfg['selected_indicators']}, Unidad: {cfg['selected_units'][0]}, Años: {selected_years if selected_years else 'Todos disponibles'}"
        )

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
                estrategia, params = self.parent_app.gui_select_imputation_strategy()
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
            logger.info(
                "Agregando visualización de componentes principales en el tiempo"
            )

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

                logger.info(
                    "Creando gráfico especializado de componentes principales..."
                )
                dl_viz.graficar_componentes_principales_tiempo(
                    results["df_componentes_principales"],
                    varianza_explicada=varianza_info,
                    titulo=f"Evolución de Componentes Principales - {cfg['selected_units'][0]}",
                )
        except Exception as e:
            logger.error(f"Error al generar gráficos: {e}")
            messagebox.showwarning(
                "Advertencia",
                f"Se completó el análisis pero hubo problemas al generar los gráficos: {e}",
            )

        # Preguntar si desea exportar los resultados
        if messagebox.askyesno(
            "¿Exportar?",
            "¿Quieres guardar los resultados del análisis de serie de tiempo?",
        ):
            filename = filedialog.asksaveasfilename(
                title="Guardar resultados serie de tiempo",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx *.xls")],
            )
            if filename:
                try:
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
                    messagebox.showinfo("Listo", f"Archivo guardado:\n{filename}")
                except Exception as e:
                    messagebox.showerror("Error", f"No se pudo guardar el archivo: {e}")

        messagebox.showinfo(
            "Info", "Análisis de serie de tiempo completado correctamente."
        )

    def run_cross_section_analysis(self):
        """Ejecuta el análisis de corte transversal."""
        from pca_cross_logic import PCAAnalysisLogic

        cfg = self.parent_app.project_config.cross_section_config
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
                    estrategia, params = (
                        self.parent_app.gui_select_imputation_strategy()
                    )
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
                msg = f"Para el año {year_to_analyze}, los 2 primeros componentes explican:\nPC1: {evr_cs[0]:.2%}\nPC2: {evr_cs[1]:.2%}\nTotal: {cum_evr_cs[1]:.2%} de la varianza"
                title = "Varianza explicada por los 2 componentes"
            elif len(evr_cs) == 1:
                msg = f"Solo se pudo calcular un componente principal para el año {year_to_analyze}.\nPC1 explica: {evr_cs[0]:.2%} de la varianza"
                title = "Varianza explicada por los 2 componentes"
            else:
                msg = "No se pudo calcular componentes principales para este año."
                title = "Varianza explicada por los 2 componentes"
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

            if messagebox.askyesno(
                "¿Exportar?",
                f"¿Quieres guardar los resultados para el año {year_to_analyze}?",
            ):
                filename = filedialog.asksaveasfilename(
                    title=f"Guardar resultados {year_to_analyze}",
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
                    messagebox.showinfo("Listo", f"Archivo guardado:\n{filename}")
        self.parent_app.status.config(text="Análisis de corte transversal completado.")

    def run_panel_analysis(self):
        """Ejecuta el análisis de panel 3D."""
        try:
            cfg = self.parent_app.project_config.panel_config
            self.parent_app.status.config(
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
                "¿Exportar?", "¿Quieres guardar los resultados del análisis Panel 3D?"
            ):
                filename = filedialog.asksaveasfilename(
                    title="Guardar resultados Panel 3D",
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx *.xls")],
                )
                if filename:
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
                    messagebox.showinfo("Listo", f"Archivo guardado:\n{filename}")
        except tk.TclError:
            pass

    def run_scatter_plot(self):
        """Ejecuta el Scatterplot PCA usando únicamente su propia configuración."""
        cfg = self.parent_app.project_config.scatter_plot_config
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
        self.parent_app.status.config(text="Generando Scatterplot PCA ...")
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

            from sklearn.preprocessing import StandardScaler
            from backend.scatter_plot import generate_scatter_plot

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

    def run_advanced_biplot_analysis(self):
        """Ejecuta el análisis de biplot avanzado."""
        try:
            if (
                not hasattr(self.parent_app, "biplot_data")
                or not self.parent_app.biplot_data
            ):
                messagebox.showerror(
                    "Error",
                    "No hay datos cargados para el biplot avanzado. Primero configura el análisis.",
                )
                return

            if (
                not hasattr(self.parent_app, "biplot_config")
                or not self.parent_app.biplot_config
            ):
                messagebox.showwarning(
                    "Configuración incompleta",
                    "Primero debes configurar el biplot avanzado usando '🎨 Configurar Biplot Avanzado'.",
                )
                return

            self.parent_app.status.config(text="Creando biplot avanzado...")

            # Obtener configuración
            config = {
                "year": str(self.parent_app.biplot_data["year"]),
                "categorization_scheme": self.parent_app.biplot_config[
                    "categorization"
                ].get(),
                "marker_scheme": self.parent_app.biplot_config["marker_scheme"].get(),
                "color_scheme": self.parent_app.biplot_config["color_scheme"].get(),
                "show_arrows": self.parent_app.biplot_config["show_arrows"].get(),
                "show_labels": self.parent_app.biplot_config["show_labels"].get(),
                "alpha": self.parent_app.biplot_config["alpha"].get(),
            }

            # Usar los datos procesados
            df_for_biplot = self.parent_app.biplot_data.get(
                "df_year_standardized", self.parent_app.biplot_data["df_year_processed"]
            )

            # Intentar con biplot avanzado, si falla usar simplificado
            try:
                from backend.biplot_advanced import create_advanced_biplot

                if (
                    "biplot_advanced_config" in self.parent_app.project_config
                    and "custom_categories"
                    in self.parent_app.project_config["biplot_advanced_config"]
                ):
                    config["custom_categories"] = self.parent_app.project_config[
                        "biplot_advanced_config"
                    ]["custom_categories"]
                success = create_advanced_biplot(df_for_biplot, config)
            except Exception as e:
                logger.warning(f"Error con biplot_advanced: {e}")
                logger.info("Intentando con versión simplificada...")
                try:
                    from backend.biplot_simple import create_advanced_biplot_simple

                    success = create_advanced_biplot_simple(df_for_biplot, config)
                except Exception as e2:
                    logger.error(f"Error con biplot simplificado: {e2}")
                    success = False

            if success:
                self.parent_app.status.config(
                    text="✅ Biplot avanzado creado exitosamente"
                )
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
                self.parent_app.status.config(text="❌ Error al crear biplot avanzado")
                messagebox.showerror(
                    "Error", "Hubo un problema al crear el biplot avanzado."
                )

        except Exception as e:
            self.parent_app.status.config(text=f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Error al crear biplot avanzado:\n{str(e)}")
        finally:
            self.parent_app.hide_loading()
