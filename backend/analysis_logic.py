# analysis_logic.py
"""
Módulo de Lógica de Análisis PCA

Este módulo contiene toda la lógica de análisis separada de la interfaz gráfica.
Permite reutilizar la funcionalidad en diferentes interfaces (vieja y moderna).
"""

import os
import json
import traceback
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

# Importar módulos existentes (backend)
from backend import data_loader_module as dl
from backend import preprocessing_module as dl_prep
from backend import visualization_module as dl_viz
from backend import pca_module as pca_mod
from backend.constants import MAPEO_INDICADORES
from backend.pca_panel3d_logic import PCAPanel3DLogic
from backend.project_save_config import ProjectConfig

# Importar funciones de biplot y scatterplot
def _import_biplot_function():
    """Import biplot function with fallback."""
    try:
        from backend.biplot_advanced import create_advanced_biplot
        return create_advanced_biplot, True
    except (ImportError, UnicodeEncodeError):
        try:
            from backend.biplot_simple import create_advanced_biplot_simple as create_advanced_biplot
            return create_advanced_biplot, True
        except ImportError:
            return None, False

def _import_scatter_function():
    """Import scatter plot function."""
    try:
        from backend.scatter_plot import generate_scatter_plot
        return generate_scatter_plot, True
    except ImportError:
        return None, False

# Initialize functions
create_advanced_biplot, BI_PLOT_AVAILABLE = _import_biplot_function()
generate_scatter_plot, SCATTER_PLOT_AVAILABLE = _import_scatter_function()

# Importar sistemas mejorados
try:
    from backend.logging_config import get_logger, setup_application_logging
    from backend.config_manager import get_config, update_config, save_config
    from backend.performance_optimizer import profiled

    setup_application_logging(debug_mode=False)
    ENHANCED_SYSTEMS_AVAILABLE = True
except ImportError:
    import logging
    ENHANCED_SYSTEMS_AVAILABLE = False

    def get_logger(name):
        return logging.getLogger(name)


logger = get_logger("analysis_logic")


class AnalysisLogic:
    """Clase principal que contiene toda la lógica de análisis PCA."""

    def __init__(self):
        self.project_config = ProjectConfig()
        self.logger = logger

    # ===== SERIES ANALYSIS =====

    @profiled if ENHANCED_SYSTEMS_AVAILABLE else lambda func: func
    def run_series_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta el análisis de serie de tiempo para la unidad y años seleccionados.

        Args:
            config: Configuración del análisis con keys:
                - data_file: str
                - selected_sheet_names: List[str]
                - country_to_analyze: str
                - years_range: Tuple[int, int]

        Returns:
            Dict con resultados del análisis
        """
        try:
            self.logger.info("Iniciando análisis de serie de tiempo")

            # Validar configuración
            required_keys = ['data_file', 'selected_sheet_names', 'country_to_analyze']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Configuración incompleta: falta {key}")

            data_file = config['data_file']
            selected_sheet_names = config['selected_sheet_names']
            country_to_analyze = config['country_to_analyze']

            # Cargar datos
            all_sheets_data = dl.load_excel_file(data_file)
            if not all_sheets_data:
                raise ValueError("No se pudieron cargar los datos del archivo")

            # Filtrar solo hojas seleccionadas
            selected_data = {k: v for k, v in all_sheets_data.items()
                           if k in selected_sheet_names}

            if not selected_data:
                raise ValueError("No se encontraron las hojas seleccionadas")

            # Transformar datos
            data_transformada_indicadores = {}
            for sheet_name, df in selected_data.items():
                df_transformado = dl.transformar_df_indicador_v1(df)
                if df_transformado is not None:
                    data_transformada_indicadores[sheet_name] = df_transformado

            if not data_transformada_indicadores:
                raise ValueError("No se pudieron transformar los datos")

            # Consolidar datos para el país
            df_consolidado = dl.consolidate_data_for_country(
                data_transformada_indicadores,
                country_to_analyze,
                selected_sheet_names
            )

            if df_consolidado.empty:
                raise ValueError("No se pudieron consolidar los datos")

            # Aplicar años si especificados
            if 'years_range' in config:
                start_year, end_year = config['years_range']
                df_consolidado = df_consolidado[
                    (df_consolidado.index >= start_year) &
                    (df_consolidado.index <= end_year)
                ]

            # Ejecutar PCA
            df_estandarizado = dl_prep.preprocess_data(df_consolidado)
            pca_model, df_componentes = pca_mod.realizar_pca(df_estandarizado)

            # Preparar diccionario de series para la visualización
            series_dict = {
                "Consolidado": df_consolidado,
                "Estandarizado": df_estandarizado,
                "Componentes Principales": df_componentes
            }

            # Return complete results package
            paquete_resultados = {
                'status': 'success',  # ✅ CRÍTICO: Campo requerido por AnalysisManager
                "visualization_function_name": "show_series_plot",  # ✅ NUEVO: Nombre de función a llamar
                "visualization_params": {  # ✅ NUEVO: Parámetros para la visualización
                    "series_dict": series_dict,
                    "title": f"Serie de Tiempo - {country_to_analyze}"
                },
                "exportable_data": {
                    'Datos_Originales': df_consolidado,
                    'Datos_Estandarizados': df_estandarizado,
                    'Componentes_PCA': df_componentes,
                    'Varianza_Explicada': pd.DataFrame({
                        'Componente': [f'PC{i+1}' for i in range(len(pca_model.explained_variance_ratio_))],
                        'Varianza_Explicada': pca_model.explained_variance_ratio_,
                        'Varianza_Acumulada': np.cumsum(pca_model.explained_variance_ratio_)
                    }),
                    'Cargas_PCA': pd.DataFrame(
                        pca_model.components_.T,
                        index=df_estandarizado.columns,
                        columns=[f'PC{i+1}' for i in range(pca_model.components_.shape[0])]
                    ),
                    'Configuracion_Analisis': pd.DataFrame({
                        'Parametro': ['Tipo_Analisis', 'Unidad_Analizada', 'Indicadores', 'Anios_Incluidos'],
                        'Valor': ['Serie_Tiempo', country_to_analyze, ', '.join(selected_sheet_names), f'{df_consolidado.index.min()}-{df_consolidado.index.max()}']
                    })
                },
                "summary": {
                    "Tipo_Analisis": "Serie de Tiempo",
                    "Unidad_Analizada": country_to_analyze,
                    "Indicadores_Analizados": len(selected_sheet_names),
                    "Anios_Incluidos": f"{df_consolidado.index.min()}-{df_consolidado.index.max()}",
                    "Componentes_PCA": pca_model.n_components_
                },
                # Legacy compatibility: Incluir campo 'data' para código antiguo
                'data': {
                    'series_data': df_consolidado,
                    'standardized_data': df_estandarizado,
                    'pca_model': pca_model,
                    'components': df_componentes,
                    'country': country_to_analyze,
                    'indicators': selected_sheet_names,
                    'config': config
                }
            }

            self.logger.info(f"✅ Análisis de serie de tiempo completado exitosamente: {paquete_resultados['summary']}")
            return paquete_resultados

        except Exception as e:
            error_msg = f"Error en análisis de serie de tiempo: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            # ❌ TOLERANCIA CERO: Re-lanzar excepción con contexto completo
            raise RuntimeError(error_msg) from e

    # ===== CROSS SECTION ANALYSIS =====

    @profiled if ENHANCED_SYSTEMS_AVAILABLE else lambda func: func
    def run_cross_section_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta el análisis de corte transversal.

        Args:
            config: Configuración con:
                - data_file: str
                - selected_sheet_names: List[str]
                - selected_countries: List[str]
                - target_year: int/str

        Returns:
            Dict con resultados del análisis
        """
        try:
            self.logger.info("Iniciando análisis de corte transversal")

            # Validar configuración
            required_keys = ['data_file', 'selected_sheet_names', 'selected_countries', 'target_year']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Configuración incompleta: falta {key}")

            data_file = config['data_file']
            selected_sheet_names = config['selected_sheet_names']
            selected_countries = config['selected_countries']
            target_year = config['target_year']
            
            # Extract group information (may be None if not provided)
            groups = config.get('groups', {})
            group_colors = config.get('group_colors', {})
            self.logger.info(f"📊 Grupos configurados: {len(groups)} unidades en {len(set(groups.values()) if groups else [])} grupos")
            self.logger.debug(f"Group colors: {group_colors}")

            # Cargar datos
            all_sheets_data = dl.load_excel_file(data_file)
            if not all_sheets_data:
                raise ValueError("No se pudieron cargar los datos del archivo")

            # Preparar datos de corte transversal
            df_cross_section = dl.preparar_datos_corte_transversal(
                all_sheets_data,
                selected_sheet_names,
                selected_countries,
                target_year
            )

            if df_cross_section.empty:
                raise ValueError("No se pudieron preparar los datos de corte transversal")

            # ✅ FIX: Filter only numeric columns to prevent PCA validation errors
            # The cross-section data may contain non-numeric identifier columns (like company names)
            # that cause "Los datos deben ser numéricos" validation error in PCA
            numeric_columns = df_cross_section.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                raise ValueError("No se encontraron columnas numéricas en los datos de corte transversal")

            df_cross_section = df_cross_section[numeric_columns]
            self.logger.info(f"✅ Datos filtrados: {len(numeric_columns)} columnas numéricas seleccionadas para PCA")

            # ✅ NUEVO: Aplicar transformaciones opcionales antes de PCA
            apply_transformations = config.get('apply_transformations', False)
            transformation_method = config.get('transformation_method', 'auto')
            skewness_threshold = config.get('skewness_threshold', 1.0)
            arrow_scale = config.get('arrow_scale', None)  # None = auto-calculate
            
            self.logger.info(f"⚙️ Transformaciones: {apply_transformations}, Método: {transformation_method}, Threshold: {skewness_threshold}")
            self.logger.info(f"🎯 Arrow scale: {'Auto' if arrow_scale is None else arrow_scale}")
            
            # Ejecutar PCA (usar 2 componentes para biplot 2D, como en la versión original)
            df_estandarizado, transformer, transformation_info = dl_prep.preprocess_data(
                df_cross_section,
                apply_transformations=apply_transformations,
                transformation_method=transformation_method,
                skewness_threshold=skewness_threshold,
                return_transformer=True
            )
            pca_model, df_componentes = pca_mod.realizar_pca(df_estandarizado, n_components=2)

            # Ejecutar clustering K-means
            from sklearn.cluster import KMeans
            n_clusters = min(3, len(df_estandarizado) - 1) if len(df_estandarizado) > 1 else 1
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(df_estandarizado)

            # ✅ NUEVA ARQUITECTURA: No generar figuras aquí, solo preparar parámetros
            # La visualización se hará en el hilo principal con plt.show()
            
            # ✅ NUEVO: Incluir información de transformaciones para notificación al usuario
            transformation_notification = None
            if apply_transformations and transformation_method == 'auto' and transformation_info:
                if transformation_info.get('auto_methods_used'):
                    methods_summary = []
                    for col, method in transformation_info['auto_methods_used'].items():
                        methods_summary.append(f"• {col}: {method}")
                    
                    transformation_notification = {
                        'title': '🤖 Transformaciones Automáticas Aplicadas',
                        'message': f"El sistema seleccionó automáticamente los siguientes métodos de transformación:\n\n" + "\n".join(methods_summary) + f"\n\nEstas transformaciones fueron aplicadas para mejorar la distribución de los datos antes del PCA.",
                        'methods_used': transformation_info['auto_methods_used'],
                        'total_transformed': transformation_info.get('total_transformed', 0)
                    }
            
            # Return complete results package with status
            paquete_resultados = {
                "status": "success",  # ✅ CRÍTICO: Campo requerido por AnalysisManager
                "visualization_function_name": "show_biplot",  # ✅ NUEVO: Nombre de función a llamar
                "visualization_params": {  # ✅ NUEVO: Parámetros para la visualización
                    "df_componentes": df_componentes,
                    "pca_model": pca_model,
                    "df_estandarizado": df_estandarizado,
                    "title": f"Análisis de Corte Transversal - {target_year}",
                    "groups": groups,
                    "group_colors": group_colors,
                    "arrow_scale": arrow_scale  # ✅ NUEVO: Pasar arrow_scale a visualización
                },
                "transformation_notification": transformation_notification,  # ✅ NUEVO: Para mostrar al usuario
                "exportable_data": {
                    'Datos_Originales': df_cross_section,
                    'Datos_Estandarizados': df_estandarizado,
                    'Componentes_PCA': df_componentes,
                    'Varianza_Explicada': pd.DataFrame({
                        'Componente': [f'PC{i+1}' for i in range(len(pca_model.explained_variance_ratio_))],
                        'Varianza_Explicada': pca_model.explained_variance_ratio_,
                        'Varianza_Acumulada': np.cumsum(pca_model.explained_variance_ratio_)
                    }),
                    'Cargas_PCA': pd.DataFrame(
                        pca_model.components_.T,
                        index=df_estandarizado.columns,
                        columns=[f'PC{i+1}' for i in range(pca_model.components_.shape[0])]
                    ),
                    'Asignacion_Clusters': pd.DataFrame({
                        'Unidad': df_estandarizado.index,
                        'Cluster': clusters
                    }),
                    'Centroides_Clusters': pd.DataFrame(
                        kmeans.cluster_centers_,
                        columns=df_estandarizado.columns,
                        index=[f'Cluster_{i}' for i in range(len(kmeans.cluster_centers_))]
                    ),
                    'Configuracion_Analisis': pd.DataFrame({
                        'Parametro': ['Tipo_Analisis', 'Anio_Analizado', 'Unidades_Incluidas', 'Indicadores', 'Numero_Clusters'],
                        'Valor': ['Corte_Transversal', str(target_year), str(len(selected_countries)), ', '.join(selected_sheet_names), str(n_clusters)]
                    })
                },
                "summary": {
                    "Tipo_Analisis": "Corte Transversal",
                    "Anio_Analizado": target_year,
                    "Unidades_Incluidas": len(selected_countries),
                    "Indicadores_Analizados": len(selected_sheet_names),
                    "Numero_Clusters": n_clusters,
                    "Componentes_PCA": pca_model.n_components_
                },
                # Legacy compatibility fields
                "data": {
                    'cross_section_data': df_cross_section,
                    'standardized_data': df_estandarizado,
                    'pca_model': pca_model,
                    'components': df_componentes,
                    'clusters': clusters,
                    'kmeans_model': kmeans,
                    'year': target_year,
                    'countries': selected_countries,
                    'indicators': selected_sheet_names,
                    'groups': groups,
                    'group_colors': group_colors,
                    'config': config
                }
            }

            self.logger.info(f"✅ Análisis de corte transversal completado exitosamente: {paquete_resultados['summary']}")
            # Debug: Verificar estructura del paquete
            self.logger.debug(f"Paquete keys: {list(paquete_resultados.keys())}")
            self.logger.debug(f"Visualization function: {paquete_resultados.get('visualization_function_name')}")
            self.logger.debug(f"Exportable data keys: {list(paquete_resultados['exportable_data'].keys())}")
            
            return paquete_resultados

        except Exception as e:
            error_msg = f"Error en análisis de corte transversal: {str(e)}"
            self.logger.error(error_msg, exc_info=True)  # ← CRÍTICO: Full traceback
            
            # ← CRÍTICO: Re-raise para que AnalysisManager sepa que falló
            raise RuntimeError(error_msg) from e

    # ===== PANEL ANALYSIS (PCA 3D) =====

    @profiled if ENHANCED_SYSTEMS_AVAILABLE else lambda func: func
    def run_panel_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta el análisis PCA 3D (panel).

        Args:
            config: Configuración con:
                - data_file: str
                - selected_sheet_names: List[str]
                - selected_countries: List[str]
                - years_range: Tuple[int, int]

        Returns:
            Dict con resultados del análisis
        """
        try:
            self.logger.info("Iniciando análisis PCA 3D (panel)")

            # Validar configuración
            required_keys = ['data_file', 'selected_sheet_names', 'selected_countries']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Configuración incompleta: falta {key}")

            data_file = config['data_file']
            selected_sheet_names = config['selected_sheet_names']
            selected_countries = config['selected_countries']

            # Cargar datos
            all_sheets_data = dl.load_excel_file(data_file)
            if not all_sheets_data:
                raise ValueError("No se pudieron cargar los datos del archivo")

            # Preparar datos de panel
            df_panel = dl.preparar_datos_panel_longitudinal(
                all_sheets_data,
                selected_sheet_names,
                selected_countries
            )

            if df_panel.empty:
                raise ValueError("No se pudieron preparar los datos de panel")

            # Aplicar filtro de años si especificado
            if 'years_range' in config:
                start_year, end_year = config['years_range']
                df_panel = df_panel[
                    (df_panel.index.get_level_values('Año') >= start_year) &
                    (df_panel.index.get_level_values('Año') <= end_year)
                ]

            # Ejecutar PCA con 3 componentes para visualización 3D (como en la versión original)
            df_panel_estandarizado = dl_prep.preprocess_data(df_panel)
            pca_model_panel, df_pc_scores_panel = pca_mod.realizar_pca(df_panel_estandarizado, n_components=3)

            # Extraer grupos y colores del config (si están disponibles)
            groups_config = config.get('groups', {})
            group_colors = config.get('group_colors', {})

            # Convertir dict de grupos a Pandas Series que create_3d_scatter_plot espera
            groups_series = None
            if groups_config:
                # Invertir el dict: {'USA': 'Grupo 1', 'CAN': 'Grupo 1', 'MEX': 'Grupo 2'}
                inverted_groups = {country: group_name
                                   for group_name, countries in groups_config.items()
                                   for country in countries}
                groups_series = pd.Series(inverted_groups, name="Group")

            self.logger.info(f"📊 Grupos configurados para PCA 3D: {len(groups_config)} unidades en {len(set(groups_config.values())) if groups_config else 0} grupos")

            # ✅ NUEVA ARQUITECTURA: No generar figuras aquí, solo preparar parámetros
            # La visualización se hará en el hilo principal con plt.show()

            # Return complete results package
            paquete_resultados = {
                'status': 'success',  # ✅ CRÍTICO: Campo requerido por AnalysisManager
                "visualization_function_name": "show_3d_scatter_plot",  # ✅ NUEVO: Nombre de función a llamar
                "visualization_params": {  # ✅ NUEVO: Parámetros para la visualización
                    "df_pc_scores": df_pc_scores_panel,
                    "title": f"PCA 3D - Panel Longitudinal",
                    "groups": groups_series,  # ¡Pasa la Serie, no el dict!
                    "group_colors": group_colors
                },
                "exportable_data": {
                    'Datos_Panel_Originales': df_panel,
                    'Datos_Panel_Estandarizados': df_panel_estandarizado,
                    'Componentes_PCA_3D': df_pc_scores_panel,
                    'Varianza_Explicada_3D': pd.DataFrame({
                        'Componente': [f'PC{i+1}' for i in range(len(pca_model_panel.explained_variance_ratio_))],
                        'Varianza_Explicada': pca_model_panel.explained_variance_ratio_,
                        'Varianza_Acumulada': np.cumsum(pca_model_panel.explained_variance_ratio_)
                    }),
                    'Cargas_PCA_3D': pd.DataFrame(
                        pca_model_panel.components_.T,
                        index=df_panel_estandarizado.columns,
                        columns=[f'PC{i+1}' for i in range(pca_model_panel.components_.shape[0])]
                    ),
                    'Configuracion_Analisis': pd.DataFrame({
                        'Parametro': ['Tipo_Analisis', 'Unidades_Incluidas', 'Indicadores', 'Anios_Incluidos', 'Componentes_PCA'],
                        'Valor': ['Panel_3D', str(len(selected_countries)), ', '.join(selected_sheet_names), f'{df_panel.index.get_level_values("Año").min()}-{df_panel.index.get_level_values("Año").max()}', '3']
                    })
                },
                "summary": {
                    "Tipo_Analisis": "Panel 3D",
                    "Unidades_Incluidas": len(selected_countries),
                    "Indicadores_Analizados": len(selected_sheet_names),
                    "Anios_Incluidos": f"{df_panel.index.get_level_values('Año').min()}-{df_panel.index.get_level_values('Año').max()}",
                    "Componentes_PCA": 3
                },
                # Legacy compatibility: Incluir campo 'data' para código antiguo
                'data': {
                    'panel_data': df_panel,
                    'standardized_data': df_panel_estandarizado,
                    'pca_model': pca_model_panel,
                    'components': df_pc_scores_panel,
                    'countries': selected_countries,
                    'indicators': selected_sheet_names,
                    'groups': groups_config,
                    'group_colors': group_colors,
                    'config': config
                }
            }

            self.logger.info(f"✅ Análisis PCA 3D completado exitosamente: {paquete_resultados['summary']}")
            return paquete_resultados

        except Exception as e:
            error_msg = f"Error en análisis PCA 3D: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            # ❌ TOLERANCIA CERO: Re-lanzar excepción con contexto completo
            raise RuntimeError(error_msg) from e

    # ===== ADVANCED BIPLOT ANALYSIS =====

    @profiled if ENHANCED_SYSTEMS_AVAILABLE else lambda func: func
    def run_advanced_biplot_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta el análisis de biplot avanzado.

        Args:
            config: Configuración del biplot avanzado

        Returns:
            Dict con resultados del análisis
        """
        try:
            self.logger.info("Iniciando análisis de biplot avanzado")

            # Validar configuración básica
            if 'data_file' not in config:
                raise ValueError("Configuración incompleta: falta data_file")

            # Cargar datos
            data_file = config['data_file']
            all_sheets_data = dl.load_excel_file(data_file)
            if not all_sheets_data:
                raise ValueError("No se pudieron cargar los datos del archivo")

            # Preparar datos según configuración
            if config.get('analysis_type') == 'cross_section':
                # Análisis de corte transversal
                df_prepared = dl.preparar_datos_corte_transversal(
                    all_sheets_data,
                    config.get('selected_indicators', []),
                    config.get('selected_countries', []),
                    config.get('target_year')
                )
            else:
                # Análisis de serie de tiempo (usar primer país disponible)
                country = config.get('country_to_analyze', list(all_sheets_data.values())[0].columns[0])
                df_prepared = dl.consolidate_data_for_country(
                    {k: dl.transformar_df_indicador_v1(v) for k, v in all_sheets_data.items()
                     if dl.transformar_df_indicador_v1(v) is not None},
                    country,
                    config.get('selected_indicators', list(all_sheets_data.keys()))
                )

            if df_prepared.empty:
                raise ValueError("No se pudieron preparar los datos")

            # ✅ NUEVO: Aplicar transformaciones opcionales antes de PCA
            apply_transformations = config.get('apply_transformations', False)
            transformation_method = config.get('transformation_method', 'auto')
            skewness_threshold = config.get('skewness_threshold', 1.0)
            arrow_scale = config.get('arrow_scale', None)  # None = auto-calculate
            
            self.logger.info(f"⚙️ Transformaciones: {apply_transformations}, Método: {transformation_method}, Threshold: {skewness_threshold}")
            self.logger.info(f"🎯 Arrow scale: {'Auto' if arrow_scale is None else arrow_scale}")

            # Ejecutar PCA (usar 2 componentes por defecto para biplot, como en análisis de corte transversal)
            df_estandarizado, transformer, transformation_info = dl_prep.preprocess_data(
                df_prepared,
                apply_transformations=apply_transformations,
                transformation_method=transformation_method,
                skewness_threshold=skewness_threshold,
                return_transformer=True
            )
            pca_model, df_componentes = pca_mod.realizar_pca(df_estandarizado, n_components=2)

            # Generar biplot avanzado - preparar configuración
            biplot_config = config.get('biplot_config', {})
            # ✅ CRÍTICO: Asegurar que show_labels esté en True por defecto
            if 'show_labels' not in biplot_config:
                biplot_config['show_labels'] = True
            if 'show_arrows' not in biplot_config:
                biplot_config['show_arrows'] = True
            # ✅ NUEVO: Añadir arrow_scale a configuración del biplot
            if arrow_scale is not None:
                biplot_config['arrow_scale'] = arrow_scale
            
            if not BI_PLOT_AVAILABLE:
                raise ValueError("Biplot functionality not available - required modules not installed")
            
            # ✅ NUEVA ARQUITECTURA: No generar figuras aquí, solo preparar parámetros
            # La visualización se hará en el hilo principal con plt.show()
            
            # ✅ NUEVO: Incluir información de transformaciones para notificación al usuario
            transformation_notification = None
            if apply_transformations and transformation_method == 'auto' and transformation_info:
                if transformation_info.get('auto_methods_used'):
                    methods_summary = []
                    for col, method in transformation_info['auto_methods_used'].items():
                        methods_summary.append(f"• {col}: {method}")
                    
                    transformation_notification = {
                        'title': '🤖 Transformaciones Automáticas Aplicadas',
                        'message': f"El sistema seleccionó automáticamente los siguientes métodos de transformación:\n\n" + "\n".join(methods_summary) + f"\n\nEstas transformaciones fueron aplicadas para mejorar la distribución de los datos antes del PCA.",
                        'methods_used': transformation_info['auto_methods_used'],
                        'total_transformed': transformation_info.get('total_transformed', 0)
                    }

            # Return complete results package
            paquete_resultados = {
                'status': 'success',  # ✅ CRÍTICO: Campo requerido por AnalysisManager
                "visualization_function_name": "show_advanced_biplot",  # ✅ NUEVO: Usar biplot avanzado
                "visualization_params": {  # ✅ NUEVO: Parámetros para la visualización
                    "df": df_prepared,
                    "config": biplot_config
                },
                "transformation_notification": transformation_notification,  # ✅ NUEVO: Para mostrar al usuario
                "exportable_data": {
                    'Datos_Preparados': df_prepared,
                    'Datos_Estandarizados': df_estandarizado,
                    'Componentes_PCA_Biplot': df_componentes,
                    'Varianza_Explicada_Biplot': pd.DataFrame({
                        'Componente': [f'PC{i+1}' for i in range(len(pca_model.explained_variance_ratio_))],
                        'Varianza_Explicada': pca_model.explained_variance_ratio_,
                        'Varianza_Acumulada': np.cumsum(pca_model.explained_variance_ratio_)
                    }),
                    'Cargas_PCA_Biplot': pd.DataFrame(
                        pca_model.components_.T,
                        index=df_estandarizado.columns,
                        columns=[f'PC{i+1}' for i in range(pca_model.components_.shape[0])]
                    ),
                    'Configuracion_Biplot': pd.DataFrame({
                        'Parametro': ['Tipo_Analisis', 'Tipo_Biplot', 'Unidades_Incluidas', 'Indicadores'],
                        'Valor': ['Biplot_Avanzado', '2D', str(len(df_estandarizado)), ', '.join(config.get('selected_indicators', []))]
                    })
                },
                "summary": {
                    "Tipo_Analisis": "Biplot Avanzado",
                    "Tipo_Biplot": "2D",
                    "Unidades_Incluidas": len(df_estandarizado),
                    "Indicadores_Analizados": len(config.get('selected_indicators', [])),
                    "Componentes_PCA": pca_model.n_components_
                },
                # Legacy compatibility: Incluir campo 'data' para código antiguo
                'data': {
                    'prepared_data': df_prepared,
                    'standardized_data': df_estandarizado,
                    'pca_model': pca_model,
                    'components': df_componentes,
                    'biplot_config': biplot_config,
                    'config': config
                }
            }

            self.logger.info(f"✅ Análisis de biplot avanzado completado exitosamente: {paquete_resultados['summary']}")
            return paquete_resultados

        except Exception as e:
            error_msg = f"Error en análisis de biplot avanzado: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            # ❌ TOLERANCIA CERO: Re-lanzar excepción con contexto completo
            raise RuntimeError(error_msg) from e

    # ===== SCATTERPLOT PCA ANALYSIS =====

    @profiled if ENHANCED_SYSTEMS_AVAILABLE else lambda func: func
    def run_scatter_plot_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta el análisis de scatterplot PCA.

        Args:
            config: Configuración del scatterplot

        Returns:
            Dict con resultados del análisis
        """
        try:
            self.logger.info("Iniciando análisis de scatterplot PCA")

            # Validar configuración
            if 'data_file' not in config:
                raise ValueError("Configuración incompleta: falta data_file")

            # Cargar datos
            data_file = config['data_file']
            all_sheets_data = dl.load_excel_file(data_file)
            if not all_sheets_data:
                raise ValueError("No se pudieron cargar los datos del archivo")

            # Preparar datos
            if config.get('analysis_type') == 'cross_section':
                df_prepared = dl.preparar_datos_corte_transversal(
                    all_sheets_data,
                    config.get('selected_indicators', []),
                    config.get('selected_countries', []),
                    config.get('target_year')
                )
            else:
                country = config.get('country_to_analyze', list(all_sheets_data.values())[0].columns[0])
                df_prepared = dl.consolidate_data_for_country(
                    {k: dl.transformar_df_indicador_v1(v) for k, v in all_sheets_data.items()
                     if dl.transformar_df_indicador_v1(v) is not None},
                    country,
                    config.get('selected_indicators', list(all_sheets_data.keys()))
                )

            if df_prepared.empty:
                raise ValueError("No se pudieron preparar los datos")

            # Ejecutar PCA (usar 2 componentes por defecto para scatter plot)
            df_estandarizado = dl_prep.preprocess_data(df_prepared)
            pca_model, df_componentes = pca_mod.realizar_pca(df_estandarizado, n_components=2)

            # Generar scatterplot - preparar configuración
            scatter_config = config.get('scatter_config', {})
            if not SCATTER_PLOT_AVAILABLE:
                raise ValueError("Scatter plot functionality not available - required modules not installed")
            
            # ✅ NUEVA ARQUITECTURA: No generar figuras aquí, solo preparar parámetros
            # La visualización se hará en el hilo principal con plt.show()

            # Return complete results package
            paquete_resultados = {
                'status': 'success',  # ✅ CRÍTICO: Campo requerido por AnalysisManager
                "visualization_function_name": "show_scatter_plot",  # ✅ NUEVO: Usar scatterplot
                "visualization_params": {  # ✅ NUEVO: Parámetros para la visualización
                    "df_standardized": df_estandarizado,
                    "labels": df_estandarizado.index.tolist(),
                    "config": scatter_config,
                    "existing_model": pca_model
                },
                "exportable_data": {
                    'Datos_Preparados_Scatter': df_prepared,
                    'Datos_Estandarizados_Scatter': df_estandarizado,
                    'Componentes_PCA_Scatter': df_componentes,
                    'Varianza_Explicada_Scatter': pd.DataFrame({
                        'Componente': [f'PC{i+1}' for i in range(len(pca_model.explained_variance_ratio_))],
                        'Varianza_Explicada': pca_model.explained_variance_ratio_,
                        'Varianza_Acumulada': np.cumsum(pca_model.explained_variance_ratio_)
                    }),
                    'Cargas_PCA_Scatter': pd.DataFrame(
                        pca_model.components_.T,
                        index=df_estandarizado.columns,
                        columns=[f'PC{i+1}' for i in range(pca_model.components_.shape[0])]
                    ),
                    'Configuracion_Scatter': pd.DataFrame({
                        'Parametro': ['Tipo_Analisis', 'Tipo_Visualizacion', 'Unidades_Incluidas', 'Indicadores'],
                        'Valor': ['Scatterplot_PCA', '2D', str(len(df_estandarizado)), ', '.join(config.get('selected_indicators', []))]
                    })
                },
                "summary": {
                    "Tipo_Analisis": "Scatterplot PCA",
                    "Tipo_Visualizacion": "2D",
                    "Unidades_Incluidas": len(df_estandarizado),
                    "Indicadores_Analizados": len(config.get('selected_indicators', [])),
                    "Componentes_PCA": pca_model.n_components_
                },
                # Legacy compatibility: Incluir campo 'data' para código antiguo
                'data': {
                    'prepared_data': df_prepared,
                    'standardized_data': df_estandarizado,
                    'pca_model': pca_model,
                    'components': df_componentes,
                    'scatter_config': config.get('scatter_config', {}),
                    'config': config
                }
            }

            self.logger.info(f"✅ Análisis de scatterplot PCA completado exitosamente: {paquete_resultados['summary']}")
            return paquete_resultados

        except Exception as e:
            error_msg = f"Error en análisis de scatterplot PCA: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            # ❌ TOLERANCIA CERO: Re-lanzar excepción con contexto completo
            raise RuntimeError(error_msg) from e

    # ===== DATA CONSOLIDATION ANALYSIS =====

    @profiled if ENHANCED_SYSTEMS_AVAILABLE else lambda func: func
    def consolidate_company_data(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        Consolida datos financieros por empresa calculando estadísticas descriptivas
        y estandarizando los resultados para análisis posteriores.

        Esta función transforma un DataFrame con múltiples registros por empresa
        (uno por año) en un único perfil estadístico por empresa, calculando
        la media y desviación estándar de cada indicador financiero.

        Args:
            input_file: Ruta al archivo de entrada (CSV o Excel)
            output_file: Ruta al archivo de salida (CSV)

        Returns:
            Dict con resultados de la consolidación
        """
        try:
            self.logger.info(f"Iniciando consolidación de datos desde {input_file}")

            # Validar archivos
            if not os.path.exists(input_file):
                raise ValueError(f"Archivo de entrada no existe: {input_file}")

            # Cargar datos
            if input_file.lower().endswith('.csv'):
                # Intentar múltiples encodings para compatibilidad
                encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None
                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(input_file, encoding=encoding)
                        logger.info(f"Archivo CSV cargado exitosamente con encoding: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    raise ValueError("No se pudo leer el archivo CSV. Intente guardar como UTF-8 o use Excel.")
            elif input_file.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(input_file)
            else:
                raise ValueError("Formato de archivo no soportado. Use CSV o Excel.")

            if df.empty:
                raise ValueError("El archivo de entrada está vacío")

            # Validar estructura esperada
            required_cols = ['Empresa', 'Año']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Columnas requeridas faltantes: {missing_cols}")

            # Identificar columnas numéricas (indicadores financieros)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Excluir la columna 'Año' si es numérica
            if 'Año' in numeric_cols:
                numeric_cols.remove('Año')

            if not numeric_cols:
                raise ValueError("No se encontraron columnas numéricas (indicadores financieros)")

            self.logger.info(f"Procesando {len(df)} registros de {df['Empresa'].nunique()} empresas")
            self.logger.info(f"Indicadores identificados: {numeric_cols}")

            # Calcular estadísticas por empresa
            consolidated_data = []

            for company in df['Empresa'].unique():
                company_data = df[df['Empresa'] == company]
                company_stats = {'Empresa': company}

                # Calcular estadísticas para cada indicador
                for indicator in numeric_cols:
                    values = company_data[indicator].dropna()

                    if len(values) > 0:
                        # Media (promedio)
                        company_stats[f"{indicator}_Promedio"] = values.mean()

                        # Desviación estándar (volatilidad)
                        company_stats[f"{indicator}_DesvEst"] = values.std()
                    else:
                        # Si no hay datos, usar NaN
                        company_stats[f"{indicator}_Promedio"] = np.nan
                        company_stats[f"{indicator}_DesvEst"] = np.nan

                consolidated_data.append(company_stats)

            # Crear DataFrame consolidado
            df_consolidated = pd.DataFrame(consolidated_data)
            df_consolidated = df_consolidated.set_index('Empresa')

            # Eliminar filas con todos los valores NaN
            df_consolidated = df_consolidated.dropna(how='all')

            if df_consolidated.empty:
                raise ValueError("No se pudieron generar estadísticas válidas")

            self.logger.info(f"Datos consolidados: {df_consolidated.shape[0]} empresas, {df_consolidated.shape[1]} métricas")

            # Estandarizar los datos (fundamental para PCA y clustering)
            scaler = StandardScaler()
            df_standardized = pd.DataFrame(
                scaler.fit_transform(df_consolidated),
                index=df_consolidated.index,
                columns=df_consolidated.columns
            )

            # Agregar filas con la media y desviación estándar global usada para estandarización
            df_stats_with_globals = df_consolidated.copy()

            # Agregar fila con la media global
            global_mean = pd.Series(scaler.mean_, index=df_consolidated.columns, name='Media_Global')
            df_stats_with_globals = pd.concat([df_stats_with_globals, global_mean.to_frame().T])

            # Agregar fila con la desviación estándar global
            global_std = pd.Series(scaler.scale_, index=df_consolidated.columns, name='DesvEst_Global')
            df_stats_with_globals = pd.concat([df_stats_with_globals, global_std.to_frame().T])

            # Exportar según el formato solicitado
            if output_file.lower().endswith('.csv'):
                df_standardized.to_csv(output_file, index=True, encoding='utf-8')
                self.logger.info(f"Datos consolidados exportados a CSV: {output_file}")
            elif output_file.lower().endswith(('.xlsx', '.xls')):
                # Exportar a Excel con múltiples hojas para mostrar los pasos
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    # Hoja 1: Datos originales
                    df.to_excel(writer, sheet_name='Datos Originales', index=False)

                    # Hoja 2: Estadísticas por empresa (antes de estandarización) + globales
                    df_stats_with_globals.to_excel(writer, sheet_name='Estadísticas por Empresa', index=True)

                    # Hoja 3: Datos consolidados estandarizados
                    df_standardized.to_excel(writer, sheet_name='Datos Estandarizados', index=True)

                    # Hoja 4: Resumen del procesamiento
                    summary_data = {
                        'Métrica': [
                            'Archivo de entrada',
                            'Registros procesados',
                            'Empresas identificadas',
                            'Indicadores procesados',
                            'Métricas generadas',
                            'Archivo de salida',
                            'Estado del procesamiento'
                        ],
                        'Valor': [
                            os.path.basename(input_file),
                            len(df),
                            df['Empresa'].nunique(),
                            len(numeric_cols),
                            len(numeric_cols) * 2,  # promedio + desv est por indicador
                            os.path.basename(output_file),
                            'Completado exitosamente'
                        ]
                    }
                    df_summary = pd.DataFrame(summary_data)
                    df_summary.to_excel(writer, sheet_name='Procesamiento Completado', index=False)

                self.logger.info(f"Datos consolidados exportados a Excel con múltiples hojas: {output_file}")
            else:
                # Default to CSV
                df_standardized.to_csv(output_file, index=True, encoding='utf-8')
                self.logger.info(f"Datos consolidados exportados a CSV (default): {output_file}")

            # Return complete results package (no visualization for consolidation)
            paquete_resultados = {
                'status': 'success',  # ✅ CRÍTICO: Campo requerido por AnalysisManager
                # No hay visualización para consolidación de datos
                "exportable_data": {
                    'Datos_Originales_Consolidados': df_consolidated,
                    'Datos_Estandarizados_Consolidados': df_standardized,
                    'Parametros_Estandarizacion': pd.DataFrame({
                        'Indicador': df_consolidated.columns,
                        'Media_Global': scaler.mean_,
                        'Desviacion_Estandar_Global': scaler.scale_
                    }),
                    'Resumen_Consolidacion': pd.DataFrame({
                        'Parametro': ['Empresas_Procesadas', 'Indicadores_Procesados', 'Archivo_Salida'],
                        'Valor': [str(len(df_consolidated)), str(len(numeric_cols)), output_file]
                    })
                },
                "summary": {
                    "Tipo_Analisis": "Consolidación de Datos",
                    "Empresas_Procesadas": len(df_consolidated),
                    "Indicadores_Procesados": len(numeric_cols),
                    "Archivo_Salida": output_file
                },
                # Legacy compatibility: Incluir campo 'data' para código antiguo
                'data': {
                    'consolidated_data': df_consolidated,
                    'standardized_data': df_standardized,
                    'scaler': scaler,
                    'numeric_cols': numeric_cols,
                    'original_df': df
                }
            }

            self.logger.info(f"✅ Consolidación completada exitosamente: {paquete_resultados['summary']}")
            return paquete_resultados

        except Exception as e:
            error_msg = f"Error en consolidación de datos: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            # ❌ TOLERANCIA CERO: Re-lanzar excepción con contexto completo
            raise RuntimeError(error_msg) from e

    # ===== HIERARCHICAL CLUSTERING ANALYSIS =====

    @profiled if ENHANCED_SYSTEMS_AVAILABLE else lambda func: func
    def perform_hierarchical_clustering(self, dataframe: pd.DataFrame, method: str = 'ward', metric: str = 'euclidean',
                                       groups: Dict[str, str] = None, group_colors: Dict[str, str] = None,
                                       selected_variables: List[str] = None, selected_units: List[str] = None) -> Dict[str, Any]:
        """
        Realiza clustering jerárquico y genera un dendrograma con colores por grupos.

        Args:
            dataframe: DataFrame con componentes PCA (filas: observaciones, columnas: componentes)
            method: Método de enlace ('ward', 'complete', 'average', 'single')
            metric: Métrica de distancia ('euclidean', 'cityblock', 'cosine')
            groups: Diccionario opcional de asignación unidad -> grupo
            group_colors: Diccionario opcional de colores por grupo

        Returns:
            Figura de Matplotlib con el dendrograma
        """
        try:
            self.logger.info("Realizando clustering jerárquico")

            # Filtrar unidades seleccionadas si se especificaron
            if selected_units:
                # Verificar que las unidades seleccionadas existan en el dataframe
                available_units = [unit for unit in selected_units if unit in dataframe.index]
                if not available_units:
                    raise ValueError("Ninguna de las unidades seleccionadas está disponible en los datos")

                # Filtrar el dataframe por filas (unidades)
                dataframe = dataframe.loc[available_units]
                self.logger.info(f"Usando {len(available_units)} unidades seleccionadas para clustering")

            # Filtrar variables seleccionadas si se especificaron
            if selected_variables:
                # Verificar que las variables seleccionadas existan en el dataframe
                available_vars = [var for var in selected_variables if var in dataframe.columns]
                if not available_vars:
                    raise ValueError("Ninguna de las variables seleccionadas está disponible en los datos")

                # Filtrar el dataframe
                dataframe = dataframe[available_vars]
                self.logger.info(f"Usando {len(available_vars)} variables seleccionadas para clustering")

            # Calcular matriz de enlace
            # Convertir DataFrame a array numpy ya que linkage() requiere datos numéricos puros
            linkage_matrix = linkage(dataframe.values, method=method, metric=metric)

            # ✅ NUEVA ARQUITECTURA: No generar figuras aquí, solo preparar parámetros
            # La visualización se hará en el hilo principal con plt.show()
            
            # Calcular dendrograma para obtener información de ordenamiento (sin visualizar)
            dendro = dendrogram(linkage_matrix, labels=dataframe.index.tolist(), no_plot=True)

            # Return complete results package
            paquete_resultados = {
                'status': 'success',  # ✅ CRÍTICO: Campo requerido por AnalysisManager
                "visualization_function_name": "show_hierarchical_dendrogram",  # ✅ NUEVO: Nombre de función a llamar
                "visualization_params": {  # ✅ NUEVO: Parámetros para la visualización
                    "dataframe": dataframe,
                    "method": method,
                    "metric": metric,
                    "title": f'Dendrograma de Clustering Jerárquico\nMétodo: {method}, Métrica: {metric}',
                    "groups": groups,
                    "group_colors": group_colors
                },
                "exportable_data": {
                    'Datos_Clustering': dataframe,
                    'Configuracion_Clustering': pd.DataFrame({
                        'Parametro': ['Metodo_Enlace', 'Metrica_Distancia', 'Unidades_Seleccionadas', 'Variables_Seleccionadas'],
                        'Valor': [method, metric, str(len(dataframe)), str(len(dataframe.columns))]
                    })
                },
                "summary": {
                    "Tipo_Analisis": "Clustering Jerárquico",
                    "Metodo_Enlace": method,
                    "Metrica_Distancia": metric,
                    "Unidades_Seleccionadas": len(dataframe),
                    "Variables_Seleccionadas": len(dataframe.columns)
                },
                # Legacy compatibility: Incluir campo 'data' para código antiguo
                'data': {
                    'clustering_data': dataframe,
                    'linkage_matrix': linkage_matrix,
                    'dendrogram': dendro,
                    'method': method,
                    'metric': metric,
                    'groups': groups,
                    'group_colors': group_colors
                }
            }

            self.logger.info(f"✅ Clustering jerárquico completado exitosamente: {paquete_resultados['summary']}")
            return paquete_resultados

        except Exception as e:
            error_msg = f"Error en clustering jerárquico: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            # ❌ TOLERANCIA CERO: Re-lanzar excepción con contexto completo
            raise RuntimeError(error_msg) from e

    # ===== CORRELATION ANALYSIS =====

    @profiled if ENHANCED_SYSTEMS_AVAILABLE else lambda func: func
    def run_correlation_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta el análisis de correlación entre unidades de investigación.

        Args:
            config: Configuración del análisis con:
                - data_file: str (ruta al archivo CSV/Excel)
                - correlation_method: str ('pearson', 'spearman', 'kendall', 'dtw')
                - time_aggregated: bool (True para datos agregados, False para series temporales)
                - similarity_threshold: float (umbral para filtrar conexiones débiles)

        Returns:
            Dict con resultados del análisis
        """
        try:
            self.logger.info("Iniciando análisis de correlación")

            # Validar configuración
            required_keys = ['data_file', 'correlation_method']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Configuración incompleta: falta {key}")

            data_file = config['data_file']
            correlation_method = config.get('correlation_method', 'pearson')
            time_aggregated = config.get('time_aggregated', True)
            similarity_threshold = config.get('similarity_threshold', 0.3)

            # Cargar datos usando el nuevo formato multi-sheet
            df = dl.load_correlation_data_multisheet(data_file)
            if df is None or df.empty:
                raise ValueError("No se pudieron cargar los datos del archivo en formato multi-sheet")

            # Filtrar indicadores seleccionados
            selected_indicators = config.get('selected_indicators', [])
            if selected_indicators:
                # Keep only Unit, Year, and selected indicators
                columns_to_keep = ['Unit', 'Year'] + selected_indicators
                df = df[columns_to_keep]
            # If no indicators selected, keep all (will be filtered later if needed)

            # Filtrar unidades seleccionadas si se especificaron
            selected_units = config.get('selected_units', [])
            if selected_units:
                df = df[df['Unit'].isin(selected_units)]
                if df.empty:
                    raise ValueError("Ninguna de las unidades seleccionadas está disponible en los datos")

            if df.empty:
                raise ValueError("No se pudieron preparar los datos para análisis de correlación")

            # Calcular matriz de similitud
            analyzer = self._get_correlation_analyzer()
            similarity_matrix = analyzer.calculate_similarity_matrix(
                df,
                method=correlation_method,
                time_aggregated=time_aggregated
            )

            # Filtrar matriz si se especifica umbral
            if similarity_threshold > 0:
                similarity_matrix = analyzer.filter_similarity_matrix(
                    similarity_matrix,
                    threshold=similarity_threshold
                )

            # Aplicar filtrado de outliers a la matriz de similitud si está habilitado para heatmap
            heatmap_filtering_report = None
            if config.get('visualization_type') in ['heatmap', 'both']:
                heatmap_config = config.get('heatmap_config', {})
                if heatmap_config.get('filter_outliers', True):
                    try:
                        # Crear una red temporal para aplicar el filtrado
                        from network_visualization import NetworkVisualizer
                        temp_visualizer = NetworkVisualizer()
                        temp_result = temp_visualizer.create_network_from_correlation(
                            similarity_matrix,
                            {
                                'threshold': similarity_threshold,
                                'filter_outliers': True,
                                'min_degree': heatmap_config.get('min_degree', 1),
                                'min_weighted_degree': heatmap_config.get('min_weighted_degree', 0.5),
                                'max_isolated_ratio': 0.3  # Usar valor por defecto
                            }
                        )

                        if isinstance(temp_result, dict) and 'filtering_report' in temp_result:
                            heatmap_filtering_report = temp_result['filtering_report']
                            # Filtrar la matriz de similitud para mantener solo las unidades que pasaron el filtro
                            kept_units = temp_result['filtering_report']['kept_units']
                            similarity_matrix = similarity_matrix.loc[kept_units, kept_units]
                            self.logger.info(f"Applied heatmap outlier filtering: removed {len(temp_result['filtering_report']['removed_units'])} units")
                    except Exception as e:
                        self.logger.warning(f"Could not apply heatmap outlier filtering: {e}")

            # Crear red y aplicar filtrado de outliers si está habilitado
            network_result = None
            network_filtering_report = None
            if config.get('visualization_type') in ['network', 'both']:
                try:
                    from network_visualization import create_correlation_network
                    network_result = create_correlation_network(
                        similarity_matrix,  # Usar la matriz ya filtrada si se aplicó filtrado de heatmap
                        config.get('network_config', {})
                    )

                    # Si se aplicó filtrado, obtener el reporte
                    if hasattr(network_result, '__getitem__') and 'filtering_report' in network_result:
                        network_filtering_report = network_result['filtering_report']
                        network_result = network_result['graph']
                except Exception as e:
                    self.logger.warning(f"Could not create network for outlier filtering: {e}")

            # Combinar reportes de filtrado
            filtering_report = None
            if heatmap_filtering_report or network_filtering_report:
                filtering_report = heatmap_filtering_report or network_filtering_report
                # Si hay ambos, el reporte del heatmap tiene prioridad ya que se aplicó primero

            # Calcular estadísticas
            stats = analyzer.get_similarity_statistics(similarity_matrix)

            # ✅ NUEVA ARQUITECTURA: No generar figuras aquí, solo preparar parámetros
            # La visualización se hará en el hilo principal con plt.show()
            
            # Determinar qué función de visualización usar
            viz_type = config.get('visualization_type', 'heatmap')
            if viz_type == 'network':
                viz_function_name = "show_correlation_network"
            else:  # 'heatmap' o cualquier otro valor por defecto
                viz_function_name = "show_correlation_heatmap"
            
            # Return complete results package
            paquete_resultados = {
                'status': 'success',  # ✅ CRÍTICO: Campo requerido por AnalysisManager
                "visualization_function_name": viz_function_name,  # ✅ NUEVO: Nombre de función a llamar
                "visualization_params": {  # ✅ NUEVO: Parámetros para la visualización
                    "similarity_matrix": similarity_matrix,
                    "title": f"{'Red de' if viz_type == 'network' else 'Mapa de Calor de'} Correlación - {correlation_method.upper()}"
                },
                "exportable_data": {
                    'Datos_Originales_Correlacion': df,
                    'Matriz_Correlacion': similarity_matrix,
                    'Estadisticas_Correlacion': pd.DataFrame([stats]) if isinstance(stats, dict) else pd.DataFrame(stats),
                    'Configuracion_Correlacion': pd.DataFrame({
                        'Parametro': ['Metodo_Correlacion', 'Datos_Agregados_Tiempo', 'Umbral_Similitud', 'Tipo_Visualizacion', 'Unidades_Analizadas', 'Indicadores_Seleccionados'],
                        'Valor': [correlation_method, 'Sí' if time_aggregated else 'No', str(similarity_threshold), config.get('visualization_type', 'heatmap'), str(len(similarity_matrix)), ', '.join(selected_indicators)]
                    })
                },
                "summary": {
                    "Tipo_Analisis": "Correlación/Red",
                    "Metodo_Correlacion": correlation_method,
                    "Unidades_Analizadas": len(similarity_matrix),
                    "Indicadores_Seleccionados": len(selected_indicators),
                    "Tipo_Visualizacion": config.get('visualization_type', 'heatmap'),
                    "Umbral_Similitud": similarity_threshold
                }
            }

            # Add network-specific data if available
            if network_result is not None:
                try:
                    import networkx as nx
                    if hasattr(network_result, 'edges') and hasattr(network_result, 'nodes'):
                        # Network edges
                        edges_data = []
                        for u, v, data in network_result.edges(data=True):
                            edges_data.append({
                                'Nodo_Origen': u,
                                'Nodo_Destino': v,
                                'Peso': data.get('weight', 0),
                                'Peso_Absoluto': abs(data.get('weight', 0))
                            })
                        if edges_data:
                            paquete_resultados['exportable_data']['Red_Conexiones'] = pd.DataFrame(edges_data)

                        # Network nodes
                        nodes_data = []
                        for node, data in network_result.nodes(data=True):
                            node_info = {
                                'Nodo': node,
                                'Grado': network_result.degree(node),
                                'Grado_Ponderado': sum(d.get('abs_weight', 0) for u, v, d in network_result.edges(node, data=True))
                            }
                            for key, value in data.items():
                                node_info[f'Atributo_{key}'] = value
                            nodes_data.append(node_info)
                        if nodes_data:
                            paquete_resultados['exportable_data']['Red_Nodos'] = pd.DataFrame(nodes_data)

                        # Network statistics
                        network_stats = {
                            'Total_Nodos': len(list(network_result.nodes())),
                            'Total_Conexiones': len(list(network_result.edges())),
                            'Densidad_Red': nx.density(network_result),
                            'Red_Conectada': nx.is_connected(network_result),
                            'Componentes_Conectados': nx.number_connected_components(network_result)
                        }
                        paquete_resultados['exportable_data']['Estadisticas_Red'] = pd.DataFrame([network_stats])

                except ImportError:
                    self.logger.warning("NetworkX not available for network data export")

            # Add filtering report if available
            if filtering_report:
                filtering_df = pd.DataFrame({
                    'Unidad': filtering_report.get('removed_units', []),
                    'Razon': ['Baja conectividad'] * len(filtering_report.get('removed_units', []))
                })
                paquete_resultados['exportable_data']['Unidades_Filtradas'] = filtering_df

            # Legacy compatibility: Incluir campo 'data' para código antiguo
            paquete_resultados['data'] = {
                'correlation_data': df,
                'similarity_matrix': similarity_matrix,
                'network_result': network_result,
                'statistics': stats,
                'filtering_report': filtering_report,
                'config': config
            }

            self.logger.info(f"✅ Análisis de correlación completado exitosamente: {paquete_resultados['summary']}")
            return paquete_resultados

        except Exception as e:
            error_msg = f"Error en análisis de correlación: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            # ❌ TOLERANCIA CERO: Re-lanzar excepción con contexto completo
            raise RuntimeError(error_msg) from e

    def _get_correlation_analyzer(self):
        """Lazy initialization of correlation analyzer."""
        if not hasattr(self, '_correlation_analyzer'):
            from correlation_analysis import CorrelationAnalyzer
            self._correlation_analyzer = CorrelationAnalyzer()
        return self._correlation_analyzer


# Instancia global para uso en la interfaz
analysis_logic = AnalysisLogic()


# Funciones de conveniencia para la interfaz
def run_series_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """Función de conveniencia para análisis de serie de tiempo."""
    return analysis_logic.run_series_analysis(config)


def run_cross_section_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """Función de conveniencia para análisis de corte transversal."""
    return analysis_logic.run_cross_section_analysis(config)


def run_panel_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """Función de conveniencia para análisis PCA 3D."""
    return analysis_logic.run_panel_analysis(config)


def run_advanced_biplot_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """Función de conveniencia para biplot avanzado."""
    return analysis_logic.run_advanced_biplot_analysis(config)


def run_scatter_plot_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """Función de conveniencia para scatterplot PCA."""
    return analysis_logic.run_scatter_plot_analysis(config)


def perform_hierarchical_clustering(dataframe: pd.DataFrame, method: str = 'ward', metric: str = 'euclidean',
                                   groups: Dict[str, str] = None, group_colors: Dict[str, str] = None,
                                   selected_variables: List[str] = None, selected_units: List[str] = None) -> Dict[str, Any]:
    """Función de conveniencia para clustering jerárquico."""
    return analysis_logic.perform_hierarchical_clustering(dataframe, method, metric, groups, group_colors, selected_variables, selected_units)


def consolidate_company_data(input_file: str, output_file: str) -> Dict[str, Any]:
    """Función de conveniencia para consolidación de datos por empresa."""
    return analysis_logic.consolidate_company_data(input_file, output_file)


# ===== CORRELATION ANALYSIS =====

def run_correlation_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """Función de conveniencia para análisis de correlación."""
    return analysis_logic.run_correlation_analysis(config)