"""
Módulo de lógica para el análisis de corte transversal (biplot 2D).
Separa la lógica de negocio de la GUI.
"""

import pandas as pd
import numpy as np
from backend import data_loader_module as dl
from backend import preprocessing_module as dl_prep
from backend import pca_module as pca_mod
from backend.constants import MAPEO_INDICADORES


class PCAAnalysisLogic:
    @staticmethod
    def run_cross_section_analysis_logic(
        cfg, year_to_analyze, imputation_strategy=None, imputation_params=None
    ):
        """
        Ejecuta el flujo de análisis de corte transversal para un año (sin GUI).
        Retorna un diccionario con todos los resultados intermedios y finales.
        """
        all_sheets_data = dl.load_excel_file(cfg["data_file"])
        selected_indicators = cfg["selected_indicators"]
        selected_units = cfg["selected_units"]
        # 1. Preparar DataFrame de corte transversal
        df_year_cross_section = dl.preparar_datos_corte_transversal(
            all_sheets_data, selected_indicators, selected_units, year_to_analyze
        )
        if df_year_cross_section.empty or df_year_cross_section.isnull().all().all():
            return {
                "warning": f"No hay datos suficientes para el año {year_to_analyze}."
            }
        # 2. Manejar datos faltantes
        if df_year_cross_section.isnull().sum().sum() > 0:
            if imputation_strategy and imputation_strategy != "ninguna":
                df_imputed_cs, _ = dl_prep.manejar_datos_faltantes(
                    df_year_cross_section,
                    estrategia=imputation_strategy,
                    devolver_mascara=True,
                    **(imputation_params or {}),
                )
                df_year_processed = df_imputed_cs.dropna(axis=0, how="any")
            else:
                df_year_processed = df_year_cross_section.dropna(axis=0, how="any")
        else:
            df_year_processed = df_year_cross_section.copy()
        if df_year_processed.shape[0] < 2 or df_year_processed.shape[1] < 2:
            return {
                "warning": f"Insuficientes países/indicadores tras limpiar NaNs para el año {year_to_analyze}."
            }
        # 3. Estandarizar
        df_year_estandarizado, scaler = dl_prep.estandarizar_datos(
            df_year_processed, devolver_scaler=True
        )
        df_cov_cs = df_year_estandarizado.cov()
        # 4. Calcular PCA (si hay datos suficientes)
        if df_year_estandarizado.shape[1] < 2:
            return {
                "warning": f"No hay suficientes indicadores para PCA en {year_to_analyze}."
            }
        # Single PCA run — fit with all components, then slice for biplot
        pca_model_full, df_pc_scores_full = pca_mod.realizar_pca(
            df_year_estandarizado, n_components=None
        )
        evr, cum_evr = pca_mod.obtener_varianza_explicada(pca_model_full)
        # For biplot we only need the first 2 components (slice from full result)
        pca_model_cs = pca_model_full
        df_pc_scores_cs = df_pc_scores_full.iloc[:, :2] if df_pc_scores_full is not None else None
        evr_cs = evr[:2] if evr is not None else None
        cum_evr_cs = cum_evr[:2] if cum_evr is not None else None
        df_varianza_explicada_cs = pd.DataFrame(
            {
                "Componente": [f"PC{i+1}" for i in range(len(evr_cs))],
                "Varianza Explicada": evr_cs,
                "Varianza Acumulada": cum_evr_cs,
            }
        ).set_index("Componente")
        # Prepara resultados
        results = {
            "df_year_cross_section": df_year_cross_section,
            "df_year_processed": df_year_processed,
            "df_year_estandarizado": df_year_estandarizado,
            "scaler": scaler,
            "df_cov_cs": df_cov_cs,
            "pca_model_cs": pca_model_cs,
            "df_pc_scores_cs": df_pc_scores_cs,
            "df_varianza_explicada_cs": df_varianza_explicada_cs,
            "evr_cs": evr_cs,
            "cum_evr_cs": cum_evr_cs,
        }
        return results
