"""
Módulo de lógica para el análisis PCA y procesamiento de datos.
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
    def run_series_analysis_logic(
        cfg, imputation_strategy=None, imputation_params=None, selected_years=None
    ):
        """
        Ejecuta el flujo de análisis de serie de tiempo (sin GUI).
        Retorna un diccionario con todos los resultados intermedios y finales.
        """
        all_sheets_data = dl.load_excel_file(cfg["data_file"])
        selected_indicators = cfg["selected_indicators"]
        selected_unit = cfg["selected_units"][0]

        # 2. Transformar cada hoja
        data_transformada = {}
        for ind in selected_indicators:
            df = all_sheets_data[ind]
            df_trans = dl.transformar_df_indicador_v1(df)
            if df_trans is not None and not df_trans.empty:
                data_transformada[ind] = df_trans

        # 3. Consolidar los datos para el país elegido
        df_consolidado = dl.consolidate_data_for_country(
            data_transformada, selected_unit, selected_indicators
        )
        # Filtrar por años si se proporciona una lista
        if selected_years is not None and len(selected_years) > 0:
            # Convertir a int si es necesario
            selected_years_int = [int(y) for y in selected_years]

            # Verificar estructura de datos para filtrado
            filtered_successfully = False

            if "Año" in df_consolidado.columns:
                df_consolidado = df_consolidado[
                    df_consolidado["Año"].isin(selected_years_int)
                ]
                filtered_successfully = True
            elif df_consolidado.index.name == "Año" or "Año" in str(
                df_consolidado.index.names
            ):
                df_consolidado = df_consolidado.loc[
                    df_consolidado.index.isin(selected_years_int)
                ]
                filtered_successfully = True
            else:
                # Intentar convertir el índice a entero y filtrar por posición
                try:
                    if df_consolidado.index.dtype in ["int64", "int32"] or all(
                        isinstance(x, (int, np.integer)) for x in df_consolidado.index
                    ):
                        df_consolidado = df_consolidado.loc[
                            df_consolidado.index.isin(selected_years_int)
                        ]
                        filtered_successfully = True
                    else:
                            pass  # Could not identify year column; use all available data
                except Exception:
                    pass  # Year filtering failed; use all available data

            if filtered_successfully and df_consolidado.empty:
                return {
                    "error": f"No hay datos para los años seleccionados: {selected_years}. Verifica que los años existan en los datos."
                }
            elif not filtered_successfully:
                pass  # Year filtering not applied; all data will be used
        if df_consolidado is None or df_consolidado.empty:
            return {
                "error": "No se pudieron consolidar los datos para el país seleccionado. Verifica que el país exista en los datos de los indicadores seleccionados."
            }

        # Verificar que hay suficientes datos
        if df_consolidado.shape[0] < 3:
            return {
                "warning": f"Datos insuficientes para análisis robusto. Solo hay {df_consolidado.shape[0]} observaciones temporales. Se recomienda tener al menos 3 años de datos."
            }

        ncols = df_consolidado.shape[1]
        if ncols == 1:
            return {
                "warning": "Solo seleccionaste un indicador. El PCA no es informativo con un solo indicador. Considera agregar más indicadores para un análisis multivariado."
            }

        # Verificar datos faltantes antes de la imputación
        datos_faltantes = df_consolidado.isnull().sum().sum()
        if datos_faltantes > 0 and (
            imputation_strategy is None or imputation_strategy == "ninguna"
        ):
            return {
                "warning": f"Se encontraron {datos_faltantes} datos faltantes en el dataset. Considera aplicar una estrategia de imputación para evitar errores en el análisis PCA."
            }

        # Imputación de datos faltantes
        df_imputado = df_consolidado.copy()
        mascara_imputados = pd.DataFrame(
            False, index=df_imputado.index, columns=df_imputado.columns
        )
        if imputation_strategy and imputation_strategy != "ninguna":
            df_imputado, mascara_imputados = dl_prep.manejar_datos_faltantes(
                df_consolidado,
                estrategia=imputation_strategy,
                devolver_mascara=True,
                **(imputation_params or {}),
            )

        df_para_scaler = (
            df_imputado.dropna(axis=0, how="any")
            if df_imputado.isnull().sum().sum() > 0
            else df_imputado
        )
        if df_para_scaler.empty:
            return {
                "error": "No se puede estandarizar. DataFrame vacío tras imputación."
            }

        df_estandarizado, scaler = dl_prep.estandarizar_datos(
            df_para_scaler, devolver_scaler=True
        )
        df_covarianza = df_estandarizado.cov()

        # --- PCA sólo si hay más de una columna y suficientes observaciones ---
        df_varianza_explicada = None
        df_componentes = None
        pca_model_final = None
        evr = None
        cum_evr = None
        n_sugg_90 = None
        n_sugg_95 = None

        # Verificar requisitos mínimos para PCA
        min_components = min(df_estandarizado.shape[0], df_estandarizado.shape[1])
        if (
            df_estandarizado.shape[1] > 1
            and df_estandarizado.shape[0] >= 3
            and min_components >= 2
        ):
            try:
                # Single PCA run with all components; slice later for display
                pca_model_full, df_pc_all = pca_mod.realizar_pca(
                    df_estandarizado, n_components=None
                )

                if pca_model_full is not None:
                    evr, cum_evr = pca_mod.obtener_varianza_explicada(pca_model_full)

                    if cum_evr is not None and evr is not None:
                        sugg_90 = np.where(cum_evr >= 0.90)[0]
                        sugg_95 = np.where(cum_evr >= 0.95)[0]
                        n_sugg_90 = sugg_90[0] + 1 if len(sugg_90) > 0 else None
                        n_sugg_95 = sugg_95[0] + 1 if len(sugg_95) > 0 else None

                        df_varianza_explicada = pd.DataFrame(
                            {
                                "Componente": [f"PC{i+1}" for i in range(len(evr))],
                                "Varianza Explicada": evr,
                                "Varianza Acumulada": cum_evr,
                            }
                        ).set_index("Componente")

                        # Slice the first N components for time-series visualisation
                        n_components_to_show = min(
                            5, len(evr), df_estandarizado.shape[1]
                        )

                        pca_model_final = pca_model_full
                        df_componentes = (
                            df_pc_all.iloc[:, :n_components_to_show]
                            if df_pc_all is not None
                            else None
                        )

                        if df_componentes is not None:
                            df_componentes.index = df_estandarizado.index
                            df_componentes.index.name = "Año"
                    else:
                        df_componentes = None
                        pca_model_final = None
                else:
                    df_componentes = None
                    pca_model_final = None

            except Exception as e:
                import logging
                logging.getLogger("pca_logic").error(
                    f"PCA analysis error: {e}", exc_info=True
                )
                df_componentes = None
                pca_model_final = None
        else:
            import logging
            logging.getLogger("pca_logic").info(
                f"PCA not applicable: {df_estandarizado.shape[1]} columns, "
                f"{df_estandarizado.shape[0]} observations (min: 2 cols, 3 obs)"
            )
            df_componentes = None
            pca_model_final = None

        # Prepara resultados
        results = {
            "df_consolidado": df_consolidado,
            "df_imputado": df_imputado,
            "mascara_imputados": mascara_imputados,
            "df_estandarizado": df_estandarizado,
            "df_componentes_principales": df_componentes,  # NUEVO: Componentes principales temporales
            "pca_model": pca_model_final,  # NUEVO: Modelo PCA final
            "scaler": scaler,
            "df_covarianza": df_covarianza,
            "pca_sugerencias": {
                "n_sugg_90": n_sugg_90 if df_estandarizado.shape[1] > 1 else None,
                "n_sugg_95": n_sugg_95 if df_estandarizado.shape[1] > 1 else None,
                "evr": evr if df_estandarizado.shape[1] > 1 else None,
                "cum_evr": cum_evr if df_estandarizado.shape[1] > 1 else None,
                "df_varianza_explicada": df_varianza_explicada,
            },
            # El PCA final y componentes se calculan aparte, según la selección del usuario
        }
        return results

    @staticmethod
    def run_pca_final(df_estandarizado, n_componentes):
        pca_model_final, df_componentes = pca_mod.realizar_pca(
            df_estandarizado, n_components=n_componentes
        )
        return pca_model_final, df_componentes
