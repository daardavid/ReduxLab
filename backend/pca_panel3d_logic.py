from backend import data_loader_module as dl
from backend import preprocessing_module as dl_prep
from backend.constants import COUNTRY_GROUPS, GROUP_COLORS
import pandas as pd


class PCAPanel3DLogic:
    @staticmethod
    def run_panel3d_analysis_logic(
        all_sheets_data,
        available_sheet_names_list,
        indicators_selected,
        countries_selected,
        country_groups_map=None,
        group_colors_map=None,
    ):
        """
        Realiza el análisis de trayectorias 3D (Panel PCA 3D) y retorna los resultados necesarios para la visualización.
        Permite pasar mapeos personalizados de grupos y colores.
        """
        df_panel = dl.preparar_datos_panel_longitudinal(
            all_sheets_data, indicators_selected, countries_selected
        )
        if df_panel.empty:
            return {
                "error": "No se pudo construir el panel de datos. Revisa la selección."
            }

        df_panel_no_na = df_panel.dropna(axis=0, how="any")
        if df_panel_no_na.shape[0] < 3 or df_panel_no_na.shape[1] < 3:
            return {
                "error": "Datos insuficientes para el análisis 3D después de eliminar NaNs."
            }

        df_panel_estandarizado, scaler_panel = dl_prep.estandarizar_datos(
            df_panel_no_na, devolver_scaler=True
        )
        pca_model_panel, df_pc_scores_panel = (
            dl_prep.pca_module.realizar_pca(df_panel_estandarizado, n_components=3)
            if hasattr(dl_prep, "pca_module")
            else (None, pd.DataFrame())
        )
        if pca_model_panel is None or df_pc_scores_panel.empty:
            import pca_module as pca_mod

            pca_model_panel, df_pc_scores_panel = pca_mod.realizar_pca(
                df_panel_estandarizado, n_components=3
            )

        if pca_model_panel is None or df_pc_scores_panel.empty:
            return {"error": "No se pudo realizar el PCA sobre los datos de panel."}

        cg_map = (
            country_groups_map if country_groups_map is not None else COUNTRY_GROUPS
        )
        gc_map = group_colors_map if group_colors_map is not None else GROUP_COLORS
        selected_country_groups = {
            pais: cg_map.get(pais, "Otros") for pais in countries_selected
        }
        grupos_presentes = set(selected_country_groups.values())
        selected_group_colors = {
            grupo: gc_map.get(grupo, "#888888") for grupo in grupos_presentes
        }

        return {
            "df_pc_scores_panel": df_pc_scores_panel,
            "pca_model_panel": pca_model_panel,
            "country_groups": selected_country_groups,
            "group_colors": selected_group_colors,
        }
