# preprocessing_module.py
"""
Módulo de preprocesamiento de datos para análisis PCA socioeconómico.

Este módulo implementa técnicas robustas de limpieza y preparación de datos:
- Múltiples estrategias de imputación de valores faltantes
- Estandarización de datos (z-score)
- Validación de calidad de datos
- Interfaz interactiva para selección de métodos

Las funciones están diseñadas para manejar datos socioeconómicos que frecuentemente
presentan valores faltantes y requieren tratamiento especializado.

Estrategias de imputación disponibles:
    - Interpolación (lineal, polinomial, spline)
    - Estadísticas descriptivas (media, mediana, moda)
    - Métodos avanzados (iterativo, KNN)
    - Propagación (forward/backward fill)
    - Valores constantes personalizados

Autor: David Armando Abreu Rosique
Fecha: 2025
"""
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, Any, Union

# Importar sistema de logging
from backend.logging_config import get_logger

logger = get_logger("preprocessing_module")


def manejar_datos_faltantes(
    df: pd.DataFrame,
    estrategia: str = "interpolacion",
    valor_relleno: Optional[Union[int, float, str]] = None,
    devolver_mascara: bool = False,
    iteraciones_imputador: int = 10,
    estimador_imputador: Optional[Any] = None,
    metodo_interpolacion: str = "linear",
    orden_interpolacion: int = 3,
    knn_vecinos: int = 5,
    **kwargs,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Maneja valores faltantes en un DataFrame usando múltiples estrategias de imputación.

    Esta función implementa un conjunto completo de técnicas de imputación optimizadas
    para datos socioeconómicos longitudinales. Cada estrategia está diseñada para
    preservar las características estadísticas y temporales de los datos originales.

    Args:
        df (pd.DataFrame): DataFrame con posibles valores faltantes. Puede contener
            columnas numéricas y categóricas.
        estrategia (str): Método de imputación a aplicar:
            - 'interpolacion': Interpolación numérica (lineal por defecto)
            - 'mean': Rellena con la media de cada columna
            - 'median': Rellena con la mediana de cada columna
            - 'most_frequent': Rellena con el valor más frecuente (moda)
            - 'ffill': Forward fill (propaga valores anteriores)
            - 'bfill': Backward fill (propaga valores posteriores)
            - 'iterative': Imputación iterativa multivariada
            - 'knn': Imputación basada en k-vecinos más cercanos
            - 'valor_constante': Rellena con un valor específico
            - 'eliminar_filas': Elimina filas con cualquier NaN
            - 'ninguna': No aplica imputación
        valor_relleno (Optional[Union[int, float, str]]): Valor específico para
            estrategia 'valor_constante'.
        devolver_mascara (bool): Si True, retorna también una máscara booleana
            indicando qué valores fueron imputados.
        iteraciones_imputador (int): Número máximo de iteraciones para 'iterative'.
        estimador_imputador (Optional[Any]): Estimador base para imputación iterativa.
        metodo_interpolacion (str): Método de interpolación ('linear', 'polynomial',
            'spline', etc.).
        orden_interpolacion (int): Orden para interpolación polinomial/spline.
        knn_vecinos (int): Número de vecinos para imputación KNN.
        **kwargs: Parámetros adicionales específicos por estrategia.

    Returns:
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
            - Si devolver_mascara=False: DataFrame con valores imputados
            - Si devolver_mascara=True: Tupla (DataFrame imputado, DataFrame máscara)

    Raises:
        ValueError: Si la estrategia no es reconocida
        TypeError: Si los parámetros no son del tipo esperado

    Example:
        >>> # Imputación básica con interpolación
        >>> df_clean = manejar_datos_faltantes(df, estrategia='interpolacion')

        >>> # Imputación KNN con máscara
        >>> df_clean, mascara = manejar_datos_faltantes(
        ...     df, estrategia='knn', knn_vecinos=3, devolver_mascara=True
        ... )

        >>> # Imputación con valor constante
        >>> df_clean = manejar_datos_faltantes(
        ...     df, estrategia='valor_constante', valor_relleno=0
        ... )

    Note:
        - Solo las columnas numéricas se procesan con estrategias estadísticas
        - La imputación preserva tipos de datos originales cuando es posible
        - Para series temporales, se recomienda 'interpolacion' o 'ffill'/'bfill'
        - La estrategia 'iterative' es más robusta pero computacionalmente costosa
    """
    if df.empty:
        if devolver_mascara:
            mascara_vacia = pd.DataFrame(False, index=df.index, columns=df.columns)
            return df.copy(), mascara_vacia
        return df.copy()

    df_copia = df.copy()
    mascara_imputados_original = df_copia.isnull()
    faltantes_antes = mascara_imputados_original.sum().sum()

    if faltantes_antes == 0:
        if devolver_mascara:
            return df_copia, pd.DataFrame(False, index=df.index, columns=df.columns)
        return df_copia

    # print(f"  Datos faltantes ANTES (por columna):\n{df_copia.isnull().sum()[df_copia.isnull().sum() > 0]}")

    columnas_originales = df_copia.columns.tolist()
    indice_original = df_copia.index

    numeric_cols_original = df_copia.select_dtypes(include=np.number).columns.tolist()
    # categorical_cols_original = df_copia.select_dtypes(exclude=np.number).columns.tolist() # No se usa activamente aún

    df_procesado = False  # Bandera para saber si se aplicó alguna estrategia

    if estrategia == "eliminar_filas":
        df_copia.dropna(axis=0, how="any", inplace=True)
        # print("  Filas con valores NaN eliminadas.")
        df_procesado = True

    elif estrategia in ["mean", "median", "most_frequent", "valor_constante"]:
        nombre_estrategia_sklearn = estrategia
        fill_val_const = np.nan

        if estrategia == "valor_constante":
            nombre_estrategia_sklearn = "constant"
            if valor_relleno is None:
                valor_relleno = 0.0
                logger.warning(
                    f"Estrategia 'valor_constante' sin 'valor_relleno'. Se usará {valor_relleno} por defecto."
                )
            fill_val_const = valor_relleno

        # Columnas que tienen algunos NaNs pero no son completamente NaN
        cols_con_nan_parcial = df_copia.columns[
            df_copia.isnull().any() & df_copia.notna().any()
        ].tolist()
        # Columnas que son completamente NaN
        cols_todo_nan = df_copia.columns[df_copia.isna().all()].tolist()
        # Columnas sin ningún NaN
        cols_sin_nan = df_copia.columns[df_copia.notna().all()].tolist()

        df_imputado_final_simple = df_copia.copy()  # Empezar con una copia

        if cols_con_nan_parcial:
            # Para 'mean' y 'median', solo aplicar a numéricas dentro de cols_con_nan_parcial
            if estrategia in ["mean", "median"]:
                cols_a_imputar_simple = [
                    col for col in cols_con_nan_parcial if col in numeric_cols_original
                ]
            else:  # 'most_frequent', 'constant' pueden aplicar a todas las cols_con_nan_parcial
                cols_a_imputar_simple = cols_con_nan_parcial

            if cols_a_imputar_simple:
                df_parte_a_imputar = df_copia[cols_a_imputar_simple]
                imputador_simple = SimpleImputer(
                    strategy=nombre_estrategia_sklearn,
                    fill_value=(
                        fill_val_const if estrategia == "valor_constante" else None
                    ),
                )

                df_imputado_np = imputador_simple.fit_transform(df_parte_a_imputar)
                df_imputado_parcial = pd.DataFrame(
                    df_imputado_np, columns=cols_a_imputar_simple, index=indice_original
                )

                # Actualizar df_imputado_final_simple
                for col in cols_a_imputar_simple:
                    df_imputado_final_simple[col] = df_imputado_parcial[col]
            else:
                pass

        # Las columnas en cols_todo_nan y cols_sin_nan ya están correctas en df_imputado_final_simple
        # (o se quedan como NaN o ya estaban completas)
        df_copia = df_imputado_final_simple[columnas_originales]  # Asegurar orden
        # print(f"  Valores NaN rellenados con '{estrategia}'.")
        df_procesado = True

    elif estrategia == "ffill":
        # ... (código ffill como estaba) ...
        limit_val = kwargs.get("ffill_limit")
        df_copia.ffill(inplace=True, limit=limit_val)
        df_copia.bfill(inplace=True, limit=kwargs.get("bfill_limit_after_ffill"))
        # print(f"  Valores NaN rellenados con forward fill (limit={limit_val}).")
        df_procesado = True
    elif estrategia == "bfill":
        # ... (código bfill como estaba) ...
        limit_val = kwargs.get("bfill_limit")
        df_copia.bfill(inplace=True, limit=limit_val)
        df_copia.ffill(inplace=True, limit=kwargs.get("ffill_limit_after_bfill"))
        # print(f"  Valores NaN rellenados con backward fill (limit={limit_val}).")
        df_procesado = True
    elif estrategia == "interpolacion":
        # ... (código interpolacion como estaba, asegurando que solo aplica a numeric_cols_original) ...
        if not numeric_cols_original:
            pass
        else:
            df_copia[numeric_cols_original] = df_copia[
                numeric_cols_original
            ].interpolate(
                method=metodo_interpolacion,
                axis=0,
                limit_direction="both",
                order=(
                    orden_interpolacion
                    if metodo_interpolacion in ["polynomial", "spline"]
                    else None
                ),
            )
            df_copia[numeric_cols_original] = df_copia[numeric_cols_original].ffill()
            df_copia[numeric_cols_original] = df_copia[numeric_cols_original].bfill()
            # print(f"  Valores NaN en columnas numéricas rellenados mediante interpolación '{metodo_interpolacion}'.")
        df_procesado = True

    elif estrategia in ["iterative", "knn"]:
        # ... (código para iterative y knn como estaba, asegurando que solo aplica a
        #      columnas numéricas que no son completamente NaN, y luego reconstruye) ...
        cols_num_imputables_adv = [
            col for col in numeric_cols_original if df_copia[col].notna().any()
        ]

        if not cols_num_imputables_adv:
            pass
        else:
            df_parte_num_imputable_adv = df_copia[cols_num_imputables_adv].copy()

            if estrategia == "iterative":
                pass
                imputer_adv = IterativeImputer(
                    max_iter=iteraciones_imputador,
                    random_state=0,
                    estimator=estimador_imputador,
                )
            else:  # knn
                pass
                imputer_adv = KNNImputer(n_neighbors=knn_vecinos)

            df_imputado_np_adv = imputer_adv.fit_transform(df_parte_num_imputable_adv)
            df_imputado_parcial_adv = pd.DataFrame(
                df_imputado_np_adv,
                columns=cols_num_imputables_adv,
                index=indice_original,
            )

            # Actualizar df_copia solo con las columnas numéricas imputadas
            for col in cols_num_imputables_adv:
                df_copia[col] = df_imputado_parcial_adv[col]
            # print(f"  Valores NaN en columnas numéricas imputados usando '{estrategia}'.")
        df_procesado = True

    if not df_procesado:
        pass

    # Mensajes de advertencia eliminados para limpieza

    if devolver_mascara:
        mascara_final_imputados = mascara_imputados_original & (~df_copia.isnull())
        return df_copia, mascara_final_imputados
    else:
        return df_copia


def estandarizar_datos(df, devolver_scaler=False):
    """
    Estandariza las columnas numéricas de un DataFrame (media 0, desviación estándar 1).

    Args:
        df (pd.DataFrame): DataFrame de entrada (Años como índice, Indicadores como columnas).
                           Se espera que las columnas a estandarizar sean numéricas.
        devolver_scaler (bool): Si True, devuelve también el objeto scaler ajustado.

    Returns:
        pd.DataFrame: DataFrame con las columnas numéricas estandarizadas.
        (Opcional) sklearn.preprocessing.StandardScaler: El objeto scaler ajustado.
    """
    if df.empty:
        if devolver_scaler:
            return df.copy(), None
        return df.copy()

    df_copia = df.copy()

    # Seleccionar solo columnas numéricas para estandarizar
    numeric_cols = df_copia.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        logger.warning("No se encontraron columnas numéricas para estandarizar.")
        if devolver_scaler:
            return df_copia, None
        return df_copia

    # print(f"  Columnas numéricas a estandarizar: {numeric_cols}")

    scaler = StandardScaler()

    # Aplicar el scaler SOLO a las columnas numéricas
    df_copia[numeric_cols] = scaler.fit_transform(df_copia[numeric_cols])

    # print("  Datos estandarizados exitosamente.")

    if devolver_scaler:
        return df_copia, scaler
    else:
        return df_copia


def preprocess_data(
    df,
    apply_transformations: bool = False,
    transformation_method: str = 'auto',
    skewness_threshold: float = 1.0,
    return_transformer: bool = False
):
    """
    Función principal de preprocesamiento de datos para PCA.
    
    Ahora incluye opción de transformaciones para manejar distribuciones sesgadas
    (común en datos financieros: ingresos, activos, valor de mercado).

    Args:
        df (pd.DataFrame): DataFrame con datos a preprocesar
        apply_transformations (bool): Si aplicar transformaciones para skewness
        transformation_method (str): Método de transformación ('auto', 'log', 'log1p', 'yeo-johnson', 'none')
        skewness_threshold (float): Umbral de skewness para aplicar transformación automática
        return_transformer (bool): Si devolver el objeto transformer para análisis

    Returns:
        pd.DataFrame: DataFrame preprocesado (transformado, imputado y estandarizado)
        (Opcional) Tuple[DataTransformer, Dict]: transformer y info de transformaciones si return_transformer=True
    """
    # Paso 1: Transformaciones (NUEVO - antes de imputación)
    transformer = None
    transformation_info = {}
    if apply_transformations and transformation_method != 'none':
        try:
            from data_transformations import DataTransformer
            logger.info(f"🔄 Aplicando transformaciones (método: {transformation_method})...")
            transformer = DataTransformer(skewness_threshold=skewness_threshold)
            df_transformed, transformation_info = transformer.transform(df, method=transformation_method)
            
            # Log summary
            if transformer.transformation_info:
                logger.info(f"✅ Transformadas {len(transformer.transformation_info)} columnas")
                summary = transformer.get_transformation_summary()
                if not summary.empty:
                    logger.info(f"\n{summary.to_string(index=False)}")
        except ImportError:
            logger.warning("⚠️  Módulo data_transformations no disponible, continuando sin transformaciones")
            df_transformed = df.copy()
        except Exception as e:
            logger.warning(f"⚠️  Error en transformaciones: {e}, continuando sin transformaciones")
            df_transformed = df.copy()
    else:
        df_transformed = df.copy()

    # Paso 2: Manejar datos faltantes
    df_imputado = manejar_datos_faltantes(df_transformed)

    # Paso 3: Estandarizar datos
    df_estandarizado = estandarizar_datos(df_imputado)

    if return_transformer:
        return df_estandarizado, transformer, transformation_info
    return df_estandarizado


def prompt_select_imputation_strategy():
    """
    DEPRECATED: This console-based function blocks the GUI event loop.
    Use the GUI dialog in dialogs.py → gui_select_imputation_strategy() instead.

    Returns a safe default so any remaining callers don't hang.
    """
    import warnings
    warnings.warn(
        "prompt_select_imputation_strategy() is deprecated and should not be "
        "called from the GUI.  Use gui_select_imputation_strategy() from "
        "dialogs.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return "interpolacion", {}


# ---------------------------------------------------------------------------
# Available imputation strategies (shared constant for GUI and logic)
# ---------------------------------------------------------------------------
IMPUTATION_STRATEGIES = {
    "interpolacion": "Interpolación (lineal por defecto)",
    "mean": "Rellenar con la Media",
    "median": "Rellenar con la Mediana",
    "most_frequent": "Rellenar con la Moda",
    "ffill": "Forward Fill (valor anterior)",
    "bfill": "Backward Fill (valor siguiente)",
    "iterative": "Imputación Iterativa (multivariada)",
    "knn": "Imputación KNN (vecinos cercanos)",
    "valor_constante": "Valor Constante",
    "eliminar_filas": "Eliminar filas con NaN",
    "ninguna": "No imputar (mantener NaNs)",
}
