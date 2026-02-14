#!/usr/bin/env python3
"""
Módulo de validación de datos para análisis PCA.

Autor: David Armando Abreu Rosique
Fecha: 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import os
from pathlib import Path
import logging
from backend.error_recovery import recoverable

logger = logging.getLogger("data_validation")

class DataValidator:
    """Clase para validar datos antes del análisis PCA."""
    
    @staticmethod
    def is_numeric_data(df: pd.DataFrame) -> bool:
        """Verifica si los datos son numéricos."""
        return df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]
    
    @staticmethod
    def has_sufficient_data(df: pd.DataFrame, min_samples: int = 3) -> bool:
        """Verifica si hay suficientes muestras."""
        return len(df) >= min_samples
    
    @staticmethod
    def check_missing_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Verifica datos faltantes."""
        missing_count = df.isnull().sum().sum()
        missing_percentage = (missing_count / (df.shape[0] * df.shape[1])) * 100
        
        return {
            'has_missing': missing_count > 0,
            'missing_count': missing_count,
            'missing_percentage': missing_percentage,
            'missing_by_column': df.isnull().sum().to_dict()
        }

@recoverable
def validate_dataframe_for_pca(df: pd.DataFrame,
                              context_or_min_samples: Any = 3,
                              max_missing_percent: float = 20.0):
    """Valida un DataFrame para PCA con interfaz retrocompatible.

    Puede llamarse de dos formas:
    - validate_dataframe_for_pca(df, 3)  # donde 3 = min_samples
    - validate_dataframe_for_pca(df, "Nombre contexto")  # como hacía pca_module original

    Retorna:
        (is_valid: bool, info: dict)
    """
    # Interpretar segundo argumento
    if isinstance(context_or_min_samples, int):
        min_samples = context_or_min_samples
        context_name = "DataFrame"
    else:
        context_name = str(context_or_min_samples)
        min_samples = 3

    result = {
        'context': context_name,
        'errors': [],
        'warnings': [],
        'shape': None,
        'missing': {},
        'zero_variance_columns': [],
        'numeric_columns': [],
    }

    if df is None:
        result['errors'].append('DataFrame es None')
        return False, result
    if df.empty:
        result['errors'].append('DataFrame está vacío')
        return False, result

    # Numeric check
    if not DataValidator.is_numeric_data(df):
        result['errors'].append('Los datos deben ser numéricos')

    # Sample size
    if not DataValidator.has_sufficient_data(df, min_samples):
        result['errors'].append(f'Se requieren al menos {min_samples} muestras, se encontraron {len(df)}')

    # Missing data
    miss = DataValidator.check_missing_data(df)
    result['missing'] = miss
    if miss['has_missing']:
        if miss['missing_percentage'] > max_missing_percent:
            result['errors'].append(f'Demasiados datos faltantes: {miss['missing_percentage']:.1f}%')
        else:
            result['warnings'].append(f'Datos faltantes: {miss['missing_percentage']:.1f}%')

    # Zero variance columns
    numeric_df = df.select_dtypes(include=[np.number])
    result['numeric_columns'] = list(numeric_df.columns)
    zv = [c for c in numeric_df.columns if numeric_df[c].var() == 0]
    if zv:
        result['zero_variance_columns'] = zv
        result['warnings'].append(f'Columnas con varianza cero: {zv}')

    result['shape'] = df.shape

    is_valid = len(result['errors']) == 0
    # Agregar resumen
    if is_valid:
        result['summary'] = 'OK'
    else:
        result['summary'] = '; '.join(result['errors'])

    return is_valid, result


def validate_matrix_shape(df: pd.DataFrame) -> List[str]:
    """
    Non-blocking validation for the n×p matrix contract (observations × variables).
    Returns a list of warning messages; does not raise. Use when assigning sheet["df"].
    """
    warnings_list: List[str] = []
    if df is None or df.empty:
        return warnings_list
    n_num = df.select_dtypes(include=[np.number]).shape[1]
    n_cols = df.shape[1]
    n_rows = df.shape[0]
    if n_num < 2 and n_cols >= 2:
        warnings_list.append("Se necesitan al menos 2 columnas numéricas para el análisis.")
    if n_num < n_cols and n_cols > 0:
        non_num = n_cols - n_num
        warnings_list.append(f"Solo se usarán columnas numéricas en el análisis ({non_num} col. no numéricas).")
    missing_pct = (df.isnull().sum().sum() / (n_rows * n_cols)) * 100 if n_rows * n_cols else 0
    if missing_pct > 30:
        warnings_list.append(f"Muchos valores faltantes ({missing_pct:.1f}%). Considere imputar o filtrar.")
    return warnings_list


def clean_dataframe_for_pca(df: pd.DataFrame,
                           drop_threshold: float = 0.5,
                           fill_method: str = 'mean') -> pd.DataFrame:
    """
    Limpia un DataFrame para análisis PCA.
    
    Args:
        df: DataFrame a limpiar
        drop_threshold: Umbral para eliminar columnas/filas con muchos NaN
        fill_method: Método para llenar valores faltantes ('mean', 'median', 'drop')
        
    Returns:
        DataFrame limpio
    """
    df_clean = df.copy()
    
    # Eliminar columnas con demasiados valores faltantes
    col_missing = df_clean.isnull().sum() / len(df_clean)
    cols_to_drop = col_missing[col_missing > drop_threshold].index
    if len(cols_to_drop) > 0:
        df_clean = df_clean.drop(columns=cols_to_drop)
        warnings.warn(f"Eliminadas columnas con >50% valores faltantes: {list(cols_to_drop)}")
    
    # Eliminar filas con demasiados valores faltantes
    row_missing = df_clean.isnull().sum(axis=1) / len(df_clean.columns)
    rows_to_drop = row_missing[row_missing > drop_threshold].index
    if len(rows_to_drop) > 0:
        df_clean = df_clean.drop(index=rows_to_drop)
        warnings.warn(f"Eliminadas {len(rows_to_drop)} filas con >50% valores faltantes")
    
    # Tratar valores faltantes restantes
    if fill_method == 'mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif fill_method == 'median':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    elif fill_method == 'drop':
        df_clean = df_clean.dropna()
    
    return df_clean


# ---------------------------------------------------------------------------
# Funciones adicionales de validación (producción)
# ---------------------------------------------------------------------------

def validate_excel_schema(
    df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
    expected_types: Optional[Dict[str, type]] = None,
) -> Dict[str, Any]:
    """Valida que un DataFrame tenga las columnas y tipos esperados.

    Args:
        df: DataFrame a validar.
        expected_columns: Lista de nombres de columna que deben estar presentes.
            Si es ``None`` la verificación de columnas se omite.
        expected_types: Diccionario ``{columna: tipo_esperado}`` donde
            *tipo_esperado* es un tipo de NumPy/Python (p. ej. ``np.float64``,
            ``int``, ``object``).  Se comprueba compatibilidad con
            ``np.issubdtype`` para tipos numéricos.  Si es ``None`` la
            verificación de tipos se omite.

    Returns:
        dict con claves ``"valid"`` (bool), ``"errors"`` (list[str]) y
        ``"warnings"`` (list[str]).
    """
    errors: List[str] = []
    warn: List[str] = []

    if df is None or not isinstance(df, pd.DataFrame):
        errors.append("El objeto proporcionado no es un DataFrame válido.")
        return {"valid": False, "errors": errors, "warnings": warn}

    # --- Validar columnas esperadas ---
    if expected_columns is not None:
        actual_cols = set(df.columns)
        expected_set = set(expected_columns)
        missing = expected_set - actual_cols
        extra = actual_cols - expected_set
        if missing:
            errors.append(f"Columnas faltantes: {sorted(missing)}")
        if extra:
            warn.append(f"Columnas adicionales no esperadas: {sorted(extra)}")

    # --- Validar tipos compatibles ---
    if expected_types is not None:
        for col, exp_type in expected_types.items():
            if col not in df.columns:
                # Ya reportada como faltante arriba; no duplicar.
                continue
            col_dtype = df[col].dtype
            try:
                # Para tipos numéricos usar issubdtype
                if np.issubdtype(col_dtype, np.number) and np.issubdtype(
                    exp_type, np.number
                ):
                    if not np.issubdtype(col_dtype, exp_type):
                        warn.append(
                            f"Columna '{col}': tipo {col_dtype} no es subtipo de "
                            f"{exp_type}, pero ambos son numéricos."
                        )
                elif col_dtype.kind == "O" and exp_type in (str, object):
                    pass  # compatible
                elif not np.issubdtype(col_dtype, exp_type):
                    errors.append(
                        f"Columna '{col}': se esperaba tipo compatible con "
                        f"{exp_type}, pero tiene {col_dtype}."
                    )
            except TypeError:
                # issubdtype puede lanzar TypeError con tipos no-numpy
                if col_dtype.type is not exp_type and col_dtype.kind != "O":
                    errors.append(
                        f"Columna '{col}': tipo {col_dtype} incompatible con "
                        f"{exp_type}."
                    )

    valid = len(errors) == 0
    if valid:
        logger.debug("Esquema del DataFrame validado correctamente.")
    else:
        logger.warning("Errores de esquema encontrados: %s", errors)

    return {"valid": valid, "errors": errors, "warnings": warn}


def detect_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Detecta valores atípicos en columnas numéricas.

    Args:
        df: DataFrame con datos a analizar.
        method: Método de detección.

            * ``"iqr"`` – Rango intercuartílico.  Un valor se considera
              atípico si cae fuera de ``[Q1 - threshold*IQR, Q3 + threshold*IQR]``.
            * ``"zscore"`` – Puntuación Z.  Un valor se considera atípico si
              ``|z| > threshold``.  Valor por defecto recomendado: ``3.0``.

        threshold: Factor multiplicador para el método elegido (por defecto
            1.5 para IQR, suele usarse 3.0 para z-score).

    Returns:
        DataFrame de la misma forma que *df* con ``True`` en las celdas que
        son valores atípicos y ``False`` en el resto.  Las columnas no
        numéricas se rellenan con ``False``.

    Raises:
        ValueError: Si *method* no es ``"iqr"`` ni ``"zscore"``.
    """
    method = method.lower().strip()
    if method not in ("iqr", "zscore"):
        raise ValueError(
            f"Método '{method}' no soportado. Usa 'iqr' o 'zscore'."
        )

    outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if numeric_cols.empty:
        logger.info("No se encontraron columnas numéricas para detectar outliers.")
        return outlier_mask

    if method == "iqr":
        q1 = df[numeric_cols].quantile(0.25)
        q3 = df[numeric_cols].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outlier_mask[numeric_cols] = (
            (df[numeric_cols] < lower) | (df[numeric_cols] > upper)
        )
    else:  # zscore
        means = df[numeric_cols].mean()
        stds = df[numeric_cols].std(ddof=1)
        # Evitar división por cero en columnas con desviación estándar 0
        safe_stds = stds.replace(0, np.nan)
        z_scores = (df[numeric_cols] - means) / safe_stds
        outlier_mask[numeric_cols] = z_scores.abs() > threshold

    n_outliers = int(outlier_mask.sum().sum())
    logger.info(
        "Outliers detectados (%s, threshold=%.2f): %d valores en %d columnas.",
        method,
        threshold,
        n_outliers,
        int((outlier_mask.sum() > 0).sum()),
    )
    return outlier_mask


def validate_pca_readiness(
    df: pd.DataFrame,
    min_samples: Optional[int] = None,
    min_features: int = 2,
) -> Dict[str, Any]:
    """Verifica si un DataFrame cumple los requisitos mínimos para PCA.

    Requisitos evaluados:

    * Al menos ``min_features`` columnas numéricas (por defecto 2).
    * Al menos ``min_samples`` filas (por defecto igual a ``min_features``).

    Args:
        df: DataFrame a evaluar.
        min_samples: Número mínimo de muestras requeridas.  Si es ``None``
            se establece igual a ``min_features``.
        min_features: Número mínimo de variables numéricas requeridas (>=2).

    Returns:
        dict con claves ``"ready"`` (bool), ``"message"`` (str),
        ``"n_samples"`` (int) y ``"n_features"`` (int).
    """
    if min_samples is None:
        min_samples = min_features

    if df is None or not isinstance(df, pd.DataFrame):
        return {
            "ready": False,
            "message": "El objeto proporcionado no es un DataFrame válido.",
            "n_samples": 0,
            "n_features": 0,
        }

    numeric_df = df.select_dtypes(include=[np.number])
    n_samples = len(numeric_df)
    n_features = numeric_df.shape[1]

    issues: List[str] = []

    if n_features < min_features:
        issues.append(
            f"Se requieren al menos {min_features} variables numéricas, "
            f"pero solo hay {n_features}."
        )
    if n_samples < min_samples:
        issues.append(
            f"Se requieren al menos {min_samples} muestras, "
            f"pero solo hay {n_samples}."
        )

    ready = len(issues) == 0
    message = "Datos listos para PCA." if ready else " | ".join(issues)

    if ready:
        logger.debug(
            "PCA readiness OK: %d muestras × %d features.", n_samples, n_features
        )
    else:
        logger.warning("PCA readiness FAIL: %s", message)

    return {
        "ready": ready,
        "message": message,
        "n_samples": n_samples,
        "n_features": n_features,
    }


def check_data_loss(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    threshold: float = 0.2,
) -> Dict[str, Any]:
    """Compara dos DataFrames y advierte si se perdió demasiada información.

    Se usa típicamente para comparar un DataFrame antes y después de una
    limpieza o filtrado.

    Args:
        df_before: DataFrame original (antes de la limpieza).
        df_after: DataFrame resultante (después de la limpieza).
        threshold: Proporción máxima aceptable de pérdida de filas (0.0–1.0).
            Por defecto 0.2 (20 %).

    Returns:
        dict con claves ``"loss_ratio"`` (float), ``"rows_before"`` (int),
        ``"rows_after"`` (int) y ``"warning"`` (str o ``None``).
    """
    rows_before = len(df_before)
    rows_after = len(df_after)

    if rows_before == 0:
        loss_ratio = 0.0
    else:
        loss_ratio = 1.0 - (rows_after / rows_before)

    warning_msg: Optional[str] = None

    if loss_ratio > threshold:
        warning_msg = (
            f"Se perdió el {loss_ratio * 100:.1f}% de las filas durante la limpieza "
            f"({rows_before} → {rows_after}). Umbral permitido: {threshold * 100:.0f}%."
        )
        logger.warning(warning_msg)
    else:
        logger.debug(
            "Pérdida de datos aceptable: %.1f%% (%d → %d filas).",
            loss_ratio * 100,
            rows_before,
            rows_after,
        )

    return {
        "loss_ratio": round(loss_ratio, 4),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "warning": warning_msg,
    }


_SUPPORTED_EXTENSIONS: Dict[str, str] = {
    ".xlsx": "Excel (xlsx)",
    ".xls": "Excel (xls)",
    ".csv": "CSV",
    ".parquet": "Parquet",
}


def validate_file_format(
    file_path: Union[str, Path],
) -> Dict[str, Any]:
    """Valida que un archivo exista, no esté vacío y tenga extensión soportada.

    Extensiones soportadas: ``.xlsx``, ``.xls``, ``.csv``, ``.parquet``.

    Args:
        file_path: Ruta al archivo a validar (cadena o ``Path``).

    Returns:
        dict con claves ``"valid"`` (bool), ``"format"`` (str),
        ``"size_mb"`` (float) y ``"error"`` (str o ``None``).
    """
    file_path = Path(file_path)
    error: Optional[str] = None
    fmt: str = ""
    size_mb: float = 0.0

    if not file_path.exists():
        error = f"El archivo no existe: {file_path}"
        logger.error(error)
        return {"valid": False, "format": fmt, "size_mb": size_mb, "error": error}

    if not file_path.is_file():
        error = f"La ruta no apunta a un archivo regular: {file_path}"
        logger.error(error)
        return {"valid": False, "format": fmt, "size_mb": size_mb, "error": error}

    size_bytes = file_path.stat().st_size
    size_mb = round(size_bytes / (1024 * 1024), 4)

    if size_bytes == 0:
        error = f"El archivo está vacío (0 bytes): {file_path}"
        logger.error(error)
        return {"valid": False, "format": fmt, "size_mb": size_mb, "error": error}

    ext = file_path.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        error = (
            f"Extensión '{ext}' no soportada. "
            f"Extensiones válidas: {list(_SUPPORTED_EXTENSIONS.keys())}"
        )
        logger.error(error)
        return {"valid": False, "format": fmt, "size_mb": size_mb, "error": error}

    fmt = _SUPPORTED_EXTENSIONS[ext]
    logger.debug(
        "Archivo validado: %s | formato=%s | tamaño=%.4f MB",
        file_path.name,
        fmt,
        size_mb,
    )
    return {"valid": True, "format": fmt, "size_mb": size_mb, "error": None}
