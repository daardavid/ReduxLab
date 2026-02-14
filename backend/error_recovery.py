# error_recovery.py
"""
Sistema de recuperación de errores para la aplicación PCA.

Proporciona decoradores y utilidades para manejar errores de manera robusta,
con mecanismos de recuperación automática, reintentos y fallback.
"""

import time
import functools
import logging
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass
from backend.logging_config import get_logger

logger = get_logger("error_recovery")


@dataclass
class RecoveryConfig:
    """Configuración para recuperación de errores."""
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    max_delay: float = 30.0
    fallback_enabled: bool = True
    log_errors: bool = True


class ErrorRecovery:
    """Sistema de recuperación de errores."""

    def __init__(self, config: Optional[RecoveryConfig] = None):
        self.config = config or RecoveryConfig()

    def retry_with_recovery(self, func: Callable) -> Callable:
        """Decorador para reintentar funciones con recuperación."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(self.config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if self.config.log_errors:
                        logger.warning(
                            f"Attempt {attempt + 1}/{self.config.max_retries + 1} failed "
                            f"for {func.__name__}: {str(e)}"
                        )

                    if attempt < self.config.max_retries:
                        delay = self._calculate_delay(attempt)
                        if self.config.log_errors:
                            logger.info(f"Retrying {func.__name__} in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        # Último intento fallido, intentar recuperación
                        return self._attempt_recovery(func, args, kwargs, last_exception)

            # No debería llegar aquí, pero por si acaso
            raise last_exception

        return wrapper

    def _calculate_delay(self, attempt: int) -> float:
        """Calcula el delay para el siguiente reintento."""
        if self.config.exponential_backoff:
            delay = self.config.retry_delay * (2 ** attempt)
        else:
            delay = self.config.retry_delay

        return min(delay, self.config.max_delay)

    def _attempt_recovery(self, func: Callable, args: tuple, kwargs: dict, exception: Exception) -> Any:
        """Intenta recuperar de un error fallido."""
        if not self.config.fallback_enabled:
            raise exception

        recovery_func = getattr(self, f"_recover_{func.__name__}", None)
        if recovery_func:
            try:
                logger.info(f"Attempting recovery for {func.__name__}")
                return recovery_func(*args, **kwargs)
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {func.__name__}: {str(recovery_error)}")
                raise exception
        else:
            logger.warning(f"No recovery function found for {func.__name__}")
            raise exception

    def _recover_load_excel_file(self, file_path: str, **kwargs) -> Any:
        """Recuperación para carga de datos Excel."""
        # Intentar con diferentes encodings o formatos
        import pandas as pd

        encodings = ['utf-8', 'latin1', 'cp1252']
        for encoding in encodings:
            try:
                logger.info(f"Trying to load Excel with encoding: {encoding}")
                return pd.read_excel(file_path, encoding=encoding, **kwargs)
            except Exception as e:
                logger.debug(f"Encoding {encoding} failed: {str(e)}")
                continue

        # Fallback: crear DataFrame vacío con mensaje
        logger.warning(f"All recovery attempts failed for {file_path}, returning empty DataFrame")
        return pd.DataFrame({"error": [f"Failed to load {file_path}"]})

    def _recover_realizar_pca(self, df_estandarizado: Any, **kwargs) -> Any:
        """Recuperación para análisis PCA."""
        # Intentar con menos componentes o datos reducidos
        try:
            import numpy as np
            import pandas as pd
            from sklearn.decomposition import PCA

            # Usar el parámetro df_estandarizado como datos
            data = df_estandarizado
            
            # Reducir dimensionalidad si es posible
            if hasattr(data, 'shape') and len(data.shape) > 1:
                n_samples, n_features = data.shape
                if n_features > 10:
                    logger.info("Reducing data dimensionality for PCA recovery")
                    # Usar solo las primeras 10 características
                    reduced_data = data[:, :10] if isinstance(data, np.ndarray) else data.iloc[:, :10]
                    pca = PCA(n_components=min(5, reduced_data.shape[1]))
                    return pca.fit_transform(reduced_data), pca

            # Fallback básico
            pca = PCA(n_components=2)
            return pca.fit_transform(data), pca

        except Exception as e:
            logger.error(f"PCA recovery failed: {str(e)}")
            raise

    def _recover_validate_dataframe_for_pca(self, df: Any, **kwargs) -> Any:
        """Recuperación para validación de datos PCA."""
        # Intentar limpiar datos problemáticos
        try:
            import pandas as pd
            import numpy as np

            if isinstance(df, pd.DataFrame):
                # Remover filas con NaN excesivos
                nan_threshold = len(df.columns) * 0.5
                cleaned_data = df.dropna(thresh=nan_threshold)

                # Rellenar NaN restantes con medianas
                for col in cleaned_data.select_dtypes(include=[np.number]).columns:
                    if cleaned_data[col].isnull().any():
                        median_val = cleaned_data[col].median()
                        cleaned_data[col] = cleaned_data[col].fillna(median_val)

                logger.info(f"Data cleaned: {df.shape} -> {cleaned_data.shape}")
                return cleaned_data

            return df

        except Exception as e:
            logger.error(f"Data validation recovery failed: {str(e)}")
            raise

    def _recover_validate_data(self, data: Any, **kwargs) -> Any:
        """Recuperación para validación de datos."""
        # Intentar limpiar datos problemáticos
        try:
            import pandas as pd
            import numpy as np

            if isinstance(data, pd.DataFrame):
                # Remover filas con NaN excesivos
                nan_threshold = len(data.columns) * 0.5
                cleaned_data = data.dropna(thresh=nan_threshold)

                # Rellenar NaN restantes con medianas
                for col in cleaned_data.select_dtypes(include=[np.number]).columns:
                    if cleaned_data[col].isnull().any():
                        median_val = cleaned_data[col].median()
                        cleaned_data[col] = cleaned_data[col].fillna(median_val)

                logger.info(f"Data cleaned: {data.shape} -> {cleaned_data.shape}")
                return cleaned_data

            return data

        except Exception as e:
            logger.error(f"Data validation recovery failed: {str(e)}")
            raise


# Instancia global
_error_recovery = ErrorRecovery()


# Decoradores de conveniencia
def recoverable(func: Callable) -> Callable:
    """Decorador para funciones con recuperación automática."""
    return _error_recovery.retry_with_recovery(func)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorador parametrizable para reintentos."""
    def decorator(func: Callable) -> Callable:
        config = RecoveryConfig(max_retries=max_retries, retry_delay=delay)
        recovery = ErrorRecovery(config)
        return recovery.retry_with_recovery(func)
    return decorator


# Funciones de utilidad
def safe_operation(func: Callable, *args, fallback_value=None, **kwargs):
    """Ejecuta una operación de manera segura con valor de fallback."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Safe operation failed: {str(e)}, using fallback")
        return fallback_value


def log_errors(func: Callable) -> Callable:
    """Decorador simple para logging de errores."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper


if __name__ == "__main__":
    # Tests del sistema de recuperación
    print("Testing error recovery system...")

    @recoverable
    def failing_function():
        raise ValueError("Test error")

    @retry_on_failure(max_retries=2, delay=0.1)
    def another_failing_function():
        raise ConnectionError("Network error")

    try:
        failing_function()
    except ValueError:
        print("Recovery system working - error propagated after retries")

    try:
        another_failing_function()
    except ConnectionError:
        print("Parametrized retry working")