# performance_optimizer.py
"""
Sistema de optimización de rendimiento para la aplicación PCA.

Incluye caching inteligente, procesamiento paralelo, monitoreo de memoria
y optimizaciones específicas para análisis de datos.
"""

import time
import functools
import gc
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, Dict, Optional, Tuple, Union
from pathlib import Path
import pickle
import hashlib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from backend.logging_config import get_logger
from backend.config_manager import get_config

logger = get_logger("performance_optimizer")


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento."""

    execution_time: float
    memory_usage_mb: float
    cache_hits: int
    cache_misses: int
    cpu_usage_percent: float

    def cache_hit_ratio(self) -> float:
        """Calcula la tasa de aciertos del cache."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class MemoryMonitor:
    """Monitor de memoria del sistema."""

    def __init__(self):
        self.process = psutil.Process()
        self._peak_memory = 0
        self._start_memory = 0

    def start_monitoring(self):
        """Inicia el monitoreo de memoria."""
        self._start_memory = self.get_current_memory()
        self._peak_memory = self._start_memory

    def get_current_memory(self) -> float:
        """Retorna el uso actual de memoria en MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def update_peak(self):
        """Actualiza el pico de memoria si es necesario."""
        current = self.get_current_memory()
        if current > self._peak_memory:
            self._peak_memory = current

    def get_peak_memory(self) -> float:
        """Retorna el pico de memoria usado."""
        return self._peak_memory

    def get_memory_delta(self) -> float:
        """Retorna el cambio en memoria desde el inicio del monitoreo."""
        return self.get_current_memory() - self._start_memory

    def cleanup_if_needed(self, threshold_mb: float = 500):
        """Ejecuta garbage collection si el uso de memoria es alto."""
        if self.get_current_memory() > threshold_mb:
            logger.info("High memory usage detected, running garbage collection")
            gc.collect()


class LRUCache:
    """Cache LRU (Least Recently Used) personalizado para DataFrames."""

    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def _make_key(self, *args, **kwargs) -> str:
        """Crea una clave hash única para los argumentos."""
        # Manejar DataFrames y arrays de numpy
        processed_args = []
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                # Hash basado en shape, columns y algunos valores
                shape_str = str(arg.shape)
                cols_str = str(list(arg.columns))
                sample_str = str(
                    arg.head(3).values.tobytes() if not arg.empty else "empty"
                )
                processed_args.append(f"df_{shape_str}_{cols_str}_{sample_str}")
            elif isinstance(arg, np.ndarray):
                processed_args.append(f"array_{arg.shape}_{str(arg.flat[:5])}")
            else:
                processed_args.append(str(arg))

        processed_kwargs = {k: str(v) for k, v in sorted(kwargs.items())}
        cache_key = f"{processed_args}_{processed_kwargs}"
        return hashlib.md5(cache_key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del cache."""
        with self._lock:
            if key in self.cache:
                # Mover al final (más reciente)
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hits += 1
                logger.debug(f"Cache hit for key: {key[:8]}...")
                return self.cache[key]
            else:
                self.misses += 1
                logger.debug(f"Cache miss for key: {key[:8]}...")
                return None

    def put(self, key: str, value: Any):
        """Almacena un valor en el cache."""
        with self._lock:
            if key in self.cache:
                # Actualizar valor existente
                self.access_order.remove(key)
                self.access_order.append(key)
                self.cache[key] = value
            else:
                # Nuevo valor
                if len(self.cache) >= self.max_size:
                    # Remover el menos usado
                    oldest = self.access_order.pop(0)
                    del self.cache[oldest]
                    logger.debug(f"Evicted cache entry: {oldest[:8]}...")

                self.cache[key] = value
                self.access_order.append(key)
                logger.debug(f"Cached new entry: {key[:8]}...")

    def clear(self):
        """Limpia el cache completamente."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del cache."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_ratio = self.hits / total_requests if total_requests > 0 else 0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": hit_ratio,
            }


class PerformanceOptimizer:
    """Optimizador principal de rendimiento."""

    def __init__(self):
        self.config = get_config().performance
        self.cache = LRUCache(max_size=self.config.max_cache_size)
        self.memory_monitor = MemoryMonitor()
        self._metrics_history = []

        # Pool de threads para operaciones paralelas
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers or 4)

    def cached_function(self, func: Callable) -> Callable:
        """Decorador para cachear resultados de funciones."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Crear clave de cache
            cache_key = self.cache._make_key(func.__name__, *args, **kwargs)

            # Intentar obtener del cache
            result = self.cache.get(cache_key)
            if result is not None:
                return result

            # Ejecutar función y cachear resultado
            start_time = time.time()
            self.memory_monitor.start_monitoring()

            result = func(*args, **kwargs)

            execution_time = time.time() - start_time
            memory_used = self.memory_monitor.get_memory_delta()

            # Cachear solo si el resultado no es demasiado grande
            if memory_used < 100:  # MB
                self.cache.put(cache_key, result)

            logger.debug(
                f"Function {func.__name__} executed in {execution_time:.3f}s, "
                f"memory delta: {memory_used:.1f}MB"
            )

            return result

        return wrapper

    def parallel_apply(
        self, func: Callable, data_chunks: list, use_processes: bool = False
    ) -> list:
        """Aplica una función en paralelo a chunks de datos."""
        start_time = time.time()

        if use_processes and self.config.enable_parallel_processing:
            # Usar procesos para operaciones CPU-intensivas
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                results = list(executor.map(func, data_chunks))
        elif self.config.enable_parallel_processing:
            # Usar threads para operaciones I/O
            results = list(self.thread_pool.map(func, data_chunks))
        else:
            # Ejecución secuencial
            results = [func(chunk) for chunk in data_chunks]

        execution_time = time.time() - start_time
        logger.info(
            f"Parallel execution completed in {execution_time:.3f}s "
            f"for {len(data_chunks)} chunks"
        )

        return results

    def chunk_dataframe(
        self, df: pd.DataFrame, chunk_size: Optional[int] = None
    ) -> list:
        """Divide un DataFrame en chunks para procesamiento paralelo."""
        chunk_size = chunk_size or self.config.chunk_size
        chunks = []

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size].copy()
            chunks.append(chunk)

        logger.debug(
            f"DataFrame chunked into {len(chunks)} pieces of ~{chunk_size} rows"
        )
        return chunks

    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimiza el uso de memoria de un DataFrame."""
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024

        # Optimizar tipos de datos numéricos
        for col in df.select_dtypes(include=["int"]).columns:
            col_min = df[col].min()
            col_max = df[col].max()

            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif (
                col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max
            ):
                df[col] = df[col].astype(np.int16)
            elif (
                col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max
            ):
                df[col] = df[col].astype(np.int32)

        for col in df.select_dtypes(include=["float"]).columns:
            if (
                df[col].min() >= np.finfo(np.float32).min
                and df[col].max() <= np.finfo(np.float32).max
            ):
                df[col] = df[col].astype(np.float32)

        # Optimizar strings categóricas
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].nunique() / len(df) < 0.5:  # Si menos del 50% son únicos
                df[col] = df[col].astype("category")

        optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = (1 - optimized_memory / original_memory) * 100

        logger.info(
            f"DataFrame memory optimized: {original_memory:.1f}MB -> "
            f"{optimized_memory:.1f}MB ({reduction:.1f}% reduction)"
        )

        return df

    def profile_function(self, func: Callable) -> Callable:
        """Decorador para perfilar el rendimiento de funciones."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Iniciar monitoreo
            start_time = time.time()
            start_cpu = psutil.cpu_percent()
            self.memory_monitor.start_monitoring()

            # Ejecutar función
            result = func(*args, **kwargs)

            # Calcular métricas
            execution_time = time.time() - start_time
            memory_usage = self.memory_monitor.get_peak_memory()
            end_cpu = psutil.cpu_percent()

            # Obtener estadísticas del cache
            cache_stats = self.cache.get_stats()

            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cache_hits=cache_stats["hits"],
                cache_misses=cache_stats["misses"],
                cpu_usage_percent=(start_cpu + end_cpu) / 2,
            )

            self._metrics_history.append(
                {
                    "function": func.__name__,
                    "timestamp": time.time(),
                    "metrics": metrics,
                }
            )

            logger.info(
                f"Performance profile for {func.__name__}: "
                f"Time={execution_time:.3f}s, Memory={memory_usage:.1f}MB, "
                f"CPU={metrics.cpu_usage_percent:.1f}%, "
                f"Cache_ratio={metrics.cache_hit_ratio():.2f}"
            )

            return result

        return wrapper

    def get_performance_report(self) -> Dict[str, Any]:
        """Genera un reporte de rendimiento del sistema."""
        if not self._metrics_history:
            return {"message": "No performance data available"}

        recent_metrics = self._metrics_history[-10:]  # Últimas 10 ejecuciones

        avg_time = np.mean([m["metrics"].execution_time for m in recent_metrics])
        avg_memory = np.mean([m["metrics"].memory_usage_mb for m in recent_metrics])
        avg_cpu = np.mean([m["metrics"].cpu_usage_percent for m in recent_metrics])

        cache_stats = self.cache.get_stats()
        system_memory = psutil.virtual_memory()

        return {
            "performance_summary": {
                "avg_execution_time": f"{avg_time:.3f}s",
                "avg_memory_usage": f"{avg_memory:.1f}MB",
                "avg_cpu_usage": f"{avg_cpu:.1f}%",
                "total_functions_profiled": len(self._metrics_history),
            },
            "cache_performance": cache_stats,
            "system_resources": {
                "total_memory_gb": system_memory.total / 1024 / 1024 / 1024,
                "available_memory_gb": system_memory.available / 1024 / 1024 / 1024,
                "memory_usage_percent": system_memory.percent,
                "cpu_count": psutil.cpu_count(),
            },
            "optimization_recommendations": self._get_optimization_recommendations(),
        }

    def _get_optimization_recommendations(self) -> list:
        """Genera recomendaciones de optimización basadas en métricas."""
        recommendations = []

        cache_stats = self.cache.get_stats()
        if cache_stats["hit_ratio"] < 0.5:
            recommendations.append(
                "Consider increasing cache size for better hit ratio"
            )

        system_memory = psutil.virtual_memory()
        if system_memory.percent > 80:
            recommendations.append("High memory usage detected, consider data chunking")

        if len(self._metrics_history) > 5:
            recent_times = [
                m["metrics"].execution_time for m in self._metrics_history[-5:]
            ]
            if np.std(recent_times) > np.mean(recent_times) * 0.5:
                recommendations.append(
                    "High execution time variance, check for bottlenecks"
                )

        return recommendations if recommendations else ["Performance looks good!"]

    def cleanup(self):
        """Limpia recursos del optimizador."""
        self.cache.clear()
        self.thread_pool.shutdown(wait=True)
        self.memory_monitor.cleanup_if_needed()
        logger.info("Performance optimizer cleaned up")


# Instancia global del optimizador
_optimizer = PerformanceOptimizer()


# Funciones de conveniencia
def cached(func: Callable) -> Callable:
    """Decorador de conveniencia para cachear funciones."""
    return _optimizer.cached_function(func)


def profiled(func: Callable) -> Callable:
    """Decorador de conveniencia para perfilar funciones."""
    return _optimizer.profile_function(func)


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Función de conveniencia para optimizar memoria de DataFrames."""
    return _optimizer.optimize_dataframe_memory(df)


def parallel_process(func: Callable, data: list, **kwargs) -> list:
    """Función de conveniencia para procesamiento paralelo."""
    return _optimizer.parallel_apply(func, data, **kwargs)


def get_performance_report() -> Dict[str, Any]:
    """Función de conveniencia para obtener reporte de rendimiento."""
    return _optimizer.get_performance_report()

def benchmark_pca_operations(sample_data: pd.DataFrame, n_components_list: list = None) -> Dict[str, Any]:
    """Ejecuta benchmarks de operaciones PCA con diferentes tamaños de datos y componentes."""
    if n_components_list is None:
        n_components_list = [2, 5, 10, min(20, len(sample_data.columns))]

    results = {
        "data_shape": sample_data.shape,
        "benchmarks": [],
        "recommendations": []
    }

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import time

    # Preparar datos
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sample_data.select_dtypes(include=[np.number]))

    for n_comp in n_components_list:
        if n_comp >= scaled_data.shape[1]:
            continue

        start_time = time.time()
        memory_start = _optimizer.memory_monitor.get_current_memory()

        pca = PCA(n_components=n_comp)
        pca_result = pca.fit_transform(scaled_data)

        execution_time = time.time() - start_time
        memory_used = _optimizer.memory_monitor.get_current_memory() - memory_start

        explained_variance = sum(pca.explained_variance_ratio_)

        results["benchmarks"].append({
            "n_components": n_comp,
            "execution_time": execution_time,
            "memory_used_mb": memory_used,
            "explained_variance_ratio": explained_variance,
            "cumulative_variance": pca.explained_variance_ratio_.tolist()
        })

        logger.info(f"PCA benchmark: {n_comp} components in {execution_time:.3f}s, "
                   f"memory: {memory_used:.1f}MB, variance explained: {explained_variance:.3f}")

    # Generar recomendaciones
    if results["benchmarks"]:
        best_components = max(results["benchmarks"],
                            key=lambda x: x["explained_variance_ratio"] / x["execution_time"])
        results["recommendations"].append(
            f"Optimal components: {best_components['n_components']} "
            f"(variance: {best_components['explained_variance_ratio']:.3f}, "
            f"time: {best_components['execution_time']:.3f}s)"
        )

    return results


def run_performance_benchmarks() -> Dict[str, Any]:
    """Ejecuta un conjunto completo de benchmarks de rendimiento."""
    logger.info("Running comprehensive performance benchmarks...")

    # Crear datos de prueba
    np.random.seed(42)
    test_data = pd.DataFrame({
        'var1': np.random.randn(1000),
        'var2': np.random.randn(1000),
        'var3': np.random.randn(1000),
        'var4': np.random.randn(1000),
        'var5': np.random.randn(1000),
        'var6': np.random.randn(1000),
        'var7': np.random.randn(1000),
        'var8': np.random.randn(1000),
    })

    # Benchmark PCA
    pca_benchmarks = benchmark_pca_operations(test_data)

    # Benchmark procesamiento de datos
    @profiled
    def data_processing_benchmark():
        df = test_data.copy()
        # Simular procesamiento típico
        df = optimize_memory(df)
        df['new_col'] = df['var1'] * df['var2']
        df['processed'] = df.select_dtypes(include=[np.number]).mean(axis=1)
        return df

    processing_result = data_processing_benchmark()

    # Obtener reporte general
    report = get_performance_report()
    report["pca_benchmarks"] = pca_benchmarks
    report["data_processing_benchmark"] = {
        "input_shape": test_data.shape,
        "output_shape": processing_result.shape,
        "operations": ["memory_optimization", "column_operations", "aggregation"]
    }

    logger.info("Performance benchmarks completed")
    return report


if __name__ == "__main__":
    # Test del sistema de optimización
    print("Testing performance optimization system...")

    # Test de cache
    @cached
    def expensive_calculation(n):
        time.sleep(0.1)  # Simular cálculo costoso
        return n**2

    # Test de profiling
    @profiled
    def test_function():
        df = pd.DataFrame({"A": range(1000), "B": range(1000, 2000)})
        return optimize_memory(df)

    # Ejecutar tests
    print("Testing cache...")
    print(expensive_calculation(5))  # Cache miss
    print(expensive_calculation(5))  # Cache hit

    print("\nTesting profiling...")
    result = test_function()

    print("\nRunning benchmarks...")
    benchmark_report = run_performance_benchmarks()
    import json
    print("Benchmark Report:")
    print(json.dumps(benchmark_report, indent=2, default=str))

    print("\nPerformance Report:")
    report = get_performance_report()
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    # Test del sistema de optimización
    print("Testing performance optimization system...")

    # Test de cache
    @cached
    def expensive_calculation(n):
        time.sleep(0.1)  # Simular cálculo costoso
        return n**2

    # Test de profiling
    @profiled
    def test_function():
        df = pd.DataFrame({"A": range(1000), "B": range(1000, 2000)})
        return optimize_memory(df)

    # Ejecutar tests
    print("Testing cache...")
    print(expensive_calculation(5))  # Cache miss
    print(expensive_calculation(5))  # Cache hit

    print("\nTesting profiling...")
    result = test_function()

    print("\nPerformance Report:")
    import json

    report = get_performance_report()
    print(json.dumps(report, indent=2))
