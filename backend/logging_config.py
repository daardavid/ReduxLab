# logging_config.py
"""
Sistema de logging centralizado para la aplicación PCA.

Proporciona configuración unificada de logging con diferentes niveles,
rotación de archivos, operation IDs para trazabilidad y formateo consistente.
"""

import logging
import logging.handlers
import json
import os
import sys
import time
import uuid
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Thread-local operation context
# ---------------------------------------------------------------------------
_operation_context = threading.local()


def get_operation_id() -> str:
    """Return the current operation ID or '-' if none is active."""
    return getattr(_operation_context, "op_id", "-")


@contextmanager
def operation_context(operation_name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager that assigns a unique operation ID to the current thread.

    Usage::

        with operation_context("load-analysis", logger):
            load_data(...)
            run_pca(...)

    Every log message emitted inside the block will contain the same
    ``op_id`` so the full flow can be traced.
    """
    op_id = f"{operation_name}-{uuid.uuid4().hex[:8]}"
    _operation_context.op_id = op_id
    t0 = time.perf_counter()
    if logger:
        logger.info(f"[{op_id}] START {operation_name}")
    try:
        yield op_id
    except Exception:
        if logger:
            elapsed = time.perf_counter() - t0
            logger.error(
                f"[{op_id}] FAIL {operation_name} after {elapsed:.3f}s",
                exc_info=True,
            )
        raise
    finally:
        elapsed = time.perf_counter() - t0
        if logger:
            logger.info(f"[{op_id}] END {operation_name} ({elapsed:.3f}s)")
        _operation_context.op_id = "-"


# ---------------------------------------------------------------------------
# Custom formatter that injects op_id automatically
# ---------------------------------------------------------------------------
class OperationFormatter(logging.Formatter):
    """Logging formatter that prepends the current operation ID."""

    def format(self, record):
        record.op_id = get_operation_id()
        return super().format(record)


# ---------------------------------------------------------------------------
# JSON formatter for structured logging (machine-parseable)
# ---------------------------------------------------------------------------
class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record):
        log_entry = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "op_id": get_operation_id(),
            "func": f"{record.funcName}:{record.lineno}",
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


class LoggingConfig:
    """Configurador centralizado de logging para la aplicación."""

    def __init__(self, app_name: str = "PCA_App"):
        self.app_name = app_name
        self.log_dir = Path(__file__).parent.parent / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # Archivos de log
        self.main_log_file = self.log_dir / f"{app_name}.log"
        self.error_log_file = self.log_dir / f"{app_name}_errors.log"
        self.debug_log_file = self.log_dir / f"{app_name}_debug.log"
        self.json_log_file = self.log_dir / f"{app_name}_structured.jsonl"

        self._loggers = {}

    def setup_logger(
        self,
        name: str,
        level: str = "INFO",
        console_output: bool = True,
        file_output: bool = True,
        max_bytes: int = 5 * 1024 * 1024,  # 5MB
        backup_count: int = 3,
    ) -> logging.Logger:
        """
        Configura un logger con handlers de archivo y consola.

        Args:
            name: Nombre del logger (ej. 'pca_gui', 'data_loader')
            level: Nivel de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            console_output: Si mostrar logs en consola
            file_output: Si guardar logs en archivo
            max_bytes: Tamaño máximo del archivo antes de rotar
            backup_count: Número de archivos de backup a mantener

        Returns:
            Logger configurado
        """
        if name in self._loggers:
            return self._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))

        # Evitar duplicar handlers si ya existe
        if logger.handlers:
            return logger

        # Formatter with operation ID
        formatter = OperationFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - [%(op_id)s] %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Handler de consola
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.WARNING)  # Solo warnings+ en consola
            logger.addHandler(console_handler)

        # Handler de archivo con rotación
        if file_output:
            file_handler = logging.handlers.RotatingFileHandler(
                self.main_log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # Handler especial para errores
            error_handler = logging.handlers.RotatingFileHandler(
                self.error_log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            error_handler.setFormatter(formatter)
            error_handler.setLevel(logging.ERROR)
            logger.addHandler(error_handler)

            # JSON structured log (for automated parsing)
            json_handler = logging.handlers.RotatingFileHandler(
                self.json_log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            json_handler.setFormatter(JSONFormatter())
            logger.addHandler(json_handler)

        self._loggers[name] = logger
        return logger

    def setup_debug_logger(self, name: str) -> logging.Logger:
        """Configura un logger específico para debugging con máximo detalle."""
        debug_logger = logging.getLogger(f"{name}_debug")
        debug_logger.setLevel(logging.DEBUG)

        if debug_logger.handlers:
            return debug_logger

        # Formatter más detallado para debug (includes op_id)
        debug_formatter = OperationFormatter(
            fmt="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - [%(op_id)s] "
            "%(filename)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        debug_handler = logging.handlers.RotatingFileHandler(
            self.debug_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB para debug
            backupCount=2,
            encoding="utf-8",
        )
        debug_handler.setFormatter(debug_formatter)
        debug_logger.addHandler(debug_handler)

        return debug_logger

    def get_logger(self, name: str) -> logging.Logger:
        """Obtiene un logger existente o crea uno con configuración por defecto."""
        if name in self._loggers:
            return self._loggers[name]
        return self.setup_logger(name)

    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Limpia logs antiguos para evitar acumulación excesiva."""
        import time

        current_time = time.time()
        cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)

        for log_file in self.log_dir.glob("*.log*"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    print(f"Removed old log file: {log_file}")
            except Exception as e:
                print(f"Error removing log file {log_file}: {e}")


# Instancia global del configurador
_logging_config = LoggingConfig()


# Funciones de conveniencia
def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Función de conveniencia para obtener un logger configurado."""
    return _logging_config.setup_logger(name, level)


def get_debug_logger(name: str) -> logging.Logger:
    """Función de conveniencia para obtener un logger de debug."""
    return _logging_config.setup_debug_logger(name)


def setup_application_logging(debug_mode: bool = False):
    """Configura el logging para toda la aplicación."""
    level = "DEBUG" if debug_mode else "INFO"

    # Loggers principales
    main_logger = _logging_config.setup_logger("main", level)
    _logging_config.setup_logger("pca_gui", level)
    _logging_config.setup_logger("data_loader", level)
    _logging_config.setup_logger("pca_module", level)
    _logging_config.setup_logger("visualization", level)
    _logging_config.setup_logger("analysis", level)
    _logging_config.setup_logger("pca_logic", level)
    _logging_config.setup_logger("preprocessing", level)

    if debug_mode:
        debug_logger = _logging_config.setup_debug_logger("app_debug")
        debug_logger.info("Debug logging enabled")

    main_logger.info(f"Application logging initialized (level: {level})")
    return main_logger


if __name__ == "__main__":
    # Test del sistema de logging
    print("Testing logging system...")

    # Setup
    logger = setup_application_logging(debug_mode=True)

    # Test diferentes niveles
    logger.debug("This is a debug message")
    logger.info("Application started successfully")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test logger específico
    data_logger = get_logger("data_test")
    data_logger.info("Data processing started")

    print(f"Log files created in: {_logging_config.log_dir}")
    print("Check the log files to verify output.")
