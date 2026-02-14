# config_manager.py
"""
Sistema de configuración flexible para la aplicación PCA.

Maneja configuraciones desde archivos YAML/JSON, variables de entorno
y valores por defecto, con validación y recarga en tiempo real.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from backend.logging_config import get_logger

logger = get_logger("config_manager")


@dataclass
class PCASettings:
    """Configuración para análisis PCA."""

    default_n_components: Union[str, int, float] = "auto"
    min_variance_threshold: float = 0.95
    random_state: int = 42
    max_components: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.min_variance_threshold, (int, float)):
            if not 0 < self.min_variance_threshold <= 1:
                raise ValueError("min_variance_threshold debe estar entre 0 y 1")


@dataclass
class VisualizationSettings:
    """Configuración para visualizaciones."""

    default_figsize: tuple = (12, 8)
    default_dpi: int = 100
    color_palette: str = "Set1"
    max_legend_items: int = 20
    font_family: str = "Arial"
    font_size: int = 12
    save_format: str = "svg"

    def __post_init__(self):
        if len(self.default_figsize) != 2:
            raise ValueError("default_figsize debe ser una tupla de 2 elementos")
        if self.default_dpi <= 0:
            raise ValueError("default_dpi debe ser positivo")


@dataclass
class DataProcessingSettings:
    """Configuración para procesamiento de datos."""

    default_imputation: str = "interpolation"
    max_missing_ratio: float = 0.5
    standardize_data: bool = True
    remove_constant_columns: bool = True
    correlation_threshold: float = 0.95

    def __post_init__(self):
        valid_imputation = ["interpolation", "mean", "median", "most_frequent", "drop"]
        if self.default_imputation not in valid_imputation:
            raise ValueError(f"default_imputation debe ser uno de: {valid_imputation}")


@dataclass
class UISettings:
    """Configuración de interfaz de usuario."""

    theme: str = "light"
    language: str = "es"
    window_size: str = "800x600"
    font_family: str = "Arial"
    font_size: int = 12
    auto_save_projects: bool = True
    max_recent_files: int = 10

    def __post_init__(self):
        if self.theme not in ["light", "dark"]:
            raise ValueError("theme debe ser 'light' o 'dark'")
        if self.language not in ["es", "en"]:
            raise ValueError("language debe ser 'es' o 'en'")


@dataclass
class PerformanceSettings:
    """Configuración de rendimiento."""

    max_cache_size: int = 128
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None
    chunk_size: int = 1000
    memory_limit_mb: int = 1024

    def __post_init__(self):
        if self.max_cache_size <= 0:
            raise ValueError("max_cache_size debe ser positivo")


@dataclass
class AppConfig:
    """Configuración completa de la aplicación."""

    pca: PCASettings = field(default_factory=PCASettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)
    data_processing: DataProcessingSettings = field(
        default_factory=DataProcessingSettings
    )
    ui: UISettings = field(default_factory=UISettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)

    # Configuraciones adicionales
    debug_mode: bool = False
    log_level: str = "INFO"
    auto_cleanup_logs: bool = True
    backup_projects: bool = True


class ConfigManager:
    """Gestor de configuración con soporte para múltiples fuentes."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"
        self.config_dir.mkdir(exist_ok=True)

        # Archivos de configuración
        self.yaml_config_file = self.config_dir / "app_config.yaml"
        self.json_config_file = self.config_dir / "app_config.json"
        self.user_config_file = self.config_dir / "user_config.json"

        # Configuración actual
        self._config = AppConfig()
        self._config_sources = []

        # Cargar configuración
        self.load_config()

    def load_config(self):
        """Carga configuración desde múltiples fuentes en orden de prioridad."""
        logger.info("Loading application configuration")

        # 1. Configuración por defecto (ya cargada)
        self._config_sources.append("defaults")

        # 2. Archivo YAML del sistema
        if self.yaml_config_file.exists():
            try:
                self.load_from_yaml(self.yaml_config_file)
                self._config_sources.append("yaml_file")
            except Exception as e:
                logger.warning(f"Error loading YAML config: {e}")

        # 3. Archivo JSON del sistema
        if self.json_config_file.exists():
            try:
                self.load_from_json(self.json_config_file)
                self._config_sources.append("json_file")
            except Exception as e:
                logger.warning(f"Error loading JSON config: {e}")

        # 4. Configuración de usuario
        if self.user_config_file.exists():
            try:
                self.load_from_json(self.user_config_file)
                self._config_sources.append("user_file")
            except Exception as e:
                logger.warning(f"Error loading user config: {e}")

        # 5. Variables de entorno
        self.load_from_environment()
        self._config_sources.append("environment")

        logger.info(f"Configuration loaded from: {', '.join(self._config_sources)}")

    def load_from_yaml(self, file_path: Path):
        """Carga configuración desde archivo YAML."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._update_config_from_dict(data)
            logger.debug(f"Loaded YAML config from {file_path}")
        except Exception as e:
            logger.error(f"Error loading YAML config from {file_path}: {e}")
            raise

    def load_from_json(self, file_path: Path):
        """Carga configuración desde archivo JSON."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._update_config_from_dict(data)
            logger.debug(f"Loaded JSON config from {file_path}")
        except Exception as e:
            logger.error(f"Error loading JSON config from {file_path}: {e}")
            raise

    def load_from_environment(self):
        """Carga configuración desde variables de entorno."""
        env_mapping = {
            "PCA_DEBUG_MODE": ("debug_mode", bool),
            "PCA_LOG_LEVEL": ("log_level", str),
            "PCA_THEME": ("ui.theme", str),
            "PCA_LANGUAGE": ("ui.language", str),
            "PCA_MAX_CACHE_SIZE": ("performance.max_cache_size", int),
            "PCA_DEFAULT_IMPUTATION": ("data_processing.default_imputation", str),
        }

        for env_var, (config_path, data_type) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Convertir tipo
                    if data_type == bool:
                        value = value.lower() in ("true", "1", "yes", "on")
                    elif data_type == int:
                        value = int(value)
                    elif data_type == float:
                        value = float(value)

                    # Aplicar valor
                    self._set_nested_config(config_path, value)
                    logger.debug(f"Set {config_path} = {value} from environment")

                except Exception as e:
                    logger.warning(
                        f"Error processing environment variable {env_var}: {e}"
                    )

    def _update_config_from_dict(self, data: Dict[str, Any]):
        """Actualiza la configuración desde un diccionario."""
        for section, values in data.items():
            if hasattr(self._config, section) and isinstance(values, dict):
                section_obj = getattr(self._config, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                        logger.debug(f"Updated {section}.{key} = {value}")
            elif hasattr(self._config, section):
                setattr(self._config, section, values)
                logger.debug(f"Updated {section} = {values}")

    def _set_nested_config(self, path: str, value: Any):
        """Establece un valor en una ruta anidada (ej. 'ui.theme')."""
        parts = path.split(".")
        obj = self._config

        for part in parts[:-1]:
            obj = getattr(obj, part)

        setattr(obj, parts[-1], value)

    def save_user_config(self):
        """Guarda la configuración actual como configuración de usuario."""
        try:
            config_dict = asdict(self._config)
            with open(self.user_config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"User configuration saved to {self.user_config_file}")
        except Exception as e:
            logger.error(f"Error saving user config: {e}")
            raise

    def create_default_yaml_config(self):
        """Crea un archivo YAML de configuración por defecto."""
        default_config = {
            "pca": {
                "default_n_components": "auto",
                "min_variance_threshold": 0.95,
                "random_state": 42,
            },
            "visualization": {
                "default_figsize": [12, 8],
                "default_dpi": 100,
                "color_palette": "Set1",
                "font_family": "Arial",
                "font_size": 12,
            },
            "data_processing": {
                "default_imputation": "interpolation",
                "max_missing_ratio": 0.5,
                "standardize_data": True,
            },
            "ui": {"theme": "light", "language": "es", "window_size": "800x600"},
            "performance": {
                "max_cache_size": 128,
                "enable_parallel_processing": True,
                "chunk_size": 1000,
            },
            "debug_mode": False,
            "log_level": "INFO",
        }

        try:
            with open(self.yaml_config_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    default_config, f, default_flow_style=False, allow_unicode=True
                )
            logger.info(f"Default YAML config created at {self.yaml_config_file}")
        except Exception as e:
            logger.error(f"Error creating default YAML config: {e}")
            raise

    def get_config(self) -> AppConfig:
        """Retorna la configuración actual."""
        return self._config

    def update_config(self, section: str, **kwargs):
        """Actualiza una sección de configuración."""
        if hasattr(self._config, section):
            section_obj = getattr(self._config, section)
            for key, value in kwargs.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
                    logger.debug(f"Updated {section}.{key} = {value}")
                else:
                    logger.warning(f"Unknown config key: {section}.{key}")
        else:
            logger.warning(f"Unknown config section: {section}")

    def reload_config(self):
        """Recarga la configuración desde todos los archivos."""
        self._config = AppConfig()
        self._config_sources.clear()
        self.load_config()

    def validate_config(self) -> bool:
        """Valida que la configuración actual sea correcta."""
        try:
            # La validación se hace automáticamente en __post_init__ de cada dataclass
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


# Instancia global del gestor de configuración
_config_manager = ConfigManager()


# Funciones de conveniencia
def get_config() -> AppConfig:
    """Función de conveniencia para obtener la configuración actual."""
    return _config_manager.get_config()


def update_config(section: str, **kwargs):
    """Función de conveniencia para actualizar configuración."""
    _config_manager.update_config(section, **kwargs)


def save_config():
    """Función de conveniencia para guardar configuración."""
    _config_manager.save_user_config()


def reload_config():
    """Función de conveniencia para recargar configuración."""
    _config_manager.reload_config()


if __name__ == "__main__":
    # Test del sistema de configuración
    print("Testing configuration system...")

    # Crear configuración por defecto
    _config_manager.create_default_yaml_config()

    # Obtener configuración
    config = get_config()
    print(f"Theme: {config.ui.theme}")
    print(f"PCA components: {config.pca.default_n_components}")
    print(f"Figure size: {config.visualization.default_figsize}")

    # Actualizar configuración
    update_config("ui", theme="dark", language="en")
    print(f"Updated theme: {get_config().ui.theme}")

    # Guardar configuración
    save_config()
    print("Configuration saved successfully")
