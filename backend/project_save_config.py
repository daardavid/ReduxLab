# project_save_config.py
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict


@dataclass
class ProjectConfig:
    project_name: str = ""
    # Cada análisis tiene su propia configuración
    series_config: dict = field(
        default_factory=lambda: {
            "data_file": "",
            "selected_indicators": [],
            "selected_units": [],
            "selected_years": [],
            "color_groups": {},
            "group_labels": {},
            "custom_titles": {"biplot": "", "legend": "Grupos", "footer": ""},
            "analysis_results": {},
            "footer_note": "",
        }
    )
    cross_section_config: dict = field(
        default_factory=lambda: {
            "data_file": "",
            "selected_indicators": [],
            "selected_units": [],
            "selected_years": [],
            "color_groups": {},
            "group_labels": {},
            "custom_titles": {"biplot": "", "legend": "Grupos", "footer": ""},
            "analysis_results": {},
            "footer_note": "",
        }
    )
    panel_config: dict = field(
        default_factory=lambda: {
            "data_file": "",
            "selected_indicators": [],
            "selected_units": [],
            "selected_years": [],
            "color_groups": {},
            "group_labels": {},
            "custom_titles": {"biplot": "", "legend": "Grupos", "footer": ""},
            "analysis_results": {},
            "footer_note": "",
        }
    )
    biplot_advanced_config: dict = field(
        default_factory=lambda: {
            "data_file": "",
            "selected_indicators": [],
            "selected_units": [],
            "selected_years": [],
            "color_groups": {},
            "group_labels": {},
            "custom_titles": {"biplot": "", "legend": "Grupos", "footer": ""},
            "analysis_results": {},
            "footer_note": "",
            # Configuraciones específicas del biplot avanzado
            "categorization_scheme": "continents",
            "marker_scheme": "classic",
            "color_scheme": "viridis",
            "show_arrows": True,
            "show_labels": True,
            "alpha": 0.7,
        }
    )
    scatter_plot_config: dict = field(
        default_factory=lambda: {
            "data_file": "",
            "selected_indicators": [],
            "selected_units": [],
            "selected_years": [],
            "pc_x": 1,
            "pc_y": 2,
            "use_cmap": False,
            "cmap": "viridis",
            "density": False,
            "gradient": "",
            "alpha": 0.7,
            "point_size": 30,
            "edgecolor": "None",  # None / auto / color hex
            "HT2": False,
            "SPE": False,
            "show_labels": False,
        }
    )
    correlation_config: dict = field(
        default_factory=lambda: {
            "data_file": "",
            "correlation_method": "pearson",
            "time_aggregated": True,
            "similarity_threshold": 0.3,
            "visualization_type": "heatmap",
            "heatmap_config": {
                "cmap": "coolwarm"
            },
            "network_config": {
                "layout": "spring",
                "node_size": 20
            },
        }
    )

    def to_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "series_config": self.series_config,
            "cross_section_config": self.cross_section_config,
            "panel_config": self.panel_config,
            "biplot_advanced_config": self.biplot_advanced_config,
            "scatter_plot_config": self.scatter_plot_config,
            "correlation_config": self.correlation_config,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectConfig":
        obj = cls()
        obj.project_name = data.get("project_name", "")
        obj.series_config = data.get("series_config", obj.series_config)
        obj.cross_section_config = data.get(
            "cross_section_config", obj.cross_section_config
        )
        obj.panel_config = data.get("panel_config", obj.panel_config)
        obj.biplot_advanced_config = data.get(
            "biplot_advanced_config", obj.biplot_advanced_config
        )
        obj.scatter_plot_config = data.get(
            "scatter_plot_config", obj.scatter_plot_config
        )
        obj.correlation_config = data.get(
            "correlation_config", obj.correlation_config
        )
        return obj

    def save_to_file(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "ProjectConfig":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
