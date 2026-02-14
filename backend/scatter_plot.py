"""scatter_plot.py
Vista utilitaria para generar un scatterplot PCA configurable (PCX vs PCY) reutilizando
el modelo PCA existente si está disponible.
"""

from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.logging_config import get_logger
from backend.pca_module import realizar_pca
from backend.pca_cross_logic import PCAAnalysisLogic

logger = get_logger("scatter_plot")

try:
    from pca import pca as pca_lib

    PCA_ADV = True
except ImportError:
    PCA_ADV = False


def generate_scatter_plot(
    df_standardized: pd.DataFrame, labels, config: Dict[str, Any], existing_model=None
):
    """Genera el scatterplot PCA.

    config keys:
      pc_x, pc_y (1-based)
      use_cmap(bool), cmap(str)
      density(bool), gradient(str/empty), alpha(float), point_size(int)
      edgecolor(str), HT2(bool), SPE(bool)
    """
    if df_standardized is None or df_standardized.empty:
        logger.error("DataFrame vacío para scatter plot.")
        return None

    pc_x = max(1, int(config.get("pc_x", 1)))
    pc_y = max(1, int(config.get("pc_y", 2)))
    alpha = float(config.get("alpha", 0.7))
    point_size = int(config.get("point_size", 30))
    use_cmap = bool(config.get("use_cmap", False))
    cmap = config.get("cmap", "viridis")
    density = bool(config.get("density", False))
    gradient = config.get("gradient") or None
    edgecolor = config.get("edgecolor", "None")
    edgecolor = None if edgecolor in ("None", "none", "") else edgecolor
    HT2 = bool(config.get("HT2", False))
    SPE = bool(config.get("SPE", False))
    show_labels = bool(config.get("show_labels", False))

    # Usar modelo existente o recalcular
    model = existing_model
    scores_df = None

    if model is None:
        model, scores_df = realizar_pca(df_standardized, n_components=None)
        if model is None or scores_df is None:
            logger.error("No se pudo generar PCA para scatter plot.")
            return None
    else:
        # Reconstruir scores con transform si no se proporcionaron
        try:
            transformed = model.transform(df_standardized)
            comps = [f"PC{i+1}" for i in range(transformed.shape[1])]
            scores_df = pd.DataFrame(
                transformed, index=df_standardized.index, columns=comps
            )
        except Exception as e:
            logger.warning(
                f"Fallo al transformar con modelo existente: {e}, recalculando."
            )
            model, scores_df = realizar_pca(df_standardized, n_components=None)
            if model is None:
                return None

    max_pc = scores_df.shape[1]
    if pc_x > max_pc or pc_y > max_pc:
        logger.error("PC solicitada excede número de componentes.")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    x_vals = scores_df.iloc[:, pc_x - 1]
    y_vals = scores_df.iloc[:, pc_y - 1]

    colors = "#1f77b4"
    if use_cmap:
        cmap_obj = plt.get_cmap(cmap)
        colors = cmap_obj(np.linspace(0, 1, len(scores_df)))

    # Scatter básico
    for i, (idx, xv, yv) in enumerate(zip(scores_df.index, x_vals, y_vals)):
        c = colors[i] if isinstance(colors, np.ndarray) else colors
        ax.scatter(
            xv, yv, s=point_size, alpha=alpha, c=[c], edgecolor=edgecolor, linewidth=0.5
        )

    if gradient and isinstance(colors, np.ndarray):
        # Aplicar un overlay de gradiente simple (linea color variable)
        ax.scatter(
            x_vals, y_vals, s=point_size * 0.3, c=x_vals, cmap=cmap, alpha=alpha * 0.4
        )

    if density and len(scores_df) > 10:
        try:
            from scipy.stats import gaussian_kde

            xy = np.vstack([x_vals, y_vals])
            z = gaussian_kde(xy)(xy)
            sc = ax.scatter(x_vals, y_vals, c=z, s=point_size, cmap="Reds", alpha=0.5)
            plt.colorbar(sc, ax=ax, label="Densidad")
        except Exception as e:
            logger.warning(f"Densidad no disponible: {e}")

    # Dibujar elipses / SPE solo si usamos la librería 'pca' avanzada
    if PCA_ADV:
        if HT2:
            try:
                model.scatter(
                    PC=[pc_x - 1, pc_y - 1], legend=False, SPE=False, HT2=True
                )
            except Exception as e:
                logger.warning(f"Hotelling T2 no dibujado: {e}")
        if SPE:
            try:
                model.scatter(
                    PC=[pc_x - 1, pc_y - 1], legend=False, SPE=True, HT2=False
                )
            except Exception as e:
                logger.warning(f"SPE no dibujado: {e}")
    else:
        if HT2 or SPE:
            logger.warning(
                "HT2/SPE solicitados pero la librería 'pca' no está instalada."
            )

    # Etiquetas sobre puntos
    if show_labels:
        try:
            for i, (idx, xv, yv) in enumerate(zip(scores_df.index, x_vals, y_vals)):
                label_txt = str(labels[i]) if i < len(labels) else str(idx)
                ax.text(
                    xv, yv, label_txt, fontsize=8, ha="center", va="bottom", alpha=0.85
                )
        except Exception as e:
            logger.warning(f"No se pudieron dibujar etiquetas: {e}")

    # Varianza explicada
    explained = getattr(model, "explained_variance_ratio_", None)
    var_x = var_y = None
    if explained is not None and len(explained) >= max(pc_x, pc_y):
        var_x = explained[pc_x - 1] * 100
        var_y = explained[pc_y - 1] * 100
    xlabel = f"PC{pc_x}" + (f" ({var_x:.1f}% var)" if var_x is not None else "")
    ylabel = f"PC{pc_y}" + (f" ({var_y:.1f}% var)" if var_y is not None else "")
    title = f"Scatterplot PCA (PC{pc_x} vs PC{pc_y})"
    if var_x is not None and var_y is not None:
        title += f"  –  Varianza total: {var_x+var_y:.1f}%"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


# ✅ NUEVO: Wrapper para mostrar scatterplot con plt.show() (ventana interactiva)
def show_scatter_plot(df_standardized: pd.DataFrame, labels, config: Dict[str, Any], existing_model=None):
    """
    Wrapper para generar y mostrar scatterplot PCA en ventana interactiva.
    
    Args:
        df_standardized: DataFrame con datos estandarizados
        labels: Etiquetas para los puntos
        config: Configuración del scatterplot
        existing_model: Modelo PCA existente (opcional)
        
    Raises:
        RuntimeError: Si falla la generación del scatterplot
    """
    try:
        fig = generate_scatter_plot(df_standardized, labels, config, existing_model)
        if fig is None:
            raise RuntimeError("generate_scatter_plot retornó None - falló la creación del scatterplot")
        # La función generate_scatter_plot ya llama a plt.show(), así que no es necesario llamarlo de nuevo
    except Exception as e:
        raise RuntimeError(f"Error al mostrar scatterplot: {str(e)}") from e
