    # visualization_module.py
"""
Módulo de visualización para análisis PCA socioeconómico.

Este módulo proporciona funciones especializadas para crear visualizaciones
de alta calidad para análisis de componentes principales, incluyendo:
- Series de tiempo múltiples
- Biplots 2D interactivos
- Gráficos 3D de trayectorias
- Scree plots para varianza explicada

Todas las funciones están optimizadas para datos socioeconómicos y proporcionan
opciones avanzadas de personalización y exportación.

Autor: David Armando Abreu Rosique
Fecha: 2025
"""
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

# Import seguro de adjustText con fallback
try:
    from adjustText import adjust_text

    ADJUSTTEXT_AVAILABLE = True
except ImportError:
    ADJUSTTEXT_AVAILABLE = False

    def adjust_text(*args, **kwargs):
        """Fallback silencioso cuando adjustText no está disponible."""
        pass


def maximizar_plot() -> None:
    """
    Intenta maximizar la ventana del plot de Matplotlib para diferentes backends.

    Esta función detecta automáticamente el backend gráfico en uso y aplica
    el método de maximización correspondiente. Es compatible con los backends
    más comunes de Matplotlib.

    Backends soportados:
        - Qt5Agg/Qt4Agg: Usa showMaximized()
        - TkAgg: Usa state('zoomed')
        - Otros: Fallback silencioso

    Note:
        Si la maximización automática falla, el gráfico se mostrará en el
        tamaño especificado por el parámetro figsize de la función llamadora.

    Example:
        >>> plt.figure(figsize=(12, 8))
        >>> plt.plot([1, 2, 3], [4, 5, 6])
        >>> maximizar_plot()  # Intenta maximizar antes de mostrar
        >>> plt.show()
    """
    try:
        # Esta función busca el manejador de la figura actual y llama al método
        # de maximización correspondiente al sistema operativo y backend gráfico.
        fig_manager = plt.get_current_fig_manager()
        if hasattr(fig_manager, "window"):
            if hasattr(fig_manager.window, "showMaximized"):
                fig_manager.window.showMaximized()  # Para backend Qt
            elif hasattr(fig_manager.window, "state"):
                fig_manager.window.state("zoomed")  # Para backend Tk
        # Puedes añadir más condiciones para otros backends si es necesario (ej. WXAgg)
    except Exception as e:
        print(
            f"Advertencia: No se pudo maximizar la ventana del plot automáticamente: {e}"
        )
        print("El tamaño del gráfico se controlará con el parámetro 'figsize'.")


def graficar_series_de_tiempo(
    dfs_dict, titulo_general="Visualización de Series de Tiempo"
):
    """
    Grafica múltiples DataFrames de series de tiempo en subplots apilados verticalmente.
    Cada DataFrame en el diccionario se grafica en su propio subplot.
    Se grafican solo las columnas numéricas.
    La leyenda se mueve fuera del gráfico si hay muchos elementos.

    Args:
        dfs_dict (dict): Un diccionario donde las claves son títulos para los subplots
                         y los valores son los DataFrames correspondientes.
        titulo_general (str): Título para toda la figura.
    """
    if not dfs_dict:
        print("No hay DataFrames para graficar.")
        return

    # Filtrar y validar DataFrames válidos
    dfs_validos = {}
    for k, v in dfs_dict.items():
        if v is not None and not v.empty:
            # Verificar que tenga al menos algunas columnas numéricas
            numeric_cols = v.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                dfs_validos[k] = v
            else:
                print(
                    f"Advertencia: DataFrame '{k}' no tiene columnas numéricas, se omitirá."
                )
        else:
            print(f"Advertencia: DataFrame '{k}' está vacío o es None, se omitirá.")

    if not dfs_validos:
        print(
            "Todos los DataFrames proporcionados están vacíos, son None o no tienen datos numéricos."
        )
        return

    num_plots = len(dfs_validos)
    if num_plots == 0:
        print("No hay DataFrames válidos para graficar después del filtrado.")
        return

    print(f"Graficando {num_plots} conjunto(s) de datos: {list(dfs_validos.keys())}")

    # --- MODIFICACIÓN PRINCIPAL AQUÍ ---
    # Apilar subplots verticalmente (1 columna)
    cols_subplot = 1
    rows_subplot = num_plots  # Cada subplot en su propia fila

    # Ajustar el tamaño de la figura: más ancha y altura proporcional al número de subplots
    fig_width = 14  # Aumentado para mejor legibilidad
    height_per_subplot = 6  # Altura para cada subplot
    fig, axes = plt.subplots(
        rows_subplot,
        cols_subplot,
        figsize=(fig_width, height_per_subplot * rows_subplot),
        squeeze=False,
    )  # squeeze=False asegura que axes siempre sea 2D array

    if titulo_general:
        fig.suptitle(titulo_general, fontsize=16, y=0.99)  # Ajustar 'y' si es necesario

    ax_flat = axes.flatten()  # Para iterar fácilmente sobre los ejes

    plot_idx = 0
    for i, (sub_titulo, df) in enumerate(dfs_validos.items()):
        ax = ax_flat[plot_idx]  # Acceder al subplot actual

        if (
            df is None or df.empty
        ):  # Esto ya estaba cubierto por dfs_validos, pero doble check
            if plot_idx < len(ax_flat):
                ax_flat[plot_idx].set_visible(False)
            plot_idx += 1
            continue

        # Seleccionar solo columnas numéricas para graficar
        numeric_cols = df.select_dtypes(include=np.number).columns

        if numeric_cols.empty:
            ax.text(
                0.5,
                0.5,
                "No hay datos numéricos para graficar",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            print(f"Advertencia: DataFrame '{sub_titulo}' no tiene columnas numéricas.")
        else:
            # Verificar que hay datos válidos (no todos NaN)
            df_numeric = df[numeric_cols].dropna(how="all")
            if df_numeric.empty:
                ax.text(
                    0.5,
                    0.5,
                    f'Todos los datos son NaN en "{sub_titulo}"',
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                print(
                    f"Advertencia: Todas las observaciones en '{sub_titulo}' son NaN."
                )
            else:
                # Verificar si es un gráfico de componentes principales para usar colores especiales
                is_pca_components = (
                    "Componentes Principales" in sub_titulo or "PCA" in sub_titulo
                )

                # Graficar cada columna numérica
                if is_pca_components:
                    # Usar colores distintivos para componentes principales
                    colors = [
                        "#1f77b4",
                        "#ff7f0e",
                        "#2ca02c",
                        "#d62728",
                        "#9467bd",
                        "#8c564b",
                        "#e377c2",
                        "#7f7f7f",
                    ]
                    line_styles = ["-", "-", "-", "-", "-", "--", "--", "--"]
                    line_widths = [2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 1.0, 1.0]
                else:
                    colors = plt.cm.tab10(np.linspace(0, 1, len(numeric_cols)))
                    line_styles = ["-"] * len(numeric_cols)
                    line_widths = [1.5] * len(numeric_cols)

                for i, columna in enumerate(numeric_cols):
                    serie = df[columna].dropna()
                    if len(serie) > 0:  # Solo graficar si hay datos válidos
                        # Usar un tamaño de marcador más pequeño si hay muchas líneas/puntos
                        marker_size = 4 if len(numeric_cols) > 10 else 6
                        color = (
                            colors[i % len(colors)]
                            if isinstance(colors, list)
                            else colors[i]
                        )
                        line_style = line_styles[i % len(line_styles)]
                        line_width = line_widths[i % len(line_widths)]

                        ax.plot(
                            serie.index,
                            serie.values,
                            label=columna,
                            marker="o",
                            linestyle=line_style,
                            markersize=marker_size,
                            color=color,
                            alpha=0.9,
                            linewidth=line_width,
                        )
                    else:
                        print(
                            f"  Advertencia: '{columna}' no tiene datos válidos, se omite del gráfico."
                        )

                # Verificar si hay algo que graficar
                if ax.get_lines():
                    # Configurar leyenda y grid
                    if len(numeric_cols) > 8:  # Puedes ajustar este umbral (ej. 8 o 10)
                        ax.legend(
                            loc="center left",
                            bbox_to_anchor=(1.01, 0.5),
                            fontsize="small",
                        )
                    elif (
                        len(numeric_cols) > 0
                    ):  # Si hay columnas pero no tantas para mover la leyenda
                        ax.legend(fontsize="small")

                    ax.grid(True, linestyle=":", alpha=0.7)

                    # Mejorar aspecto para componentes principales
                    if is_pca_components:
                        ax.set_ylabel(
                            "Valor del Componente", fontsize=11, fontweight="bold"
                        )
                        # Título se manejará más abajo
                    else:
                        ax.set_ylabel("Valor", fontsize=10)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f'No hay series válidas para graficar en "{sub_titulo}"',
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

        # Configurar título y etiquetas
        is_pca_components = (
            "Componentes Principales" in sub_titulo or "PCA" in sub_titulo
        )
        if is_pca_components:
            ax.set_title(
                f"{sub_titulo} - Evolución Temporal", fontsize=14, fontweight="bold"
            )
        else:
            ax.set_title(sub_titulo, fontsize=13)

        ax.set_xlabel(
            "Año" if df.index.name == "Año" else str(df.index.name), fontsize=10
        )
        if not is_pca_components:  # Solo si no se estableció arriba
            ax.set_ylabel("Valor", fontsize=10)
        ax.tick_params(
            axis="x", rotation=45
        )  # Rotar etiquetas del eje X si son largas (años)

        plot_idx += 1

    # Ocultar ejes no utilizados si los hubiera (no debería con cols_subplot=1)
    for i in range(plot_idx, len(ax_flat)):
        ax_flat[i].set_visible(False)

    # Ajustar el layout para evitar solapamientos y hacer espacio para suptitle y leyendas externas
    # El parámetro 'right' en subplots_adjust puede necesitar ser menor si las leyendas externas son anchas
    try:
        if any(
            len(df.select_dtypes(include=np.number).columns) > 8
            for df in dfs_validos.values()
            if df is not None and not df.empty
        ):
            fig.subplots_adjust(right=0.80)  # Necesitarás experimentar con este valor
        plt.tight_layout(rect=[0, 0.03, 1, 0.97 if titulo_general else 1])
    except Exception as e_layout:
        print(f"Advertencia al aplicar tight_layout/subplots_adjust: {e_layout}")

    plt.show()


def graficar_componentes_principales_tiempo(
    df_componentes,
    varianza_explicada=None,
    titulo="Evolución de Componentes Principales en el Tiempo",
):
    """
    Crea una visualización especializada para componentes principales en serie de tiempo.

    Args:
        df_componentes (pd.DataFrame): DataFrame con componentes principales (PC1, PC2, etc.) como columnas y años como índice
        varianza_explicada (list, optional): Lista con porcentajes de varianza explicada por cada componente
        titulo (str): Título del gráfico
    """
    if df_componentes is None or df_componentes.empty:
        print("No hay datos de componentes principales para graficar.")
        return

    n_components = df_componentes.shape[1]
    if n_components == 0:
        print("No hay componentes principales para graficar.")
        return

    # Configurar el gráfico
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Colores y estilos distintivos para cada componente
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    line_styles = ["-", "-", "-", "-", "-", "--", "--", "--"]
    line_widths = [3.0, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 1.0]

    # Graficar cada componente principal
    for i, columna in enumerate(df_componentes.columns):
        serie = df_componentes[columna].dropna()
        if len(serie) > 0:
            color = colors[i % len(colors)]
            line_style = line_styles[i % len(line_styles)]
            line_width = line_widths[i % len(line_widths)]

            # Crear etiqueta con información de varianza si está disponible
            if varianza_explicada and i < len(varianza_explicada):
                label = f"{columna} ({varianza_explicada[i]:.1%} var.)"
            else:
                label = columna

            ax.plot(
                serie.index,
                serie.values,
                label=label,
                marker="o",
                linestyle=line_style,
                markersize=6,
                color=color,
                alpha=0.9,
                linewidth=line_width,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=2,
            )

    # Configurar el gráfico
    ax.set_title(titulo, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Año", fontsize=12, fontweight="bold")
    ax.set_ylabel("Valor del Componente Principal", fontsize=12, fontweight="bold")

    # Leyenda mejorada
    ax.legend(loc="best", fontsize=11, frameon=True, fancybox=True, shadow=True)

    # Grid mejorado
    ax.grid(True, linestyle=":", alpha=0.6, linewidth=0.8)

    # Añadir línea horizontal en cero para referencia
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.8)

    # Mejorar aspecto general
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.tick_params(axis="x", rotation=45)

    # Añadir información adicional si está disponible
    if varianza_explicada:
        total_var = sum(varianza_explicada[:n_components])
        info_text = f"Varianza total explicada: {total_var:.1%}"
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    plt.show()


def create_biplot(df_componentes, pca_model, df_estandarizado, title="Biplot PCA", groups=None, group_colors=None, arrow_scale=None):
    """
    Create a biplot visualization for PCA analysis.

    Args:
        df_componentes: DataFrame with PCA components
        pca_model: Fitted PCA model
        df_estandarizado: Standardized data
        title: Plot title
        groups: Dict mapping {unit_name: group_name} or Series/array of group labels
        group_colors: Color mapping {group_name: color}
        arrow_scale: Arrow scaling factor (None=auto, float=manual override)

    Returns:
        matplotlib.figure.Figure: The biplot figure or None if error
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"🎨 Creating biplot: {title}")
        
        fig, ax = plt.subplots(figsize=(12, 8))

        # Convert groups dict to Series if necessary
        if groups is not None and isinstance(groups, dict):
            # Create Series from dict using df_componentes index
            group_series = pd.Series([groups.get(idx, 'Sin Grupo') for idx in df_componentes.index],
                                    index=df_componentes.index)
        elif groups is not None:
            group_series = pd.Series(groups, index=df_componentes.index)
        else:
            group_series = None

        # Plot component scores
        if group_series is not None and group_colors is not None:
            logger.info(f"📊 Plotting with groups: {group_series.unique()}")
            for group_name in group_series.unique():
                mask = group_series == group_name
                color = group_colors.get(group_name, '#808080')  # Gray default
                ax.scatter(df_componentes.loc[mask, 'PC1'], 
                          df_componentes.loc[mask, 'PC2'],
                          c=color, label=group_name, alpha=0.7, s=100,
                          edgecolors='black', linewidth=0.5)
                logger.debug(f"  Group '{group_name}': {mask.sum()} points, color={color}")
        else:
            logger.info("📊 Plotting without groups")
            ax.scatter(df_componentes['PC1'], df_componentes['PC2'], 
                      alpha=0.7, s=100, c='steelblue',
                      edgecolors='black', linewidth=0.5)
        
        # ✅ CRÍTICO: Añadir etiquetas de las unidades junto a los puntos
        logger.info(f"📝 Adding labels for {len(df_componentes)} units")
        for idx in df_componentes.index:
            label_text = str(idx)[:20]  # Truncar etiquetas largas
            x = df_componentes.loc[idx, 'PC1']
            y = df_componentes.loc[idx, 'PC2']
            ax.annotate(label_text,
                        (x, y),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9,
                        alpha=0.9)
        logger.info(f"✅ {len(df_componentes)} labels added successfully")

        # Plot loadings as arrows with configurable scaling
        loadings = pca_model.components_.T
        
        # ✅ NUEVO: Calcular escala automática inteligente o usar manual override
        max_score_range = np.max([np.abs(df_componentes['PC1']), np.abs(df_componentes['PC2'])])
        max_loading_val = np.abs(loadings).max()
        if max_loading_val == 0:
            max_loading_val = 1
        
        if arrow_scale is None:
            # Auto-calculate: vectores al ~30% del rango de puntos
            auto_scale = (max_score_range / max_loading_val) * 0.3
            final_arrow_scale = auto_scale * 3  # Compatibilidad con código anterior
            logger.info(f"🎯 Arrow scale auto-calculated: {final_arrow_scale:.2f}")
        else:
            # Manual override
            final_arrow_scale = (max_score_range / max_loading_val) * arrow_scale
            logger.info(f"🎯 Arrow scale manual: {final_arrow_scale:.2f} (user factor: {arrow_scale})")
        
        logger.info(f"🎯 Adding {len(df_estandarizado.columns)} loading vectors")
        for i, feature in enumerate(df_estandarizado.columns):
            ax.arrow(0, 0, loadings[i, 0]*final_arrow_scale, loadings[i, 1]*final_arrow_scale,
                     head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
            ax.text(loadings[i, 0]*final_arrow_scale*1.07, loadings[i, 1]*final_arrow_scale*1.07, feature,
                    fontsize=10, ha='center', va='center',
                    color='red')        # Configure plot
        ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}% variance)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)

        if group_series is not None and group_colors is not None:
            ax.legend(title="Grupos", loc='best')

        plt.tight_layout()
        logger.info("✅ Biplot created successfully")
        return fig

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Error creating biplot: {e}", exc_info=True)
        # NO retornar None silenciosamente - re-raise para que el caller sepa que falló
        raise RuntimeError(f"Failed to create biplot: {e}") from e


def create_3d_scatter_plot(df_pc_scores, title="PCA 3D Scatter Plot", groups=None, group_colors=None):
    """
    Create a 3D scatter plot for PCA analysis.

    Args:
        df_pc_scores: DataFrame with PC scores (PC1, PC2, PC3)
        title: Plot title
        groups: Group labels for observations
        group_colors: Color mapping for groups

    Returns:
        matplotlib.figure.Figure: The 3D scatter plot figure
    """
    try:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot 3D scatter
        if groups is not None and group_colors is not None:
            for group in groups.unique():
                mask = groups == group
                ax.scatter(df_pc_scores.loc[mask, 'PC1'],
                           df_pc_scores.loc[mask, 'PC2'],
                           df_pc_scores.loc[mask, 'PC3'],
                           c=group_colors.get(group, 'gray'), label=group, alpha=0.7)
        else:
            ax.scatter(df_pc_scores['PC1'], df_pc_scores['PC2'], df_pc_scores['PC3'], alpha=0.7)

        # Configure plot
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(title)

        if groups is not None and group_colors is not None:
            ax.legend()

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error creating 3D scatter plot: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_correlation_network_visualization(similarity_matrix, title="Correlation Network"):
    """
    Create a network visualization for correlation analysis.

    Args:
        similarity_matrix: Correlation/similarity matrix
        title: Plot title

    Returns:
        matplotlib.figure.Figure: The network visualization figure
    """
    try:
        import networkx as nx

        fig, ax = plt.subplots(figsize=(12, 8))

        # Create network from correlation matrix
        G = nx.from_pandas_adjacency(similarity_matrix.abs())

        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw network
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=500, font_size=8, font_weight='bold',
                edge_color='gray', width=1, alpha=0.7, ax=ax)

        ax.set_title(title)
        plt.axis('off')
        plt.tight_layout()
        return fig

    except ImportError:
        print("NetworkX not available for network visualization")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Error creating correlation network: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_correlation_heatmap(similarity_matrix, title="Correlation Heatmap"):
    """
    Create a heatmap visualization for correlation analysis.

    Args:
        similarity_matrix: Correlation/similarity matrix
        title: Plot title

    Returns:
        matplotlib.figure.Figure: The heatmap figure
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation')

        # Configure labels
        ax.set_xticks(range(len(similarity_matrix.columns)))
        ax.set_yticks(range(len(similarity_matrix.index)))
        ax.set_xticklabels(similarity_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(similarity_matrix.index)

        ax.set_title(title)
        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error creating correlation heatmap: {e}")
        import traceback
        traceback.print_exc()
        return None


def graficar_cada_df_en_ventana_separada(dfs_dict, titulo_base_ventana="Análisis para"):
    """
    Crea una figura separada de Matplotlib para cada DataFrame en el diccionario.
    """
    if not dfs_dict:
        print("No hay DataFrames para graficar.")
        return

    dfs_validos = {k: v for k, v in dfs_dict.items() if v is not None and not v.empty}
    if not dfs_validos:
        print("Todos los DataFrames proporcionados están vacíos o son None.")
        return

    for key_df, df_actual in dfs_validos.items():
        plt.figure(figsize=(12, 6))  # Nueva figura para cada DF
        ax = plt.gca()  # Obtener ejes de la figura actual

        numeric_cols = df_actual.select_dtypes(include=np.number).columns

        if numeric_cols.empty:
            ax.text(
                0.5,
                0.5,
                f'No hay datos numéricos en "{key_df}"',
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            for columna in numeric_cols:
                marker_size = 3 if len(numeric_cols) > 10 else 5
                ax.plot(
                    df_actual.index,
                    df_actual[columna],
                    label=columna,
                    marker="o",
                    linestyle="-",
                    markersize=marker_size,
                )

            if len(numeric_cols) > 8:
                ax.legend(
                    loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize="small"
                )
                plt.subplots_adjust(right=0.80)  # Ajuste para leyenda externa
            elif len(numeric_cols) > 0:
                ax.legend(fontsize="small")

            ax.grid(True, linestyle=":", alpha=0.7)

        ax.set_title(f"{titulo_base_ventana}: {key_df}", fontsize=14)
        ax.set_xlabel(
            "Año" if df_actual.index.name == "Año" else str(df_actual.index.name),
            fontsize=10,
        )
        ax.set_ylabel("Valor", fontsize=10)
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Ajuste para el título
        if len(numeric_cols) > 8:  # Re-aplicar ajuste si la leyenda está fuera
            plt.subplots_adjust(right=0.80, top=0.95)

        plt.show()  # Mostrar cada figura individualmente


def graficar_biplot_corte_transversal(
    # --- Datos Esenciales ---
    pca_model,
    df_pc_scores,
    nombres_indicadores_originales,
    # --- Etiquetas y Colores ---
    nombres_indicadores_etiquetas,
    nombres_individuos_etiquetas,
    grupos_individuos=None,
    mapa_de_colores=None,
    # --- Parámetros de Configuración del Gráfico ---
    titulo="Biplot PCA",
    pc_x=0,
    pc_y=1,
    figsize=(18, 14),
    arrow_scale=None,  # ✅ NUEVO: None = auto-calculate, float = manual override
    fontsize_paises=10,
    fontsize_indicadores=11,
    # --- Parámetro para Guardar  ---
    ruta_guardado=None,
    # --- Nota al pie personalizada ---
    footer_note=None,
    legend_title="Grupos",
):
    """
    [OPTIMIZADA] Crea un biplot maximizado, flexible y legible para un análisis de PCA de corte transversal.
    
    **NUEVO (2025-11-10)**: 
    - arrow_scale=None usa cálculo automático inteligente basado en dispersión de datos
    - arrow_scale=float permite override manual para casos especiales
    
    Ajusta automáticamente el tamaño y la legibilidad cuando hay muchas unidades de investigación.
    """
    # --- 1. Verificaciones Iniciales ---
    if pca_model is None or df_pc_scores.empty:
        print("Modelo PCA o scores no disponibles para generar biplot.")
        return
    if pca_model.n_features_in_ != len(nombres_indicadores_originales):
        print("Error: El número de indicadores no coincide con el modelo PCA.")
        return

    # --- 2. Ajuste dinámico del tamaño de figura basado en número de unidades ---
    n_unidades = len(nombres_individuos_etiquetas)

    # Ajustar tamaño de figura para muchas unidades
    if n_unidades > 50:
        # Para muchas unidades, hacer figura mucho más grande
        base_width = max(24, n_unidades * 0.3)  # Aumentar ancho proporcionalmente
        base_height = max(18, n_unidades * 0.25)  # Aumentar alto proporcionalmente
        figsize = (base_width, base_height)
        fontsize_paises = max(6, 12 - (n_unidades // 20))  # Reducir fuente para muchas unidades
        fontsize_indicadores = max(8, 11 - (n_unidades // 25))
        arrow_scale = max(0.5, 0.8 - (n_unidades // 100) * 0.1)  # Reducir escala de flechas
    elif n_unidades > 30:
        figsize = (20, 16)
        fontsize_paises = 8
        fontsize_indicadores = 10
    elif n_unidades > 20:
        figsize = (18, 14)
        fontsize_paises = 9
        fontsize_indicadores = 10

    # --- 3. Preparación de Coordenadas ---
    scores = df_pc_scores.iloc[:, [pc_x, pc_y]].values
    xs = scores[:, 0]
    ys = scores[:, 1]
    loadings = pca_model.components_[[pc_x, pc_y], :].T
    
    # ✅ NUEVO: Cálculo inteligente de arrow_scale
    max_score_range = np.max([np.abs(xs), np.abs(ys)])
    max_loading_val = np.abs(loadings).max()
    if max_loading_val == 0:
        max_loading_val = 1
    
    if arrow_scale is None:
        # Auto-calculate: vectores al ~30-40% del rango de puntos
        auto_scale = (max_score_range / max_loading_val) * 0.35
        scalefactor = auto_scale
        print(f"🎯 Arrow scale auto-calculado: {scalefactor:.2f}")
    else:
        # Manual override
        scalefactor = max_score_range / max_loading_val * arrow_scale
        print(f"🎯 Arrow scale manual: {scalefactor:.2f} (base={arrow_scale})")
    
    loadings_scaled = loadings * scalefactor

    # --- 4. Creación del Gráfico ---
    fig, ax = plt.subplots(figsize=figsize)

    # Ajustar tamaño de puntos basado en número de unidades
    point_size = 60 if n_unidades <= 30 else max(20, 60 - (n_unidades // 10))

    colores_individuos = "gray"
    if grupos_individuos and mapa_de_colores:
        colores_individuos = [
            mapa_de_colores.get(grupo, "gray") for grupo in grupos_individuos
        ]
    ax.scatter(xs, ys, s=point_size, alpha=0.7, c=colores_individuos, zorder=3)

    variable_texts = []
    for i, name in enumerate(nombres_indicadores_etiquetas):
        x_loading, y_loading = loadings_scaled[i, 0], loadings_scaled[i, 1]
        ax.arrow(
            0,
            0,
            x_loading,
            y_loading,
            color="red",
            alpha=0.8,
            head_width=0.08,
            zorder=4,
        )
        text_obj = ax.text(
            x_loading * 1.15,
            y_loading * 1.15,
            name,
            color="maroon",
            ha="center",
            va="center",
            fontsize=fontsize_indicadores,
            zorder=5,
        )
        variable_texts.append(text_obj)

    # --- 5. Ajuste de Etiquetas y Leyenda ---
    # Mostrar todas las etiquetas de unidades de investigación
    if n_unidades > 100:
        # Para muchas unidades, usar fuente muy pequeña pero mostrar todas
        current_fontsize = max(6, fontsize_paises - 3)
        textos_paises = [
            ax.text(xs[i], ys[i], name, fontsize=current_fontsize, alpha=0.8, ha='center', va='center')
            for i, name in enumerate(nombres_individuos_etiquetas)
        ]
    elif n_unidades > 50:
        # Para unidades moderadamente muchas, usar fuente pequeña pero mostrar todas
        current_fontsize = max(7, fontsize_paises - 2)
        textos_paises = [
            ax.text(xs[i], ys[i], name, fontsize=current_fontsize, alpha=0.85, ha='center', va='center')
            for i, name in enumerate(nombres_individuos_etiquetas)
        ]
    elif n_unidades > 30:
        # Para unidades algo numerosas, usar fuente ligeramente más pequeña
        current_fontsize = max(8, fontsize_paises - 1)
        textos_paises = [
            ax.text(xs[i], ys[i], name, fontsize=current_fontsize, alpha=0.9, ha='center', va='center')
            for i, name in enumerate(nombres_individuos_etiquetas)
        ]
    else:
        # Para menos unidades, mostrar todas las etiquetas normalmente
        textos_paises = [
            ax.text(xs[i], ys[i], name, fontsize=fontsize_paises, ha='center', va='center')
            for i, name in enumerate(nombres_individuos_etiquetas)
        ]

    # Aplicar ajuste de texto para evitar solapamientos
    if ADJUSTTEXT_AVAILABLE and textos_paises:
        try:
            adjust_text(
                textos_paises,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6),
                force_text=0.5,  # Fuerza para mover texto
                force_points=0.2,  # Fuerza para alejar de puntos
                expand_text=(1.2, 1.2),  # Expandir área de texto
                expand_points=(1.2, 1.2),  # Expandir área de puntos
            )
        except Exception as e:
            print(f"Warning: adjust_text failed: {e}. Labels may overlap.")

    # Siempre ajustar las etiquetas de variables (indicadores)
    if ADJUSTTEXT_AVAILABLE and variable_texts:
        try:
            adjust_text(variable_texts, force_text=0.8, expand_text=(1.5, 1.5))
        except Exception as e:
            print(f"Warning: adjust_text for variables failed: {e}")

    if grupos_individuos and mapa_de_colores:
        unique_groups_in_plot = sorted(list(set(grupos_individuos)))
        legend_patches = [
            mpatches.Patch(color=mapa_de_colores.get(group, "gray"), label=group)
            for group in unique_groups_in_plot
        ]
        ax.legend(
            handles=legend_patches,
            title=legend_title,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
        )

    # --- 5. Formato Final del Gráfico ---
    pc_x_label = df_pc_scores.columns[pc_x]
    pc_y_label = df_pc_scores.columns[pc_y]
    var_pc_x = pca_model.explained_variance_ratio_[pc_x] * 100
    var_pc_y = pca_model.explained_variance_ratio_[pc_y] * 100
    ax.set_xlabel(f"{pc_x_label} ({var_pc_x:.2f}% varianza explicada)", fontsize=12)
    ax.set_ylabel(f"{pc_y_label} ({var_pc_y:.2f}% varianza explicada)", fontsize=12)
    ax.set_title(titulo, fontsize=15)
    ax.grid(True, linestyle=":", alpha=0.7, zorder=0)
    ax.axhline(0, color="dimgray", linewidth=1.5, linestyle="-", zorder=1)
    ax.axvline(0, color="dimgray", linewidth=1.5, linestyle="-", zorder=1)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])

    # --- Pie de gráfico: solo mostrar si hay leyenda personalizada ---
    if footer_note:
        fig.text(
            0.03,
            0.0007,
            footer_note,
            ha="left",
            va="bottom",
            fontsize=9,
            color="dimgray",
        )

    # --- 7. Añadir información sobre navegación interactiva ---
    if n_unidades > 30:
        # Añadir texto informativo sobre navegación
        info_text = "💡 Use zoom/pan para navegar. Clic derecho + arrastrar para hacer zoom en área."
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=8, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
               verticalalignment='bottom')

    # --- 8. Guardar y Mostrar ---
    # Guardar el gráfico ANTES de mostrarlo
    if ruta_guardado:
        try:
            # Aseguramos que el directorio de guardado exista
            Path(ruta_guardado).parent.mkdir(parents=True, exist_ok=True)
            # Guardamos la figura
            fig.savefig(
                ruta_guardado, format="svg", bbox_inches="tight", pad_inches=0.1
            )
            print(f"\nGráfico SVG guardado exitosamente en: {ruta_guardado}")
        except Exception as e:
            print(f"Error al guardar el gráfico en {ruta_guardado}: {e}")

    # Mostrar el gráfico maximizado con navegación interactiva
    maximizar_plot()

    # Habilitar navegación interactiva (zoom, pan)
    try:
        # Intentar añadir toolbar de navegación si está disponible
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        # La navegación interactiva ya está disponible con maximizar_plot()
    except ImportError:
        pass

    plt.show()


def graficar_trayectorias_3d(
    df_pc_scores,
    pca_model,
    grupos_paises,
    mapa_de_colores,
    titulo="Trayectorias en el espacio de componentes",
):
    """
    Crea un gráfico 3D que muestra las trayectorias de las unidades de investigación a través del tiempo
    en el espacio de los 3 primeros componentes principales.
    Ajusta automáticamente para manejar muchas unidades.
    """
    if df_pc_scores.shape[1] < 3:
        print(
            "Error: Se necesitan al menos 3 componentes principales para un gráfico 3D."
        )
        return

    # Ajustar tamaño de figura basado en número de países
    paises_unicos = df_pc_scores.index.get_level_values("País").unique()
    n_paises = len(paises_unicos)

    if n_paises > 30:
        figsize = (20, 16)
        marker_size_start = 30
        marker_size_end = 60
        fontsize_labels = 7
    elif n_paises > 20:
        figsize = (18, 14)
        marker_size_start = 40
        marker_size_end = 80
        fontsize_labels = 8
    else:
        figsize = (16, 12)
        marker_size_start = 40
        marker_size_end = 100
        fontsize_labels = 9

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")

    for pais_code in paises_unicos:
        datos_pais = df_pc_scores.loc[pais_code]
        if datos_pais.empty:
            continue

        grupo = grupos_paises.get(pais_code, "Otros")
        color = mapa_de_colores.get(grupo, "gray")

        ax.plot(
            datos_pais["PC1"],
            datos_pais["PC2"],
            datos_pais["PC3"],
            color=color,
            alpha=0.6,
            label=f"_line_{pais_code}",
        )

        ax.scatter(
            datos_pais["PC1"].iloc[0],
            datos_pais["PC2"].iloc[0],
            datos_pais["PC3"].iloc[0],
            color=color,
            marker="o",
            s=marker_size_start,
            label=f"_start_{pais_code}",
        )
        ax.scatter(
            datos_pais["PC1"].iloc[-1],
            datos_pais["PC2"].iloc[-1],
            datos_pais["PC3"].iloc[-1],
            color=color,
            marker="^",
            s=marker_size_end,
            label=f"_end_{pais_code}",
        )
        # Añadir etiqueta solo al punto final para no saturar el gráfico
        # Para muchos países, mostrar menos etiquetas
        if n_paises <= 30 or (n_paises > 30 and hash(pais_code) % 3 == 0):  # Mostrar ~1/3 de las etiquetas
            ax.text(
                datos_pais["PC1"].iloc[-1],
                datos_pais["PC2"].iloc[-1],
                datos_pais["PC3"].iloc[-1],
                f" {pais_code}",
                color="black",
                fontsize=fontsize_labels,
            )

    legend_patches = [
        mpatches.Patch(color=color, label=grupo)
        for grupo, color in mapa_de_colores.items()
    ]
    ax.legend(
        handles=legend_patches,
        title="Grupos",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
    )

    var_pc1 = pca_model.explained_variance_ratio_[0] * 100
    var_pc2 = pca_model.explained_variance_ratio_[1] * 100
    var_pc3 = pca_model.explained_variance_ratio_[2] * 100
    ax.set_xlabel(f"\nPC1 ({var_pc1:.2f}%)", fontsize=10)
    ax.set_ylabel(f"\nPC2 ({var_pc2:.2f}%)", fontsize=10)
    ax.set_zlabel(f"\nPC3 ({var_pc3:.2f}%)", fontsize=10)

    ax.set_title(titulo, fontsize=16)
    fig.tight_layout()
    maximizar_plot()
    plt.show()


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN CON VENTANAS EMERGENTES (REVERTIDO)
# ============================================================================
# Estas funciones muestran gráficos en ventanas emergentes interactivas de matplotlib
# con zoom, pan y guardado funcionales. NO retornan figuras para incrustar.

def show_biplot(df_componentes, pca_model, df_estandarizado, title="Biplot PCA", 
                groups=None, group_colors=None, arrow_scale=None):
    """
    Muestra un biplot en una ventana emergente interactiva.
    
    Args:
        df_componentes: DataFrame con componentes PCA
        pca_model: Modelo PCA fitted
        df_estandarizado: Datos estandarizados
        title: Título del gráfico
        groups: Dict {nombre_unidad: nombre_grupo} o Series de etiquetas
        group_colors: Dict {nombre_grupo: color}
        arrow_scale: Arrow scaling factor (None=auto, float=manual override)
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"🎨 Showing biplot in interactive window: {title}")
        
        fig = create_biplot(df_componentes, pca_model, df_estandarizado, 
                          title, groups, group_colors, arrow_scale)
        plt.figure(fig.number)
        plt.show()
        logger.info("✅ Biplot window opened")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Error showing biplot: {e}", exc_info=True)
        raise


def show_3d_scatter_plot(df_pc_scores, title="PCA 3D Scatter Plot",
                         groups=None, group_colors=None):
    """
    Muestra un scatter plot 3D en una ventana emergente interactiva.

    Args:
        df_pc_scores: DataFrame con scores PCA (debe tener PC1, PC2, PC3)
        title: Título del gráfico
        groups: Dict {nombre_unidad: nombre_grupo} o Series de etiquetas
        group_colors: Dict {nombre_grupo: color}
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"🎨 Showing 3D scatter plot in interactive window: {title}")

        fig = create_3d_scatter_plot(df_pc_scores, title, groups, group_colors)

        if fig is not None:
            plt.figure(fig.number)
            plt.show()
            logger.info("✅ 3D scatter window opened")
        else:
            logger.warning("El objeto de figura es None, no se mostrará el gráfico.")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Error showing 3D scatter: {e}", exc_info=True)
        raise


def show_correlation_heatmap(similarity_matrix, title="Correlation Heatmap", **kwargs):
    """
    Muestra un heatmap de correlación en una ventana emergente interactiva.
    
    Args:
        similarity_matrix: Matriz de similitud/correlación (DataFrame)
        title: Título del gráfico
        **kwargs: Argumentos adicionales para create_correlation_heatmap
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"🎨 Showing correlation heatmap in interactive window: {title}")
        
        fig = create_correlation_heatmap(similarity_matrix, title, **kwargs)
        plt.figure(fig.number)
        plt.show()
        logger.info("✅ Heatmap window opened")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Error showing heatmap: {e}", exc_info=True)
        raise


def show_correlation_network(similarity_matrix, title="Correlation Network", **kwargs):
    """
    Muestra una red de correlación en una ventana emergente interactiva.

    Args:
        similarity_matrix: Matriz de similitud/correlación (DataFrame)
        title: Título del gráfico
        **kwargs: Argumentos adicionales para create_correlation_network_visualization
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"🎨 Showing correlation network in interactive window: {title}")

        fig = create_correlation_network_visualization(similarity_matrix, title, **kwargs)
        if fig is not None:
            plt.figure(fig.number)
            plt.show()
            logger.info("✅ Network window opened")
        else:
            logger.warning("El objeto de figura es None, no se mostrará el gráfico.")
            raise RuntimeError("Failed to create correlation network visualization - figure is None")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Error showing network: {e}", exc_info=True)
        raise


def show_series_plot(series_dict, title="Time Series Analysis"):
    """
    Muestra gráficos de series de tiempo en una ventana emergente interactiva.

    Args:
        series_dict: Dict con DataFrames de series temporales
        title: Título general
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"🎨 Showing time series in interactive window: {title}")

        # graficar_series_de_tiempo() no retorna una figura, crea y muestra directamente
        graficar_series_de_tiempo(series_dict, titulo_general=title)
        # No necesitamos plt.figure() ni plt.show() adicionales ya que graficar_series_de_tiempo() los maneja
        logger.info("✅ Series plot window opened")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Error showing series plot: {e}", exc_info=True)
        raise


def show_hierarchical_dendrogram(dataframe, method='ward', metric='euclidean',
                                 title="Hierarchical Clustering Dendrogram",
                                 groups=None, group_colors=None):
    """
    Muestra un dendrograma en una ventana emergente interactiva.
    
    Args:
        dataframe: DataFrame con datos para clustering
        method: Método de enlace ('ward', 'complete', 'average', 'single')
        metric: Métrica de distancia ('euclidean', 'cityblock', 'cosine')
        title: Título del gráfico
        groups: Dict {nombre_unidad: nombre_grupo}
        group_colors: Dict {nombre_grupo: color}
    """
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        
        logger = logging.getLogger(__name__)
        logger.info(f"🎨 Showing dendrogram in interactive window: {title}")
        
        # Calcular matriz de enlace
        linkage_matrix = linkage(dataframe.values, method=method, metric=metric)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Generar dendrograma
        dendro = dendrogram(linkage_matrix, labels=dataframe.index.tolist(), ax=ax,
                          leaf_rotation=90, leaf_font_size=8)
        
        # Colorear las etiquetas de las hojas según los grupos asignados
        if groups and group_colors:
            leaves_order = dendro['leaves']
            original_labels = dataframe.index.tolist()
            
            for i, xtick in enumerate(ax.get_xticklabels()):
                if i < len(leaves_order):
                    original_idx = leaves_order[i]
                    if original_idx < len(original_labels):
                        original_label = original_labels[original_idx]
                        group = groups.get(original_label, '')
                        color = group_colors.get(group, 'black')
                        xtick.set_color(color)
        
        # Agregar leyenda si hay grupos
        if groups and group_colors:
            legend_elements = []
            for group, color in group_colors.items():
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=color, markersize=8, label=group))
            ax.legend(handles=legend_elements, title="Grupos", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.set_title(f'{title}\nMétodo: {method}, Métrica: {metric}')
        ax.set_xlabel('Observaciones')
        ax.set_ylabel('Distancia')
        
        plt.tight_layout()
        plt.show()
        logger.info("✅ Dendrogram window opened")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Error showing dendrogram: {e}", exc_info=True)
        raise
