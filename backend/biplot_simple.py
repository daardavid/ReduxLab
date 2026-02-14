#!/usr/bin/env python3
"""
Módulo simplificado para biplots avanzados usando solo scikit-learn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple
import warnings


def create_advanced_biplot_simple(df, config):
    """
    Función simplificada para crear biplot avanzado usando solo scikit-learn.

    Args:
        df: DataFrame con datos ya procesados
        config: Diccionario con configuración:
            - year: Año del análisis
            - show_arrows: Mostrar flechas de variables
            - show_labels: Mostrar etiquetas de unidades
            - alpha: Transparencia
            - arrow_scale: Factor de escala para vectores (None=auto, float=manual)
            - groups: Diccionario de grupos
            - group_colors: Colores de grupos

    Returns:
        bool: True si se creó exitosamente, False si hubo error
    """
    try:
        print("🚀 Iniciando biplot avanzado simplificado...")

        # Extraer configuración
        year = config.get("year", "2022")
        show_arrows = config.get("show_arrows", True)
        # ✅ SIEMPRE mostrar etiquetas por defecto (unless explicitly disabled)
        show_labels = config.get("show_labels", True)
        if show_labels:
            print("✅ Etiquetas de unidades HABILITADAS")
        else:
            print("⚠️ Etiquetas de unidades DESHABILITADAS")
        alpha = config.get("alpha", 0.7)
        arrow_scale_manual = config.get("arrow_scale", None)  # ✅ NUEVO
        
        # Extraer información de grupos
        groups = config.get("groups", {})
        group_colors = config.get("group_colors", {})
        
        print(f"📊 Grupos disponibles: {list(groups.keys()) if groups else 'Ninguno'}")
        print(f"🎨 Colores de grupos: {list(group_colors.keys()) if group_colors else 'Ninguno'}")

        # Preparar datos
        data = df.copy()
        data = data.dropna()

        if data.empty:
            print("❌ No hay datos disponibles")
            return False

        print(f"📊 Procesando {len(data)} unidades con {len(data.columns)} indicadores")

        # Estandarizar datos
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Aplicar PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data_scaled)

        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 10))

        # Colores para diferentes grupos (simplificado)
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        # Scatter plot de las unidades
        for i, unit in enumerate(data.index):
            # Determinar color según grupo
            if groups and unit in groups:
                group_name = groups[unit]
                if group_colors and group_name in group_colors:
                    color = group_colors[group_name]
                    print(f"🎨 {unit} -> Grupo: {group_name}, Color: {color}")
                else:
                    # Usar color por defecto si no hay color definido para el grupo
                    color_idx = hash(group_name) % len(colors)
                    color = colors[color_idx]
                    print(f"🎨 {unit} -> Grupo: {group_name}, Color por defecto: {color}")
            else:
                # Sin grupo definido, usar color por índice
                color_idx = i % len(colors)
                color = colors[color_idx]
                print(f"🎨 {unit} -> Sin grupo, Color por índice: {color}")
            
            ax.scatter(
                pca_result[i, 0],
                pca_result[i, 1],
                c=color,
                alpha=alpha,
                s=100,
                marker="o",
                edgecolors="black",
                linewidth=0.5,
                label=groups.get(unit, 'Sin Grupo') if groups else None
            )

            # Etiquetas
            if show_labels:
                label_text = unit[:15] + ("..." if len(unit) > 15 else "")
                ax.annotate(
                    label_text,
                    (pca_result[i, 0], pca_result[i, 1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    alpha=0.9,
                    bbox=None
                )
                print(f"📝 Etiqueta añadida: '{label_text}' en posición ({pca_result[i, 0]:.2f}, {pca_result[i, 1]:.2f})")

        # Verificar cuántas etiquetas se añadieron
        if show_labels:
            num_labels = len(data.index)
            print(f"✅ Total de etiquetas añadidas: {num_labels} de {len(data.index)} unidades")

        # Dibujar vectores de variables (arrows)
        if show_arrows:
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            
            # ✅ NUEVO: Cálculo inteligente de arrow_scale
            max_score_range = np.max([np.abs(pca_result[:, 0]), np.abs(pca_result[:, 1])])
            max_loading_val = np.abs(loadings).max()
            if max_loading_val == 0:
                max_loading_val = 1
            
            if arrow_scale_manual is None:
                # Auto-calculate: vectores al ~30% del rango de puntos
                arrow_scale = (max_score_range / max_loading_val) * 0.3
                print(f"🎯 Arrow scale auto-calculado: {arrow_scale:.2f}")
            else:
                # Manual override
                arrow_scale = arrow_scale_manual
                print(f"🎯 Arrow scale manual: {arrow_scale:.2f}")

            for i, var in enumerate(data.columns):
                ax.arrow(
                    0,
                    0,
                    loadings[i, 0] * arrow_scale,
                    loadings[i, 1] * arrow_scale,
                    head_width=0.1,
                    head_length=0.1,
                    fc="red",
                    ec="red",
                    alpha=0.7,
                )
                ax.text(
                    loadings[i, 0] * arrow_scale * 1.07,
                    loadings[i, 1] * arrow_scale * 1.07,
                    var,
                    fontsize=10,
                    ha="center",
                    va="center",
                    color="red"
                )

        # Configurar ejes
        ax.set_xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza explicada)"
        )
        ax.set_ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza explicada)"
        )
        ax.set_title(
            f"Biplot Avanzado - {year}\n"
            f"Varianza total explicada: {sum(pca.explained_variance_ratio_):.1%}"
        )

        # Líneas de referencia
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)

        # Grid
        ax.grid(True, alpha=0.3)
        
        # Crear leyenda de grupos si existen
        if groups and group_colors:
            # Crear handles únicos para la leyenda
            unique_groups = set(groups.values())
            legend_handles = []
            
            for group_name in unique_groups:
                if group_name in group_colors:
                    color = group_colors[group_name]
                else:
                    # Usar color por defecto para grupos sin color definido
                    color_idx = hash(group_name) % len(colors)
                    color = colors[color_idx]
                
                handle = ax.scatter([], [], c=color, s=100, marker="o", 
                                  edgecolors="black", linewidth=0.5, 
                                  label=group_name)
                legend_handles.append(handle)
            
            # Agregar leyenda
            ax.legend(handles=legend_handles, title="Grupos", 
                     loc='upper right', bbox_to_anchor=(1.15, 1))
            print(f"📋 Leyenda creada para {len(unique_groups)} grupos")

        # Mostrar gráfico
        plt.tight_layout()

        # Configurar matplotlib para tkinter
        import matplotlib

        matplotlib.use("TkAgg")

        plt.show()
        print("✅ Biplot avanzado creado exitosamente")
        return True

    except Exception as e:
        print(f"❌ Error al crear biplot: {e}")
        import traceback

        traceback.print_exc()
        return False


def show_advanced_biplot(df, config):
    """
    Wrapper para crear y mostrar biplot simplificado en ventana interactiva.
    Compatible con la arquitectura de analysis_manager.py
    
    Args:
        df: DataFrame con datos ya procesados
        config: Diccionario con configuración del biplot
        
    Raises:
        RuntimeError: Si falla la creación del biplot
    """
    try:
        print("🎨 show_advanced_biplot() llamado desde analysis_manager")
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"⚙️ Config keys: {list(config.keys())}")
        
        # Llamar a la función principal
        success = create_advanced_biplot_simple(df, config)
        
        if not success:
            raise RuntimeError("create_advanced_biplot_simple retornó False - falló la creación del biplot")
        
        print("✅ show_advanced_biplot() completado exitosamente")
        
    except Exception as e:
        error_msg = f"Error al mostrar biplot simplificado: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(error_msg) from e


if __name__ == "__main__":
    # Test básico
    print("🧪 Probando biplot simplificado...")

    # Datos de prueba
    np.random.seed(42)
    data = np.random.randn(8, 4)
    df = pd.DataFrame(
        data,
        index=[f"Unidad_{i}" for i in range(8)],
        columns=[f"Indicador_{i}" for i in range(4)],
    )

    config = {"year": "2023", "show_arrows": True, "show_labels": True, "alpha": 0.7}

    result = create_advanced_biplot_simple(df, config)
    print(f"Resultado: {result}")
