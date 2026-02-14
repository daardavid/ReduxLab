# biplot_advanced.py
"""
Módulo para crear biplots avanzados con marcadores personalizados.

Utiliza la librería 'pca' para crear visualizaciones sofisticadas con:
- Marcadores únicos por país/grupo
- Configuración de colores por categorías
- Leyendas personalizadas
- Etiquetas de variables (loadings)

**NOTA SOBRE ESCALADO DE VECTORES (2025-11-10)**:
Este módulo usa la librería externa 'pca' que maneja el escalado de vectores
internamente. Para datos con alta dispersión (común en datos financieros), 
se recomienda aplicar transformaciones ANTES del PCA usando el nuevo módulo
data_transformations.py (ver preprocessing_module.preprocess_data).

Para control manual del escalado de vectores, usar biplot_simple.py o
visualization_module.py que permiten arrow_scale configurable.

Autor: David Armando Abreu Rosique
Fecha: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple
import warnings
from matplotlib.patches import FancyArrowPatch
import os
import datetime

# Importar la librería PCA con manejo de errores (lazy loading)
PCA_LIBRARY_AVAILABLE = None


def _get_pca():
    """Importa la librería pca de manera lazy."""
    global PCA_LIBRARY_AVAILABLE
    if PCA_LIBRARY_AVAILABLE is None:
        try:
            from pca import pca

            PCA_LIBRARY_AVAILABLE = True
            print("✅ Librería 'pca' importada exitosamente")
            return pca
        except ImportError as e:
            PCA_LIBRARY_AVAILABLE = False
            print(f"❌ Error al importar librería 'pca': {e}")
            warnings.warn(
                "La librería 'pca' no está disponible. Instala con: pip install pca"
            )
            return None
    elif PCA_LIBRARY_AVAILABLE:
        from pca import pca

        return pca
    else:
        return None


# Inicializar la librería al cargar el módulo
pca_lib = _get_pca()

# Importar módulos locales (con manejo de errores)
try:
    from logging_config import get_logger

    logger = get_logger("biplot_advanced")
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("biplot_advanced")

try:
    from performance_optimizer import cached, profiled
except ImportError:
    # Si no está disponible, usar decoradores dummy
    def cached(func):
        return func

    def profiled(func):
        return func


# Diccionario de marcadores disponibles
AVAILABLE_MARKERS = {
    "punto": ".",
    "circulo": "o",
    "triangulo_arriba": "^",
    "triangulo_abajo": "v",
    "triangulo_izq": "<",
    "triangulo_der": ">",
    "cuadrado": "s",
    "pentagono": "p",
    "estrella": "*",
    "hexagono": "h",
    "hexagono_alt": "H",
    "diamante": "D",
    "diamante_delgado": "d",
    "plus": "P",
    "x": "X",
    "octagon": "8",
}

"""NOTA SOBRE ESQUEMAS:
Se manejan alias bilingües para permitir que la GUI use claves en español.
Internamente se normalizan a inglés para reutilizar mapas de marcadores.
"""

# Esquemas de colores predefinidos (claves en español conservadas para retrocompatibilidad)
COLOR_SCHEMES = {
    "continentes": {
        "América": "#1f77b4",
        "Europa": "#ff7f0e",
        "Asia": "#2ca02c",
        "África": "#d62728",
        "Oceanía": "#9467bd",
    },
    "desarrollo": {
        "Desarrollado": "#2ca02c",
        "En desarrollo": "#ff7f0e",
        "Menos desarrollado": "#d62728",
    },
    "ingreso": {
        "Alto": "#2ca02c",
        "Medio-alto": "#17becf",
        "Medio-bajo": "#ff7f0e",
        "Bajo": "#d62728",
    },
}

# Alias de esquemas (GUI -> interno)
SCHEME_ALIAS = {
    "continentes": "continents",
    "desarrollo": "development",
    "ingreso": "income",
    # Identidades (si ya viene en inglés no cambia)
    "continents": "continents",
    "development": "development",
    "income": "income",
}

# Esquemas de marcadores por categorías
MARKER_SCHEMES = {
    "classic": ["o", "s", "^", "D", "*", "p", "h"],  # Clásico
    "geometric": ["D", "p", "*", "h", "H", "d", "+"],  # Geométrico
    "varied": ["o", "^", "s", "D", "*", "p", "h", "v", "<", ">"],  # Variado
}

# Esquemas de marcadores por categorías específicas
CATEGORY_MARKER_SCHEMES = {
    "continents": {
        "América del Norte": "^",
        "América del Sur": "v",
        "Europa": "o",
        "Asia": "s",
        "África": "D",
        "Oceanía": "*",
        "Otros": "+",
    },
    "development": {"Desarrollados": "o", "Emergentes": "s", "En Desarrollo": "^"},
    "income": {
        "Ingresos Altos": "o",
        "Ingresos Medios-Altos": "s",
        "Ingresos Medios-Bajos": "^",
        "Ingresos Bajos": "D",
    },
}

# Duplicar claves en español para marcadores (apuntan a los mismos dicts)
CATEGORY_MARKER_SCHEMES["continentes"] = CATEGORY_MARKER_SCHEMES["continents"]
CATEGORY_MARKER_SCHEMES["desarrollo"] = CATEGORY_MARKER_SCHEMES["development"]
CATEGORY_MARKER_SCHEMES["ingreso"] = CATEGORY_MARKER_SCHEMES["income"]


def create_country_categorization(
    countries: List[str], scheme: str = "continentes"
) -> Dict[str, str]:
    """
    Crea una categorización automática de países según el esquema especificado.

    Args:
        countries: Lista de nombres de países
        scheme: Esquema de categorización ('continentes', 'desarrollo', 'ingreso')

    Returns:
        Diccionario {país: categoría}
    """

    # Mapeo básico por continentes (expandir según necesidades)
    continents_map = {
        # América
        "Argentina": "América",
        "Brasil": "América",
        "Chile": "América",
        "Colombia": "América",
        "México": "América",
        "Perú": "América",
        "Venezuela": "América",
        "Ecuador": "América",
        "Uruguay": "América",
        "Paraguay": "América",
        "Bolivia": "América",
        "Costa Rica": "América",
        "Panamá": "América",
        "Guatemala": "América",
        "Honduras": "América",
        "Nicaragua": "América",
        "El Salvador": "América",
        "Estados Unidos": "América",
        "Canadá": "América",
        "United States": "América",
        "Canada": "América",
        "USA": "América",
        # Europa
        "España": "Europa",
        "Francia": "Europa",
        "Alemania": "Europa",
        "Italia": "Europa",
        "Reino Unido": "Europa",
        "Portugal": "Europa",
        "Países Bajos": "Europa",
        "Bélgica": "Europa",
        "Suiza": "Europa",
        "Austria": "Europa",
        "Suecia": "Europa",
        "Noruega": "Europa",
        "Dinamarca": "Europa",
        "Finlandia": "Europa",
        "Polonia": "Europa",
        "República Checa": "Europa",
        "Germany": "Europa",
        "France": "Europa",
        "Spain": "Europa",
        "Italy": "Europa",
        "United Kingdom": "Europa",
        "Netherlands": "Europa",
        "Sweden": "Europa",
        # Asia
        "China": "Asia",
        "Japón": "Asia",
        "India": "Asia",
        "Corea del Sur": "Asia",
        "Indonesia": "Asia",
        "Tailandia": "Asia",
        "Filipinas": "Asia",
        "Vietnam": "Asia",
        "Malasia": "Asia",
        "Singapur": "Asia",
        "Hong Kong": "Asia",
        "Taiwán": "Asia",
        "Japan": "Asia",
        "South Korea": "Asia",
        "Thailand": "Asia",
        "Philippines": "Asia",
        "Malaysia": "Asia",
        "Singapore": "Asia",
        "Taiwan": "Asia",
        # África
        "Sudáfrica": "África",
        "Nigeria": "África",
        "Egipto": "África",
        "Marruecos": "África",
        "Túnez": "África",
        "Argelia": "África",
        "Kenia": "África",
        "Ghana": "África",
        "South Africa": "África",
        "Egypt": "África",
        "Morocco": "África",
        "Tunisia": "África",
        "Algeria": "África",
        "Kenya": "África",
        # Oceanía
        "Australia": "Oceanía",
        "Nueva Zelanda": "Oceanía",
        "Nueva Guinea": "Oceanía",
        "New Zealand": "Oceanía",
        "Papua New Guinea": "Oceanía",
    }

    # Mapeo por nivel de desarrollo
    development_map = {
        # Desarrollados
        "Estados Unidos": "Desarrollado",
        "Canadá": "Desarrollado",
        "Alemania": "Desarrollado",
        "Francia": "Desarrollado",
        "Reino Unido": "Desarrollado",
        "Japón": "Desarrollado",
        "Australia": "Desarrollado",
        "Suiza": "Desarrollado",
        "Suecia": "Desarrollado",
        "Noruega": "Desarrollado",
        "Dinamarca": "Desarrollado",
        "Países Bajos": "Desarrollado",
        "United States": "Desarrollado",
        "Canada": "Desarrollado",
        "Germany": "Desarrollado",
        "France": "Desarrollado",
        "United Kingdom": "Desarrollado",
        "Japan": "Desarrollado",
        "Switzerland": "Desarrollado",
        "Sweden": "Desarrollado",
        "Norway": "Desarrollado",
        "Denmark": "Desarrollado",
        "Netherlands": "Desarrollado",
        # En desarrollo
        "Brasil": "En desarrollo",
        "México": "En desarrollo",
        "Argentina": "En desarrollo",
        "Chile": "En desarrollo",
        "China": "En desarrollo",
        "India": "En desarrollo",
        "Rusia": "En desarrollo",
        "Sudáfrica": "En desarrollo",
        "Turquía": "En desarrollo",
        "Brazil": "En desarrollo",
        "Mexico": "En desarrollo",
        "Argentina": "En desarrollo",
        "Chile": "En desarrollo",
        "China": "En desarrollo",
        "India": "En desarrollo",
        "Russia": "En desarrollo",
        "South Africa": "En desarrollo",
        "Turkey": "En desarrollo",
        # Menos desarrollados (por defecto para países no categorizados)
    }

    categorization = {}

    for country in countries:
        if scheme == "continentes":
            categorization[country] = continents_map.get(country, "Otros")
        elif scheme == "desarrollo":
            categorization[country] = development_map.get(country, "Menos desarrollado")
        elif scheme == "ingreso":
            # Simplificado - en una implementación real usarías datos del Banco Mundial
            if country in [
                "Estados Unidos",
                "Alemania",
                "Francia",
                "Japón",
                "Australia",
                "Suiza",
                "United States",
                "Germany",
                "France",
                "Japan",
                "Switzerland",
            ]:
                categorization[country] = "Alto"
            elif country in [
                "Brasil",
                "México",
                "Argentina",
                "Chile",
                "China",
                "Rusia",
                "Brazil",
                "Mexico",
                "Argentina",
                "Chile",
                "China",
                "Russia",
            ]:
                categorization[country] = "Medio-alto"
            else:
                categorization[country] = "Medio-bajo"
        else:
            categorization[country] = "Default"

    return categorization


def create_advanced_biplot(df, config):
    """
    Función principal para crear biplot avanzado desde la GUI.

    Args:
        df: DataFrame con datos ya procesados para el año específico
        config: Diccionario con configuración de la GUI

    Returns:
        bool: True si se creó exitosamente, False si hubo error
    """
    try:
        year = config.get("year", "2022")
        categorization_scheme = config.get("categorization_scheme", "continents")
        marker_scheme = config.get("marker_scheme", "classic")
        color_scheme = config.get("color_scheme", "viridis")
        show_arrows = config.get("show_arrows", True)
        show_labels = config.get("show_labels", True)
        alpha = config.get("alpha", 0.7)
        custom_categories = config.get("custom_categories")  # nuevo
        
        # Extraer información de grupos de la configuración de análisis
        groups = config.get("groups", {})
        group_colors = config.get("group_colors", {})
        
        # Si hay grupos configurados, usarlos en lugar del esquema de categorización automático
        if groups and group_colors:
            print(f"🏷️ Usando grupos configurados: {list(set(groups.values()))}")
            custom_categories = groups

        # El DataFrame ya viene procesado para el año específico
        year_data = df.copy()
        year_data = year_data.dropna()

        if year_data.empty:
            print(f"❌ No hay datos disponibles para el análisis.")
            return False

        print(
            f"📊 Creando biplot para {len(year_data)} unidades de investigación en {year}"
        )
        print(
            f"🎨 Configuración: {categorization_scheme}, {marker_scheme}, {color_scheme}"
        )

        # Crear biplot avanzado
        fig, ax = create_advanced_biplot_core(
            df_standardized=year_data,
            categorization_scheme=categorization_scheme,
            marker_scheme=marker_scheme,
            color_scheme=color_scheme,
            custom_categories=custom_categories,
            custom_colors=group_colors if groups and group_colors else None,
            show_arrows=show_arrows,
            show_labels=show_labels,
            alpha=alpha,
            title=f"Biplot Avanzado - {year}",
        )

        if fig is not None:
            print("✅ Biplot creado exitosamente")
            plt.tight_layout()

            # Crear nombre de archivo único con timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"biplot_avanzado_{year}_{timestamp}.png"

            try:
                # Guardar la figura
                fig.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"✅ Gráfico guardado como {filename}")

                # Abrir el archivo con el visor de imágenes predeterminado del sistema
                try:
                    os.startfile(filename)  # Para Windows
                    print("✅ Gráfico abierto en el visor de imágenes")
                    return True
                except AttributeError:
                    # Si no es Windows, intentar otros métodos
                    try:
                        os.system(f"start {filename}")  # Alternativa para Windows
                        print("✅ Gráfico abierto en el visor de imágenes (método alternativo)")
                        return True
                    except Exception as e3:
                        print(f"⚠️ No se pudo abrir automáticamente el gráfico: {e3}")
                        print(f"📁 El archivo se guardó en: {os.path.abspath(filename)}")
                        return True
                except Exception as e3:
                    print(f"⚠️ No se pudo abrir automáticamente el gráfico: {e3}")
                    print(f"📁 El archivo se guardó en: {os.path.abspath(filename)}")
                    return True

            except Exception as e:
                print(f"❌ Error al guardar gráfico: {e}")
                return False
        else:
            print("❌ Error: No se pudo crear el gráfico")
            return False

    except Exception as e:
        print(f"❌ Error al crear biplot avanzado: {e}")
        import traceback

        traceback.print_exc()
        return False


@profiled
@cached
def create_advanced_biplot_core(
    df_standardized: pd.DataFrame,
    countries: Optional[List[str]] = None,
    categorization_scheme: str = "continents",
    marker_scheme: str = "classic",
    color_scheme: str = "viridis",
    custom_categories: Optional[Dict[str, str]] = None,
    custom_colors: Optional[Dict[str, str]] = None,
    n_components: int = 2,
    n_features_show: int = 5,
    title: str = "Biplot Avanzado - Análisis PCA",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    show_arrows: bool = True,
    show_labels: bool = True,
    alpha: float = 0.7,
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Crea un biplot avanzado con marcadores y colores personalizados por categorías.

    Args:
        df_standardized: DataFrame con datos estandarizados (países como filas, indicadores como columnas)
        countries: Lista de países a incluir (si None, usa todos)
        categorization_scheme: Esquema de categorización ('continents', 'development', 'income')
        marker_scheme: Esquema de marcadores ('classic', 'geometric', 'varied')
        color_scheme: Esquema de colores ('viridis', 'plasma', 'tab10', 'set3')
        custom_categories: Diccionario personalizado {país: categoría}
        n_components: Número de componentes principales a usar
        n_features_show: Número de features (vectores) a mostrar
        title: Título del gráfico
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la imagen (opcional)
        show_arrows: Si mostrar flechas de variables
        show_labels: Si mostrar etiquetas de países
        alpha: Transparencia de los marcadores

    Returns:
        Tupla (figura, ejes) de matplotlib
    """

    if not PCA_LIBRARY_AVAILABLE:
        logger.error(
            "La librería 'pca' no está disponible. No se puede crear biplot avanzado."
        )
        print("❌ Para usar biplots avanzados, instala: pip install pca")
        return None, None

    if df_standardized is None or df_standardized.empty:
        logger.error("DataFrame vacío o None para biplot avanzado")
        return None, None

    try:
        # Preparar datos (verificar posible no estandarización)
        if countries is None:
            countries = df_standardized.index.tolist()

        df_filtered = df_standardized.loc[df_standardized.index.isin(countries)].copy()
        if df_filtered.empty:
            logger.error("No hay datos después del filtrado por países")
            return None, None

        try:
            mean_std = df_filtered.std().mean()
            if mean_std > 5:
                print(
                    f"⚠️ Aviso: La desviación estándar media ({mean_std:.2f}) sugiere que los datos podrían no estar estandarizados."
                )
        except Exception:
            pass

        print(
            f"📊 Creando biplot avanzado con {len(df_filtered)} países y {len(df_filtered.columns)} indicadores"
        )

        normalized_scheme = SCHEME_ALIAS.get(
            categorization_scheme, categorization_scheme
        )
        if custom_categories:
            categories = custom_categories
        else:
            categories = {}
            categorization_func = {
                "continents": categorize_by_continent,
                "development": categorize_by_development,
                "income": categorize_by_income,
            }.get(normalized_scheme, categorize_by_continent)
            for country in df_filtered.index:
                categories[country] = categorization_func(country)

        unique_categories = sorted(set(categories.values()))
        n_categories = len(unique_categories)

        print(f"📋 Categorías encontradas: {unique_categories}")

        # Obtener esquemas de colores y marcadores
        # Usar colores personalizados si están disponibles
        if custom_colors:
            print(f"🎨 Usando colores personalizados de grupos: {custom_colors}")
            color_map = custom_colors.copy()
            # Agregar colores por defecto para categorías no definidas
            for cat in unique_categories:
                if cat not in color_map:
                    color_map[cat] = '#808080'  # Gris por defecto
        # Determinar mapa de colores (priorizar selección de colormap del usuario si existe)
        elif color_scheme in plt.colormaps():
            cmap = plt.get_cmap(color_scheme)
            colors_auto = cmap(np.linspace(0, 1, n_categories))
            color_map = {cat: colors_auto[i] for i, cat in enumerate(unique_categories)}
        elif categorization_scheme in COLOR_SCHEMES:
            color_map = COLOR_SCHEMES[categorization_scheme]
        else:
            colors_auto = plt.cm.Set1(np.linspace(0, 1, n_categories))
            color_map = {cat: colors_auto[i] for i, cat in enumerate(unique_categories)}

        if categorization_scheme in CATEGORY_MARKER_SCHEMES:
            marker_map = CATEGORY_MARKER_SCHEMES[categorization_scheme]
        else:
            # Generar marcadores automáticamente usando el esquema seleccionado
            available_markers = MARKER_SCHEMES.get(
                marker_scheme, MARKER_SCHEMES["classic"]
            )
            marker_map = {
                cat: available_markers[i % len(available_markers)]
                for i, cat in enumerate(unique_categories)
            }

        # Crear arrays para cada observación
        colors_array = []
        markers_array = []

        for country in df_filtered.index:
            cat = categories.get(country, "Default")
            colors_array.append(color_map.get(cat, "#000000"))
            markers_array.append(marker_map.get(cat, "o"))

        # Convertir a numpy arrays
        colors_array = np.array(colors_array)
        markers_array = np.array(markers_array)

        print(
            f"✅ Configuración completada: {len(np.unique(markers_array))} marcadores únicos, {len(np.unique(colors_array))} colores únicos"
        )

        # Obtener librería pca
        pca_lib = _get_pca()
        if pca_lib is None:
            print("❌ La librería 'pca' no está disponible.")
            return None, None

        # Inicializar modelo PCA
        model = pca_lib(n_components=n_components, verbose=False)

        # Ajustar modelo
        results = model.fit_transform(df_filtered.values)

        # Preparar labels array (siempre, incluso si no se van a mostrar)
        labels_array = np.array(df_filtered.index.tolist())
        
        # Crear biplot con marcadores personalizados
        # NOTA: La librería 'pca' a veces ignora el parámetro labels=None
        # Por eso siempre pasamos las etiquetas y las removemos después si es necesario
        fig, ax = model.biplot(
            c=colors_array,
            marker=markers_array,
            title=title,
            n_feat=n_features_show,
            legend=False,
            figsize=figsize,
            labels=labels_array,  # Siempre pasar etiquetas
        )

        # Crear leyenda personalizada
        legend_elements = []
        for cat in unique_categories:
            color = color_map.get(cat, "#000000")
            marker = marker_map.get(cat, "o")
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=cat,
                    linestyle="None",
                )
            )

        # Añadir leyenda
        ax.legend(
            handles=legend_elements,
            loc="best",
            title=f"Categorización: {categorization_scheme.title()}",
            title_fontsize=12,
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        # Opcional: ocultar flechas (intento heurístico) si show_arrows es False
        if not show_arrows:
            removed = 0
            for patch in list(ax.patches):
                if isinstance(patch, FancyArrowPatch):
                    patch.remove()
                    removed += 1
            if removed:
                print(f"🧹 Flechas removidas: {removed}")

        # Gestión de etiquetas de puntos (nombres de unidades/países)
        idx_set = set(df_filtered.index.astype(str))
        
        if show_labels:
            # ASEGURAR que las etiquetas estén visibles
            print(f"✅ Etiquetas de puntos habilitadas para {len(idx_set)} unidades")
            
            # Verificar si las etiquetas existen en el axes
            text_labels_found = 0
            for txt in ax.texts:
                if txt.get_text() in idx_set:
                    # Asegurar que la etiqueta sea visible
                    txt.set_visible(True)
                    txt.set_fontsize(9)
                    txt.set_alpha(1.0)
                    text_labels_found += 1
            
            # Si la librería no añadió las etiquetas, añadirlas manualmente
            if text_labels_found == 0:
                print(f"⚠️ La librería pca no añadió etiquetas. Añadiéndolas manualmente...")
                
                # Obtener los scores PCA (posiciones de los puntos)
                pc_scores = results["PC"]  # DataFrame con PC1, PC2, etc.
                
                for i, label in enumerate(labels_array):
                    # Obtener coordenadas del punto
                    x = pc_scores.iloc[i, 0]  # PC1
                    y = pc_scores.iloc[i, 1]  # PC2
                    
                    # Añadir etiqueta con offset pequeño
                    ax.annotate(
                        label,
                        (x, y),
                        xytext=(5, 5),  # Offset en píxeles
                        textcoords='offset points',
                        fontsize=9,
                        alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7)
                    )
                
                print(f"✅ {len(labels_array)} etiquetas añadidas manualmente")
            else:
                print(f"✅ {text_labels_found} etiquetas encontradas y configuradas")
        
        else:
            # Si no se quieren etiquetas, removerlas
            removed_labels = 0
            for txt in list(ax.texts):
                if txt.get_text() in idx_set:
                    txt.remove()
                    removed_labels += 1
            if removed_labels:
                print(f"🧹 Etiquetas removidas: {removed_labels}")

        # Añadir información de varianza explicada
        if hasattr(model, "results") and "explained_var" in model.results:
            var_explained = model.results["explained_var"]
            var_text = f"PC1: {var_explained[0]:.1%}, PC2: {var_explained[1]:.1%}"
            ax.text(
                0.02,
                0.98,
                var_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        # Mejorar aspecto general
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Guardar si se especifica ruta
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"💾 Biplot guardado en: {save_path}")

        print("✅ Biplot avanzado creado exitosamente")
        return fig, ax

    except Exception as e:
        logger.error(f"Error creando biplot avanzado: {e}")
        print(f"❌ Error en biplot avanzado: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def demo_advanced_biplot():
    """Función de demostración del biplot avanzado."""
    print("🎨 Demostración de Biplot Avanzado")
    print("=" * 40)

    # Crear datos de ejemplo
    np.random.seed(42)
    countries = [
        "México",
        "Brasil",
        "Argentina",
        "España",
        "Francia",
        "Alemania",
        "China",
        "Japón",
        "India",
        "Australia",
        "Estados Unidos",
        "Canadá",
    ]
    indicators = [
        "PIB_per_capita",
        "Inflacion",
        "Desempleo",
        "Inversion",
        "Educacion",
        "Salud",
    ]

    # Generar datos correlacionados
    data = np.random.multivariate_normal(
        mean=[0] * len(indicators),
        cov=np.eye(len(indicators)) + 0.3 * np.ones((len(indicators), len(indicators))),
        size=len(countries),
    )

    df_test = pd.DataFrame(data, index=countries, columns=indicators)

    print(f"📊 Datos de prueba: {df_test.shape}")
    print(f"🌍 Países: {countries[:5]}...")

    # Crear biplot con diferentes esquemas
    schemes = ["continentes", "desarrollo", "ingreso"]

    for scheme in schemes:
        print(f"\n🎯 Creando biplot con esquema: {scheme}")
        fig, ax = create_advanced_biplot(
            df_test,
            categorization_scheme=scheme,
            title=f"Biplot - Categorización por {scheme.title()}",
            n_features_show=4,
        )

        if fig is not None:
            plt.show()
        else:
            print(f"❌ No se pudo crear biplot para {scheme}")


def categorize_by_continent(country):
    """Categoriza países por continente."""
    continents = {
        "Europa": [
            "Germany",
            "France",
            "Italy",
            "Spain",
            "United Kingdom",
            "Netherlands",
            "Belgium",
            "Austria",
            "Switzerland",
            "Sweden",
            "Norway",
            "Denmark",
            "Finland",
            "Poland",
            "Czech Republic",
            "Hungary",
            "Greece",
            "Portugal",
            "Ireland",
            "Luxembourg",
            "Slovakia",
            "Slovenia",
            "Estonia",
            "Latvia",
            "Lithuania",
            "Croatia",
            "Bulgaria",
            "Romania",
            "Cyprus",
            "Malta",
        ],
        "Asia": [
            "China",
            "Japan",
            "India",
            "South Korea",
            "Singapore",
            "Hong Kong",
            "Taiwan",
            "Thailand",
            "Malaysia",
            "Indonesia",
            "Philippines",
            "Vietnam",
            "Bangladesh",
            "Pakistan",
            "Sri Lanka",
            "Kazakhstan",
            "Uzbekistan",
            "Mongolia",
            "Cambodia",
            "Laos",
            "Myanmar",
            "Brunei",
            "Bhutan",
            "Nepal",
        ],
        "América del Norte": ["United States", "Canada", "Mexico"],
        "América del Sur": [
            "Brazil",
            "Argentina",
            "Chile",
            "Colombia",
            "Peru",
            "Venezuela",
            "Ecuador",
            "Bolivia",
            "Paraguay",
            "Uruguay",
            "Guyana",
            "Suriname",
        ],
        "África": [
            "South Africa",
            "Nigeria",
            "Egypt",
            "Morocco",
            "Kenya",
            "Ghana",
            "Tanzania",
            "Uganda",
            "Ethiopia",
            "Tunisia",
            "Algeria",
            "Angola",
            "Cameroon",
            "Ivory Coast",
            "Senegal",
            "Zimbabwe",
            "Zambia",
            "Botswana",
            "Namibia",
            "Mozambique",
            "Madagascar",
            "Mali",
            "Burkina Faso",
            "Niger",
        ],
        "Oceanía": [
            "Australia",
            "New Zealand",
            "Fiji",
            "Papua New Guinea",
            "Solomon Islands",
            "Vanuatu",
            "Samoa",
            "Tonga",
            "Kiribati",
            "Tuvalu",
            "Nauru",
            "Palau",
        ],
    }

    for continent, countries in continents.items():
        if country in countries:
            return continent
    return "Otros"


def categorize_by_development(country):
    """Categoriza países por nivel de desarrollo."""
    developed = [
        "United States",
        "Germany",
        "Japan",
        "United Kingdom",
        "France",
        "Italy",
        "Canada",
        "South Korea",
        "Spain",
        "Australia",
        "Netherlands",
        "Belgium",
        "Switzerland",
        "Austria",
        "Sweden",
        "Norway",
        "Denmark",
        "Finland",
        "Ireland",
        "Luxembourg",
        "Singapore",
        "Hong Kong",
        "New Zealand",
        "Taiwan",
    ]

    emerging = [
        "China",
        "India",
        "Brazil",
        "Russia",
        "Mexico",
        "Indonesia",
        "Turkey",
        "Saudi Arabia",
        "Argentina",
        "South Africa",
        "Thailand",
        "Malaysia",
        "Chile",
        "Poland",
        "Egypt",
        "Philippines",
        "Vietnam",
        "Bangladesh",
        "Nigeria",
        "Ukraine",
        "Peru",
        "Colombia",
        "Morocco",
        "Kazakhstan",
    ]

    if country in developed:
        return "Desarrollados"
    elif country in emerging:
        return "Emergentes"
    else:
        return "En Desarrollo"


def categorize_by_income(country):
    """Categoriza países por nivel de ingresos (según Banco Mundial)."""
    high_income = [
        "United States",
        "Germany",
        "Japan",
        "United Kingdom",
        "France",
        "Italy",
        "Canada",
        "South Korea",
        "Spain",
        "Australia",
        "Netherlands",
        "Belgium",
        "Switzerland",
        "Austria",
        "Sweden",
        "Norway",
        "Denmark",
        "Finland",
        "Ireland",
        "Luxembourg",
        "Singapore",
        "Hong Kong",
        "New Zealand",
        "Taiwan",
        "Israel",
        "Czech Republic",
        "Slovenia",
        "Slovakia",
        "Estonia",
        "Latvia",
        "Lithuania",
        "Croatia",
        "Hungary",
        "Poland",
        "Chile",
        "Uruguay",
    ]

    upper_middle = [
        "China",
        "Brazil",
        "Russia",
        "Mexico",
        "Turkey",
        "Argentina",
        "Malaysia",
        "Thailand",
        "South Africa",
        "Colombia",
        "Peru",
        "Ecuador",
        "Dominican Republic",
        "Costa Rica",
        "Panama",
        "Romania",
        "Bulgaria",
        "Montenegro",
        "Serbia",
        "North Macedonia",
        "Albania",
        "Bosnia and Herzegovina",
        "Belarus",
        "Kazakhstan",
        "Azerbaijan",
        "Turkmenistan",
        "Iran",
        "Iraq",
        "Jordan",
        "Lebanon",
        "Libya",
        "Algeria",
        "Tunisia",
        "Botswana",
        "Mauritius",
        "Gabon",
        "Equatorial Guinea",
    ]

    lower_middle = [
        "India",
        "Indonesia",
        "Philippines",
        "Vietnam",
        "Egypt",
        "Morocco",
        "Ukraine",
        "Nigeria",
        "Kenya",
        "Ghana",
        "Ivory Coast",
        "Senegal",
        "Cameroon",
        "Angola",
        "Zambia",
        "Zimbabwe",
        "Honduras",
        "El Salvador",
        "Guatemala",
        "Nicaragua",
        "Bolivia",
        "Paraguay",
        "Sri Lanka",
        "Bangladesh",
        "Pakistan",
        "Myanmar",
        "Cambodia",
        "Laos",
        "Mongolia",
        "Uzbekistan",
        "Kyrgyzstan",
        "Tajikistan",
        "Georgia",
        "Armenia",
        "Moldova",
    ]

    if country in high_income:
        return "Ingresos Altos"
    elif country in upper_middle:
        return "Ingresos Medios-Altos"
    elif country in lower_middle:
        return "Ingresos Medios-Bajos"
    else:
        return "Ingresos Bajos"


def get_categorization_preview(df, config):
    """
    Genera una vista previa de la configuración de categorización.

    Args:
        df: DataFrame con datos
        config: Diccionario con configuración

    Returns:
        str: Texto de vista previa
    """
    try:
        scheme = config.get("categorization_scheme", "continents")
        marker_scheme = config.get("marker_scheme", "classic")
        color_scheme = config.get("color_scheme", "viridis")

        preview_text = f"📊 VISTA PREVIA DE CONFIGURACIÓN\n"
        preview_text += f"=" * 40 + "\n\n"

        preview_text += f"🌍 Esquema de Categorización: {scheme.title()}\n"
        preview_text += f"🔵 Esquema de Marcadores: {marker_scheme.title()}\n"
        preview_text += f"🎨 Esquema de Colores: {color_scheme.title()}\n\n"

        # Obtener categorías para países en el dataset
        countries = df.index.tolist()
        categorization_func = {
            "continents": categorize_by_continent,
            "development": categorize_by_development,
            "income": categorize_by_income,
        }.get(scheme, categorize_by_continent)

        # Agrupar países por categoría
        categories = {}
        for country in countries:
            cat = categorization_func(country)
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(country)

        preview_text += f"📋 PAÍSES POR CATEGORÍA:\n"
        preview_text += f"-" * 25 + "\n"

        for category, country_list in categories.items():
            preview_text += f"\n🏷️ {category}:\n"
            for country in sorted(country_list):
                preview_text += f"   • {country}\n"

        preview_text += f"\n📈 RESUMEN:\n"
        preview_text += f"-" * 15 + "\n"
        preview_text += f"Total de países: {len(countries)}\n"
        preview_text += f"Total de categorías: {len(categories)}\n"

        # Información sobre marcadores y colores
        markers = MARKER_SCHEMES[marker_scheme]
        preview_text += f"\n🔵 Marcadores disponibles: {', '.join(markers)}\n"
        preview_text += f"🎨 Esquema de colores: {color_scheme}\n"

        return preview_text

    except Exception as e:
        return f"Error al generar vista previa: {str(e)}"


if __name__ == "__main__":
    demo_advanced_biplot()


# ✅ NUEVO: Wrapper para mostrar biplot avanzado con plt.show() (ventana interactiva)
def show_advanced_biplot(df, config):
    """
    Wrapper para crear y mostrar biplot avanzado en ventana interactiva.
    
    Args:
        df: DataFrame con datos ya procesados
        config: Diccionario con configuración del biplot
        
    Raises:
        RuntimeError: Si falla la creación del biplot
    """
    try:
        # Extraer parámetros de configuración
        categorization_scheme = config.get("categorization_scheme", "continents")
        marker_scheme = config.get("marker_scheme", "classic")
        color_scheme = config.get("color_scheme", "viridis")
        show_arrows = config.get("show_arrows", True)
        show_labels = config.get("show_labels", True)
        alpha = config.get("alpha", 0.7)
        custom_categories = config.get("custom_categories")
        year = config.get("year", "2022")
        
        # Extraer grupos y colores si están presentes
        groups = config.get("groups", {})
        group_colors = config.get("group_colors", {})
        
        # Si hay grupos configurados, usarlos
        if groups and group_colors:
            custom_categories = groups
        
        # Crear biplot usando la función core
        fig, ax = create_advanced_biplot_core(
            df_standardized=df,
            categorization_scheme=categorization_scheme,
            marker_scheme=marker_scheme,
            color_scheme=color_scheme,
            custom_categories=custom_categories,
            custom_colors=group_colors if groups and group_colors else None,
            show_arrows=show_arrows,
            show_labels=show_labels,
            alpha=alpha,
            title=f"Biplot Avanzado - {year}",
        )
        
        if fig is None:
            raise RuntimeError("create_advanced_biplot_core retornó None - falló la creación del biplot")
        
        # ✅ ARQUITECTURA NUEVA: Mostrar con plt.show() en ventana interactiva
        plt.figure(fig.number)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        raise RuntimeError(f"Error al mostrar biplot avanzado: {str(e)}") from e
