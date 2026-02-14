# Uso de las funciones matemáticas (Phase D – advanced_analytics)

El módulo `backend/advanced_analytics.py` incluye 19 funciones en 5 categorías. Se pueden usar desde la **interfaz** o desde **código**.

---

## Desde la interfaz (GUI)

1. **Carga datos** en la hoja activa: en la barra de herramientas, **Cargar datos** y elige un archivo (Excel, CSV o Parquet).
2. Abre el menú **Herramientas** (en la misma barra).
3. Elige la función que quieras. Se aplica sobre las **columnas numéricas** de la hoja activa.
4. El resultado se muestra en una ventana de texto.

### Menú Herramientas

| Categoría | Función | Descripción breve |
|-----------|--------|-------------------|
| **Tests estadísticos** | Bartlett | Esfericidad; si p < 0.05, los datos son adecuados para PCA. |
| | KMO | Adecuación muestral; > 0.6 suele indicar que PCA es apropiado. |
| | Alfa de Cronbach | Consistencia interna entre ítems. |
| | Distancia de Mahalanobis | Distancias por observación (outliers multivariados). |
| **Reducción de dimensionalidad** | t-SNE | Reducción 2D/3D para visualización. |
| | UMAP | Reducción 2D/3D (requiere `pip install umap-learn`). |
| | Análisis factorial (FA) | Factores latentes con rotación. |
| | ICA | Componentes independientes. |
| **Clustering** | K-Means | K óptimo por codo/silueta. |
| | DBSCAN | Clustering por densidad. |
| | GMM | Mezcla de gaussianas; número de componentes por BIC. |
| **Transformaciones** | Yeo-Johnson | Transformación de potencia (admite negativos). |
| | Winsorización | Recorte por percentiles (p. ej. 5 %–95 %). |
| | Rank inverse normal | Transformación a normal vía rangos. |
| | Robust scaling | Escalado por mediana e IQR. |

**Nota:** Las funciones de **red** (centralidad, detección de comunidades, coeficiente de clustering) reciben un **grafo** (NetworkX), no un DataFrame. Se usan típicamente sobre la matriz de correlación o similar; en la app se pueden llamar desde código una vez tengas el grafo (p. ej. desde el análisis de correlación/redes).

---

## Desde código (Python)

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import advanced_analytics as adv
import pandas as pd

# Ejemplo: datos numéricos
df = pd.read_excel("mis_datos.xlsx", sheet_name=0)
df_num = df.select_dtypes(include=["number"])

# Tests pre-PCA
bartlett = adv.bartlett_test(df_num)
kmo = adv.kmo_test(df_num)

# Reducción
tsne_result = adv.run_tsne(df_num, n_components=2)
umap_result = adv.run_umap(df_num, n_components=2)

# Clustering
kmeans_result = adv.kmeans_analysis(df_num, max_k=10)
dbscan_result = adv.dbscan_analysis(df_num, eps=0.5)
gmm_result = adv.gmm_analysis(df_num, max_components=10)

# Transformaciones
df_winsor = adv.winsorize(df_num, limits=(0.05, 0.05))
df_scaled = adv.robust_scale(df_num)
df_transformed, lambdas = adv.yeo_johnson_transform(df_num)
```

### Análisis de redes (requiere grafo NetworkX)

```python
import networkx as nx
from backend import advanced_analytics as adv

# Ejemplo: grafo desde matriz de correlación
# (en la app, el módulo de correlación/redes puede construir este grafo)
# G = nx.from_pandas_adjacency(corr_matrix)
# centralidad = adv.centrality_measures(G)
# comunidades = adv.community_detection(G, method='louvain')
# clustering = adv.network_clustering_coefficient(G)
```

---

## Resumen

- **En la app:** Cargar datos → **Herramientas** → elegir función. Los resultados se muestran en una ventana.
- **En código:** `from backend import advanced_analytics as adv` y llamar a la función que necesites con un `DataFrame` numérico (o un grafo para las de red).
