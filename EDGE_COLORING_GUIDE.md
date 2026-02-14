# Guía de Configuración de Coloreo de Aristas

## Descripción
Esta guía explica cómo configurar el coloreo de aristas (líneas) en las visualizaciones de redes basado en la fuerza de correlación.

## Configuraciones Disponibles

### Parámetros Principales

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `edge_coloring_enabled` | bool | `True` | Habilita o deshabilita el coloreo de aristas |
| `edge_colorscale` | str | `'Blues'` | Esquema de colores para las aristas |
| `edge_color_intensity_range` | tuple | `(0.2, 1.0)` | Rango de intensidad de colores (min, max) |
| `edge_opacity` | float | `0.8` | Transparencia de las aristas (0-1) |
| `edge_colorbar_enabled` | bool | `True` | Mostrar barra de colores |
| `edge_colorbar_title` | str | `'Correlation Strength'` | Título de la barra de colores |
| `edge_default_color` | str | `'#888888'` | Color por defecto cuando el coloreo está deshabilitado |

### Esquemas de Color Disponibles

#### Monocromáticos (Un solo color)
- **`'Blues'`**: Gradiente azul (claro a oscuro)
- **`'Reds'`**: Gradiente rojo (claro a oscuro)
- **`'Greens'`**: Gradiente verde (claro a oscuro)
- **`'Oranges'`**: Gradiente naranja (claro a oscuro)
- **`'Purples'`**: Gradiente morado (claro a oscuro)
- **`'Greys'`**: Gradiente gris (claro a oscuro)

#### Perceptualmente Uniformes
- **`'Viridis'`**: Esquema viridis (científico)
- **`'Plasma'`**: Esquema plasma (científico)
- **`'Inferno'`**: Esquema inferno (científico)
- **`'Magma'`**: Esquema magma (científico)

#### Especialistas
- **`'Cividis'`**: Esquema amigable para daltónicos
- **`'Turbo'`**: Alto contraste
- **`'RdBu'`**: Rojo-Azul divergente
- **`'RdYlBu'`**: Rojo-Amarillo-Azul divergente
- **`'Spectral'`**: Colores espectrales
- **`'Coolwarm'`**: Colores fríos-cálidos

## Ejemplos de Uso

### 1. Análisis Financiero (Azul)
```python
config = {
    'edge_coloring_enabled': True,
    'edge_colorscale': 'Blues',
    'edge_color_intensity_range': (0.3, 1.0),
    'edge_opacity': 0.8,
    'edge_colorbar_title': 'Fuerza de Correlación',
    'edge_colorbar_enabled': True
}
```

### 2. Análisis de Riesgo (Rojo)
```python
config = {
    'edge_coloring_enabled': True,
    'edge_colorscale': 'Reds',
    'edge_color_intensity_range': (0.2, 0.9),
    'edge_opacity': 0.9,
    'edge_colorbar_title': 'Correlación de Riesgo',
    'edge_colorbar_enabled': True
}
```

### 3. Análisis de Crecimiento (Verde)
```python
config = {
    'edge_coloring_enabled': True,
    'edge_colorscale': 'Greens',
    'edge_color_intensity_range': (0.1, 0.8),
    'edge_opacity': 0.7,
    'edge_colorbar_title': 'Correlación de Crecimiento',
    'edge_colorbar_enabled': True
}
```

### 4. Presentación Académica (Viridis)
```python
config = {
    'edge_coloring_enabled': True,
    'edge_colorscale': 'Viridis',
    'edge_color_intensity_range': (0.25, 1.0),
    'edge_opacity': 0.85,
    'edge_colorbar_title': 'Coeficiente de Correlación',
    'edge_colorbar_enabled': True
}
```

### 5. Vista Simple (Sin coloreo)
```python
config = {
    'edge_coloring_enabled': False,
    'edge_default_color': '#666666'
}
```

## Métodos Auxiliares

### `get_available_edge_colorschemes()`
Obtiene todos los esquemas de color disponibles con sus descripciones:

```python
visualizer = NetworkVisualizer()
schemes = visualizer.get_available_edge_colorschemes()
for name, description in schemes.items():
    print(f"{name}: {description}")
```

### `set_edge_color_config(config_updates)`
Valida y combina configuraciones de coloreo:

```python
visualizer = NetworkVisualizer()
config = {
    'edge_colorscale': 'Reds',
    'edge_opacity': 0.9
}
merged_config = visualizer.set_edge_color_config(config)
```

## Uso Completo

```python
from network_visualization import NetworkVisualizer
import pandas as pd

# Crear visualizador
visualizer = NetworkVisualizer()

# Configurar coloreo personalizado
edge_config = {
    'edge_coloring_enabled': True,
    'edge_colorscale': 'Blues',
    'edge_color_intensity_range': (0.3, 1.0),
    'edge_opacity': 0.8,
    'edge_colorbar_title': 'Correlación',
    'edge_colorbar_enabled': True
}

# Validar y combinar configuración
config = visualizer.set_edge_color_config(edge_config)

# Crear red desde matriz de correlación
graph = visualizer.create_network_from_correlation(correlation_matrix, config)

# Crear visualización con coloreo personalizado
fig = visualizer.create_plotly_network(graph, config=config)
```

## Consejos de Uso

1. **Análisis Financiero**: Usa `'Blues'` para correlaciones generales
2. **Análisis de Riesgo**: Usa `'Reds'` para enfatizar correlaciones de riesgo
3. **Análisis de Crecimiento**: Usa `'Greens'` para correlaciones positivas
4. **Presentaciones Académicas**: Usa `'Viridis'` por ser perceptualmente uniforme
5. **Correlaciones Divergentes**: Usa `'RdBu'` para mostrar correlaciones positivas/negativas
6. **Control de Contraste**: Ajusta `edge_color_intensity_range` para controlar el contraste
7. **Visualización Simple**: Desactiva el coloreo con `edge_coloring_enabled: False`

## Validaciones

El sistema incluye validaciones para:
- Esquemas de color válidos
- Rangos de intensidad válidos (0-1)
- Valores de opacidad válidos (0-1)
- Rangos de intensidad lógicos (min < max)