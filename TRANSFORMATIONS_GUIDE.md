# Guía de Transformaciones de Datos para PCA

## 📊 ¿Qué son las transformaciones de datos?

Las transformaciones de datos son operaciones matemáticas que modifican la distribución de las variables para hacerlas más apropiadas para el análisis de componentes principales (PCA). Son especialmente importantes cuando trabajamos con **datos financieros** o **indicadores con diferentes órdenes de magnitud**.

## 🎯 ¿Por qué son necesarias?

### Problema común en datos financieros

Imagina que estás analizando empresas con estas variables:
- **Ingresos**: de 100 millones a 500,000 millones (rango: x5000)
- **Empleados**: de 1,000 a 2,000,000 (rango: x2000)
- **Margen de utilidad**: de 2% a 15% (rango: x7)

**Problema**: Las variables de magnitud (ingresos, empleados) tienen distribuciones **log-normales** extremadamente sesgadas, mientras que los ratios financieros ya están normalizados.

**Consecuencia en el PCA**:
- Los componentes principales quedan dominados por las variables con mayor varianza absoluta
- Los vectores en el biplot se comprimen hacia el origen
- Es imposible interpretar las relaciones entre variables
- La varianza explicada es artificialmente baja

## 🔧 Solución: Sistema Dual de Transformaciones

Nuestro sistema implementa **DOS soluciones complementarias**:

### 1. Transformaciones Estadísticas (Pre-PCA)
Normaliza las distribuciones ANTES de aplicar PCA.

```python
from data_transformations import DataTransformer

# Crear transformador
transformer = DataTransformer(skewness_threshold=1.0)

# Analizar datos
analysis = transformer.analyze_data(df)
print(analysis)

# Aplicar transformaciones automáticas
df_transformed, metadata = transformer.transform(df, method='auto')
```

### 2. Escalado Visual de Vectores (Post-PCA)
Ajusta la longitud de los vectores en el biplot para mejorar visualización.

```python
# En visualization_module.py
create_biplot(pca_result, arrow_scale=None)  # None = auto-calculate
create_biplot(pca_result, arrow_scale=0.5)   # Manual override
```

## 📈 Métodos de Transformación Disponibles

### `'auto'` (Recomendado)
Selecciona automáticamente el mejor método según las características de cada columna:
- **Log**: Para datos estrictamente positivos con alta asimetría
- **Log1p**: Para datos con zeros (log(x + 1))
- **Yeo-Johnson**: Para datos con valores negativos
- **Box-Cox**: Para datos estrictamente positivos

```python
df_transformed, metadata = transformer.transform(df, method='auto')
```

### `'log'`
Transformación logarítmica: `y = log(x)`
- **Cuándo usar**: Datos estrictamente positivos (ingresos, activos, valor de mercado)
- **Ventaja**: Simple e interpretable
- **Limitación**: No funciona con zeros o negativos

```python
df_transformed, metadata = transformer.transform(df, method='log')
```

### `'log1p'`
Logaritmo con desplazamiento: `y = log(x + 1)`
- **Cuándo usar**: Datos con algunos zeros (ventas, empleados despedidos)
- **Ventaja**: Maneja zeros correctamente
- **Limitación**: No funciona con valores negativos

```python
df_transformed, metadata = transformer.transform(df, method='log1p')
```

### `'sqrt'`
Raíz cuadrada: `y = √x`
- **Cuándo usar**: Asimetría moderada, conteos (número de empleados, tiendas)
- **Ventaja**: Menos agresiva que log
- **Limitación**: Solo valores no negativos

```python
df_transformed, metadata = transformer.transform(df, method='sqrt')
```

### `'box-cox'`
Transformación Box-Cox: encuentra el λ óptimo para normalizar
- **Cuándo usar**: Datos estrictamente positivos, quieres máxima normalidad
- **Ventaja**: Óptima estadísticamente
- **Limitación**: Requiere x > 0, menos interpretable

```python
df_transformed, metadata = transformer.transform(df, method='box-cox')
```

### `'yeo-johnson'`
Extensión de Box-Cox que permite negativos
- **Cuándo usar**: Datos con valores negativos (ganancias/pérdidas, flujo de caja)
- **Ventaja**: Funciona con cualquier valor
- **Limitación**: Menos interpretable

```python
df_transformed, metadata = transformer.transform(df, method='yeo-johnson')
```

## 🎓 Detección Automática de Tipos de Columna

El sistema clasifica automáticamente las columnas en:

### Columnas de Magnitud (SE TRANSFORMAN)
Variables con valores absolutos grandes y distribución sesgada:

**Keywords detectados**:
- Financieros: `ingreso`, `revenue`, `venta`, `sale`, `activo`, `asset`, `pasivo`, `liability`, `capital`, `valor`, `value`, `precio`, `price`
- Operativos: `empleado`, `employee`, `personal`, `staff`, `cliente`, `customer`, `unidad`, `unit`, `producción`, `production`
- Métricas absolutas: `cantidad`, `amount`, `volumen`, `volume`, `tamaño`, `size`

**Ejemplo**: `Ingresos_Millones`, `Total_Activos`, `Empleados_Totales`

### Columnas de Ratio/Porcentaje (NO SE TRANSFORMAN)
Variables ya normalizadas que no deben transformarse:

**Keywords detectados**:
- Ratios: `ratio`, `tasa`, `rate`, `índice`, `index`
- Rentabilidad: `roe`, `roa`, `roi`, `margen`, `margin`
- Porcentajes: `porcentaje`, `percentage`, `pct`, `%`

**Ejemplo**: `ROE_Porcentaje`, `Margen_Operativo`, `Ratio_Deuda_Capital`

**Rango detectado**:
- Valores en [0, 1] → Porcentaje decimal
- Valores en [0, 100] → Porcentaje

## 🚀 Integración con Preprocessing Pipeline

### Uso Básico
```python
from preprocessing_module import preprocess_data

# SIN transformaciones (comportamiento antiguo)
df_preprocessed = preprocess_data(
    df,
    apply_transformations=False
)

# CON transformaciones automáticas
df_preprocessed = preprocess_data(
    df,
    apply_transformations=True,
    transformation_method='auto',
    skewness_threshold=1.0
)
```

### Uso Avanzado con Metadata
```python
# Obtener detalles de transformaciones aplicadas
df_preprocessed, transformer = preprocess_data(
    df,
    apply_transformations=True,
    return_transformer=True
)

# Ver qué se transformó
print("Metadata:", transformer.transformation_metadata)
```

### Configurar Skewness Threshold
```python
# Más agresivo (transforma más columnas)
df_preprocessed = preprocess_data(
    df,
    apply_transformations=True,
    skewness_threshold=0.5  # Default: 1.0
)

# Menos agresivo (transforma menos columnas)
df_preprocessed = preprocess_data(
    df,
    apply_transformations=True,
    skewness_threshold=2.0
)
```

## 📊 Análisis de Distribución Antes/Después

### Función de Análisis
```python
from data_transformations import analyze_data_distribution

# Analizar distribución de cada columna
analysis = analyze_data_distribution(df, skewness_threshold=1.0)

for column, info in analysis.items():
    print(f"\n{column}:")
    print(f"  Tipo: {info['type']}")
    print(f"  Skewness: {info['skewness']:.2f}")
    print(f"  Necesita transformación: {info['needs_transform']}")
```

**Interpretación del Skewness**:
- **|skewness| < 0.5**: Distribución simétrica (buena para PCA)
- **0.5 ≤ |skewness| < 1.0**: Moderadamente sesgada (aceptable)
- **|skewness| ≥ 1.0**: Altamente sesgada (transformación recomendada)

### Visualización
```python
import matplotlib.pyplot as plt

# Comparar distribuciones antes/después
fig, axes = plt.subplots(2, len(columns), figsize=(15, 8))

for i, col in enumerate(columns):
    # Original
    axes[0, i].hist(df[col], bins=30, edgecolor='black')
    axes[0, i].set_title(f'{col} (Original)')
    
    # Transformada
    axes[1, i].hist(df_transformed[col], bins=30, edgecolor='black')
    axes[1, i].set_title(f'{col} (Transformada)')

plt.tight_layout()
plt.savefig('comparacion_distribuciones.png', dpi=300)
```

## 🎯 Escalado de Vectores en Biplot

### Escalado Automático (Recomendado)
El sistema calcula automáticamente el factor de escala basado en el rango de los datos:

```python
# En visualization_module.py
max_score_range = np.max([abs(PC1), abs(PC2)])
max_loading_val = np.abs(loadings).max()
arrow_scale = (max_score_range / max_loading_val) * 0.35
```

**Fórmula**: `arrow_scale = (rango_puntos / rango_vectores) × 0.35`

**Objetivo**: Vectores ocupan ~30-35% del rango total del biplot

### Escalado Manual
Si el escalado automático no es adecuado:

```python
from visualization_module import create_biplot

# Vectores más largos
create_biplot(pca_result, arrow_scale=0.8)

# Vectores más cortos
create_biplot(pca_result, arrow_scale=0.2)

# Auto (default)
create_biplot(pca_result, arrow_scale=None)
```

## ✅ Cuándo Usar Transformaciones

### ✅ USAR transformaciones cuando:

1. **Datos financieros de empresas**:
   - Ingresos, activos, pasivos, valor de mercado
   - Número de empleados, clientes, unidades vendidas
   - Precios, costos, inversiones

2. **Indicadores de países**:
   - PIB, población, superficie
   - Exportaciones, importaciones
   - Gasto público, deuda

3. **Datos con distribución log-normal**:
   - Skewness > 1.0
   - Rango de valores > 100x (ej: 100 a 10,000)
   - Algunos valores mucho mayores que la media

4. **Biplots con vectores comprimidos**:
   - No se pueden interpretar direcciones
   - Todos los vectores apuntan al origen
   - Puntos muy dispersos pero vectores pequeños

### ❌ NO USAR transformaciones cuando:

1. **Ratios y porcentajes financieros**:
   - ROE, ROA, ROI (ya normalizados)
   - Márgenes de utilidad, rentabilidad
   - Ratio deuda/capital, liquidez corriente

2. **Variables ya normalizadas**:
   - Índices (0-100)
   - Calificaciones (1-5 estrellas)
   - Probabilidades (0-1)

3. **Datos con distribución simétrica**:
   - Skewness < 0.5
   - Distribución normal o uniforme
   - Sin outliers extremos

4. **Conteos pequeños**:
   - Número de productos (< 100)
   - Categorías (1-10)
   - Binarios (0/1)

## 📚 Ejemplo Completo: Análisis de Fortune 500

### Paso 1: Cargar y Explorar Datos
```python
import pandas as pd
from data_transformations import DataTransformer, analyze_data_distribution

# Cargar datos
df = pd.read_csv('fortune500.csv')

# Ver columnas
print(df.columns)
# ['Empresa', 'Ingresos_Millones', 'Utilidad_Millones', 'Activos_Millones',
#  'Valor_Mercado_Millones', 'Empleados', 'ROE_%', 'Margen_Operativo_%', 
#  'Ratio_Deuda_Capital', 'Liquidez_Corriente']

# Analizar distribución
analysis = analyze_data_distribution(df, skewness_threshold=1.0)
for col, info in analysis.items():
    if info['needs_transform']:
        print(f"{col}: skewness={info['skewness']:.2f} ⚠️ TRANSFORMAR")
    else:
        print(f"{col}: skewness={info['skewness']:.2f} ✓ OK")
```

**Output esperado**:
```
Ingresos_Millones: skewness=3.55 ⚠️ TRANSFORMAR
Utilidad_Millones: skewness=6.69 ⚠️ TRANSFORMAR
Activos_Millones: skewness=1.99 ⚠️ TRANSFORMAR
Valor_Mercado_Millones: skewness=5.02 ⚠️ TRANSFORMAR
Empleados: skewness=4.15 ⚠️ TRANSFORMAR
ROE_%: skewness=0.27 ✓ OK
Margen_Operativo_%: skewness=-0.14 ✓ OK
Ratio_Deuda_Capital: skewness=0.45 ✓ OK
Liquidez_Corriente: skewness=0.33 ✓ OK
```

### Paso 2: Aplicar Transformaciones
```python
# Crear transformador
transformer = DataTransformer(skewness_threshold=1.0)

# Transformar automáticamente
df_transformed, metadata = transformer.transform(df, method='auto')

# Ver qué se transformó
print("\nTransformaciones aplicadas:")
for col, info in metadata.items():
    print(f"{col}:")
    print(f"  Método: {info['method']}")
    print(f"  Skewness antes: {info['original_skewness']:.2f}")
    print(f"  Skewness después: {info['transformed_skewness']:.2f}")
```

### Paso 3: Comparar PCA Sin/Con Transformaciones
```python
from sklearn.decomposition import PCA
from preprocessing_module import preprocess_data
import matplotlib.pyplot as plt

# PCA sin transformaciones
df_std_1 = preprocess_data(df, apply_transformations=False)
pca_1 = PCA(n_components=2)
scores_1 = pca_1.fit_transform(df_std_1)
var_1 = pca_1.explained_variance_ratio_.sum()

# PCA con transformaciones
df_std_2 = preprocess_data(df, apply_transformations=True)
pca_2 = PCA(n_components=2)
scores_2 = pca_2.fit_transform(df_std_2)
var_2 = pca_2.explained_variance_ratio_.sum()

# Comparar
print(f"\nVarianza explicada:")
print(f"  Sin transformaciones: {var_1*100:.1f}%")
print(f"  Con transformaciones: {var_2*100:.1f}%")
print(f"  Mejora: +{(var_2-var_1)*100:.1f}%")

# Visualizar
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(scores_1[:, 0], scores_1[:, 1], alpha=0.6)
axes[0].set_title(f'Sin Transformaciones ({var_1*100:.1f}%)')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

axes[1].scatter(scores_2[:, 0], scores_2[:, 1], alpha=0.6)
axes[1].set_title(f'Con Transformaciones ({var_2*100:.1f}%)')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')

plt.tight_layout()
plt.savefig('comparacion_pca.png', dpi=300)
```

### Paso 4: Generar Biplot con Vectores Escalados
```python
from visualization_module import create_biplot

# Biplot con transformaciones y auto-scale
fig, ax = plt.subplots(figsize=(12, 10))
create_biplot(
    pca_result=pca_2,
    df=df_std_2,
    feature_names=df.columns[1:],  # Excluir 'Empresa'
    labels=df['Empresa'],
    arrow_scale=None,  # Auto-calculate
    ax=ax
)
plt.title('Fortune 500 - Biplot con Transformaciones')
plt.savefig('biplot_fortune500.png', dpi=300, bbox_inches='tight')
```

## 🔍 Interpretación de Resultados

### Antes de las Transformaciones
```
PC1 (18.5%): Dominado por Ingresos y Activos (alta varianza absoluta)
PC2 (14.3%): Mezclado, difícil interpretar
Vectores: Comprimidos, todos apuntan al origen
Total: 32.8% varianza explicada
```

### Después de las Transformaciones
```
PC1 (21.4%): "Tamaño de empresa" (ingresos, empleados, activos)
PC2 (14.4%): "Eficiencia operativa" (márgenes, ROE, liquidez)
Vectores: Claros, interpretables, bien distribuidos
Total: 35.8% varianza explicada (+3%)
```

### Mejoras Observadas
1. **+3% varianza explicada**: Componentes capturan mejor la estructura
2. **Vectores interpretables**: Claras direcciones de variables
3. **Clusters visibles**: Empresas similares agrupadas
4. **Outliers identificables**: Empresas con perfiles únicos

## ⚙️ Configuración Avanzada

### Transformar Solo Columnas Específicas
```python
# Especificar columnas manualmente
columns_to_transform = ['Ingresos_Millones', 'Activos_Millones', 'Empleados']

df_transformed, metadata = transformer.transform(
    df,
    method='log',
    columns=columns_to_transform
)
```

### Ajustar Threshold de Skewness
```python
# Más sensible (transforma con skewness > 0.5)
transformer_aggressive = DataTransformer(skewness_threshold=0.5)

# Menos sensible (solo skewness > 2.0)
transformer_conservative = DataTransformer(skewness_threshold=2.0)
```

### Manejar Edge Cases
```python
# El sistema maneja automáticamente:
# - Valores negativos → usa yeo-johnson
# - Zeros → usa log1p en lugar de log
# - NaN → los preserva sin transformar
# - Std=0 → no transforma (constante)
# - n < 3 → no calcula skewness
```

## 🐛 Troubleshooting

### Problema: "No se ven los vectores en el biplot"
**Solución**:
1. Verificar que `arrow_scale=None` (auto)
2. Si manual, aumentar: `arrow_scale=0.5` o `arrow_scale=0.8`
3. Aplicar transformaciones: `apply_transformations=True`

### Problema: "Los vectores son demasiado largos"
**Solución**:
1. Reducir arrow_scale: `arrow_scale=0.2`
2. Verificar que no haya outliers extremos en los datos

### Problema: "Error: cannot take logarithm of negative values"
**Solución**:
```python
# Usar método que acepta negativos
df_transformed, _ = transformer.transform(df, method='yeo-johnson')
# O usar auto para detección automática
df_transformed, _ = transformer.transform(df, method='auto')
```

### Problema: "La varianza explicada disminuyó con transformaciones"
**Causa**: Probablemente los datos ya estaban bien distribuidos
**Solución**:
1. Verificar skewness original: `analyze_data_distribution(df)`
2. Si |skewness| < 1.0, NO usar transformaciones
3. Considerar usar threshold más alto: `skewness_threshold=1.5`

### Problema: "Los clusters cambiaron con las transformaciones"
**Esto es normal**: Las transformaciones revelan la estructura real de los datos
**Interpretación**:
- Clusters antes: Dominados por magnitud absoluta
- Clusters después: Basados en relaciones proporcionales

## 📖 Referencias

### Documentación Relacionada
- `data_transformations.py`: Implementación del transformador
- `preprocessing_module.py`: Integración con pipeline
- `visualization_module.py`: Escalado de vectores
- `test_transformations.py`: Tests y ejemplos

### Recursos Externos
- [Scikit-learn: Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Box-Cox Transformation](https://en.wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation)
- [PCA con datos no-normales](https://stats.stackexchange.com/questions/69157/why-do-we-need-to-normalize-data-before-principal-component-analysis-pca)

## 📞 Soporte

Si tienes dudas o problemas:
1. Revisa los ejemplos en `test_transformations.py`
2. Ejecuta `analyze_data_distribution(df)` para diagnóstico
3. Consulta los logs del sistema (carpeta `logs/`)
4. Verifica que `scipy` y `scikit-learn` estén actualizados

---

**Versión**: 1.0  
**Última actualización**: 2024  
**Autor**: Sistema PCA Socioeconomics
