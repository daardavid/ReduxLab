# Guía de Soporte para Archivos Parquet en PCA

## Descripción General

Esta guía explica cómo usar archivos Parquet (.parquet) en la aplicación PCA para análisis de componentes principales y correlaciones. Los archivos Parquet ofrecen ventajas significativas en términos de velocidad de carga, tamaño de archivo y compatibilidad.

## ¿Qué es Parquet?

Apache Parquet es un formato de archivo columnar que proporciona:

- **Compresión eficiente**: Los archivos Parquet son generalmente más pequeños que CSV/Excel
- **Carga rápida**: Acceso optimizado a columnas específicas
- **Compatibilidad**: Soportado por pandas, PyArrow, y muchos otros frameworks
- **Preservación de tipos**: Mantiene tipos de datos más precisos que CSV

## Formatos de Datos Soportados

La aplicación PCA ahora soporta los siguientes formatos:

1. **Excel** (.xlsx, .xls)
2. **CSV** (.csv)
3. **Parquet** (.parquet) - ✨ Nuevo

## Estructura de Datos Requerida

### Para Análisis de Series Temporales
```
| País    | 2020 | 2021 | 2022 |
|---------|------|------|------|
| México  | 1.2  | 2.3  | 1.8  |
| Brasil  | -3.4 | 1.7  | 0.9  |
| ...     | ...  | ...  | ...  |
```

### Para Análisis de Correlación
```
| Unit    | Year | Indicator1 | Indicator2 | Indicator3 |
|---------|------|------------|------------|------------|
| Company_A| 2020 | 100        | 0.15       | 0.05       |
| Company_A| 2021 | 120        | 0.17       | 0.06       |
| ...     | ...  | ...        | ...        | ...        |
```

### Para Datos de Panel
```
| Country | Year | GDP  | Inflation | Unemployment |
|---------|------|------|-----------|-------------|
| USA     | 2020 | 100  | 2.1       | 5.2         |
| USA     | 2021 | 102  | 2.3       | 5.0         |
| ...     | ...  | ...  | ...       | ...         |
```

## Usando Archivos Parquet

### 1. Conversión de Excel/CSV a Parquet

```python
import pandas as pd

# Desde Excel
df = pd.read_excel('datos_socioeconomicos.xlsx')
df.to_parquet('datos_socioeconomicos.parquet')

# Desde CSV
df = pd.read_csv('datos.csv')
df.to_parquet('datos.parquet')
```

### 2. Carga en la Aplicación PCA

1. **Abrir la aplicación PCA**
2. **Seleccionar archivo**: El diálogo de selección ahora incluye "Parquet files (*.parquet)"
3. **Los archivos Parquet se detectan automáticamente**
4. **Proceder con el análisis normal**

### 3. Funciones de Carga Disponibles

```python
# Carga básica de archivos Parquet
from data_loader_module import load_parquet_file
data = load_parquet_file('mi_archivo.parquet')

# Carga de correlación específica
from data_loader_module import load_correlation_data_parquet
df = load_correlation_data_parquet('correlacion.parquet')

# Carga automática (detecta formato)
from data_loader_module import load_any_file
data = load_any_file('cualquier_archivo.xlsx')  # También funciona con .parquet
```

## Ventajas de Parquet

### Rendimiento
- **Carga 2-5x más rápida** que Excel
- **Archivos 50-80% más pequeños**
- **Acceso directo a columnas**

### Compatibilidad
- **Interoperable** con R, Python, Spark, etc.
- **Preservación de tipos de datos**
- **Soporte para valores nulos**

### Funciones de Compresión
- **Snappy** (rápido, recomendado)
- **Gzip** (máxima compresión)
- **Brotli** (navegador web)

## Ejemplos de Uso

### Ejemplo 1: Datos Socioeconómicos
```python
# Crear archivo Parquet
data = pd.DataFrame({
    'País': ['México', 'Brasil', 'Argentina'],
    '2020': [1.2, -3.4, 2.1],
    '2021': [2.3, 1.7, -0.5],
    '2022': [1.8, 0.9, 1.4]
})
data.to_parquet('socioeconomicos.parquet')

# Usar en PCA
from data_loader_module import load_any_file
resultado = load_any_file('socioeconomicos.parquet')
```

### Ejemplo 2: Análisis de Correlación
```python
# Datos de empresas
correlation_data = pd.DataFrame({
    'Unit': ['Company_A', 'Company_A', 'Company_B', 'Company_B'],
    'Year': [2020, 2021, 2020, 2021],
    'Revenue': [100, 120, 150, 180],
    'Profit_Margin': [0.15, 0.17, 0.12, 0.14]
})
correlation_data.to_parquet('correlation.parquet')

# Cargar para análisis
from data_loader_module import load_correlation_data_parquet
df = load_correlation_data_parquet('correlation.parquet')
```

## Configuración de Compresión

### Para Análisis Rápidos (Recomendado)
```python
df.to_parquet('datos.parquet', compression='snappy')
```

### Para Máxima Compresión
```python
df.to_parquet('datos.parquet', compression='gzip')
```

### Para Desarrollo Web
```python
df.to_parquet('datos.parquet', compression='brotli')
```

## Validación de Archivos

La aplicación incluye validación de seguridad para archivos Parquet:

- **Verificación de tamaño** (máximo 1M filas, 1000 columnas)
- **Detección de contenido malicioso** en nombres de columnas
- **Validación de tipos de datos**
- **Manejo robusto de errores**

## Compatibilidad con Análisis Existentes

Los archivos Parquet funcionan con **todos** los análisis PCA:

### ✅ Análisis de Series Temporales
- Carga de datos por país/unidad
- Transformación automática de formato
- Análisis PCA estándar

### ✅ Análisis de Corte Transversal  
- Selección de año específico
- Múltiples indicadores
- Visualización comparativa

### ✅ Análisis de Panel 3D
- Datos longitudinales
- Análisis multivariado
- Representaciones 3D

### ✅ Análisis de Correlación
- Matrices de correlación
- Redes de correlación
- Análisis de comunidades

### ✅ Análisis de Biplot
- Visualización avanzada
- Categorización de unidades
- Análisis exploratorio

## Solución de Problemas

### Error: "Formato no soportado"
- Verificar que el archivo tenga extensión `.parquet`
- Asegurarse de que el archivo no esté corrupto

### Error: "Columnas requeridas faltantes"
- Verificar estructura de datos
- Para correlación: usar columnas `Unit` y `Year`
- Para otros análisis: verificar estructura esperada

### Error: "Archivo corrupto"
- El archivo Parquet puede estar dañado
- Intentar recrear el archivo desde la fuente original
- Verificar que sea un archivo Parquet válido

### Rendimiento lento
- Considerar compresión Snappy para mejor velocidad
- Verificar que el archivo no sea excesivamente grande
- Usar tipos de datos apropiados

## Mejores Prácticas

### 1. Preparación de Datos
```python
# Limpiar datos antes de guardar
df = df.dropna(subset=['Important_Column'])  # Eliminar filas críticas vacías
df['Year'] = df['Year'].astype(int)  # Asegurar tipo correcto
df.to_parquet('clean_data.parquet', compression='snappy')
```

### 2. Organización de Archivos
- Usar nombres descriptivos: `analisis_correlacion_2024.parquet`
- Mantener metadatos en nombres de archivo
- Versionar archivos importantes

### 3. Validación Antes de Carga
```python
# Verificar archivo antes de usar
import pandas as pd
df_test = pd.read_parquet('archivo.parquet', nrows=5)  # Solo primeras 5 filas
print(f"Columnas: {list(df_test.columns)}")
print(f"Forma: {df_test.shape}")
```

## Rendimiento Comparativo

| Formato | Tamaño Relativo | Velocidad de Carga | Memoria |
|---------|----------------|-------------------|---------|
| Excel   | 100% (base)    | 1x (base)         | Media   |
| CSV     | 60-80%         | 1.5-2x            | Baja    |
| Parquet | 20-40%         | 3-5x              | Muy Baja|

## Conclusión

El soporte para archivos Parquet en la aplicación PCA proporciona:

- **Mejora significativa** en rendimiento de carga
- **Reducción drástica** en tamaño de archivos
- **Compatibilidad completa** con análisis existentes
- **Funcionalidad de seguridad** robusta
- **Facilidad de uso** con detección automática

Los usuarios pueden continuar usando sus flujos de trabajo existentes mientras aprovechan las ventajas de rendimiento de Parquet.

---

**Desarrollado por**: Equipo de Desarrollo PCA  
**Fecha**: 2025  
**Versión**: 1.0  
**Soporte**: Verificar logs de aplicación para detalles de carga