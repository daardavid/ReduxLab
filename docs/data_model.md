# Modelo de datos: matriz n×p

La aplicación trabaja con un **modelo de datos estrictamente matricial**. No existe un modelo relacional interno (entidades, relaciones, joins a nivel de aplicación).

## Contrato n×p

- **Entrada de trabajo**: una única **matriz de datos** de dimensiones **n×p**:
  - **n** = número de observaciones (filas): unidades de análisis, países, años, productos, etc.
  - **p** = número de variables o indicadores (columnas): las características medidas.

- Todo el álgebra lineal (SVD, descomposición en autovalores, PCA, clustering, etc.) opera sobre esta única tabla numérica. Los datos deben estar **consolidados** en esa estructura antes de usarse en los análisis.

- **Implicación**: si tus datos vienen de varias tablas o hojas (relacionales), los joins/merges o consultas SQL deben hacerse:
  - **Fuera** de la aplicación (pre-procesamiento externo), o
  - **Dentro** de la aplicación usando el flujo **"Preparar datos"** (botón en la barra de herramientas): cargar un Excel, elegir **una hoja** como matriz o **combinar dos hojas** por una columna clave (merge). El resultado es una sola matriz n×p asignada a la hoja activa.

## Formatos soportados como fuente

- **Archivos** (botón "Cargar datos"): Excel (.xlsx, .xls), CSV, Parquet, SQLite (.db, .sqlite, .sqlite3). Si el Excel tiene varias hojas, se muestra un diálogo para elegir qué hoja usar como matriz. Si el archivo es SQLite con varias tablas, se elige una tabla. Esa hoja o tabla se interpreta como la matriz n×p.
- **Preparar datos**: desde un Excel, elegir una hoja o combinar dos hojas por una columna clave (merge). El resultado se carga en la hoja activa.
- **Bases de datos** (PostgreSQL, MySQL): se pueden usar desde código con `backend.data_connectors`; en la interfaz, "Cargar datos" soporta directamente archivos SQLite.

En todos los casos, el resultado final en memoria (`sheet["df"]`) es una sola tabla plana lista para el análisis.
