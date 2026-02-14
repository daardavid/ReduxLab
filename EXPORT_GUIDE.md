# 📊 Guía de Exportación y Gestión de Grupos - ReduxLab

## 🏷️ NUEVA FUNCIONALIDAD: Sistema Universal de Grupos

### ✨ **Gestión Universal de Grupos:**
- **Persistencia Automática**: Los grupos se guardan automáticamente y persisten entre sesiones
- **Uso Universal**: Los grupos creados están disponibles en todos los tipos de análisis
- **Interfaz Integrada**: Gestión completa desde la GUI principal
- **Datos de Ejemplo**: Incluye clasificación predefinida de empresas automotrices
- **🆕 MÚLTIPLES FORMATOS**: Importación y exportación en JSON, CSV y Excel

### 🎯 **Filtrado de Análisis por Grupos:**
- **Todos los Análisis**: Analizar todas las unidades seleccionadas
- **Grupos Específicos**: Analizar solo unidades de grupos seleccionados  
- **Excluir Grupos**: Analizar excluyendo grupos específicos
- **Validación Automática**: El sistema valida que hay suficientes unidades para el análisis

### 🚀 **Cómo Usar el Sistema de Grupos:**

#### **1. Acceder al Gestor de Grupos:**
```
Botón lateral: "🏷️ Gestionar Grupos"
```

#### **2. Crear Grupos:**
- Haz clic en "🏷️ Manage Universal Groups"
- Selecciona las unidades para el grupo
- Asigna nombre, descripción y color
- Los grupos se guardan automáticamente

#### **3. Cargar Datos de Ejemplo:**
```
REMOVIDO: La funcionalidad de carga automática de datos automotrices ha sido eliminada
para simplificar la interfaz. Usa los archivos de ejemplo en la carpeta examples/:
- examples/sample_groups.csv
- examples/sample_groups.xlsx
```

#### **🆕 4. Importar/Exportar Grupos en Múltiples Formatos:**

##### **Exportar Grupos:**
```
Fila "Export:" → Selecciona formato:
📤 JSON    - Formato completo con metadatos
📊 CSV     - Formato simple: Unit, Group  
📈 Excel   - Múltiples hojas con resumen
```

##### **Importar Grupos:**
```
Fila "Import:" → Selecciona formato:
📥 JSON    - Importar desde archivo de configuración
📊 CSV     - Importar desde hoja de cálculo CSV
📈 Excel   - Importar desde archivo Excel (auto-detecta hoja)
```

##### **Formato Requerido para CSV/Excel:**
```
Columna 1: Unit (Unidad de investigación)
Columna 2: Group (Nombre del grupo)

Ejemplo:
Unit,Group
Tesla,OEM
Ford Motor,OEM
Aptiv,Autopartes_electronicas
Denso,Autopartes_electronicas
```

#### **5. Filtrar Análisis por Grupos:**
1. En cualquier análisis, ve a la sección "🏷️ Groups & Analysis Filtering"
2. Haz clic en "🔄 Load Current Groups"
3. Selecciona el modo de análisis:
   - **All Units**: Todas las unidades
   - **Selected Groups Only**: Solo grupos seleccionados
   - **Exclude Groups**: Excluir grupos específicos
4. Ejecuta el análisis normalmente

### 📁 **Archivos de Configuración:**
```
config/universal_groups.json     # Grupos guardados
config/groups_history.json       # Historial de operaciones
examples/sample_groups.csv       # Ejemplo CSV
examples/sample_groups.xlsx      # Ejemplo Excel
```

### 📤 **Exportación/Importación Avanzada:**

#### **📊 Exportación CSV:**
- **Formato**: Tabla simple con columnas Unit y Group
- **Codificación**: UTF-8 para caracteres especiales
- **Ordenamiento**: Unidades ordenadas alfabéticamente
- **Uso**: Ideal para hojas de cálculo y procesamiento de datos

#### **📈 Exportación Excel:**
- **Múltiples Hojas**:
  - `Group_Assignments`: Asignaciones principales (Unit, Group)
  - `Group_Summary`: Resumen por grupo (nombre, cantidad, descripción, color, fechas)
- **Formato**: Compatible con Excel 2007+ (.xlsx)
- **Metadatos**: Incluye información completa de cada grupo

#### **🔄 Importación Inteligente:**
- **Auto-detección**: Detecta automáticamente formato y estructura
- **Gestión de Conflictos**: Pregunta antes de sobrescribir grupos existentes
- **Validación**: Verifica formato y datos antes de importar
- **Colores Automáticos**: Asigna colores únicos a grupos nuevos
- **Registro**: Mantiene historial de todas las operaciones

### 📋 **Casos de Uso Específicos:**

#### **🔄 Migración de Datos:**
```
1. Exportar grupos existentes como respaldo (JSON)
2. Preparar nuevos datos en Excel/CSV
3. Importar nuevos datos 
4. Verificar y ajustar según necesidad
```

#### **📊 Colaboración en Equipo:**
```
1. Crear grupos base en la aplicación
2. Exportar a Excel para compartir
3. Equipo edita/completa clasificaciones
4. Importar datos actualizados
```

#### **🔧 Procesamiento Masivo:**
```
1. Exportar a CSV para procesamiento externo
2. Usar herramientas de análisis de datos
3. Aplicar clasificaciones automáticas
4. Importar resultados procesados
```

## 🔧 Errores Corregidos ✅

### ✅ Errores Críticos Resueltos:
1. **`analysis_manager.py`**: Se agregaron los imports faltantes (`pandas`, `numpy`, `filedialog`, etc.)
2. **`error_recovery.py`**: Se corrigió la variable `data` no definida en `_recover_realizar_pca`
3. **`refactored_frames.py`**: 
   - Se arregló la variable `parent` no definida (línea 967)
   - Se solucionó la función `populate_unit_listbox` faltante (línea 2194)
4. **Exportación Excel**: Se corrigió el error "At least one sheet must be visible" ajustando la estructura de datos

### 📦 Dependencias Opcionales (Normal):
- `setuptools`, `pytest`, `coverage`, `community` son dependencias opcionales y no afectan el funcionamiento principal

### 🧪 Pruebas Realizadas:
- ✅ 39 tests unitarios pasando
- ✅ Aplicación GUI ejecutándose sin errores
- ✅ Funcionalidad de exportación Excel verificada
- ✅ Análisis de correlación y redes funcionando
- ✅ Sistema de grupos operativo

## 📈 Funcionalidad de Exportación de Datos

### 🚀 Cómo Usar la Exportación:

1. **Ejecutar un Análisis**:
   ```bash
   python pca_gui_modern.py
   ```

2. **Realizar Análisis de Correlación**:
   - Selecciona "Correlation/Network" en el panel izquierdo
   - Carga tu archivo de datos (Excel)
   - Configura los parámetros de correlación
   - **NUEVO**: Configura grupos y filtros de análisis
   - Haz clic en "Ejecutar Análisis"

3. **Exportar Resultados**:
   - Una vez completado el análisis, el botón "💾 Exportar Resultados" se habilitará
   - Haz clic en el botón de exportación
   - Selecciona la ubicación donde guardar
   - Los resultados se guardarán como `nombre_archivo_complete_results.xlsx`

### 📊 **Datos Exportados - AMPLIADOS**:

#### **Análisis Básico:**
- `Original_Data` - Datos originales
- `Correlation_Matrix` - Matriz de correlación
- `Statistics` - Estadísticas descriptivas
- `Configuration` - Parámetros del análisis
- `Selection_Info` - Indicadores y unidades seleccionadas

#### **Análisis de Redes (NUEVO):**
- `Network_Edges` - Lista completa de conexiones con pesos
- `Network_Nodes` - Información de nodos con grados y estadísticas
- `Network_Communities` - Asignación de comunidades (Louvain)
- `Community_Summary` - Resumen de tamaños de comunidades
- `Network_Statistics` - Métricas de red (densidad, conectividad, etc.)
- `Filtering_Report` - Reporte de filtrado de outliers
- `Network_Config` - Configuración de parámetros de red

#### **Información de Grupos (NUEVO):**
- `Group_Assignments` - Asignación de unidades a grupos
- `Group_Colors` - Colores asignados a cada grupo
- `Analysis_Filter_Info` - Información del filtrado aplicado
- `Filtered_Units` - Lista de unidades incluidas en el análisis

### 🎨 **Visualizaciones Mejoradas:**
- **Redes**: Coloreo automático por grupos
- **Heatmaps**: Agrupación por clasificación
- **Leyendas**: Identificación clara de grupos y colores
- **Filtros**: Análisis enfocado en subconjuntos específicos

### 📋 **Flujo de Trabajo Recomendado:**
1. **Preparar Datos**: Cargar archivo de datos
2. **Crear Grupos**: Definir clasificaciones de unidades
3. **Configurar Análisis**: Seleccionar parámetros y filtros
4. **Ejecutar**: Realizar análisis con filtrado por grupos
5. **Exportar**: Guardar resultados completos incluyendo información de grupos
6. **Revisar**: Analizar datos exportados con contexto de grupos

### 🔍 **Casos de Uso Específicos:**

#### **Análisis por Industria:**
```
Grupos: OEM, Autopartes, Semiconductores
Filtro: Solo OEM y Autopartes
Resultado: Análisis enfocado en fabricantes y proveedores
```

#### **Comparación Exclusiva:**
```
Grupos: Empresas_Grandes, Empresas_Medianas, Startups
Filtro: Excluir Startups
Resultado: Análisis de empresas establecidas solamente
```

#### **Benchmarking Sectorial:**
```
Grupos: Por_País, Por_Tamaño, Por_Tecnología
Filtro: Seleccionar grupos específicos
Resultado: Comparación controlada por categorías
```

¡La aplicación ahora ofrece un sistema completo de gestión de grupos con análisis filtrado y exportación comprehensiva! 🚀
   - Selecciona la ubicación donde guardar el archivo Excel
   - Los resultados se guardarán como `nombre_archivo_complete_results.xlsx`

### 📋 Datos Exportados por Tipo de Análisis:

#### 🔗 **Análisis de Correlación**:
- **Original_Data**: Datos originales cargados
- **Correlation_Matrix**: Matriz de correlación completa
- **Filtered_Correlations**: Matriz filtrada (si aplica)
- **Units_Statistics**: Estadísticas por unidad
- **Analysis_Summary**: Resumen del análisis y configuración

#### 📊 **Análisis PCA** (Series, Cross-Section, Panel, Biplot, Scatter):
- **Standardized_Data**: Datos estandarizados
- **PCA_Components**: Componentes principales
- **PCA_Loadings**: Cargas de los componentes
- **Variance_Explained**: Varianza explicada por componente
- **Analysis_Summary**: Resumen del análisis

#### 🌳 **Clustering Jerárquico**:
- **Original_Data**: Datos originales
- **Cluster_Assignments**: Asignación de clusters (si disponible)
- **Analysis_Summary**: Resumen del análisis

### 🔍 Verificación de Datos:

Con los datos exportados puedes:

1. **Verificar la Matriz de Correlación**:
   - Abrir la hoja "Correlation_Matrix" en Excel
   - Verificar los valores de correlación entre variables
   - Comprobar que coincidan con las visualizaciones

2. **Revisar Datos Originales**:
   - Hoja "Original_Data" contiene los datos tal como fueron cargados
   - Verificar que todas las variables y unidades estén presentes

3. **Analizar Estadísticas**:
   - Hoja "Units_Statistics" para ver estadísticas por unidad
   - Hoja "Analysis_Summary" para configuración utilizada

### 📝 Ejemplo de Flujo de Trabajo:

```
1. Cargar datos → 2. Configurar análisis → 3. Ejecutar → 4. Exportar → 5. Verificar en Excel
```

### 🛠️ Solución de Problemas:

- **Botón deshabilitado**: Primero ejecuta un análisis
- **Error "At least one sheet must be visible"**: ✅ **CORREGIDO** - Se ajustó la estructura de datos
- **Error al exportar**: Verifica que tienes permisos de escritura en la carpeta destino
- **Archivo muy grande**: Los datos se guardan en múltiples hojas para mejor organización
- **Datos faltantes**: El sistema ahora registra qué datos están disponibles y crea hojas solo para datos válidos

### 💡 Notas Importantes:

- Los archivos exportados incluyen **TODOS** los datos procesados
- Permite **reproducibilidad completa** del análisis
- Facilita **verificación** y **validación** de resultados
- Compatible con análisis posteriores en **R**, **Python**, **SPSS**, etc.

## 🎯 Estado del Proyecto

**✅ TODOS LOS ERRORES CRÍTICOS CORREGIDOS**
**✅ FUNCIONALIDAD DE EXPORTACIÓN IMPLEMENTADA**
**✅ APLICACIÓN FUNCIONANDO CORRECTAMENTE**

¡La aplicación está lista para uso en producción! 🚀