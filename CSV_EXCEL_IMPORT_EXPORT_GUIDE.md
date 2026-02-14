# 📁 Guía de Importación/Exportación de Grupos - CSV y Excel

## 🎯 **Funcionalidades Implementadas**

### ✅ **Formatos Soportados:**
- **📤 JSON**: Formato completo con metadatos
- **📊 CSV**: Formato simple para hojas de cálculo  
- **📈 Excel**: Múltiples hojas con resumen detallado

### 🔧 **Características Técnicas:**

#### **📊 Formato CSV:**
```csv
Unit,Group
Tesla,OEM
Ford Motor,OEM
Aptiv,Autopartes_electronicas
Denso,Autopartes_electronicas
Huawei,Electronics_Suppliers
```

**Especificaciones:**
- **Codificación**: UTF-8
- **Separador**: Coma (,)
- **Encabezados**: Primera fila con nombres de columnas
- **Columnas Requeridas**: Mínimo 2 (Unit, Group)
- **Orden**: Las unidades se exportan ordenadas alfabéticamente

#### **📈 Formato Excel:**
```
Archivo: groups.xlsx
Hojas:
├── Group_Assignments (Principal)
│   ├── Unit: Nombre de la unidad
│   └── Group: Nombre del grupo
├── Group_Summary (Resumen)
│   ├── Group: Nombre del grupo
│   ├── Unit_Count: Cantidad de unidades
│   ├── Description: Descripción del grupo
│   ├── Color: Color asignado
│   ├── Created: Fecha de creación
│   └── Last_Modified: Última modificación
```

**Especificaciones:**
- **Formato**: Excel 2007+ (.xlsx)
- **Múltiples Hojas**: Datos principales + resumen
- **Metadatos**: Información completa de grupos
- **Compatible**: LibreOffice, Google Sheets, Excel

## 🚀 **Cómo Usar**

### **1. Exportar Grupos**

#### **Desde la Aplicación:**
```
1. Abrir ReduxLab
2. Clic en "🏷️ Gestionar Grupos"
3. Clic en "🏷️ Manage Universal Groups"
4. En fila "Export:" seleccionar formato:
   - 📤 JSON: Exportación completa
   - 📊 CSV: Para hojas de cálculo
   - 📈 Excel: Múltiples hojas
5. Elegir ubicación y nombre del archivo
```

#### **Formatos de Salida:**

**CSV Export:**
- **Archivo**: `groups.csv`
- **Contenido**: Tabla simple Unit,Group
- **Uso**: Procesamiento de datos, importación a otras herramientas

**Excel Export:**
- **Archivo**: `groups.xlsx`
- **Contenido**: Múltiples hojas con datos y metadatos
- **Uso**: Análisis detallado, presentaciones, colaboración

### **2. Importar Grupos**

#### **Preparar Datos:**

**Para CSV:**
```csv
Unit,Group
Company_A,Technology
Company_B,Technology  
Company_C,Manufacturing
Company_D,Manufacturing
Company_E,Services
```

**Para Excel:**
```
Crear archivo .xlsx con hoja que contenga:
- Columna A: Unit (nombres de unidades)
- Columna B: Group (nombres de grupos)
- Opcional: Hojas adicionales serán ignoradas
```

#### **Desde la Aplicación:**
```
1. Preparar archivo CSV o Excel
2. Abrir ReduxLab → "🏷️ Gestionar Grupos"
3. Clic en "🏷️ Manage Universal Groups"
4. En fila "Import:" seleccionar formato:
   - 📥 JSON: Importar configuración completa
   - 📊 CSV: Importar desde tabla CSV
   - 📈 Excel: Importar desde archivo Excel
5. Seleccionar archivo preparado
6. Confirmar importación
```

### **3. Gestión de Conflictos**

#### **Conflictos de Nombres:**
```
Si el grupo ya existe:
1. El sistema detecta el conflicto
2. Muestra diálogo de confirmación
3. Opciones:
   ✅ Sobrescribir: Reemplaza grupo existente
   ❌ Mantener: Conserva grupo original
```

#### **Validaciones Automáticas:**
- ✅ **Formato de archivo**: Verifica estructura correcta
- ✅ **Columnas requeridas**: Al menos Unit y Group
- ✅ **Datos válidos**: No valores vacíos o nulos
- ✅ **Codificación**: Manejo correcto de caracteres especiales

## 📋 **Ejemplos Prácticos**

### **Ejemplo 1: Clasificación por Industria**
```csv
Unit,Group
Microsoft,Technology
Apple,Technology
Google,Technology
Ford,Automotive
Tesla,Automotive
ExxonMobil,Energy
Chevron,Energy
```

### **Ejemplo 2: Clasificación por Tamaño**
```csv
Unit,Group
Amazon,Large_Corp
Microsoft,Large_Corp
Apple,Large_Corp
Zoom,Medium_Corp
Slack,Medium_Corp
StartupX,Small_Corp
StartupY,Small_Corp
```

### **Ejemplo 3: Clasificación Regional**
```csv
Unit,Group
Toyota,Asia_Pacific
Samsung,Asia_Pacific
Sony,Asia_Pacific
BMW,Europe
Volkswagen,Europe
Mercedes-Benz,Europe
Ford,North_America
General Motors,North_America
Tesla,North_America
```

## 🔧 **Solución de Problemas**

### **Errores Comunes:**

#### **"Formato de archivo inválido"**
- **Causa**: Archivo CSV mal formateado o Excel corrupto
- **Solución**: Verificar que el archivo tenga al menos 2 columnas (Unit, Group)

#### **"Error de codificación"**
- **Causa**: Caracteres especiales no soportados
- **Solución**: Guardar CSV como UTF-8

#### **"No se encontraron datos"**
- **Causa**: Archivo vacío o solo encabezados
- **Solución**: Verificar que existan filas de datos

#### **"Conflictos de grupos"**
- **Causa**: Grupos con nombres duplicados
- **Solución**: Elegir sobrescribir o mantener según necesidad

### **Mejores Prácticas:**

1. **Nombres de Grupos**: Usar nombres descriptivos sin espacios (usar _ o -)
2. **Backup**: Exportar grupos existentes antes de importar nuevos
3. **Validación**: Revisar datos en hoja de cálculo antes de importar
4. **Incrementales**: Para cambios grandes, hacer importaciones en lotes pequeños

## 📊 **Casos de Uso Avanzados**

### **Migración de Sistemas:**
```
Fuente: Sistema externo → CSV export
Proceso: Formatear datos → ReduxLab import
Resultado: Grupos integrados en ReduxLab
```

### **Colaboración en Equipo:**
```
1. Analista exporta grupos base → Excel
2. Equipo revisa/edita clasificaciones
3. Analista importa datos actualizados
4. Análisis con nuevas clasificaciones
```

### **Automatización:**
```
1. Script externo genera CSV con clasificaciones
2. ReduxLab importa automáticamente
3. Análisis ejecutado con grupos actualizados
```

¡Las funcionalidades de importación/exportación CSV y Excel están completamente implementadas y probadas! 🎉