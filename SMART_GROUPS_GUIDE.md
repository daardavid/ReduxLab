# 🧠 Guía de Grupos Inteligentes (Smart Group Detection)

## 📋 Descripción

La funcionalidad de **Grupos Inteligentes** permite que el sistema detecte automáticamente coincidencias entre los nombres de unidades en tu base de datos y los grupos existentes, incluso cuando los nombres no coinciden exactamente.

## ✨ Características

### ✅ Casos de Uso Resueltos

El sistema Smart Group Detection puede identificar correctamente:

| Base de Datos | Grupo Guardado | Match Type |
|---------------|---------------|------------|
| `Huawei 2011` | `Huawei` | ✓ Contains |
| `Huawei 2012` | `Huawei` | ✓ Contains |
| `TESLA` | `Tesla` | ✓ Case Insensitive |
| `Tesla Inc` | `Tesla` | ✓ Fuzzy Match |
| `Ford Motor Company` | `Ford Motor` | ✓ Word Match |
| `General Motors Corp` | `General Motors` | ✓ Fuzzy Match |
| `apple inc.` | `Apple` | ✓ Case + Fuzzy |

### 🎯 Tipos de Matching

1. **Exact Match**: Coincidencia exacta de nombres
2. **Case Insensitive**: Ignora mayúsculas/minúsculas
3. **Contains Match**: Detecta cuando un nombre contiene otro (ej: "Huawei 2011" contiene "Huawei")
4. **Fuzzy Match**: Usa similitud de texto para detectar variaciones

## 🔧 Configuración

### Acceso

1. Abrir PCA-SS
2. Ir a cualquier análisis (Series, Cross-Section, Panel 3D)
3. Clic en **"🏷️ Gestionar Grupos Universales"**
4. Sección **"🧠 Smart Matching Settings"**

### Opciones de Configuración

#### ✅ Enable Smart Group Detection
- **Toggle**: Habilita o deshabilita la detección inteligente
- **Default**: ✅ Habilitado
- **Recomendación**: Mantener habilitado para máxima flexibilidad

#### 🎚️ Similarity Threshold (Umbral de Similitud)
- **Rango**: 50% - 95%
- **Default**: 75%
- **Función**: Controla qué tan similar debe ser un nombre para considerarse un match

**Recomendaciones de Threshold:**

| Threshold | Uso Recomendado | Ejemplo |
|-----------|-----------------|---------|
| **90-95%** | Nombres muy similares | "Tesla Inc" ↔ "Tesla" |
| **75-85%** | Variaciones comunes (RECOMENDADO) | "Ford Motor Co" ↔ "Ford Motor" |
| **60-75%** | Matching más permisivo | "Microsoft Corp" ↔ "Microsoft" |
| **50-60%** | Muy permisivo (usar con cuidado) | Puede dar falsos positivos |

## 📖 Cómo Usar

### Paso 1: Crear Grupos Base

1. En el Group Manager, crear grupos con nombres **base** de las empresas:
   ```
   Grupo: "Tech Companies"
   Unidades: Tesla, Apple, Google, Microsoft
   ```

2. No es necesario agregar todas las variaciones manualmente

### Paso 2: Habilitar Smart Matching

1. En la sección "🧠 Smart Matching Settings"
2. Marcar ✅ "Enable Smart Group Detection"
3. Ajustar threshold según necesidad (recomendado: 75%)

### Paso 3: Preview de Matches (Opcional)

Antes de ejecutar un análisis, puedes ver qué se va a detectar:

1. Cargar tu archivo de datos
2. En Group Manager, clic en **"🔍 Preview Smart Matches"**
3. Se mostrará una tabla con:
   - Unidad en Base de Datos
   - Unidad Coincidente en Grupo
   - Nombre del Grupo
   - Nivel de Confianza

**Ejemplo de Preview:**

```
Database Unit          Matched To      Group          Confidence
─────────────────────────────────────────────────────────────────
Tesla 2020            Tesla           Tech Companies  95% - High ✓
Huawei 2021           Huawei          Chinese Tech    95% - High ✓
Ford Motor Co         Ford Motor      Auto Makers     95% - High ✓
```

### Paso 4: Ejecutar Análisis

1. Ir al módulo de análisis deseado
2. Cargar datos
3. Clic en **"🔄 Load Current Groups"**
4. El sistema automáticamente:
   - Intenta matches exactos primero
   - Si no encuentra, usa Smart Matching
   - Asigna colores automáticamente
   - Muestra resumen de grupos cargados

### Paso 5: Verificar Resultados

En el frame de análisis verás:
```
✅ 3 groups loaded (45/50 units grouped)
```

Esto indica:
- 3 grupos detectados
- 45 de 50 unidades fueron asignadas a grupos
- 5 unidades quedaron sin grupo (aparecerán como "Ungrouped")

## 🔧 Funciones Avanzadas

### Limpiar Cache

El sistema guarda en caché los matches encontrados para performance.

**Cuándo limpiar:**
- Cambios en nombres de grupos
- Cambio de threshold
- Problemas de detección

**Cómo limpiar:**
1. Group Manager
2. Clic en **"🗑️ Clear Cache"**
3. Los matches se recalcularán en el próximo análisis

### Ajustar Threshold Dinámicamente

Si encuentras que:
- **Demasiados falsos positivos**: Subir threshold (→ 85-90%)
- **Muy pocas detecciones**: Bajar threshold (→ 65-70%)

Puedes ajustar el slider y usar "Preview Smart Matches" para ver el efecto.

## 📊 Ejemplos Prácticos

### Ejemplo 1: Análisis de Empresas por Año

**Situación:**
- Grupos: `{Automotive: [Tesla, Ford, GM]}`
- Base de datos: `Tesla 2010, Tesla 2011, ..., Tesla 2023`

**Con Smart Matching:**
✅ Todas las variaciones de Tesla se asignan automáticamente al grupo "Automotive"

**Sin Smart Matching:**
❌ Ninguna detección - habría que crear manualmente "Tesla 2010", "Tesla 2011", etc.

### Ejemplo 2: Nombres con Variaciones Legales

**Situación:**
- Grupos: `{Tech: [Apple, Google, Microsoft]}`
- Base de datos: `Apple Inc., GOOGLE LLC, Microsoft Corporation`

**Con Smart Matching (75%):**
✅ Detecta correctamente:
- "Apple Inc." → "Apple"
- "GOOGLE LLC" → "Google" 
- "Microsoft Corporation" → "Microsoft"

### Ejemplo 3: Importación desde Excel

**Situación:**
Tienes un Excel con columnas:
```
Company                    | Year | Revenue
──────────────────────────────────────────
Huawei Technologies Inc    | 2020 | 100M
Huawei Technologies Inc    | 2021 | 120M
Tesla Motors              | 2020 | 50M
```

**Proceso:**
1. Crear grupo: `{Chinese Tech: [Huawei]}`
2. Habilitar Smart Matching (75%)
3. Cargar Excel
4. Sistema detecta "Huawei Technologies Inc" → "Huawei"
5. Análisis aplica colores y agrupación automáticamente

## ⚙️ Configuración Persistente

Las configuraciones se guardan automáticamente en:
```
config/universal_groups.json
```

Contenido:
```json
{
  "groups": { ... },
  "smart_matching_settings": {
    "enabled": true,
    "threshold": 0.75
  }
}
```

Esto asegura que tus preferencias se mantengan entre sesiones.

## 🐛 Troubleshooting

### Problema: No se detectan matches esperados

**Soluciones:**
1. ✅ Verificar que Smart Matching esté habilitado
2. 📉 Reducir threshold (ej: de 75% a 65%)
3. 🔍 Usar "Preview Smart Matches" para verificar
4. 🗑️ Limpiar cache

### Problema: Demasiados falsos positivos

**Soluciones:**
1. 📈 Aumentar threshold (ej: de 75% a 85%)
2. 📝 Revisar nombres en grupos (asegurar que sean específicos)
3. ❌ Desactivar Smart Matching temporalmente

### Problema: Sistema lento

**Soluciones:**
1. 🗑️ Limpiar cache
2. 📊 Reducir número de grupos o unidades
3. ⚡ El sistema cachea resultados - solo lento en primera ejecución

## 📈 Mejores Prácticas

### ✅ DO's

1. **Usar nombres base en grupos**: "Tesla" en lugar de "Tesla Inc"
2. **Preview antes de análisis**: Verificar matches con datos reales
3. **Threshold en 75%**: Balance óptimo para la mayoría de casos
4. **Mantener Smart Matching habilitado**: Máxima flexibilidad

### ❌ DON'Ts

1. **No usar threshold muy bajo**: < 60% puede causar falsos positivos
2. **No mezclar idiomas**: Mantener consistencia (todo inglés o todo español)
3. **No ignorar el preview**: Siempre verificar antes de análisis importantes

## 🎓 Algoritmo Técnico

Para usuarios avanzados, el sistema usa:

1. **Normalización**: 
   - Lowercase
   - Remoción de caracteres especiales
   - Consolidación de espacios

2. **Contains Match** (Prioridad Alta):
   - Detección de palabras contenidas
   - Word boundary matching
   - Score: 95%

3. **Fuzzy Match** (Prioridad Media):
   - SequenceMatcher (difflib)
   - Calcula ratio de similitud
   - Score: 0-100%

4. **Threshold Check**:
   - Solo retorna match si score ≥ threshold
   - Cache de resultados para performance

## 📚 Referencias

- **Archivo**: `group_manager.py` (líneas 333-502)
- **Test Suite**: `test_smart_groups.py`
- **Configuración**: `config/universal_groups.json`

## 🆘 Soporte

Si tienes problemas o sugerencias:
1. Ejecutar `test_smart_groups.py` para diagnóstico
2. Revisar logs en `logs/`
3. Contactar soporte con detalles de configuración

---

**Versión**: 2.1.0  
**Última actualización**: Noviembre 2025  
**Autor**: Sistema PCA-SS
