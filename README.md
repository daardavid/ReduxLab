# PCA-SS: Análisis de Componentes Principales para Datos Socioeconómicos

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-2.0.0-brightgreen.svg)

Una aplicación completa para realizar análisis PCA (Análisis de Componentes Principales) sobre datos socioeconómicos con interfaz gráfica intuitiva.

## 🎯 Características Principales

- **Interfaz Gráfica Intuitiva**: GUI moderna construida con Tkinter
- **Múltiples Tipos de Análisis**: 
  - Serie de tiempo (análisis longitudinal)
  - Corte transversal (comparación entre países)
  - Panel 3D (trayectorias temporales)
- **Gestión Robusta de Datos Faltantes**: 10+ estrategias de imputación
- **Visualizaciones Profesionales**: Gráficos interactivos y exportables
- **Scatter Plot PCA Independiente (Nuevo en 2.0.0)**: Flujo autónomo de selección (archivo → indicadores → unidades → años → configuración) con ejecución automática.
- **Etiquetas Opcionales de Puntos (Nuevo)**: Muestra nombres de unidades/países con opción activable.
- **Varianza Explicada en Ejes (Nuevo)**: Los ejes muestran porcentaje de varianza explicada (PC1 / PC2).
- **Auto-run Config (Nuevo)**: El scatter se ejecuta automáticamente al aplicar configuración reduciendo fricción de uso.
- **Sistema de Proyectos**: Guarda y carga configuraciones completas
- **Soporte Multiidioma**: Español e Inglés
- **Exportación de Resultados**: Formatos Excel y SVG

## 📥 Descargar la aplicación (sin instalar Python)

Para usar **ReduxLab** sin instalar Python ni dependencias:

1. Ve a **[Releases](https://github.com/daardavid/PCA-SS/releases)** del repositorio.
2. En la última versión descarga:
   - **ReduxLab-Setup-X.X.X.exe**: instalador. Doble clic, sigue el asistente y listo.
   - **ReduxLab-X.X.X-portable.zip**: versión portable. Descomprime y ejecuta `ReduxLab.exe` dentro de la carpeta.

El instalador y el portable se generan con `build.bat` y la aplicación abre correctamente tras la instalación.

**Requisitos:** Windows 10 o superior, 64 bits.

## 🚀 Instalación Rápida (desde código fuente)

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clona o descarga el proyecto**:
```bash
git clone https://github.com/daardavid/PCA-SS.git
cd PCA-SS
```

2. **Instala las dependencias** (solo las necesarias para ejecutar la app):
```bash
pip install -r requirements.txt
```
   Para desarrollo, tests y documentación: `pip install -r requirements-dev.txt`

3. **Verifica la instalación**:
```bash
python check_dependencies.py
```

4. **Ejecuta la aplicación**:
```bash
python pca_gui.py
```

## 📋 Dependencias

- **Producción**: `requirements.txt` contiene todo lo necesario para ejecutar ReduxLab (pandas, numpy, scikit-learn, matplotlib, ttkbootstrap, openpyxl, etc.). No incluye pathlib2 (redundante en Python 3.4+); el análisis usa `sklearn.decomposition.PCA`, no el paquete `pca`.
- **Desarrollo**: `requirements-dev.txt` añade pytest, black, flake8, sphinx y sphinx-rtd-theme; no se empaquetan en el instalador.

## 🎮 Guía de Uso

### 1. Crear un Nuevo Proyecto
1. Ejecuta `python pca_gui.py`
2. Ve a **Proyecto → Nuevo proyecto**
3. Asigna un nombre descriptivo

### 2. Cargar Datos
- Usa archivos Excel (.xlsx, .xls)
- Formato esperado: Primera columna con códigos de países, columnas siguientes con años
- Cada hoja representa un indicador socioeconómico

### 3. Configurar Análisis

#### Serie de Tiempo (1 país, múltiples años)
```
Proyecto → Serie de Tiempo → Seleccionar:
- Archivo de datos
- Indicadores (múltiples)
- País (uno)
- Años (múltiples)
```

#### Corte Transversal (múltiples países, años específicos)
```
Proyecto → Corte Transversal → Seleccionar:
- Archivo de datos  
- Indicadores (múltiples)
- Países (múltiples)
- Años (uno o varios)
```

#### Panel 3D (múltiples países y años)
```
Proyecto → Panel 3D → Seleccionar:
- Archivo de datos
- Indicadores (múltiples)
- Países (múltiples)  
- Años (múltiples)
```

### 4. Ejecutar Análisis
- Haz clic en **Ejecutar** junto al tipo de análisis configurado
- La aplicación manejará automáticamente datos faltantes
- Se generarán visualizaciones interactivas

## 📊 Tipos de Visualización

### Biplots 2D
- Visualiza relaciones entre países e indicadores
- Vectores muestran dirección e intensidad de indicadores
- Puntos representan países coloreados por grupos

### Gráficos 3D
- Trayectorias de países a través del tiempo
- Primeros 3 componentes principales
- Animación interactiva

### Series de Tiempo
- Evolución temporal de indicadores
- Datos originales, imputados y estandarizados
- Múltiples subplots organizados

## 🔧 Gestión de Datos Faltantes

La aplicación incluye estrategias avanzadas de imputación:

- **Interpolación**: Lineal, polinomial, spline
- **Estadísticas**: Media, mediana, moda
- **Propagación**: Forward fill, backward fill
- **Métodos Avanzados**: Imputación iterativa, KNN
- **Personalizado**: Valores constantes, eliminación de filas

## 🎨 Personalización

### Colores y Grupos
- Asigna colores personalizados a grupos de países
- Edita títulos, leyendas y pies de página
- Configura unidades y etiquetas

### Configuración Global
- Tema claro/oscuro
- Idioma (español/inglés)
- Fuentes y tamaños personalizados

## 📁 Estructura del Proyecto

```
PCA-SS/
├── pca_gui.py              # Interfaz gráfica principal
├── data_loader_module.py   # Carga y transformación de datos
├── preprocessing_module.py # Limpieza e imputación
├── pca_module.py          # Algoritmos PCA
├── visualization_module.py # Generación de gráficos
├── constants.py           # Constantes y mapeos
├── dependency_manager.py  # Gestión de dependencias
├── check_dependencies.py  # Verificador de instalación
├── project_save_config.py # Configuración de proyectos
├── i18n_es.py            # Traducciones español
├── i18n_en.py            # Traducciones inglés
├── requirements.txt       # Dependencias del proyecto
└── README.md             # Esta documentación
```

## 🧪 Novedades 2.0.0

Principales mejoras respecto a 1.x:

- Nuevo módulo de Scatter Plot PCA independiente del resto de análisis.
- Añadida opción `show_labels` para anotar puntos.
- Inclusión de porcentaje de varianza explicada en títulos/ejes.
- Reescritura robusta del diálogo de configuración del Scatter para evitar errores de indentación.
- Ejecución automática tras aplicar configuración (elimina necesidad de botón extra de Run).
- Archivo `THIRD_PARTY_LICENSES.txt` para mayor claridad de cumplimiento.
- Compleción del texto de la licencia MIT (añadida cláusula de exoneración de garantías).

## 🧪 Testing

Para ejecutar las pruebas (próximamente):
```bash
pytest tests/
```

Para verificar el estilo de código:
```bash
black --check .
flake8 .
```

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

Transparencia de terceros: ver [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES.txt).

Historial de cambios: ver [CHANGELOG](CHANGELOG.md).

## 👨‍💻 Autor

**David Armando Abreu Rosique**
- Email: davidabreu1110@gmail.com
- GitHub: [@daardavid](https://github.com/daardavid)
- Ko-fi: [Invítame un café ☕](https://ko-fi.com/daardavid)

## 🙏 Agradecimientos

- Instituto de Investigaciones Económicas de la UNAM
- Equipo de desarrollo de scikit-learn
- Comunidad de matplotlib y pandas

## 📚 Referencias

- [Documentación de scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Análisis de Componentes Principales - Wikipedia](https://es.wikipedia.org/wiki/An%C3%A1lisis_de_componentes_principales)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

**¿Te gusta el proyecto?** ⭐ ¡Dale una estrella en GitHub!

**¿Necesitas ayuda?** 📧 Contacta al autor o abre un issue.
