# Cómo publicar un Release de ReduxLab

Para que los usuarios puedan descargar la aplicación desde GitHub sin clonar el repositorio:

1. **Generar los archivos** (en tu PC):
   - Ejecuta `build.bat` desde la raíz del proyecto.
   - Tras el build, en la carpeta `Output/` tendrás:
     - `ReduxLab-Setup-X.X.X.exe` (instalador)
     - `ReduxLab-X.X.X-portable.zip` (portable)

2. **Crear el Release en GitHub**:
   - Repositorio: `daardavid/PCA-SS` (o el tuyo).
   - Pestaña **Releases** → "Create a new release".
   - **Tag:** p. ej. `v2.1.0` (crear el tag si no existe).
   - **Título:** p. ej. "ReduxLab 2.1.0".
   - **Descripción:** opcional; puedes indicar qué incluye la versión.

3. **Adjuntar los dos archivos**:
   - En "Attach binaries by dropping them here or selecting them", arrastra o selecciona desde tu carpeta `Output/`:
     - `ReduxLab-Setup-2.1.0.exe`
     - `ReduxLab-2.1.0-portable.zip`
   - Publica el release.

Los enlaces de descarga quedarán en la página del release. No subas estos archivos al repositorio (están en `.gitignore`); solo como adjuntos del release.
