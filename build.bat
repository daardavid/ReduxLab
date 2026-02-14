@echo off
setlocal enabledelayedexpansion
:: Forzar UTF-8 para evitar caracteres corruptos en consola (Tamaño -> no Tama├▒o)
chcp 65001 >nul

echo ===============================================
echo    ReduxLab - Build Script
echo ===============================================
set "VERSION=2.1.0"
echo.

:: (opcional pero útil) trabajar relativo a la carpeta del script
pushd "%~dp0"

:: Verificar que estamos en el directorio correcto (raiz del proyecto)
if not exist "frontend\pca_gui_modern.py" (
    echo ERROR: No se encontro frontend\pca_gui_modern.py
    echo Ejecuta build.bat desde la carpeta del proyecto PCA (donde esta build.bat^)
    pause
    exit /b 1
)

:: Resolver ejecutable de Python (python o py)
for %%P in (python,py) do (
    %%P --version >nul 2>&1 && set "PYEXE=%%P"
)
if not defined PYEXE (
    echo ERROR: No se encontro Python en PATH
    pause
    exit /b 1
)

:: Asegurar que todas las dependencias estan instaladas (para que el exe lleve todo)
echo [1/7] Comprobando e instalando dependencias...
%PYEXE% -c "import tkinter, pandas, matplotlib, sklearn, numpy, seaborn, openpyxl, PIL, yaml, ttkbootstrap, scipy" 2>nul
if errorlevel 1 (
    echo Instalando dependencias desde requirements.txt...
    %PYEXE% -m pip install -r requirements.txt -q
    if errorlevel 1 (
        echo ERROR: No se pudieron instalar las dependencias
        echo Ejecuta manualmente: pip install -r requirements.txt
        pause
        exit /b 1
    )
)
%PYEXE% -c "import tkinter, pandas, matplotlib, sklearn, numpy, seaborn, openpyxl, ttkbootstrap, scipy" 2>nul
if errorlevel 1 (
    echo ERROR: Faltan dependencias. Ejecuta: pip install -r requirements.txt
    pause
    exit /b 1
)

:: Verificar PyInstaller
%PYEXE% -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Instalando PyInstaller...
    %PYEXE% -m pip install --user --upgrade pyinstaller
    if errorlevel 1 (
        echo ERROR: No se pudo instalar PyInstaller
        pause
        exit /b 1
    )
)

:: Limpiar directorios anteriores
echo [2/7] Limpiando directorios anteriores...
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"
if exist "Output" rmdir /s /q "Output"
if exist "__pycache__" rmdir /s /q "__pycache__"

:: Crear directorio de salida
mkdir "Output" 2>nul

:: Compilar con PyInstaller
echo [3/7] Compilando aplicacion con PyInstaller...
echo Esto puede tomar varios minutos...
%PYEXE% -m PyInstaller ReduxLab.spec --clean --noconfirm
if errorlevel 1 (
    echo ERROR: Fallo la compilacion con PyInstaller
    echo Revisa los errores arriba
    pause
    exit /b 1
)

:: Verificar que se genero el ejecutable
if not exist "dist\ReduxLab\ReduxLab.exe" (
    echo ERROR: No se genero el ejecutable
    pause
    exit /b 1
)

:: Copiar archivos adicionales para el usuario final
echo [4/7] Copiando archivos adicionales...
copy "README.md" "dist\ReduxLab\" >nul 2>&1
copy "THIRD_PARTY_LICENSES.txt" "dist\ReduxLab\" >nul 2>&1

:: Crear LICENSE.txt si no existe
if not exist "LICENSE.txt" (
    echo Creando LICENSE.txt basico...
    (
    echo MIT License
    echo.
    echo Copyright ^(c^) 2024 David Armando Abreu Rosique
    echo.
    echo Permission is hereby granted, free of charge, to any person obtaining a copy
    echo of this software and associated documentation files ^(the "Software"^), to deal
    echo in the Software without restriction, including without limitation the rights
    echo to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    echo copies of the Software, and to permit persons to whom the Software is
    echo furnished to do so, subject to the following conditions:
    echo.
    echo The above copyright notice and this permission notice shall be included in all
    echo copies or substantial portions of the Software.
    ) > LICENSE.txt
)
copy "LICENSE.txt" "dist\ReduxLab\" >nul 2>&1

:: Crear LEEME para quien reciba la app (no necesita Python)
(
echo ReduxLab - Como usar
echo.
echo EJECUTAR: Doble clic en ReduxLab.exe. No necesitas instalar Python.
echo.
echo COMPARTIR CON OTRA PC:
echo - Opcion 1: Enviale el instalador ReduxLab-Setup-X.X.X.exe ^(en carpeta Output^).
echo - Opcion 2: Comprime la carpeta completa "ReduxLab" y enviala; la otra persona descomprime y ejecuta ReduxLab.exe.
echo.
echo Requisitos: Windows 10 o superior, 64 bits.
) > "dist\ReduxLab\LEEME.txt"

:: Probar el ejecutable
echo [5/7] Probando el ejecutable generado...
echo Iniciando aplicacion por 5 segundos para verificar que funciona...
start "" "dist\ReduxLab\ReduxLab.exe"
timeout /t 5 /nobreak >nul
taskkill /f /im "ReduxLab.exe" 2>nul

:: Crear instalador con Inno Setup
echo [6/7] Creando instalador con Inno Setup...

:: Buscar Inno Setup en ubicaciones comunes
set "INNO_PATH="
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set "INNO_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if exist "C:\Program Files\Inno Setup 6\ISCC.exe" set "INNO_PATH=C:\Program Files\Inno Setup 6\ISCC.exe"
if exist "C:\Program Files (x86)\Inno Setup 5\ISCC.exe" set "INNO_PATH=C:\Program Files (x86)\Inno Setup 5\ISCC.exe"

if "!INNO_PATH!"=="" (
    echo ADVERTENCIA: No se encontro Inno Setup
    echo Descargalo desde: https://jrsoftware.org/isinfo.php
    echo.
    echo El ejecutable se encuentra en: dist\ReduxLab\ReduxLab.exe
    echo Puedes crear el instalador manualmente ejecutando ReduxLab-Setup.iss
) else (
    echo Encontrado Inno Setup en: !INNO_PATH!
    "!INNO_PATH!" "/OOutput" "ReduxLab-Setup.iss"
    
    if errorlevel 1 (
        echo ADVERTENCIA: Error al crear el instalador
        echo El ejecutable se encuentra en: dist\ReduxLab\
    ) else (
        echo.
        echo ===============================================
        echo           BUILD COMPLETADO EXITOSAMENTE
        echo ===============================================
        echo.
        echo Archivos generados:
        echo - Aplicacion: dist\ReduxLab\ReduxLab.exe
        echo - Instalador: Output\ReduxLab-Setup-%VERSION%.exe
        echo.
        
        :: Mostrar tamanos de archivo
        for %%F in ("dist\ReduxLab\ReduxLab.exe") do (
            set size=%%~zF
            set /a sizeMB=!size!/1024/1024
            echo Tamaño del ejecutable: !sizeMB! MB
        )
        
    for %%F in ("Output\ReduxLab-Setup-%VERSION%.exe") do (
            set size=%%~zF
            set /a sizeMB=!size!/1024/1024
            echo Tamaño del instalador: !sizeMB! MB
        )
        
        echo.
        echo Para compartir con otra computadora:
        echo   1. Instalador: Output\ReduxLab-Setup-%VERSION%.exe
        echo   2. Portable: Output\ReduxLab-%VERSION%-portable.zip
        echo La otra persona NO necesita instalar Python.
    )
)

:: Crear ZIP portable para compartir (carpeta lista para copiar o enviar)
echo [7/7] Creando paquete portable (ZIP)...
if exist "dist\ReduxLab" (
    powershell -NoProfile -Command "Compress-Archive -Path 'dist\ReduxLab' -DestinationPath 'Output\ReduxLab-%VERSION%-portable.zip' -Force" 2>nul
    if exist "Output\ReduxLab-%VERSION%-portable.zip" (
        for %%F in ("Output\ReduxLab-%VERSION%-portable.zip") do (
            set size=%%~zF
            set /a sizeMB=!size!/1024/1024
            echo Paquete portable creado: Output\ReduxLab-%VERSION%-portable.zip ^(!sizeMB! MB^)
        )
        echo La otra persona descomprime el ZIP y ejecuta ReduxLab.exe dentro de la carpeta.
    ) else (
        echo AVISO: No se pudo crear el ZIP. Puedes comprimir manualmente la carpeta dist\ReduxLab
    )
)

echo.
echo ===============================================
echo   Para compartir: usa el instalador .exe o el .zip portable
echo   En la otra PC no hace falta instalar Python.
echo ===============================================

echo.
echo ===============================================
popd
pause
