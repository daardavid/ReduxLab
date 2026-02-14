# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Recopilar datos de matplotlib
matplotlib_datas = []
try:
    import matplotlib
    matplotlib_datas = collect_data_files('matplotlib')
except ImportError:
    pass

# Datos adicionales de tu aplicación
app_datas = [
    ('config', 'config'),
    ('i18n_es.py', '.'),
    ('i18n_en.py', '.'),
    ('requirements.txt', '.'),
    ('README.md', '.'),
    ('THIRD_PARTY_LICENSES.txt', '.'),
]

# Módulos ocultos que PyInstaller podría no detectar automáticamente
hidden_imports = [
    'tkinter.messagebox',
    'tkinter.simpledialog', 
    'tkinter.filedialog',
    'tkinter.ttk',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.figure',
    'matplotlib.pyplot',
    'pandas',
    'numpy',
    'sklearn.decomposition',
    'sklearn.preprocessing', 
    'sklearn.impute',
    'seaborn',
    'openpyxl',
    'PIL',
    'yaml',
    'json',
    'logging',
    'threading',
    'queue',
    'webbrowser',
    'platform',
    'subprocess',
    'sys',
    'os'
]

# Análisis principal
a = Analysis(
    ['pca_gui.py'],  # Tu archivo principal
    pathex=[],
    binaries=[],
    datas=app_datas + matplotlib_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['test', 'tests', '__pycache__'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filtrar archivos innecesarios para reducir tamaño
a.datas = [x for x in a.datas if not x[0].startswith('matplotlib/tests')]
a.datas = [x for x in a.datas if not x[0].startswith('numpy/tests')]
a.datas = [x for x in a.datas if not x[0].startswith('pandas/tests')]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ReduxLab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Sin ventana de consola
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None  # Cambia por 'app_icon.ico' si tienes un icono
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ReduxLab'
)
