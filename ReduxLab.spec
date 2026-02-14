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

# Datos adicionales (rutas desde la raíz del proyecto)
# frontend.frames se incluye como paquete Python al rastrear imports
app_datas = [
    ('config', 'config'),
    ('backend/i18n_es.py', '.'),
    ('backend/i18n_en.py', '.'),
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
    'ttkbootstrap',
    'ttkbootstrap.constants',
    'ttkbootstrap.themes',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.figure',
    'matplotlib.pyplot',
    'pandas',
    'numpy',
    'scipy.stats',
    'sklearn.decomposition',
    'sklearn.preprocessing', 
    'sklearn.impute',
    'seaborn',
    'openpyxl',
    'pyarrow',
    'sqlalchemy',
    'sqlite3',
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

# Análisis principal (ejecutar PyInstaller desde la raíz del proyecto)
a = Analysis(
    ['frontend/pca_gui_modern.py'],  # Entry point
    pathex=['.'],
    binaries=[],
    datas=app_datas + matplotlib_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'test', 'tests', '__pycache__',
        'pytest', 'pytest_core', '_pytest',
        'sphinx', 'IPython', 'notebook', 'jupyter',
        'matplotlib.tests', 'numpy.testing', 'pandas.tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filtrar archivos innecesarios para reducir tamaño
def _filter_datas(datas):
    out = []
    for x in datas:
        p = x[0] if isinstance(x[0], str) else ''
        if p.startswith('matplotlib/tests') or p.startswith('numpy/tests') or p.startswith('pandas/tests'):
            continue
        if p.startswith('matplotlib/sample_data') or p.startswith('matplotlib/mpl-data/sample_data'):
            continue
        out.append(x)
    return out
a.datas = _filter_datas(a.datas)

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
