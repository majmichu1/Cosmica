# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect torch data files (kernels, libs)
torch_datas = collect_data_files('torch', include_py_files=False)
# Collect astropy data
astropy_datas = collect_data_files('astropy')
# App resources and bundled model — paths are relative to spec file location
extra_datas = [
    (str(Path('cosmica') / 'resources'), str(Path('cosmica') / 'resources')),
    (str(Path('cosmica') / 'ai' / 'models' / 'cosmica_denoise_v1.pt'),
     str(Path('cosmica') / 'ai' / 'models')),
]

hidden_imports = [
    # PyQt6
    'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets', 'PyQt6.QtOpenGL',
    'PyQt6.QtOpenGLWidgets', 'PyQt6.QtPrintSupport',
    # PyTorch
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.cuda',
    'torch.backends.cudnn', 'torch.backends.cuda',
    # numpy / scipy / astropy
    'numpy', 'scipy', 'scipy.ndimage', 'scipy.optimize', 'scipy.signal',
    'astropy', 'astropy.io.fits', 'astropy.wcs', 'astropy.stats',
    # image libs
    'cv2', 'PIL', 'PIL.Image', 'tifffile', 'rawpy',
    # cosmica internals
    'cosmica', 'cosmica.core', 'cosmica.ui', 'cosmica.ai',
    'cosmica.ai.inference.denoise', 'cosmica.ai.inference.sharpen',
    'cosmica.ai.models.denoise_model', 'cosmica.ai.models.sharpen_model',
    'cosmica.ai.models.unet',
    # misc
    'platformdirs', 'requests', 'packaging',
] + collect_submodules('cosmica')

a = Analysis(
    ['cosmica/__main__.py'],
    pathex=['.'],
    binaries=[],
    datas=torch_datas + astropy_datas + extra_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'notebook', 'IPython'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Cosmica',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon='cosmica/resources/icon.ico' if Path('cosmica/resources/icon.ico').exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Cosmica',
)
