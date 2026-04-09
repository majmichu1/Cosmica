# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Cosmica — Windows build

import sys
from pathlib import Path

block_cipher = None
project_root = Path(SPECPATH).parent.parent

a = Analysis(
    [str(project_root / 'cosmica' / '__main__.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / 'cosmica' / 'resources'), 'cosmica/resources'),
    ],
    hiddenimports=[
        'cosmica.core.device_manager',
        'cosmica.core.image_io',
        'cosmica.core.calibration',
        'cosmica.core.stacking',
        'cosmica.core.stretch',
        'cosmica.core.background',
        'cosmica.core.project',
        'cosmica.ui.app',
        'cosmica.ui.main_window',
        'cosmica.ui.theme',
        'cosmica.licensing.license_manager',
        'cosmica.updater.auto_updater',
        'torch',
        'torchvision',
        'astropy',
        'ccdproc',
        'scipy',
        'cv2',
        'PIL',
        'PyQt6',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'notebook', 'jupyter'],
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
    console=False,  # GUI app, no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / 'cosmica' / 'resources' / 'icons' / 'cosmica.ico'),
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
