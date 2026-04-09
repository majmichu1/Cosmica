# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Cosmica — macOS build

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
    excludes=['tkinter', 'matplotlib', 'notebook', 'jupyter'],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Cosmica',
    debug=False,
    strip=False,
    upx=False,  # UPX doesn't work well on macOS arm64
    console=False,
    target_arch='universal2',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='Cosmica',
)

app = BUNDLE(
    coll,
    name='Cosmica.app',
    icon=str(project_root / 'cosmica' / 'resources' / 'icons' / 'cosmica.icns'),
    bundle_identifier='com.cosmica.app',
    info_plist={
        'CFBundleShortVersionString': '0.1.0',
        'NSHighResolutionCapable': True,
    },
)
