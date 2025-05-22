# -*- mode: python ; coding: utf-8 -*-

datas = [
    ("config.json", "."),
    ("icons/*", "icons"),
    ("Hey-Igloo_en_windows_v3_0_0/*", "Hey-Igloo_en_windows_v3_0_0"),
    ("porcupine_resources/windows/*", "pvporcupine/resources/keyword_files/windows"),

    ("venv/Lib/site-packages/pvporcupine/lib/windows/amd64/*", "pvporcupine/lib/windows/amd64"),
    ("venv/Lib/site-packages/pvporcupine/lib/common/*", "pvporcupine/lib/common"),
    ("venv/Lib/site-packages/en_core_web_sm/en_core_web_sm-3.8.0", "en_core_web_sm")
]

a = Analysis(
    ['server_windows.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='server_windows',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='server_windows',
)
