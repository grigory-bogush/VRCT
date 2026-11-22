# -*- mode: python ; coding: utf-8 -*-

import os

# UPX compression roughly doubles the PyInstaller collect step and adds
# a few seconds to sidecar startup. Default to disabled so ordinary
# rebuilds are fast; release scripts can opt in by setting
# VRCT_PYINSTALLER_UPX=1.
_use_upx = os.environ.get("VRCT_PYINSTALLER_UPX") == "1"


# whisper.cpp (Vulkan) runtime for the local Whisper engine on AMD/Intel GPUs.
# Optional: built by tools/whisper_cpp/build_whisper_cpp_vulkan.ps1. When it is
# absent the build still succeeds and "(Vulkan)" devices simply don't appear.
_whisper_cpp_bin = os.path.join(SPECPATH, '..', 'tools', 'whisper_cpp', 'bin')
_whisper_cpp_datas = [(_whisper_cpp_bin, 'whisper_cpp/')] if os.path.isfile(os.path.join(_whisper_cpp_bin, 'whisper-server.exe')) else []
# llama.cpp (Vulkan) runtime for the local AI translation engine's GGUF
# (TranslateGemma) weights. Optional: built by
# tools/llama_cpp/build_llama_cpp_vulkan.ps1.
_llama_cpp_bin = os.path.join(SPECPATH, '..', 'tools', 'llama_cpp', 'bin')
_llama_cpp_datas = [(_llama_cpp_bin, 'llama_cpp/')] if os.path.isfile(os.path.join(_llama_cpp_bin, 'llama-server.exe')) else []


a = Analysis(
    ['..\\src-python\\mainloop.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('./../src-python/models/overlay/fonts', 'fonts/'),
        ('./../src-python/models/translation/translation_settings/prompt', 'translation_settings/prompt/'),
        ('./../src-python/models/translation/translation_settings/languages', 'translation_settings/languages/'),
        ('./../src-python/models/ocr/onnx', 'ocr_onnx/'),
        ('./../.venv/Lib/site-packages/zeroconf', 'zeroconf/'),
        ('./../.venv/Lib/site-packages/openvr', 'openvr/'),
        ('./../.venv/Lib/site-packages/glfw', 'glfw/'),  # glfw3.dll (VR UI overlay texture) is loaded via ctypes, not found by analysis
        ('./../.venv/Lib/site-packages/faster_whisper', 'faster_whisper/'),
        ('./../.venv/Lib/site-packages/hf_xet', 'hf_xet/'),
        ('./../.venv/Lib/site-packages/rapidocr', 'rapidocr/'),
        ] + _whisper_cpp_datas + _llama_cpp_datas,
    hiddenimports=['telemetry_keys', 'faster_whisper.vad', 'models.transcription.audio_pipeline', 'rapidocr', 'cv2', 'models.ocr', 'windows_capture'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pandas', 'matplotlib', 'PyQt5'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VRCT-sidecar-x86_64-pc-windows-msvc',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=_use_upx,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=_use_upx,
    upx_exclude=[],
    name='.',
)
