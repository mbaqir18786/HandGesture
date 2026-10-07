"""
Automated Build Script for ClassroomGestureMouse Standalone Windows App
Bundles Python code, PyTorch/YOLO models, OpenCV ONNX models, and MediaPipe assets
into a standalone portable Windows application (no Python required on target PC).
"""

import os
import sys
import subprocess
import shutil


def check_and_install_pyinstaller():
    print("[1/4] Checking PyInstaller...")
    try:
        import PyInstaller
        print("      PyInstaller already installed.")
    except ImportError:
        print("      Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "pyinstaller"])


def create_spec_file():
    print("[2/4] Generating PyInstaller specification...")
    spec_content = """# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect ultralytics and mediapipe data files
extra_datas = [
    ('config.json', '.'),
    ('version.json', '.'),
    ('models', 'models'),
    ('yolov8n-pose.pt', '.'),
    ('live_mouse_control_using_hand_gestures/hand_landmarker.task', 'live_mouse_control_using_hand_gestures'),
]

try:
    extra_datas += collect_data_files('ultralytics')
except Exception:
    pass

hidden_imports = [
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'ultralytics',
    'lap',
    'lapx',
    'cv2',
    'mediapipe',
    'pyttsx3',
    'pyttsx3.drivers',
    'pyttsx3.drivers.sapi5',
    'pyautogui',
    'numpy',
]

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=extra_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'unittest', 'test'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ClassroomGestureMouse',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""
    with open("ClassroomGestureMouse.spec", "w", encoding="utf-8") as f:
        f.write(spec_content)
    print("      ClassroomGestureMouse.spec created (Single Portable .EXE mode).")


def build_app():
    print("[3/4] Compiling Single Standalone Windows Executable (.exe)...")
    subprocess.check_call([sys.executable, "-m", "PyInstaller", "--noconfirm", "ClassroomGestureMouse.spec"])


def package_output():
    print("[4/4] Finalizing Standalone Executable...")
    single_exe = os.path.abspath("dist/ClassroomGestureMouse.exe")
    if os.path.exists(single_exe):
        size_mb = os.path.getsize(single_exe) / (1024 * 1024)
        print("\n" + "=" * 70)
        print("BUILD SUCCESSFUL!")
        print("=" * 70)
        print(f"Direct Executable File: {single_exe} ({size_mb:.1f} MB)")
        print("\nHow to use:")
        print("1. Transfer 'ClassroomGestureMouse.exe' directly via Pen Drive to any computer.")
        print("2. Double-click to run! (No zip extraction, no setup, no Python needed).")
        print("3. When you release updates on GitHub, users just click 'Updates -> Update Now'!")
        print("=" * 70)


if __name__ == "__main__":
    check_and_install_pyinstaller()
    create_spec_file()
    build_app()
    package_output()
