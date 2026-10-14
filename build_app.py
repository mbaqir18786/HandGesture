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


def create_spec_file(mode: str = "onedir"):
    print(f"[2/4] Generating PyInstaller specification (Mode: {mode.upper()})...")
    
    is_onefile = (mode == "onefile")
    
    spec_content = """# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

# Collect all binaries, data, and submodules for mediapipe, ultralytics, and torchvision
mp_datas, mp_binaries, mp_hiddenimports = [], [], []
try:
    mp_datas, mp_binaries, mp_hiddenimports = collect_all('mediapipe')
except Exception:
    pass

ultra_datas, ultra_binaries, ultra_hiddenimports = [], [], []
try:
    ultra_datas, ultra_binaries, ultra_hiddenimports = collect_all('ultralytics')
except Exception:
    pass

tv_datas, tv_binaries, tv_hiddenimports = [], [], []
try:
    tv_datas, tv_binaries, tv_hiddenimports = collect_all('torchvision')
except Exception:
    pass

# Ensure torchvision native C++ operator extensions (_C_*.pyd, image_*.pyd) are included
import torchvision
tv_dir = os.path.dirname(torchvision.__file__)
tv_custom_binaries = []
for f in os.listdir(tv_dir):
    if f.endswith(('.pyd', '.dll')):
        tv_custom_binaries.append((os.path.join(tv_dir, f), 'torchvision'))

extra_datas = [
    ('config.json', '.'),
    ('version.json', '.'),
    ('models', 'models'),
    ('yolov8n-pose.pt', '.'),
    ('live_mouse_control_using_hand_gestures/hand_landmarker.task', 'live_mouse_control_using_hand_gestures'),
] + mp_datas + ultra_datas + tv_datas

extra_binaries = mp_binaries + ultra_binaries + tv_binaries + tv_custom_binaries

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
    'torch',
    'torchvision',
    'matplotlib',
    'scipy',
    'speech_recognition',
    'pyaudio',
    'win32clipboard',
] + mp_hiddenimports + ultra_hiddenimports + tv_hiddenimports

# Exclude unnecessary heavy tools to shrink size without breaking dependencies
excludes = [
    'tkinter',
    'IPython',
    'notebook',
    'pytest',
]

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=extra_binaries,
    datas=extra_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
"""

    if is_onefile:
        spec_content += """
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
    else:
        spec_content += """
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ClassroomGestureMouse',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ClassroomGestureMouse',
)
"""

    with open("ClassroomGestureMouse.spec", "w", encoding="utf-8") as f:
        f.write(spec_content)
    print(f"      ClassroomGestureMouse.spec created ({mode.upper()} mode).")


def build_app():
    print("[3/4] Compiling Windows Executable with PyInstaller...")
    # Clean up stale single-file builds from dist to avoid user confusion
    for stale in ["dist/ClassroomGestureMouse.exe", "dist/ClassroomGestureMouse_App.zip"]:
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except Exception:
                pass
    subprocess.check_call([sys.executable, "-m", "PyInstaller", "--noconfirm", "ClassroomGestureMouse.spec"])


def package_output(mode: str = "onedir"):
    print("[4/4] Finalizing Output Package...")
    print("\n" + "=" * 70)
    print("BUILD SUCCESSFUL!")
    print("=" * 70)

    if mode == "onedir":
        dist_dir = os.path.abspath("dist/ClassroomGestureMouse")
        exe_file = os.path.join(dist_dir, "ClassroomGestureMouse.exe")
        print(f"Fast-Launch Application Folder: {dist_dir}")
        print(f"Executable: {exe_file}")
        print("\n[+] Performance Benefit:")
        print("- OPENS IN 1-2 SECONDS! (No decompression lag on startup)")
        print("- To share or put on a USB pen drive: Copy the 'ClassroomGestureMouse' folder directly.")
    else:
        single_exe = os.path.abspath("dist/ClassroomGestureMouse.exe")
        if os.path.exists(single_exe):
            size_mb = os.path.getsize(single_exe) / (1024 * 1024)
            print(f"Single Portable Binary: {single_exe} ({size_mb:.1f} MB)")
            print("\nNote: Single-file .exe extracts DLLs to Temp on every start (takes 15-30s).")
            print("Use 'python build_app.py --onedir' for instant 1-2s launch!")
    print("=" * 70)


if __name__ == "__main__":
    mode = "onedir"
    if "--onefile" in sys.argv:
        mode = "onefile"
    elif "--onedir" in sys.argv:
        mode = "onedir"

    check_and_install_pyinstaller()
    create_spec_file(mode)
    build_app()
    package_output(mode)
