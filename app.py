import os
import sys
import traceback

# 0. PyInstaller GUI Safety: When console=False on Windows, sys.stdout and sys.stderr are None.
# Calling print() will raise AttributeError: 'NoneType' object has no attribute 'write'.
# Redirect all output and unhandled exceptions safely to app_debug.log and crash.log.
_APP_DIR = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))
_DEBUG_LOG_PATH = os.path.join(_APP_DIR, "app_debug.log")
_CRASH_LOG_PATH = os.path.join(_APP_DIR, "crash.log")


class SafeStream:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._f = None

    def write(self, s: str):
        try:
            if not self._f:
                self._f = open(self.log_path, "a", encoding="utf-8", buffering=1)
            self._f.write(s)
        except Exception:
            pass

    def flush(self):
        try:
            if self._f:
                self._f.flush()
        except Exception:
            pass


if sys.stdout is None or getattr(sys, "frozen", False):
    _safe_stream = SafeStream(_DEBUG_LOG_PATH)
    sys.stdout = _safe_stream
    sys.stderr = _safe_stream


def _handle_exception(exc_type, exc_value, exc_tb):
    err_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        with open(_CRASH_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(err_text + "\n")
    except Exception:
        pass
    try:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "ClassroomGestureMouse Error", f"Fatal Error:\n{err_text}")
    except Exception:
        pass


sys.excepthook = _handle_exception

# Ensure workspace root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QPixmap, QColor, QPainter, QFont
from core.config_manager import ConfigManager
from ui.main_window import TeacherControlHub


def main():
    print("=" * 70)
    print("ClassroomGestureMouse • Starting Desktop Application")
    print("=" * 70)

    # Enable Windows Per-Monitor DPI Awareness so SetCursorPos coordinates land accurately
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass

    # 1. Launch PyQt6 Application
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # Keep running in system tray

    # 2. Instant Startup Splash Screen (eliminates perceived startup delay)
    splash = QSplashScreen()
    splash_pix = QPixmap(420, 200)
    splash_pix.fill(QColor(14, 17, 22))
    painter = QPainter(splash_pix)
    painter.setPen(QColor(0, 255, 102))
    painter.drawRect(0, 0, 419, 199)
    painter.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
    painter.drawText(24, 55, "ClassroomGestureMouse")
    painter.setFont(QFont("Segoe UI", 9))
    painter.setPen(QColor(160, 175, 195))
    painter.drawText(24, 85, "Intelligent Vision Controller for Smart Displays")
    painter.setPen(QColor(0, 220, 255))
    painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
    painter.drawText(24, 145, "Loading AI models & vision pipeline...")
    painter.end()
    splash.setPixmap(splash_pix)
    splash.show()
    app.processEvents()

    # 3. Load Master Configuration & Initialize Core Hub
    config = ConfigManager.get_instance()
    window = TeacherControlHub(config)
    window.show()
    splash.finish(window)

    print("[+] ClassroomGestureMouse Desktop App running.")
    print("    - Floating status dot active on screen.")
    print("    - System tray icon enabled.")
    print("    - Use 'Lock Me' on dashboard or press tray icon to enroll.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
