import os
import sys

# Ensure workspace root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from PyQt6.QtWidgets import QApplication
from core.config_manager import ConfigManager
from ui.main_window import TeacherControlHub


def main():
    print("=" * 70)
    print("ClassroomGestureMouse • Starting Desktop Application")
    print("=" * 70)

    # 1. Load Master Configuration
    config = ConfigManager.get_instance()

    # 2. Launch PyQt6 Application
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # Keep running in system tray

    # 3. Create & Show Main Dashboard
    window = TeacherControlHub(config)
    window.show()

    print("[+] ClassroomGestureMouse Desktop App running.")
    print("    - Floating status dot active on screen.")
    print("    - System tray icon enabled.")
    print("    - Use 'Lock Me' on dashboard or press tray icon to enroll.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
