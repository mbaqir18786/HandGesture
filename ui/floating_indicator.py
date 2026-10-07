import sys
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QFont


class FloatingStatusIndicator(QWidget):
    """
    Minimalist, Always-on-Top Floating Status Pill for Classroom Smart Screens.
    - Draggable anywhere on the screen.
    - Vibrant status dot (Green = Locked, Yellow = Lost, Gray = Inactive).
    - Displays Mode (NEAR / FAR) and distance.
    - Clicking toggles the Teacher Control Hub.
    """

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._drag_position = QPoint()
        self._is_dragging = False

        # Status properties
        self.status_color = QColor(100, 100, 100)  # Default Gray
        self.status_text = "OFF"
        self.mode_text = "NEAR"
        self.distance_text = "0.0m"

        self.setFixedSize(170, 42)
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(8)

        # Spacer for custom drawn status dot
        self.dot_spacer = QLabel()
        self.dot_spacer.setFixedSize(14, 14)
        layout.addWidget(self.dot_spacer)

        # Text labels
        self.label_info = QLabel("ClassroomMouse")
        self.label_info.setStyleSheet("color: #E0E0E0; font-family: 'Segoe UI', Arial; font-size: 11px; font-weight: bold;")
        layout.addWidget(self.label_info)

    def update_status(self, state_str: str, mode_str: str, distance_m: float):
        """Updates the indicator colors and text."""
        self.mode_text = mode_str
        self.distance_text = f"{distance_m:.1f}m"

        if state_str in ("LOCKED", "RE_LOCKED"):
            self.status_color = QColor(0, 255, 102)  # Vibrant Green
            self.status_text = "ACTIVE"
            self.label_info.setText(f"LOCKED • {self.mode_text} {self.distance_text}")
            self.label_info.setStyleSheet("color: #00FF66; font-family: 'Segoe UI', Arial; font-size: 11px; font-weight: bold;")
        elif state_str == "LOST":
            self.status_color = QColor(255, 204, 0)  # Amber Yellow
            self.status_text = "FROZEN"
            self.label_info.setText("LOST • MOUSE FROZEN")
            self.label_info.setStyleSheet("color: #FFCC00; font-family: 'Segoe UI', Arial; font-size: 11px; font-weight: bold;")
        else:
            self.status_color = QColor(120, 120, 120)  # Gray
            self.status_text = "IDLE"
            self.label_info.setText("IDLE • UNENROLLED")
            self.label_info.setStyleSheet("color: #AAAAAA; font-family: 'Segoe UI', Arial; font-size: 11px; font-weight: bold;")

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw Pill Glassmorphic Background
        bg_color = QColor(18, 18, 22, 230)
        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(QColor(40, 40, 50), 1.5))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 20, 20)

        # Draw Glowing Status Dot
        painter.setBrush(QBrush(self.status_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(14, 14, 12, 12)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = True
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._is_dragging and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        if self._is_dragging:
            self._is_dragging = False
            # Check if this was a simple click (not dragged)
            self.clicked.emit()
            event.accept()
