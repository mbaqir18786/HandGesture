import sys
import os
import time
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QInputDialog, QLineEdit, QSystemTrayIcon,
    QMenu, QMessageBox, QCheckBox, QSlider, QDialog, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize, QUrl
from PyQt6.QtGui import QImage, QPixmap, QIcon, QColor, QFont, QDesktopServices

from core.config_manager import ConfigManager
from core.camera_stream import CameraStream
from core.tracker import PersonTracker, PersonDetection
from core.security_engine import SecurityEngine, SecurityState
from core.mouse_controller import DualModeMouseController
from core.voice_feedback import VoiceFeedback
from core.update_manager import UpdateManager
from ui.floating_indicator import FloatingStatusIndicator


class UpdateDialog(QDialog):
    """Modern modal dialog showing update details with live download progress."""
    def __init__(self, parent, version_data: dict, updater: UpdateManager):
        super().__init__(parent)
        self.updater = updater
        self.version_data = version_data
        self.setWindowTitle("ClassroomGestureMouse • Update Available")
        self.setFixedSize(500, 270)
        self.setStyleSheet("""
            QDialog {
                background-color: #0E1116;
                color: #FFFFFF;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel { color: #CBD5E1; }
            QProgressBar {
                background-color: #171B22;
                border: 1px solid #28303E;
                border-radius: 6px;
                text-align: center;
                color: #FFFFFF;
                height: 22px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #00FF66;
                border-radius: 5px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        new_ver = version_data.get("latest_version", "New Version")
        notes = version_data.get("release_notes", "Performance improvements and bug fixes.")

        title_lbl = QLabel(f"🚀 New Update Available: v{new_ver}")
        title_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #00FF66;")
        layout.addWidget(title_lbl)

        notes_lbl = QLabel(f"<b>What's New:</b><br>{notes}")
        notes_lbl.setWordWrap(True)
        notes_lbl.setStyleSheet("font-size: 12px; color: #94A3B8;")
        layout.addWidget(notes_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("font-size: 12px; color: #00E5FF; font-weight: bold;")
        self.status_lbl.hide()
        layout.addWidget(self.status_lbl)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.btn_cancel = QPushButton("Later")
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #212733; color: #CBD5E1; border: 1px solid #333C4D;
                border-radius: 6px; padding: 8px 16px; font-size: 12px;
            }
            QPushButton:hover { background-color: #2B3342; }
        """)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_update = QPushButton("⚡ Update Now")
        self.btn_update.setStyleSheet("""
            QPushButton {
                background-color: #00FF66; color: #0A0D12; font-weight: bold;
                border: none; border-radius: 6px; padding: 8px 18px; font-size: 12px;
            }
            QPushButton:hover { background-color: #00E55C; }
        """)
        self.btn_update.clicked.connect(self._start_download)
        btn_row.addWidget(self.btn_update)

        layout.addLayout(btn_row)

    def _start_download(self):
        download_url = self.version_data.get("download_url", "")
        if not download_url:
            QMessageBox.warning(self, "Update", "No direct download URL provided in release.")
            return

        self.btn_update.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.show()
        self.status_lbl.show()
        self.status_lbl.setText("Downloading update from GitHub...")

        def on_progress(pct):
            QTimer.singleShot(0, lambda: self._update_progress(pct))

        def on_complete(success, result):
            if success:
                QTimer.singleShot(0, lambda: self._on_finish(result))
            else:
                QTimer.singleShot(0, lambda: self._on_failed(result))

        self.updater.download_and_apply_update_async(download_url, on_progress, on_complete)

    def _update_progress(self, pct):
        self.progress_bar.setValue(int(pct))
        self.status_lbl.setText(f"Downloading update: {int(pct)}%")

    def _on_finish(self, new_exe_path):
        self.status_lbl.setText("Update downloaded! Restarting app into new version...")
        self.status_lbl.setStyleSheet("font-size: 12px; color: #00FF66; font-weight: bold;")
        self.progress_bar.setValue(100)
        QTimer.singleShot(1200, lambda: UpdateManager.apply_update_and_restart(new_exe_path))

    def _on_failed(self, error_msg):
        self.btn_update.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.status_lbl.setText(f"Update failed: {error_msg}")
        self.status_lbl.setStyleSheet("font-size: 12px; color: #FF4444; font-weight: bold;")


class TeacherControlHub(QMainWindow):
    """
    Teacher Control Hub & Desktop Application for ClassroomGestureMouse.
    - Full-screen / windowed dashboard with live vision feed.
    - Teacher PIN enrollment & biometric zero-trust management.
    - System Tray & Always-On-Top floating pill integration.
    - 'Hide Camera Feed' toggle for privacy during classroom lectures.
    """

    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.setWindowTitle("ClassroomGestureMouse • Teacher Control Hub")
        self.resize(1080, 720)
        self.setMinimumSize(800, 560)

        # Core Vision & Control Engines
        self._init_engines()

        # UI Setup
        self._init_styles()
        self._init_ui()
        self._init_tray()

        # Always-On-Top Floating Pill Indicator
        self.pill = FloatingStatusIndicator()
        self.pill.move(40, 40)
        self.pill.clicked.connect(self._toggle_visibility)
        self.pill.show()

        # Main processing timer (30 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_frame)
        self.timer.start(16)  # ~60 Hz event loop poll

        self.frame_idx = 0
        self.persons = []
        self.is_teacher_active = False
        self.current_mode = "NEAR"
        self.distance_m = 1.0

        # Background Update Check
        if self.config.get("updates.check_on_startup", True):
            QTimer.singleShot(2500, self._check_updates_background)

    def _init_engines(self):
        """Initializes backend computer vision & audio engines."""
        near_thresh = self.config.get("near_mode.max_distance_meters", 3.5)
        max_dist = self.config.get("far_mode.max_distance_meters", 10.0)
        box_cfg = self.config.get("near_mode.interaction_box", {"x_min": 0.22, "x_max": 0.78, "y_min": 0.20, "y_max": 0.60})

        # Tracker
        self.tracker = PersonTracker(
            model_name=self.config.get("tracking.yolo_model", "yolov8n-pose.pt"),
            min_person_conf=self.config.get("tracking.min_person_confidence", 0.50),
            near_threshold_m=near_thresh,
            max_distance_m=max_dist,
        )

        # Security Engine
        self.security = SecurityEngine(
            lost_timeout_sec=self.config.get("security.lost_timeout_sec", 0.8),
        )

        # Mouse Controller
        self.mouse = DualModeMouseController(
            interaction_box=box_cfg,
            smoothing=self.config.get("near_mode.smoothing", 5.0),
            deadzone_px=self.config.get("near_mode.deadzone_px", 5.0),
            pinch_close_threshold=self.config.get("near_mode.pinch_close_threshold", 32.0),
            pinch_open_threshold=self.config.get("near_mode.pinch_open_threshold", 45.0),
            double_tap_window_sec=self.config.get("near_mode.double_tap_window_sec", 0.40),
            dwell_time_sec=self.config.get("far_mode.dwell_time_sec", 1.0),
            dwell_radius_px=self.config.get("far_mode.dwell_radius_px", 25.0),
        )

        # Voice Feedback
        voice_cfg = self.config.get("voice_feedback", {})
        self.voice = VoiceFeedback(
            enabled=voice_cfg.get("enabled", True),
            speech_rate=voice_cfg.get("speech_rate", 180),
            volume=voice_cfg.get("volume", 0.9),
            alerts=voice_cfg.get("alerts", {}),
        )

        # Camera Stream
        self.stream = CameraStream.from_config(self.config.get_all())
        self.stream.start()

        # In-App Update Manager
        update_cfg = self.config.get("updates", {})
        self.updater = UpdateManager(
            current_version=self.config.get("version", "1.0.0"),
            update_url=update_cfg.get("update_url", ""),
        )

    def _init_styles(self):
        """Sleek modern dark-mode stylesheet."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0E1116;
            }
            QWidget {
                color: #E2E8F0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QFrame.card {
                background-color: #171B22;
                border: 1px solid #28303E;
                border-radius: 12px;
                padding: 12px;
            }
            QPushButton.primary-btn {
                background-color: #00FF66;
                color: #0A0D12;
                font-weight: bold;
                font-size: 13px;
                border: none;
                border-radius: 8px;
                padding: 10px 18px;
            }
            QPushButton.primary-btn:hover {
                background-color: #00E55C;
            }
            QPushButton.secondary-btn {
                background-color: #212733;
                color: #CBD5E1;
                font-size: 12px;
                border: 1px solid #333C4D;
                border-radius: 8px;
                padding: 8px 14px;
            }
            QPushButton.secondary-btn:hover {
                background-color: #2B3342;
                color: #FFFFFF;
            }
            QLabel.title {
                font-size: 18px;
                font-weight: bold;
                color: #FFFFFF;
            }
            QLabel.subtitle {
                font-size: 12px;
                color: #8C9BAE;
            }
            QLabel.stat-val {
                font-size: 20px;
                font-weight: bold;
                color: #00FF66;
            }
            QLabel.stat-lbl {
                font-size: 11px;
                color: #8C9BAE;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
        """)

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # ---------------- Top Navigation Bar ----------------
        top_bar = QHBoxLayout()
        title_box = QVBoxLayout()
        title_lbl = QLabel("ClassroomGestureMouse")
        title_lbl.setProperty("class", "title")
        subtitle_lbl = QLabel("Intelligent Touch-Free Vision Controller for Smart Displays")
        subtitle_lbl.setProperty("class", "subtitle")
        title_box.addWidget(title_lbl)
        title_box.addWidget(subtitle_lbl)
        top_bar.addLayout(title_box)

        top_bar.addStretch()

        self.btn_hide_feed = QPushButton("👁️ Hide Feed (Classroom Mode)")
        self.btn_hide_feed.setProperty("class", "secondary-btn")
        self.btn_hide_feed.setCheckable(True)
        self.btn_hide_feed.toggled.connect(self._toggle_feed_visibility)
        top_bar.addWidget(self.btn_hide_feed)

        self.btn_check_update = QPushButton("🔄 Updates")
        self.btn_check_update.setProperty("class", "secondary-btn")
        self.btn_check_update.clicked.connect(self._check_updates_clicked)
        top_bar.addWidget(self.btn_check_update)

        self.btn_minimize_tray = QPushButton("Minimize to Tray")
        self.btn_minimize_tray.setProperty("class", "secondary-btn")
        self.btn_minimize_tray.clicked.connect(self.hide)
        top_bar.addWidget(self.btn_minimize_tray)

        main_layout.addLayout(top_bar)

        # ---------------- Metrics Summary Bar ----------------
        metrics_frame = QFrame()
        metrics_frame.setProperty("class", "card")
        metrics_layout = QHBoxLayout(metrics_frame)
        metrics_layout.setContentsMargins(16, 12, 16, 12)

        # Card 1: State
        c1 = QVBoxLayout()
        self.val_state = QLabel("UNENROLLED")
        self.val_state.setProperty("class", "stat-val")
        self.val_state.setStyleSheet("color: #AAAAAA;")
        lbl_s = QLabel("Security State")
        lbl_s.setProperty("class", "stat-lbl")
        c1.addWidget(self.val_state)
        c1.addWidget(lbl_s)
        metrics_layout.addLayout(c1)

        metrics_layout.addStretch()

        # Card 2: Mode & Distance
        c2 = QVBoxLayout()
        self.val_mode = QLabel("NEAR (0.0m)")
        self.val_mode.setProperty("class", "stat-val")
        lbl_m = QLabel("Interaction Mode")
        lbl_m.setProperty("class", "stat-lbl")
        c2.addWidget(self.val_mode)
        c2.addWidget(lbl_m)
        metrics_layout.addLayout(c2)

        metrics_layout.addStretch()

        # Card 3: Live FPS
        c3 = QVBoxLayout()
        self.val_fps = QLabel("30.0")
        self.val_fps.setProperty("class", "stat-val")
        lbl_fps = QLabel("Stream Rate")
        lbl_fps.setProperty("class", "stat-lbl")
        c3.addWidget(self.val_fps)
        c3.addWidget(lbl_fps)
        metrics_layout.addLayout(c3)

        main_layout.addWidget(metrics_frame)

        # ---------------- Center: Video Viewport & Controls ----------------
        center_layout = QHBoxLayout()
        center_layout.setSpacing(16)

        # Left: Video Screen
        video_card = QFrame()
        video_card.setProperty("class", "card")
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(8, 8, 8, 8)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: #080A0D; border-radius: 8px;")
        video_layout.addWidget(self.video_label)
        center_layout.addWidget(video_card, stretch=3)

        # Right: Teacher Controls Sidebar
        sidebar = QFrame()
        sidebar.setProperty("class", "card")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(14)

        sec_header = QLabel("Teacher Controls")
        sec_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFFFFF;")
        sidebar_layout.addWidget(sec_header)

        self.btn_lock = QPushButton("🔒 Lock Me (Enroll Teacher)")
        self.btn_lock.setProperty("class", "primary-btn")
        self.btn_lock.clicked.connect(self._handle_lock_request)
        sidebar_layout.addWidget(self.btn_lock)

        self.btn_unlock = QPushButton("🔓 Unlock / Reset")
        self.btn_unlock.setProperty("class", "secondary-btn")
        self.btn_unlock.clicked.connect(self._handle_unlock)
        sidebar_layout.addWidget(self.btn_unlock)

        sidebar_layout.addSpacing(10)

        # Settings
        pref_header = QLabel("Preferences")
        pref_header.setStyleSheet("font-size: 13px; font-weight: bold; color: #CBD5E1;")
        sidebar_layout.addWidget(pref_header)

        self.chk_flip = QCheckBox("Mirror Camera Feed")
        self.chk_flip.setChecked(self.config.get("camera.flip_horizontal", True))
        self.chk_flip.toggled.connect(self._toggle_flip)
        sidebar_layout.addWidget(self.chk_flip)

        self.chk_voice = QCheckBox("Voice Feedback Alerts")
        self.chk_voice.setChecked(self.config.get("voice_feedback.enabled", True))
        self.chk_voice.toggled.connect(self._toggle_voice)
        sidebar_layout.addWidget(self.chk_voice)

        sidebar_layout.addStretch()

        # Exit
        btn_exit = QPushButton("Quit Application")
        btn_exit.setStyleSheet("background-color: #381818; color: #FF6666; border-radius: 6px; padding: 8px;")
        btn_exit.clicked.connect(self.close)
        sidebar_layout.addWidget(btn_exit)

        center_layout.addWidget(sidebar, stretch=1)
        main_layout.addLayout(center_layout)

    def _init_tray(self):
        """Windows System Tray Icon integration."""
        self.tray_icon = QSystemTrayIcon(self)
        # Generate custom colored pixmap icon
        pix = QPixmap(16, 16)
        pix.fill(QColor(0, 255, 102))
        self.tray_icon.setIcon(QIcon(pix))

        tray_menu = QMenu()
        show_action = tray_menu.addAction("Open Control Hub")
        show_action.triggered.connect(self.showNormal)

        lock_action = tray_menu.addAction("Lock Me (Enroll)")
        lock_action.triggered.connect(self._handle_lock_request)

        unlock_action = tray_menu.addAction("Unlock Teacher")
        unlock_action.triggered.connect(self._handle_unlock)

        tray_menu.addSeparator()
        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(self.close)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._toggle_visibility()

    def _toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.showNormal()
            self.activateWindow()

    def _toggle_feed_visibility(self, is_hidden: bool):
        if is_hidden:
            self.btn_hide_feed.setText("👁️ Show Feed")
            self.video_label.setText("🔒 Camera Feed Hidden for Classroom Presentation\n(Gesture Mouse is Active in Background)")
            self.video_label.setStyleSheet("color: #00FF66; font-size: 14px; font-weight: bold; background-color: #080A0D; border-radius: 8px;")
        else:
            self.btn_hide_feed.setText("👁️ Hide Feed (Classroom Mode)")
            self.video_label.setText("")

    def _toggle_flip(self, checked: bool):
        self.stream.set_flip(checked)
        self.config.set("camera.flip_horizontal", checked, auto_save=True)

    def _toggle_voice(self, checked: bool):
        self.voice.enabled = checked
        self.config.set("voice_feedback.enabled", checked, auto_save=True)

    def _handle_lock_request(self):
        # Prompt for Teacher PIN if configured
        teacher_pin = self.config.get("security.teacher_pin", "1234")
        pin_input, ok = QInputDialog.getText(
            self, "Teacher PIN Verification", "Enter Teacher PIN to Lock / Enroll:",
            QLineEdit.EchoMode.Password
        )
        if ok and pin_input == teacher_pin:
            ret, frame = self.stream.read()
            if ret and frame is not None and self.persons:
                best = max(self.persons, key=lambda p: p.distance_info.get("proximity_score", 0.0))
                self.security.enroll(frame, best)
                self.voice.trigger_alert("locked", force=True)
            else:
                QMessageBox.warning(self, "Enrollment", "Please ensure you are standing in front of the camera.")
        elif ok:
            QMessageBox.critical(self, "Security Denied", "Incorrect Teacher PIN.")
            self.voice.trigger_alert("unauthorized", force=True)

    def _handle_unlock(self):
        self.security.unlock()
        self.voice.trigger_alert("lost", force=True)

    def _check_updates_background(self):
        """Silently checks for updates in the background on startup."""
        self.updater.check_for_updates_async(self._on_background_update_result)

    def _on_background_update_result(self, is_avail: bool, data: dict, msg: str):
        if is_avail and data:
            QTimer.singleShot(0, lambda: self._show_update_dialog(data, is_manual=False))

    def _check_updates_clicked(self):
        """User clicked manual 'Check for Updates' button."""
        self.btn_check_update.setText("🔄 Checking...")
        self.btn_check_update.setEnabled(False)

        def _on_manual_result(is_avail, data, msg):
            QTimer.singleShot(0, lambda: self._handle_manual_update_ui(is_avail, data, msg))

        self.updater.check_for_updates_async(_on_manual_result)

    def _handle_manual_update_ui(self, is_avail: bool, data: dict, msg: str):
        self.btn_check_update.setText("🔄 Updates")
        self.btn_check_update.setEnabled(True)
        if is_avail and data:
            self._show_update_dialog(data, is_manual=True)
        else:
            QMessageBox.information(
                self,
                "ClassroomGestureMouse Updates",
                f"You are up to date! (Current version: v{self.config.get('version', '1.0.0')})\n{msg}"
            )

    def _show_update_dialog(self, data: dict, is_manual: bool = False):
        new_ver = data.get("latest_version", "New Version")
        notes = data.get("release_notes", "Performance improvements and bug fixes.")
        url = data.get("download_url", self.config.get("updates.releases_page", "https://github.com/"))

        reply = QMessageBox.question(
            self,
            "Software Update Available",
            f"<b>A new version of ClassroomGestureMouse is available!</b><br><br>"
            f"<b>Installed:</b> v{self.config.get('version', '1.0.0')}<br>"
            f"<b>Latest:</b> v{new_ver}<br><br>"
            f"<b>What's New:</b><br>{notes}<br><br>"
            f"Would you like to open the download page now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply == QMessageBox.StandardButton.Yes:
            QDesktopServices.openUrl(QUrl(url))

    def _process_frame(self):
        """Main processing loop executing AI pipeline and updating UI."""
        ret, frame = self.stream.read()
        if not ret or frame is None:
            return

        self.frame_idx += 1
        h, w = frame.shape[:2]

        # Tracking Cadence: full speed when LOST/Far; interleaved in Near Mode
        should_run_yolo = (
            not self.is_teacher_active
            or self.current_mode == "FAR"
            or (self.frame_idx % 2 == 0)
            or len(self.persons) == 0
        )

        prev_state = self.security.state

        if should_run_yolo:
            self.persons = self.tracker.track(frame)
            state, teacher_id, metadata = self.security.update(frame, self.persons)
            self.is_teacher_active = state in (SecurityState.LOCKED, SecurityState.RE_LOCKED) and teacher_id is not None

            # Audio state transitions
            if state in (SecurityState.LOCKED, SecurityState.RE_LOCKED) and prev_state == SecurityState.LOST:
                self.voice.trigger_alert("relocked")
            elif state == SecurityState.LOST and prev_state in (SecurityState.LOCKED, SecurityState.RE_LOCKED):
                self.voice.trigger_alert("lost")

        # Active teacher resolution
        teacher_id = self.security.teacher_track_id
        teacher_person = next((p for p in self.persons if p.track_id == teacher_id), None) if self.is_teacher_active else None

        if teacher_person is not None:
            self.distance_m = teacher_person.distance_info.get("distance_m", 1.0)
            self.current_mode = teacher_person.distance_info.get("mode", "NEAR")
        elif self.persons:
            self.distance_m = self.persons[0].distance_info.get("distance_m", 1.0)
            self.current_mode = self.persons[0].distance_info.get("mode", "NEAR")

        # Execute Mouse Control
        if self.current_mode == "NEAR":
            self.mouse.process_near_mode(frame, (h, w), is_security_unlocked=self.is_teacher_active)
        elif self.current_mode == "FAR" and teacher_person is not None:
            self.mouse.process_far_mode(teacher_person.right_wrist, (h, w), is_security_unlocked=self.is_teacher_active)

        # Update Metrics Bar & Floating Pill
        state_str = self.security.state.value
        self.pill.update_status(state_str, self.current_mode, self.distance_m)

        if state_str in ("LOCKED", "RE_LOCKED"):
            self.val_state.setText(f"LOCKED (ID #{teacher_id})")
            self.val_state.setStyleSheet("color: #00FF66; font-size: 20px; font-weight: bold;")
        elif state_str == "LOST":
            self.val_state.setText("LOST (FROZEN)")
            self.val_state.setStyleSheet("color: #FFCC00; font-size: 20px; font-weight: bold;")
        else:
            self.val_state.setText("UNENROLLED")
            self.val_state.setStyleSheet("color: #AAAAAA; font-size: 20px; font-weight: bold;")

        self.val_mode.setText(f"{self.current_mode} ({self.distance_m:.1f}m)")
        fps_val = self.stream.get_fps()
        self.val_fps.setText(f"{fps_val:.1f} FPS")

        # Render Video Viewport if feed is visible
        if not self.btn_hide_feed.isChecked():
            display_frame = self.tracker.draw_visuals(frame.copy(), self.persons, teacher_id=teacher_id)
            # Render QImage
            rgb_img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            ih, iw, ich = rgb_img.shape
            bytes_per_line = ich * iw
            q_img = QImage(rgb_img.data, iw, ih, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.video_label.setPixmap(pixmap)

    def _check_updates_background(self):
        """Asynchronously checks for updates in the background on startup."""
        update_url = self.config.get("updates.update_url", "")
        current_ver = self.config.get("version", "1.0.0")
        updater = UpdateManager(current_version=current_ver, update_url=update_url)

        def _callback(is_avail, data, msg):
            if is_avail and data:
                QTimer.singleShot(0, lambda: self._show_update_dialog(data, updater))

        updater.check_for_updates_async(_callback)

    def _check_updates_clicked(self):
        """User triggered update check via header button."""
        update_url = self.config.get("updates.update_url", "")
        current_ver = self.config.get("version", "1.0.0")
        updater = UpdateManager(current_version=current_ver, update_url=update_url)

        self.btn_check_update.setText("Checking...")
        self.btn_check_update.setEnabled(False)

        def _callback(is_avail, data, msg):
            def _ui_action():
                self.btn_check_update.setText("🔄 Updates")
                self.btn_check_update.setEnabled(True)
                if is_avail and data:
                    self._show_update_dialog(data, updater)
                else:
                    QMessageBox.information(
                        self,
                        "Check for Updates",
                        f"You are running the latest version (v{current_ver})!\n\nStatus: {msg}",
                    )

            QTimer.singleShot(0, _ui_action)

        updater.check_for_updates_async(_callback)

    def _show_update_dialog(self, data: dict, updater: UpdateManager):
        """Opens the update prompt dialog."""
        dialog = UpdateDialog(self, data, updater)
        dialog.exec()

    def closeEvent(self, event):
        """Graceful shutdown of all resources."""
        self.timer.stop()
        self.stream.stop()
        self.voice.stop()
        self.pill.close()
        self.tray_icon.hide()
        event.accept()
