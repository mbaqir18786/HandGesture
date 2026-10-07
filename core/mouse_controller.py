import os
import time
import math
from typing import Optional, Tuple, Dict, Any, List
import ctypes
import numpy as np
import cv2
import pyautogui

from core.one_euro_filter import OneEuroFilter2D

# Disable pyautogui failsafe
pyautogui.FAILSAFE = False

# Windows Direct Mouse Injection API for 0ms input lag
try:
    user32 = ctypes.windll.user32
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_ABSOLUTE = 0x8000
    HAS_WIN32 = True
except Exception:
    HAS_WIN32 = False


class DualModeMouseController:
    """
    Dual-Mode Touch-Free Mouse Engine with Zero-Trust Security Gating:
    - Near Mode (< 3.5m): MediaPipe Hand Landmarker tracking Index Knuckle (Landmark 5) + Single/Double Pinch Tap.
    - Far Mode (3.5m - 10m): YOLO Pose Wrist Tracking + One Euro Filter Smoothing + 1.0s Dwell Click.
    - Zero-Trust: Mouse strictly disabled if teacher is LOST.
    """

    def __init__(
        self,
        screen_size: Optional[Tuple[int, int]] = None,
        interaction_box: Optional[Dict[str, float]] = None,
        smoothing: float = 5.0,
        deadzone_px: float = 5.0,
        pinch_close_threshold: float = 32.0,
        pinch_open_threshold: float = 45.0,
        double_tap_window_sec: float = 0.40,
        dwell_time_sec: float = 1.0,
        dwell_radius_px: float = 25.0,
    ):
        if screen_size is None:
            self.screen_width, self.screen_height = pyautogui.size()
        else:
            self.screen_width, self.screen_height = screen_size

        self.box_cfg = interaction_box or {"x_min": 0.22, "x_max": 0.78, "y_min": 0.20, "y_max": 0.60}
        self.smoothing = smoothing
        self.deadzone_px = deadzone_px
        self.pinch_close_threshold = pinch_close_threshold
        self.pinch_open_threshold = pinch_open_threshold
        self.double_tap_window_sec = double_tap_window_sec
        self.dwell_time_sec = dwell_time_sec
        self.dwell_radius_px = dwell_radius_px

        # Cursor State
        self.curr_cursor_x = self.screen_width // 2
        self.curr_cursor_y = self.screen_height // 2
        self.prev_x = self.curr_cursor_x
        self.prev_y = self.curr_cursor_y

        # Near Mode Hand Tap State
        self.is_pinched = False
        self.last_pinch_time = 0.0
        self.feedback_text = "Idle"
        self.feedback_color = (0, 255, 0)

        # Far Mode Dwell State
        self.one_euro = OneEuroFilter2D(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
        self.dwell_start_pos: Optional[Tuple[float, float]] = None
        self.dwell_start_time: float = 0.0
        self.dwell_progress: float = 0.0
        self.last_wrist_pos: Optional[Tuple[float, float]] = None

        # MediaPipe Hand Landmarker
        self._hand_landmarker = None
        self._mp_timestamp_ms = 0
        self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        """Initializes MediaPipe Hand Landmarker with hand_landmarker.task model."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            from core.config_manager import get_resource_path
            model_path = get_resource_path(os.path.join("live_mouse_control_using_hand_gestures", "hand_landmarker.task"))

            if os.path.exists(model_path):
                base_options = mp_python.BaseOptions(model_asset_path=model_path)
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=1,
                    min_hand_detection_confidence=0.6,
                    min_hand_presence_confidence=0.6,
                    min_tracking_confidence=0.6,
                )
                self._hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
                print("[DualModeMouseController] MediaPipe Hand Landmarker initialized.")
        except Exception as e:
            print(f"[DualModeMouseController] MediaPipe init error: {e}")

    def _move_cursor(self, target_x: float, target_y: float) -> None:
        """Low-latency OS cursor movement."""
        tx = int(np.clip(target_x, 0, self.screen_width - 1))
        ty = int(np.clip(target_y, 0, self.screen_height - 1))

        diff_x = tx - self.prev_x
        diff_y = ty - self.prev_y

        if abs(diff_x) > self.deadzone_px or abs(diff_y) > self.deadzone_px:
            curr_x = int(self.prev_x + diff_x / self.smoothing)
            curr_y = int(self.prev_y + diff_y / self.smoothing)

            if HAS_WIN32:
                user32.SetCursorPos(curr_x, curr_y)
            else:
                pyautogui.moveTo(curr_x, curr_y, _pause=False)

            self.prev_x, self.prev_y = curr_x, curr_y
            self.curr_cursor_x, self.curr_cursor_y = curr_x, curr_y

    def _execute_click(self, is_double: bool = False) -> None:
        """Low-latency OS click execution."""
        if is_double:
            pyautogui.doubleClick()
            self.feedback_text = "DOUBLE CLICK!"
            self.feedback_color = (255, 255, 0)
        else:
            pyautogui.click()
            self.feedback_text = "SINGLE CLICK!"
            self.feedback_color = (0, 0, 255)

    def process_near_mode(
        self,
        frame: np.ndarray,
        frame_shape: Tuple[int, int],
        is_security_unlocked: bool = True,
    ) -> Dict[str, Any]:
        """
        Near Mode (< 3.5m): Knuckle Tracking & Pinch Single/Double Tap.
        """
        result_meta = {
            "mode": "NEAR",
            "active": False,
            "cursor_pos": (self.curr_cursor_x, self.curr_cursor_y),
            "feedback": self.feedback_text,
            "feedback_color": self.feedback_color,
            "hand_landmarks": None,
        }

        if self._hand_landmarker is None or not is_security_unlocked:
            self.feedback_text = "FROZEN (Zero-Trust)" if not is_security_unlocked else "Idle"
            self.feedback_color = (0, 180, 255) if not is_security_unlocked else (150, 150, 150)
            return result_meta

        h, w = frame_shape[:2]
        current_time = time.time()

        import mediapipe as mp
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detect_result = self._hand_landmarker.detect_for_video(mp_image, self._mp_timestamp_ms)
        self._mp_timestamp_ms += 33

        if detect_result.hand_landmarks:
            hand = detect_result.hand_landmarks[0]
            result_meta["hand_landmarks"] = hand
            result_meta["active"] = True

            thumb_tip = hand[4]
            index_tip = hand[8]
            tracking_point = hand[5]  # Index Knuckle (Landmark 5)

            # Calculate pinch distance
            pinch_dist = math.hypot((thumb_tip.x - index_tip.x) * w, (thumb_tip.y - index_tip.y) * h)

            # Detect Pinch Down Event
            if pinch_dist < self.pinch_close_threshold:
                if not self.is_pinched:
                    time_since_last = current_time - self.last_pinch_time
                    if time_since_last < self.double_tap_window_sec:
                        self._execute_click(is_double=True)
                        self.last_pinch_time = 0.0
                    else:
                        self._execute_click(is_double=False)
                        self.last_pinch_time = current_time
                    self.is_pinched = True
            elif pinch_dist > self.pinch_open_threshold:
                self.is_pinched = False
                self.feedback_text = "Aiming Cursor"
                self.feedback_color = (0, 255, 0)

                # Cursor Movement (when not pinching)
                raw_x = np.interp(tracking_point.x, [self.box_cfg["x_min"], self.box_cfg["x_max"]], [0, self.screen_width])
                raw_y = np.interp(tracking_point.y, [self.box_cfg["y_min"], self.box_cfg["y_max"]], [0, self.screen_height])
                self._move_cursor(raw_x, raw_y)

        result_meta["cursor_pos"] = (self.curr_cursor_x, self.curr_cursor_y)
        result_meta["feedback"] = self.feedback_text
        result_meta["feedback_color"] = self.feedback_color
        return result_meta

    def process_far_mode(
        self,
        wrist_coord: Tuple[float, float, float],
        frame_shape: Tuple[int, int],
        is_security_unlocked: bool = True,
    ) -> Dict[str, Any]:
        """
        Far Mode (3.5m - 10m): Wrist Tracking + One Euro Filter Smoothing + 1.0s Dwell Click.
        """
        result_meta = {
            "mode": "FAR",
            "active": False,
            "cursor_pos": (self.curr_cursor_x, self.curr_cursor_y),
            "dwell_progress": self.dwell_progress,
            "feedback": self.feedback_text,
        }

        if not is_security_unlocked or wrist_coord[2] < 0.35:
            self.dwell_start_pos = None
            self.dwell_progress = 0.0
            return result_meta

        h, w = frame_shape[:2]
        wx, wy = wrist_coord[0] / float(w), wrist_coord[1] / float(h)

        # Map wrist coordinates inside interaction zone to screen
        raw_x = np.interp(wx, [self.box_cfg["x_min"], self.box_cfg["x_max"]], [0, self.screen_width])
        raw_y = np.interp(wy, [self.box_cfg["y_min"], self.box_cfg["y_max"]], [0, self.screen_height])

        # Apply One Euro Filter smoothing
        smooth_x, smooth_y = self.one_euro.filter(raw_x, raw_y)
        self._move_cursor(smooth_x, smooth_y)

        # -------------------------------------------------------------
        # Dwell Click Logic (Hold steady for 1.0s)
        # -------------------------------------------------------------
        now = time.time()
        if self.dwell_start_pos is None:
            self.dwell_start_pos = (smooth_x, smooth_y)
            self.dwell_start_time = now
            self.dwell_progress = 0.0
        else:
            dx = smooth_x - self.dwell_start_pos[0]
            dy = smooth_y - self.dwell_start_pos[1]
            dist_moved = math.hypot(dx, dy)

            if dist_moved <= self.dwell_radius_px:
                # Holding steady within dwell radius
                elapsed = now - self.dwell_start_time
                self.dwell_progress = min(1.0, elapsed / self.dwell_time_sec)
                self.feedback_text = f"Dwell Clicking: {int(self.dwell_progress * 100)}%"
                self.feedback_color = (0, 229, 255)

                if elapsed >= self.dwell_time_sec:
                    self._execute_click(is_double=False)
                    self.dwell_start_pos = None
                    self.dwell_progress = 0.0
                    self.dwell_start_time = now + 0.5  # Cooldown
            else:
                # Moved beyond dwell radius -> Reset dwell timer
                self.dwell_start_pos = (smooth_x, smooth_y)
                self.dwell_start_time = now
                self.dwell_progress = 0.0
                self.feedback_text = "Far Mode Tracking"
                self.feedback_color = (0, 255, 100)

        result_meta["active"] = True
        result_meta["cursor_pos"] = (self.curr_cursor_x, self.curr_cursor_y)
        result_meta["dwell_progress"] = self.dwell_progress
        result_meta["feedback"] = self.feedback_text
        return result_meta
