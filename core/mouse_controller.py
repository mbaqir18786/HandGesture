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

        self.box_cfg = interaction_box or {"x_min": 0.25, "x_max": 0.75, "y_min": 0.28, "y_max": 0.68}
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

        # Dedicated One-Euro Filters for Near Mode (micro-movements) and Far Mode (macro-movements)
        # beta=0.65 eliminates cursor lag when moving fast, min_cutoff=1.2 eliminates jitter when still
        self.near_one_euro = OneEuroFilter2D(min_cutoff=1.2, beta=0.65, d_cutoff=1.0)
        self.one_euro = OneEuroFilter2D(min_cutoff=0.9, beta=0.45, d_cutoff=1.0)
        self.dwell_start_pos: Optional[Tuple[float, float]] = None
        self.dwell_start_time: float = 0.0
        self.dwell_progress: float = 0.0
        self.last_wrist_pos: Optional[Tuple[float, float]] = None

        # Anti-slip anchor during pinch clicks
        self.click_anchor_pos: Optional[Tuple[int, int]] = None

        # Scroll Gesture State
        self.last_scroll_y: Optional[float] = None

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
                    min_hand_detection_confidence=0.55,
                    min_hand_presence_confidence=0.55,
                    min_tracking_confidence=0.55,
                )
                self._hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
                print("[DualModeMouseController] MediaPipe Hand Landmarker initialized.")
        except Exception as e:
            print(f"[DualModeMouseController] MediaPipe init error: {e}")

    def _move_cursor(self, target_x: float, target_y: float) -> None:
        """Zero-deadzone, direct OS cursor placement with native Windows API for 0ms lag."""
        curr_x = int(np.clip(target_x, 0, self.screen_width - 1))
        curr_y = int(np.clip(target_y, 0, self.screen_height - 1))

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
        Near Mode (< 3.5m): Scale-Invariant Normalized Pinch + OneEuroFilter Anti-Jitter + Scroll Support.
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
            self.is_pinched = False
            self.click_anchor_pos = None
            return result_meta

        h, w = frame_shape[:2]
        current_time = time.time()

        import mediapipe as mp
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # MediaPipe video mode requires strictly monotonic real-world millisecond timestamps
        frame_ts_ms = max(self._mp_timestamp_ms + 1, int(current_time * 1000))
        self._mp_timestamp_ms = frame_ts_ms

        detect_result = self._hand_landmarker.detect_for_video(mp_image, frame_ts_ms)

        if detect_result.hand_landmarks:
            hand = detect_result.hand_landmarks[0]
            result_meta["hand_landmarks"] = hand
            result_meta["active"] = True

            # Anatomical Landmarks
            wrist = hand[0]
            thumb_tip = hand[4]
            index_mcp = hand[5]   # Knuckle
            index_tip = hand[8]
            middle_tip = hand[12]

            # 1. Scale-invariant reference: rigid distance between Wrist (0) and Index MCP (5)
            # This reference bone distance never changes regardless of finger flexing
            hand_scale = math.hypot((wrist.x - index_mcp.x) * w, (wrist.y - index_mcp.y) * h)
            hand_scale = max(hand_scale, 25.0)

            # 2. Normalized distances (independent of distance from camera or resolution)
            raw_pinch_px = math.hypot((thumb_tip.x - index_tip.x) * w, (thumb_tip.y - index_tip.y) * h)
            pinch_ratio = raw_pinch_px / hand_scale

            # Distance between index tip and middle tip (for scroll mode)
            index_middle_ratio = math.hypot((index_tip.x - middle_tip.x) * w, (index_tip.y - middle_tip.y) * h) / hand_scale

            # Check if 2 fingers are raised together (Index + Middle extended for scrolling)
            is_two_finger_scroll = (
                index_middle_ratio < 0.35
                and index_tip.y < index_mcp.y
                and middle_tip.y < hand[9].y
                and pinch_ratio > 0.45
            )

            if is_two_finger_scroll:
                # Two-finger vertical scroll mode
                self.is_pinched = False
                self.click_anchor_pos = None
                self.feedback_text = "Scrolling"
                self.feedback_color = (0, 229, 255)

                if self.last_scroll_y is not None:
                    delta_y = (index_tip.y - self.last_scroll_y) * h
                    if abs(delta_y) > 3.0:
                        scroll_amount = int(-delta_y * 3.5)
                        pyautogui.scroll(scroll_amount)
                self.last_scroll_y = index_tip.y

            else:
                self.last_scroll_y = None

                # 3. Normalized Pinch Triggering (Hysteresis: 0.30 close / 0.45 open)
                PINCH_CLOSE_RATIO = 0.32
                PINCH_OPEN_RATIO = 0.48

                if pinch_ratio < PINCH_CLOSE_RATIO:
                    # Pinch DOWN event
                    if not self.is_pinched:
                        time_since_last = current_time - self.last_pinch_time
                        if time_since_last < self.double_tap_window_sec:
                            self._execute_click(is_double=True)
                            self.last_pinch_time = 0.0
                        else:
                            self._execute_click(is_double=False)
                            self.last_pinch_time = current_time
                        self.is_pinched = True
                        # Anchor cursor so pinching does NOT cause click to slip
                        self.click_anchor_pos = (self.curr_cursor_x, self.curr_cursor_y)
                    else:
                        # While holding pinch, keep cursor anchored to prevent slipping off buttons
                        if self.click_anchor_pos is not None:
                            self._move_cursor(self.click_anchor_pos[0], self.click_anchor_pos[1])

                elif pinch_ratio > PINCH_OPEN_RATIO:
                    # Pinch RELEASE event
                    self.is_pinched = False
                    self.click_anchor_pos = None
                    self.feedback_text = "Aiming Cursor"
                    self.feedback_color = (0, 255, 0)

                    # 4. Laser-smooth cursor positioning via OneEuroFilter
                    # Scale-adaptive interaction zone ensures fingertip never leaves camera to reach top/corners
                    # and automatically scales when user is seated further back ("long place")
                    norm_scale = float(np.clip(hand_scale / 110.0, 0.40, 1.0))

                    cx = (self.box_cfg["x_max"] + self.box_cfg["x_min"]) / 2.0
                    cy = (self.box_cfg["y_max"] + self.box_cfg["y_min"]) / 2.0

                    # Dynamic span: contracts when further away so comfortable hand gestures cover entire screen
                    span_x = (self.box_cfg["x_max"] - self.box_cfg["x_min"]) * norm_scale
                    span_y = (self.box_cfg["y_max"] - self.box_cfg["y_min"]) * norm_scale

                    curr_x_min = cx - span_x / 2.0
                    curr_x_max = cx + span_x / 2.0
                    curr_y_min = cy - span_y / 2.0
                    curr_y_max = cy + span_y / 2.0

                    tracking_point = index_tip  # Direct fingertip tracking for intuitive reach
                    raw_x = np.interp(tracking_point.x, [curr_x_min, curr_x_max], [0, self.screen_width])
                    raw_y = np.interp(tracking_point.y, [curr_y_min, curr_y_max], [0, self.screen_height])

                    # Strictly clamp to screen bounds to reach all edges and corners effortlessly
                    raw_x = float(np.clip(raw_x, 0, self.screen_width - 1))
                    raw_y = float(np.clip(raw_y, 0, self.screen_height - 1))

                    smooth_x, smooth_y = self.near_one_euro.filter(raw_x, raw_y, timestamp=current_time)
                    self._move_cursor(smooth_x, smooth_y)

        result_meta["cursor_pos"] = (self.curr_cursor_x, self.curr_cursor_y)
        result_meta["feedback"] = self.feedback_text
        result_meta["feedback_color"] = self.feedback_color
        return result_meta

    def reset_far_anchor(self) -> None:
        """Reset far mode relative wrist anchor when arm is lowered."""
        self.last_wrist_pos = None

    def process_far_mode(
        self,
        wrist_coord: Tuple[float, float, float],
        frame_shape: Tuple[int, int],
        is_security_unlocked: bool = True,
    ) -> Dict[str, Any]:
        """
        Far Mode (3.5m - 10m): High-sensitivity Wrist Tracking + One Euro Filter Smoothing + 1.0s Dwell Click.
        Allows reaching all corners and top of screen from across the room without excessive body travel.
        """
        result_meta = {
            "mode": "FAR",
            "active": False,
            "cursor_pos": (self.curr_cursor_x, self.curr_cursor_y),
            "dwell_progress": self.dwell_progress,
            "feedback": self.feedback_text,
        }

        if not is_security_unlocked or wrist_coord[2] < 0.30:
            self.last_wrist_pos = None
            self.dwell_start_pos = None
            self.dwell_progress = 0.0
            return result_meta

        h, w = frame_shape[:2]
        wx, wy = wrist_coord[0] / float(w), wrist_coord[1] / float(h)

        # High-sensitivity relative motion: allows full screen navigation from across the room
        if self.last_wrist_pos is None:
            self.last_wrist_pos = (wx, wy)
            smooth_x, smooth_y = float(self.curr_cursor_x), float(self.curr_cursor_y)
        else:
            dx = (wx - self.last_wrist_pos[0]) * self.screen_width * 2.8
            dy = (wy - self.last_wrist_pos[1]) * self.screen_height * 2.6
            self.last_wrist_pos = (wx, wy)

            target_x = float(np.clip(self.curr_cursor_x + dx, 0, self.screen_width - 1))
            target_y = float(np.clip(self.curr_cursor_y + dy, 0, self.screen_height - 1))

            smooth_x, smooth_y = self.one_euro.filter(target_x, target_y)
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
