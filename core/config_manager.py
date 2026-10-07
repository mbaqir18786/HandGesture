import os
import sys
import json
import copy
import threading
from typing import Any, Dict, Optional, Union


DEFAULT_CONFIG: Dict[str, Any] = {
    "app_name": "ClassroomGestureMouse",
    "version": "1.0.0",
    "camera": {
        "device_index": 0,
        "fallback_indices": [1, 2],
        "backend": "DSHOW",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "flip_horizontal": True,
        "buffer_size": 1,
        "auto_reconnect": True,
        "reconnect_interval_sec": 2.0,
    },
    "security": {
        "teacher_pin": "1234",
        "scan_duration_sec": 3.5,
        "zero_trust_policy": True,
        "lost_timeout_sec": 2.0,
        "re_lock_cooldown_sec": 1.0,
        "face_match_threshold": 0.68,
        "body_hist_threshold": 0.72,
        "body_aspect_ratio_tolerance": 0.20,
        "standing_detection_ratio": 1.4,
    },
    "near_mode": {
        "enabled": True,
        "max_distance_meters": 3.5,
        "model_path": "live_mouse_control_using_hand_gestures/hand_landmarker.task",
        "tracking_landmark": 5,
        "smoothing": 5.0,
        "deadzone_px": 5.0,
        "interaction_box": {
            "x_min": 0.22,
            "x_max": 0.78,
            "y_min": 0.20,
            "y_max": 0.60,
        },
        "pinch_close_threshold": 32.0,
        "pinch_open_threshold": 45.0,
        "double_tap_window_sec": 0.40,
        "detection_confidence": 0.60,
        "tracking_confidence": 0.60,
    },
    "far_mode": {
        "enabled": True,
        "min_distance_meters": 3.5,
        "max_distance_meters": 10.0,
        "tracking_joint": "RIGHT_WRIST",
        "relative_movement_gain_x": 2.5,
        "relative_movement_gain_y": 2.2,
        "dwell_time_sec": 1.0,
        "dwell_radius_px": 25.0,
        "one_euro_filter": {
            "min_cutoff": 1.0,
            "beta": 0.05,
            "d_cutoff": 1.0,
        },
    },
    "tracking": {
        "yolo_model": "yolov8n-pose.pt",
        "min_person_confidence": 0.50,
        "bytetrack_track_thresh": 0.45,
        "bytetrack_match_thresh": 0.80,
        "bytetrack_buffer": 30,
    },
    "voice_feedback": {
        "enabled": True,
        "tts_engine": "pyttsx3",
        "speech_rate": 180,
        "volume": 0.9,
        "alerts": {
            "locked": "Teacher locked. Gesture control active.",
            "lost": "Teacher lost. Mouse control frozen.",
            "relocked": "Welcome back teacher. Control restored.",
            "unauthorized": "Access denied.",
        },
    },
    "ui": {
        "theme": "dark",
        "always_on_top_indicator": True,
        "indicator_size_px": 16,
        "show_debug_feed": True,
        "status_colors": {
            "locked": "#00FF66",
            "lost": "#FFCC00",
            "inactive": "#888888",
            "scanning": "#00E5FF",
        },
    },
}



def get_resource_path(relative_path: str) -> str:
    """
    Resolves the absolute path to a resource file, compatible with:
    1. PyInstaller single-file bundle (sys._MEIPASS)
    2. PyInstaller directory bundle (_internal or next to .exe)
    3. Normal Python development workspace
    """
    if hasattr(sys, "_MEIPASS"):
        bundle_path = os.path.join(sys._MEIPASS, relative_path)
        if os.path.exists(bundle_path):
            return bundle_path

    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        local_path = os.path.join(exe_dir, relative_path)
        if os.path.exists(local_path):
            return local_path
        internal_path = os.path.join(exe_dir, "_internal", relative_path)
        if os.path.exists(internal_path):
            return internal_path

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(workspace_root, relative_path)


def _deep_merge(target: dict, source: dict) -> dict:
    """Recursively merges source dict into target dict."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


class ConfigManager:
    """
    Thread-safe Configuration Manager with JSON persistence and nested key lookups.
    Supports dot-notation keys (e.g. 'camera.width') and deep merge with default configuration.
    """

    _instance: Optional["ConfigManager"] = None
    _singleton_lock = threading.Lock()

    def __init__(self, config_path: Optional[str] = None):
        self._lock = threading.RLock()
        self._listeners = []

        if config_path is None:
            if getattr(sys, "frozen", False):
                exe_dir = os.path.dirname(sys.executable)
                self.config_path = os.path.join(exe_dir, "config.json")
            else:
                workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                self.config_path = os.path.join(workspace_root, "config.json")
        else:
            self.config_path = os.path.abspath(config_path)

        self._config: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
        self.load()

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> "ConfigManager":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(config_path)
            return cls._instance

    def load(self) -> bool:
        """Loads configuration from JSON file. Auto-creates file if absent."""
        with self._lock:
            if not os.path.exists(self.config_path):
                bundled_cfg = get_resource_path("config.json")
                if os.path.exists(bundled_cfg) and os.path.abspath(bundled_cfg) != os.path.abspath(self.config_path):
                    try:
                        import shutil
                        shutil.copy2(bundled_cfg, self.config_path)
                        print(f"[ConfigManager] Copied bundled template to '{self.config_path}'.")
                    except Exception as e:
                        print(f"[ConfigManager] Warning copying bundled config: {e}")

                if not os.path.exists(self.config_path):
                    print(f"[ConfigManager] Config file '{self.config_path}' not found. Generating default...")
                    self.save()
                    return True

            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Merge loaded data over defaults so missing fields are preserved
                merged = copy.deepcopy(DEFAULT_CONFIG)
                _deep_merge(merged, data)
                self._config = merged
                return True
            except Exception as e:
                print(f"[ConfigManager] Error loading config from {self.config_path}: {e}. Keeping existing values.")
                return False

    def save(self) -> bool:
        """Saves current configuration to disk safely."""
        with self._lock:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
                temp_path = self.config_path + ".tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(self._config, f, indent=2)
                # Atomic replace
                os.replace(temp_path, self.config_path)
                return True
            except Exception as e:
                print(f"[ConfigManager] Error saving config: {e}")
                return False

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieves a value using dot-notation or slash-notation path.
        Example: config.get('camera.fps', 30)
        """
        with self._lock:
            keys = key_path.replace("/", ".").split(".")
            curr = self._config
            for k in keys:
                if isinstance(curr, dict) and k in curr:
                    curr = curr[k]
                else:
                    return default
            return copy.deepcopy(curr)

    def set(self, key_path: str, value: Any, auto_save: bool = False) -> None:
        """
        Updates a configuration key using dot-notation path.
        Example: config.set('camera.flip_horizontal', False, auto_save=True)
        """
        with self._lock:
            keys = key_path.replace("/", ".").split(".")
            curr = self._config
            for k in keys[:-1]:
                if k not in curr or not isinstance(curr[k], dict):
                    curr[k] = {}
                curr = curr[k]
            curr[keys[-1]] = copy.deepcopy(value)

            if auto_save:
                self.save()

            self._notify_listeners(key_path, value)

    def get_all(self) -> Dict[str, Any]:
        """Returns deep copy of entire config dictionary."""
        with self._lock:
            return copy.deepcopy(self._config)

    def add_listener(self, callback) -> None:
        """Adds a callback function(key_path, new_value) for changes."""
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def _notify_listeners(self, key_path: str, value: Any) -> None:
        for cb in self._listeners:
            try:
                cb(key_path, value)
            except Exception as e:
                print(f"[ConfigManager] Listener error on '{key_path}': {e}")

    def __getitem__(self, item: str) -> Any:
        with self._lock:
            return self._config[item]

    def __repr__(self) -> str:
        return f"<ConfigManager path={self.config_path} sections={list(self._config.keys())}>"
