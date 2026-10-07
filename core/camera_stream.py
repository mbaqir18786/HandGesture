import time
import threading
from typing import Tuple, Optional, Dict, Any, List
import cv2
import numpy as np


class CameraStream:
    """
    High-performance threaded camera stream manager.
    Eliminates frame buffering lag, supports Windows DirectShow (cv2.CAP_DSHOW),
    automatic device fallback, auto-reconnection, and real-time FPS benchmarking.
    """

    def __init__(
        self,
        device_index: int = 0,
        fallback_indices: Optional[List[int]] = None,
        backend: str = "DSHOW",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        flip_horizontal: bool = True,
        auto_reconnect: bool = True,
        reconnect_interval_sec: float = 2.0,
    ):
        self.device_index = device_index
        self.fallback_indices = fallback_indices if fallback_indices is not None else [1, 2]
        self.backend_str = backend.upper()
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.flip_horizontal = flip_horizontal
        self.auto_reconnect = auto_reconnect
        self.reconnect_interval_sec = reconnect_interval_sec

        self._active_device_index: int = device_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._running = False
        self._frame: Optional[np.ndarray] = None
        self._has_new_frame = False
        self._frame_timestamp: float = 0.0
        self._frame_count: int = 0

        # FPS & Performance metrics
        self._last_fps_calc_time = time.time()
        self._fps_frame_count = 0
        self._current_fps: float = 0.0
        self._actual_width = 0
        self._actual_height = 0
        self._last_read_latency_ms: float = 0.0

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "CameraStream":
        """Factory constructor from config dictionary."""
        cam_cfg = config_dict.get("camera", {})
        return cls(
            device_index=cam_cfg.get("device_index", 0),
            fallback_indices=cam_cfg.get("fallback_indices", [1, 2]),
            backend=cam_cfg.get("backend", "DSHOW"),
            width=cam_cfg.get("width", 1280),
            height=cam_cfg.get("height", 720),
            fps=cam_cfg.get("fps", 30),
            flip_horizontal=cam_cfg.get("flip_horizontal", True),
            auto_reconnect=cam_cfg.get("auto_reconnect", True),
            reconnect_interval_sec=cam_cfg.get("reconnect_interval_sec", 2.0),
        )

    def _get_backend_flag(self) -> int:
        if self.backend_str == "DSHOW" and hasattr(cv2, "CAP_DSHOW"):
            return cv2.CAP_DSHOW
        elif self.backend_str == "MSMF" and hasattr(cv2, "CAP_MSMF"):
            return cv2.CAP_MSMF
        elif self.backend_str == "V4L2" and hasattr(cv2, "CAP_V4L2"):
            return cv2.CAP_V4L2
        return cv2.CAP_ANY

    def _open_capture(self) -> bool:
        """Attempts to open camera device with configured backend and fallbacks."""
        candidates = [self.device_index] + [i for i in self.fallback_indices if i != self.device_index]
        backend = self._get_backend_flag()

        for idx in candidates:
            print(f"[CameraStream] Trying camera device index {idx} with backend {self.backend_str}...")
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened() and backend != cv2.CAP_ANY:
                print(f"[CameraStream] Direct backend failed for index {idx}. Retrying with cv2.CAP_ANY...")
                cap = cv2.VideoCapture(idx, cv2.CAP_ANY)

            if cap.isOpened():
                # Configure resolution & MJPEG format for max FPS
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Warm-up grab
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self._cap = cap
                    self._active_device_index = idx
                    self._actual_height, self._actual_width = test_frame.shape[:2]
                    print(
                        f"[CameraStream] Connected to camera index {idx} "
                        f"({self._actual_width}x{self._actual_height} @ requested {self.target_fps} FPS)"
                    )
                    return True
                else:
                    cap.release()

        print("[CameraStream] ERROR: Failed to open any camera device.")
        return False

    def start(self) -> "CameraStream":
        """Starts background frame reader thread."""
        if self._running:
            return self

        if not self._open_capture():
            raise RuntimeError("Could not initialize video capture stream.")

        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, name="CameraStreamWorker", daemon=True)
        self._thread.start()
        return self

    def _capture_worker(self) -> None:
        """Background thread continuously grabbing latest frame with zero queue delay."""
        while self._running:
            start_t = time.perf_counter()
            if self._cap is None or not self._cap.isOpened():
                if self.auto_reconnect and self._running:
                    print(f"[CameraStream] Connection lost. Attempting reconnect in {self.reconnect_interval_sec}s...")
                    time.sleep(self.reconnect_interval_sec)
                    self._open_capture()
                    continue
                else:
                    break

            ret, frame = self._cap.read()
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            if ret and frame is not None:
                if self.flip_horizontal:
                    frame = cv2.flip(frame, 1)

                with self._lock:
                    self._frame = frame
                    self._has_new_frame = True
                    self._frame_timestamp = time.time()
                    self._frame_count += 1
                    self._last_read_latency_ms = elapsed_ms
                    self._fps_frame_count += 1

                # Update FPS meter once every second
                now = time.time()
                dt = now - self._last_fps_calc_time
                if dt >= 1.0:
                    self._current_fps = self._fps_frame_count / dt
                    self._fps_frame_count = 0
                    self._last_fps_calc_time = now
            else:
                # Give CPU small yield if no frame
                time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Thread-safe retrieval of the most recent frame.
        Returns (success, frame_bgr).
        """
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def get_latest_frame_ref(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Retrieves reference to frame without deep copy for high-speed pipelines.
        Returns (success, frame, timestamp).
        """
        with self._lock:
            if self._frame is None:
                return False, None, 0.0
            return True, self._frame, self._frame_timestamp

    def get_fps(self) -> float:
        """Returns calculated live stream FPS."""
        return self._current_fps

    def get_stats(self) -> Dict[str, Any]:
        """Returns diagnostic metadata about the camera stream."""
        with self._lock:
            return {
                "active_device_index": self._active_device_index,
                "backend": self.backend_str,
                "resolution": (self._actual_width, self._actual_height),
                "fps": round(self._current_fps, 1),
                "total_frames": self._frame_count,
                "latency_ms": round(self._last_read_latency_ms, 2),
                "is_running": self._running,
            }

    def set_flip(self, flip: bool) -> None:
        self.flip_horizontal = flip

    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        """Stops background thread and releases camera resource."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None
        print("[CameraStream] Stream stopped and camera released.")

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
