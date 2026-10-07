import os
import sys
import time
import cv2
import numpy as np

# Ensure root directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from core.config_manager import ConfigManager
from core.camera_stream import CameraStream


def draw_hud(frame: np.ndarray, stats: dict, config: ConfigManager, flip_state: bool) -> np.ndarray:
    """Draws a modern diagnostic overlay on top of the camera frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top HUD bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 25), -1)
    # Semi-transparent bottom HUD bar
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Top Bar: Title & Status
    cv2.putText(
        frame,
        "ClassroomGestureMouse | Phase 1 Camera & Config Diagnostics",
        (15, 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (0, 229, 255),
        1,
        cv2.LINE_AA,
    )

    # Metrics Display
    fps_val = stats.get("fps", 0.0)
    fps_color = (0, 255, 100) if fps_val >= 25.0 else (0, 200, 255) if fps_val >= 15.0 else (0, 100, 255)
    metrics_str = (
        f"FPS: {fps_val:4.1f} | Latency: {stats.get('latency_ms', 0):.1f}ms | "
        f"Device: #{stats.get('active_device_index', 0)} ({stats.get('backend', 'DSHOW')}) | "
        f"Res: {stats.get('resolution', (w, h))[0]}x{stats.get('resolution', (w, h))[1]}"
    )
    cv2.putText(
        frame,
        metrics_str,
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        fps_color,
        1,
        cv2.LINE_AA,
    )

    # Frame counter on top right
    count_str = f"Frames: {stats.get('total_frames', 0)}"
    cv2.putText(
        frame,
        count_str,
        (w - 180, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Interaction Box Preview (from Near Mode config)
    near_cfg = config.get("near_mode.interaction_box", {})
    x_min = int(near_cfg.get("x_min", 0.22) * w)
    x_max = int(near_cfg.get("x_max", 0.78) * w)
    y_min = int(near_cfg.get("y_min", 0.20) * h)
    y_max = int(near_cfg.get("y_max", 0.60) * h)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "Active Gesture Zone (Near Mode)",
        (x_min + 8, y_min + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Bottom Bar: Instructions
    flip_text = "ON" if flip_state else "OFF"
    instructions = f"Keys: [Q] Quit  |  [F] Toggle Horizontal Mirror (Currently: {flip_text})  |  [S] Save Frame"
    cv2.putText(
        frame,
        instructions,
        (15, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    return frame


def run_diagnostics():
    print("=" * 70)
    print("ClassroomGestureMouse - Phase 1 Diagnostics & Stream Benchmark")
    print("=" * 70)

    # 1. Test ConfigManager
    print("\n[1/3] Initializing ConfigManager...")
    config = ConfigManager.get_instance()
    app_name = config.get("app_name", "Unknown")
    version = config.get("version", "0.0.0")
    print(f"      Loaded config for: {app_name} v{version}")
    print(f"      Camera Target: {config.get('camera.width')}x{config.get('camera.height')} @ {config.get('camera.fps')} FPS")
    print(f"      Security PIN configured: {'****' if config.get('security.teacher_pin') else 'None'}")
    print(f"      Near Mode Interaction Box: {config.get('near_mode.interaction_box')}")
    print("      [+] ConfigManager test PASSED.")

    # 2. Test CameraStream
    print("\n[2/3] Launching Threaded CameraStream...")
    camera_cfg = config.get("camera", {})
    flip_horizontal = camera_cfg.get("flip_horizontal", True)

    try:
        stream = CameraStream.from_config(config.get_all())
        stream.start()
    except Exception as e:
        print(f"\n[-] Camera initialization error: {e}")
        print("    If no physical camera is connected, verify device index in config.json.")
        return

    print("      [+] CameraStream running on background thread.")
    print("\n[3/3] Displaying live diagnostic window...")
    print("      - Press 'q' or 'ESC' to exit")
    print("      - Press 'f' to toggle mirror flip")
    print("      - Press 's' to save a snapshot screenshot\n")

    window_name = "ClassroomGestureMouse - Phase 1 Diagnostics"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    snapshot_counter = 0

    try:
        while stream.is_running():
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            stats = stream.get_stats()
            display_frame = draw_hud(frame, stats, config, flip_horizontal)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                print("[+] Quit key pressed. Shutting down...")
                break
            elif key in (ord("f"), ord("F")):
                flip_horizontal = not flip_horizontal
                stream.set_flip(flip_horizontal)
                config.set("camera.flip_horizontal", flip_horizontal, auto_save=True)
                print(f"[+] Flip horizontal toggled to: {flip_horizontal} (saved to config.json)")
            elif key in (ord("s"), ord("S")):
                snapshot_counter += 1
                filename = f"snapshot_phase1_{int(time.time())}_{snapshot_counter}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[+] Saved screenshot to: {filename}")

    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print("\n[+] Phase 1 diagnostic run complete. Stream cleanly released.")


if __name__ == "__main__":
    run_diagnostics()
