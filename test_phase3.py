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
from core.tracker import PersonTracker
from core.security_engine import SecurityEngine, SecurityState


def draw_security_hud(
    frame: np.ndarray,
    stats: dict,
    state: SecurityState,
    teacher_id: int | None,
    metadata: dict,
) -> np.ndarray:
    """Renders sleek, modern security diagnostic HUD for Phase 3."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 26), -1)
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 26), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(
        frame,
        "ClassroomGestureMouse | Phase 3 Biometric Security & Deep SFace Re-ID",
        (15, 26),
        cv2.FONT_HERSHEY_DUPLEX,
        0.58,
        (0, 229, 255),
        1,
        cv2.LINE_AA,
    )

    state_colors = {
        SecurityState.UNENROLLED: (150, 150, 150),
        SecurityState.LOCKED: (0, 255, 100),         # Green
        SecurityState.RE_LOCKED: (0, 255, 100),      # Green
        SecurityState.LOST: (0, 180, 255),           # Amber
    }
    badge_color = state_colors.get(state, (200, 200, 200))

    if state in (SecurityState.LOCKED, SecurityState.RE_LOCKED):
        tag = "RE-LOCKED" if state == SecurityState.RE_LOCKED else "LOCKED"
        state_text = f"SECURITY: TEACHER {tag} (ID #{teacher_id}) [MOUSE ACTIVE]"
    elif state == SecurityState.LOST:
        match_score = int(metadata.get("best_match_score", 0.0) * 100)
        state_text = f"SECURITY: ZERO-TRUST FROZEN (Teacher Lost | Searching Re-Entry... Match: {match_score}%)"
    else:
        state_text = "SECURITY: UNENROLLED (Press 'L' to Lock Teacher)"

    cv2.putText(
        frame,
        state_text,
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        badge_color,
        2 if state in (SecurityState.LOCKED, SecurityState.RE_LOCKED, SecurityState.LOST) else 1,
        cv2.LINE_AA,
    )

    fps_val = stats.get("fps", 0.0)
    fps_color = (0, 255, 100) if fps_val >= 20.0 else (0, 200, 255)
    stats_str = f"FPS: {fps_val:4.1f} | Latency: {stats.get('latency_ms', 0):.1f}ms"
    cv2.putText(
        frame,
        stats_str,
        (w - 280, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        fps_color,
        1,
        cv2.LINE_AA,
    )

    instructions = "Keys: [L] Lock Me (Enroll Face & Body)  |  [U] Unlock / Reset  |  [Q] Quit"
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


def run_phase3_diagnostics():
    print("=" * 70)
    print("ClassroomGestureMouse - Phase 3 Deep SFace Biometric Security")
    print("=" * 70)

    config = ConfigManager.get_instance()
    lost_timeout = config.get("security.lost_timeout_sec", 0.6)

    print("\n[1/3] Initializing YOLOv8-Pose + ByteTrack...")
    tracker = PersonTracker(
        model_name=config.get("tracking.yolo_model", "yolov8n-pose.pt"),
        min_person_conf=config.get("tracking.min_person_confidence", 0.50),
        near_threshold_m=config.get("near_mode.max_distance_meters", 3.5),
        max_distance_m=config.get("far_mode.max_distance_meters", 10.0),
    )

    print("[2/3] Initializing SFace Deep Neural Face & Biometric Security Engine...")
    security = SecurityEngine(
        lost_timeout_sec=lost_timeout,
    )

    print("[3/3] Initializing CameraStream...")
    try:
        stream = CameraStream.from_config(config.get_all())
        stream.start()
    except Exception as e:
        print(f"[-] Camera error: {e}")
        return

    window_name = "ClassroomGestureMouse - Phase 3 Biometric Security"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\n[+] Phase 3 Ready:")
    print("    1. Press 'L' to enroll your SFace Biometric Face Embedding & Clothing Signature.")
    print("    2. Status turns GREEN -> TEACHER LOCKED.")
    print("    3. Step OUT of camera view -> Status turns ORANGE/LOST (Zero-Trust protection).")
    print("    4. Step BACK in -> Instantly RE-LOCKED (Green) on the 1st frame!")
    print("    - Press 'U' to Unlock / Reset")
    print("    - Press 'Q' to Quit\n")

    try:
        while stream.is_running():
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            persons = tracker.track(frame)
            state, teacher_id, metadata = security.update(frame, persons)
            frame = tracker.draw_visuals(frame, persons, teacher_id=teacher_id)

            stats = stream.get_stats()
            frame = draw_security_hud(frame, stats, state, teacher_id, metadata)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                print("[+] Quit requested. Exiting...")
                break
            elif key in (ord("l"), ord("L")):
                if persons:
                    best = max(persons, key=lambda p: p.distance_info.get("proximity_score", 0.0))
                    security.enroll(frame, best)
                else:
                    print("[-] Please stand in front of camera to enroll.")
            elif key in (ord("u"), ord("U")):
                security.unlock()

    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print("\n[+] Phase 3 stream cleanly released.")


if __name__ == "__main__":
    run_phase3_diagnostics()
