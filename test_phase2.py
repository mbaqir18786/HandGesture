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
from core.tracker import PersonTracker, PersonDetection
from core.security_engine import SecurityEngine, SecurityState


def draw_phase2_hud(
    frame: np.ndarray,
    stats: dict,
    persons: list,
    state: SecurityState,
    locked_teacher_id: int | None,
    metadata: dict,
) -> np.ndarray:
    """Draws sleek diagnostic HUD overlay for Multi-Person Tracking & Instant Teacher Re-ID."""
    h, w = frame.shape[:2]

    # Semi-transparent top HUD bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 26), -1)
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 26), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Top Bar: Title & Status
    cv2.putText(
        frame,
        "ClassroomGestureMouse | Multi-Person Tracking & Instant Teacher Re-ID",
        (15, 26),
        cv2.FONT_HERSHEY_DUPLEX,
        0.58,
        (0, 229, 255),
        1,
        cv2.LINE_AA,
    )

    # Stats Row
    fps_val = stats.get("fps", 0.0)
    fps_color = (0, 255, 100) if fps_val >= 20.0 else (0, 200, 255) if fps_val >= 10.0 else (0, 100, 255)
    person_count = len(persons)
    status_str = f"People: {person_count} | FPS: {fps_val:4.1f} | Latency: {stats.get('latency_ms', 0):.1f}ms"
    cv2.putText(
        frame,
        status_str,
        (15, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        fps_color,
        1,
        cv2.LINE_AA,
    )

    # Security Lock Status Badge on top right
    if state in (SecurityState.LOCKED, SecurityState.RE_LOCKED) and locked_teacher_id is not None:
        tag = "RE-LOCKED" if state == SecurityState.RE_LOCKED else "LOCKED"
        lock_text = f"TEACHER {tag}: ID #{locked_teacher_id} [ACTIVE]"
        lock_color = (0, 255, 100)
    elif state == SecurityState.LOST:
        match_score = int(metadata.get("best_match_score", 0.0) * 100)
        lock_text = f"TEACHER: LOST [SEARCHING RE-ENTRY... Match: {match_score}%]"
        lock_color = (0, 180, 255)
    else:
        lock_text = "TEACHER: UNLOCKED (Press 'L' to Lock Me)"
        lock_color = (180, 180, 180)

    cv2.putText(
        frame,
        lock_text,
        (w - 560, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        lock_color,
        2 if state in (SecurityState.LOCKED, SecurityState.RE_LOCKED, SecurityState.LOST) else 1,
        cv2.LINE_AA,
    )

    # Bottom Instructions Bar
    instructions = "Keys: [L] Lock Me (Instant Enrollment)  |  [U] Unlock / Reset  |  [S] Snapshot  |  [Q] Quit"
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


def run_phase2_diagnostics():
    print("=" * 70)
    print("ClassroomGestureMouse - Multi-Person Tracking & Instant Teacher Re-ID")
    print("=" * 70)

    # 1. Load Config
    config = ConfigManager.get_instance()
    near_thresh = config.get("near_mode.max_distance_meters", 3.5)
    max_dist = config.get("far_mode.max_distance_meters", 10.0)
    yolo_model = config.get("tracking.yolo_model", "yolov8n-pose.pt")
    min_conf = config.get("tracking.min_person_confidence", 0.50)

    # 2. Init YOLO Tracker
    print(f"\n[1/3] Initializing YOLOv8-Pose ({yolo_model}) + ByteTrack...")
    tracker = PersonTracker(
        model_name=yolo_model,
        min_person_conf=min_conf,
        near_threshold_m=near_thresh,
        max_distance_m=max_dist,
    )

    # 3. Init Security Engine (Instant Biometric Re-ID with SFace)
    print("[2/3] Initializing Deep SFace Biometric Security Engine...")
    security = SecurityEngine(
        lost_timeout_sec=0.6,
    )

    # 4. Init Camera
    print("[3/3] Initializing CameraStream...")
    try:
        stream = CameraStream.from_config(config.get_all())
        stream.start()
    except Exception as e:
        print(f"[-] Camera error: {e}")
        return

    window_name = "ClassroomGestureMouse - Multi-Person Tracking & Teacher Re-ID"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    snapshot_count = 0

    print("\n[+] Instant Re-ID System Ready:")
    print("    1. Press 'L' to Lock yourself as Teacher.")
    print("    2. Your skeleton instantly turns GREEN.")
    print("    3. Step OUT of camera view -> System switches to LOST.")
    print("    4. Step BACK in -> Instantly RE-LOCKS you in GREEN on the 1st frame!")
    print("    - Press 'U' to Unlock / Reset")
    print("    - Press 'Q' to Quit\n")

    try:
        while stream.is_running():
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # Run Multi-Person Tracking & Distance Estimation
            persons = tracker.track(frame)

            # Run Biometric Security State Machine & 1-Frame Instant Re-ID
            state, teacher_id, metadata = security.update(frame, persons)

            # Render Skeletons & Badges (Green for Teacher, Cyan/Orange for Others)
            frame = tracker.draw_visuals(frame, persons, teacher_id=teacher_id)

            # Draw Top HUD
            stats = stream.get_stats()
            frame = draw_phase2_hud(frame, stats, persons, state, teacher_id, metadata)

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
            elif key in (ord("s"), ord("S")):
                snapshot_count += 1
                fname = f"snapshot_{int(time.time())}_{snapshot_count}.jpg"
                cv2.imwrite(fname, frame)
                print(f"[+] Saved snapshot to: {fname}")

    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print("\n[+] Stream cleanly released.")


if __name__ == "__main__":
    run_phase2_diagnostics()
