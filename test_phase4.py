import os
import sys
import time
import math
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
from core.mouse_controller import DualModeMouseController


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def draw_phase4_hud(
    frame: np.ndarray,
    stats: dict,
    state: SecurityState,
    teacher_id: int | None,
    distance_m: float,
    current_mode: str,
    mouse_feedback: str,
    mouse_feedback_color: tuple,
    dwell_progress: float,
) -> np.ndarray:
    """Renders comprehensive Phase 4 Mouse Engine HUD."""
    h, w = frame.shape[:2]

    # Semi-transparent top HUD bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 88), (20, 20, 26), -1)
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 26), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Top Bar: Title & Mode
    mode_color = (0, 255, 100) if current_mode == "NEAR" else (0, 229, 255)
    cv2.putText(
        frame,
        f"ClassroomGestureMouse | Phase 4 Dual-Mode Mouse Engine [{current_mode} MODE ({distance_m:.1f}m)]",
        (15, 26),
        cv2.FONT_HERSHEY_DUPLEX,
        0.58,
        mode_color,
        1,
        cv2.LINE_AA,
    )

    # Security & Mouse Status
    if state in (SecurityState.LOCKED, SecurityState.RE_LOCKED):
        sec_text = f"TEACHER ACTIVE (ID #{teacher_id}) | Mouse: {mouse_feedback}"
        sec_color = (0, 255, 100)
    elif state == SecurityState.LOST:
        sec_text = "TEACHER LOST | MOUSE FROZEN (Zero-Trust Security)"
        sec_color = (0, 180, 255)
    else:
        sec_text = "TEACHER UNENROLLED (Press 'L' to Lock Teacher)"
        sec_color = (180, 180, 180)

    cv2.putText(
        frame,
        sec_text,
        (15, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        sec_color,
        2 if state in (SecurityState.LOCKED, SecurityState.RE_LOCKED) else 1,
        cv2.LINE_AA,
    )

    # Performance Stats on top right
    fps_val = stats.get("fps", 0.0)
    fps_color = (0, 255, 100) if fps_val >= 20.0 else (0, 200, 255)
    stats_str = f"FPS: {fps_val:4.1f} | Latency: {stats.get('latency_ms', 0):.1f}ms"
    cv2.putText(
        frame,
        stats_str,
        (w - 280, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        fps_color,
        1,
        cv2.LINE_AA,
    )

    # Draw Dwell Click Progress Bar (if active in Far Mode)
    if dwell_progress > 0.0:
        bar_w = int((w - 40) * dwell_progress)
        cv2.rectangle(frame, (20, 83), (20 + bar_w, 88), (0, 229, 255), -1)

    # Bottom Instructions Bar
    instructions = "Near (<3.5m): Knuckle Move + Pinch Tap  |  Far (3.5-10m): Wrist Move + 1.0s Dwell Click  |  [L] Lock  |  [Q] Quit"
    cv2.putText(
        frame,
        instructions,
        (15, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    return frame


def run_phase4_diagnostics():
    print("=" * 70)
    print("ClassroomGestureMouse - Phase 4 Dual-Mode Touch-Free Mouse Engine")
    print("=" * 70)

    # 1. Config & Services
    config = ConfigManager.get_instance()
    near_thresh = config.get("near_mode.max_distance_meters", 3.5)
    max_dist = config.get("far_mode.max_distance_meters", 10.0)
    interaction_box = config.get("near_mode.interaction_box", {"x_min": 0.22, "x_max": 0.78, "y_min": 0.20, "y_max": 0.60})

    # Tracker & Biometrics
    tracker = PersonTracker(
        model_name=config.get("tracking.yolo_model", "yolov8n-pose.pt"),
        min_person_conf=config.get("tracking.min_person_confidence", 0.50),
        near_threshold_m=near_thresh,
        max_distance_m=max_dist,
    )

    security = SecurityEngine(
        lost_timeout_sec=config.get("security.lost_timeout_sec", 0.8),
    )

    # Dual-Mode Mouse Controller
    mouse = DualModeMouseController(
        interaction_box=interaction_box,
        smoothing=config.get("near_mode.smoothing", 5.0),
        deadzone_px=config.get("near_mode.deadzone_px", 5.0),
        pinch_close_threshold=config.get("near_mode.pinch_close_threshold", 32.0),
        pinch_open_threshold=config.get("near_mode.pinch_open_threshold", 45.0),
        double_tap_window_sec=config.get("near_mode.double_tap_window_sec", 0.40),
        dwell_time_sec=config.get("far_mode.dwell_time_sec", 1.0),
        dwell_radius_px=config.get("far_mode.dwell_radius_px", 25.0),
    )

    # Camera Stream
    try:
        stream = CameraStream.from_config(config.get_all())
        stream.start()
    except Exception as e:
        print(f"[-] Camera error: {e}")
        return

    window_name = "ClassroomGestureMouse - Phase 4 Dual-Mode Mouse Engine"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\n[+] Phase 4 Live Mouse Control Active:")
    print("    1. Press 'L' to enroll and lock yourself as Teacher.")
    print("    2. NEAR MODE (< 3.5m):")
    print("       - Move hand inside interaction zone: Smooth cursor tracking.")
    print("       - Tap Thumb + Index ONCE: Single Click.")
    print("       - Tap Thumb + Index TWICE within 0.4s: Double Click.")
    print("    3. FAR MODE (3.5m - 10m):")
    print("       - Move right wrist: Relative cursor tracking.")
    print("       - Hold steady for 1.0s: Dwell Click.")
    print("    4. ZERO-TRUST:")
    print("       - When you step away (LOST), mouse immediately freezes.")
    print("    - Press 'U' to Unlock / Reset")
    print("    - Press 'Q' to Quit\n")

    current_mode = "NEAR"
    distance_m = 1.0
    mouse_feedback = "Idle"
    mouse_feedback_color = (0, 255, 0)
    dwell_progress = 0.0

    frame_idx = 0
    persons = []
    is_teacher_active = False
    state = SecurityState.UNENROLLED
    teacher_id = None
    metadata = {}

    try:
        while stream.is_running():
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame_idx += 1
            h, w = frame.shape[:2]

            # 1. Multi-Person Tracking (Full speed when LOST or in Far Mode; interleaved in Near Mode for 30 FPS cursor)
            should_run_yolo = (
                not is_teacher_active
                or current_mode == "FAR"
                or (frame_idx % 2 == 0)
                or len(persons) == 0
            )

            if should_run_yolo:
                persons = tracker.track(frame)
                state, teacher_id, metadata = security.update(frame, persons)
                is_teacher_active = state in (SecurityState.LOCKED, SecurityState.RE_LOCKED) and teacher_id is not None

            # 3. Find Teacher Detection
            teacher_person = next((p for p in persons if p.track_id == teacher_id), None) if is_teacher_active else None

            if teacher_person is not None:
                distance_m = teacher_person.distance_info.get("distance_m", 1.0)
                current_mode = teacher_person.distance_info.get("mode", "NEAR")
            elif persons:
                distance_m = persons[0].distance_info.get("distance_m", 1.0)
                current_mode = persons[0].distance_info.get("mode", "NEAR")

            # 4. Draw Interaction Box
            bx1, by1 = int(interaction_box["x_min"] * w), int(interaction_box["y_min"] * h)
            bx2, by2 = int(interaction_box["x_max"] * w), int(interaction_box["y_max"] * h)
            box_color = (0, 255, 100) if is_teacher_active else (100, 100, 100)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2, cv2.LINE_AA)

            # 5. Execute Mouse Engine (Zero-Trust Gated)
            if current_mode == "NEAR":
                m_res = mouse.process_near_mode(frame, (h, w), is_security_unlocked=is_teacher_active)
                mouse_feedback = m_res["feedback"]
                mouse_feedback_color = m_res["feedback_color"]
                dwell_progress = 0.0

                # Render Hand Skeleton if detected
                if m_res.get("hand_landmarks") is not None:
                    hand = m_res["hand_landmarks"]
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                    for (a, b) in HAND_CONNECTIONS:
                        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)
                    for pt in pts:
                        cv2.circle(frame, pt, 3, (0, 0, 255), -1, cv2.LINE_AA)

                    # Highlight Index Knuckle (Landmark 5)
                    knuckle_pt = (int(hand[5].x * w), int(hand[5].y * h))
                    cv2.circle(frame, knuckle_pt, 7, (255, 0, 0), -1, cv2.LINE_AA)

            elif current_mode == "FAR" and teacher_person is not None:
                wrist_coord = teacher_person.right_wrist
                m_res = mouse.process_far_mode(wrist_coord, (h, w), is_security_unlocked=is_teacher_active)
                mouse_feedback = m_res["feedback"]
                dwell_progress = m_res.get("dwell_progress", 0.0)

                # Draw Dwell Ring around wrist
                if wrist_coord[2] > 0.35 and dwell_progress > 0.0:
                    wx, wy = int(wrist_coord[0]), int(wrist_coord[1])
                    cv2.circle(frame, (wx, wy), 25, (0, 229, 255), 2, cv2.LINE_AA)
                    cv2.ellipse(frame, (wx, wy), (25, 25), 0, 0, int(dwell_progress * 360), (0, 255, 100), 3, cv2.LINE_AA)
            else:
                mouse_feedback = "FROZEN (Zero-Trust)" if not is_teacher_active else "Idle"
                dwell_progress = 0.0

            # 6. Render Person Skeletons & Top HUD
            frame = tracker.draw_visuals(frame, persons, teacher_id=teacher_id)
            stats = stream.get_stats()
            frame = draw_phase4_hud(
                frame, stats, state, teacher_id, distance_m, current_mode,
                mouse_feedback, mouse_feedback_color, dwell_progress
            )

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
        print("\n[+] Phase 4 stream cleanly released.")


if __name__ == "__main__":
    run_phase4_diagnostics()
