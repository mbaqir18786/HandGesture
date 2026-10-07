import os
import time
import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pyautogui
import numpy as np

# Disable failsafe
pyautogui.FAILSAFE = False

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Active interaction zone (easy reach across whole screen)
X_MIN, X_MAX = 0.22, 0.78
Y_MIN, Y_MAX = 0.20, 0.60

# Precision tuning
SMOOTHING = 5          # Cursor smoothness
DEADZONE = 5           # Deadzone to eliminate hand vibration
prev_x, prev_y = screen_width // 2, screen_height // 2

# Tap / Double-Tap detection states
is_pinched = False
last_pinch_time = 0
DOUBLE_TAP_WINDOW = 0.40  # 0.40 seconds window to detect double click

# -----------------------------------------------------------
# Setup MediaPipe Hand Landmarker
# -----------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "hand_landmarker.task")

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def get_distance(pt1, pt2, w, h):
    """Calculates pixel distance between two landmarks."""
    return math.hypot((pt2.x - pt1.x) * w, (pt2.y - pt1.y) * h)


def main():
    global prev_x, prev_y, is_pinched, last_pinch_time

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[-] Error: Could not open camera.")
            return

    timestamp_ms = 0
    feedback_text = "Aiming Cursor"
    feedback_color = (0, 255, 0)

    print("[+] Single / Double Pinch Mouse Active:")
    print("    - 🎯 Move hand: Steady cursor")
    print("    - 🤏 Tap Thumb + Index ONCE: Single Click")
    print("    - 🤏 Tap Thumb + Index TWICE quickly: Double Click")
    print("    - Press 'q' to exit")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            current_time = time.time()

            # Draw the active screen interaction box
            box_x1, box_y1 = int(X_MIN * w), int(Y_MIN * h)
            box_x2, box_y2 = int(X_MAX * w), int(Y_MAX * h)
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += 33

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                # Draw skeleton
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                for (a, b) in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                for pt in pts:
                    cv2.circle(frame, pt, 4, (0, 0, 255), -1)

                thumb_tip = hand[4]
                index_tip = hand[8]
                tracking_point = hand[5]  # Index knuckle for steady tracking

                cv2.circle(frame, (int(tracking_point.x * w), int(tracking_point.y * h)), 8, (255, 0, 0), cv2.FILLED)

                # Distance between Thumb and Index
                pinch_dist = get_distance(thumb_tip, index_tip, w, h)

                PINCH_CLOSE_THRESHOLD = 32
                PINCH_OPEN_THRESHOLD = 45

                # Detect Pinch DOWN event
                if pinch_dist < PINCH_CLOSE_THRESHOLD:
                    cv2.circle(frame, (int(index_tip.x * w), int(index_tip.y * h)), 14, (0, 0, 255), cv2.FILLED)

                    if not is_pinched:
                        time_since_last = current_time - last_pinch_time

                        # Check if this tap is a DOUBLE CLICK (second tap within window)
                        if time_since_last < DOUBLE_TAP_WINDOW:
                            pyautogui.doubleClick()
                            last_pinch_time = 0  # Reset
                            feedback_text = "DOUBLE CLICK!"
                            feedback_color = (255, 255, 0)
                        else:
                            pyautogui.click()
                            last_pinch_time = current_time
                            feedback_text = "SINGLE CLICK!"
                            feedback_color = (0, 0, 255)

                        is_pinched = True

                # Detect Pinch RELEASE (hysteresis)
                elif pinch_dist > PINCH_OPEN_THRESHOLD:
                    is_pinched = False
                    feedback_text = "Aiming Cursor"
                    feedback_color = (0, 255, 0)

                    # Move Cursor smoothly when not pinching
                    raw_x = np.interp(tracking_point.x, [X_MIN, X_MAX], [0, screen_width])
                    raw_y = np.interp(tracking_point.y, [Y_MIN, Y_MAX], [0, screen_height])

                    target_x = np.clip(raw_x, 0, screen_width)
                    target_y = np.clip(raw_y, 0, screen_height)

                    diff_x = target_x - prev_x
                    diff_y = target_y - prev_y

                    if abs(diff_x) > DEADZONE or abs(diff_y) > DEADZONE:
                        curr_x = prev_x + diff_x / SMOOTHING
                        curr_y = prev_y + diff_y / SMOOTHING
                        pyautogui.moveTo(int(curr_x), int(curr_y), _pause=False)
                        prev_x, prev_y = curr_x, curr_y

                cv2.putText(frame, feedback_text, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, feedback_color, 2)

            cv2.imshow('Hand Gesture Mouse Control', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            landmarker.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
