import os
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import cv2

from core.distance_estimator import DistanceEstimator


# COCO 17 Keypoint Connection Pairs
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # Facial links
    (5, 6),                                  # Shoulders
    (5, 7), (7, 9),                          # Left Arm
    (6, 8), (8, 10),                         # Right Arm
    (5, 11), (6, 12), (11, 12),              # Torso
    (11, 13), (13, 15),                      # Left Leg
    (12, 14), (14, 16)                       # Right Leg
]


@dataclass
class PersonDetection:
    """Structured container for tracked individual in the classroom."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    keypoints: np.ndarray             # (17, 3) -> [x, y, conf]
    is_standing: bool
    distance_info: Dict[str, Any]
    torso_bbox: Optional[Tuple[int, int, int, int]]
    face_bbox: Optional[Tuple[int, int, int, int]]
    right_wrist: Tuple[float, float, float]  # (x, y, conf)
    left_wrist: Tuple[float, float, float]   # (x, y, conf)


class PersonTracker:
    """
    Multi-Person YOLOv8-Pose + ByteTrack Engine.
    Detects all individuals, tracks stable IDs across frames,
    computes standing posture, and estimates real-time distance.
    """

    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        min_person_conf: float = 0.50,
        near_threshold_m: float = 3.5,
        max_distance_m: float = 10.0,
    ):
        self.model_name = model_name
        self.min_person_conf = min_person_conf
        self.distance_estimator = DistanceEstimator(
            near_threshold_m=near_threshold_m,
            max_distance_m=max_distance_m,
        )

        self._model = None
        self._init_model()

    def _init_model(self) -> None:
        """Lazily imports and initializes YOLOv8-pose."""
        try:
            from ultralytics import YOLO
            from core.config_manager import get_resource_path
            model_file = get_resource_path(self.model_name)
            target = model_file if os.path.exists(model_file) else self.model_name
            print(f"[PersonTracker] Loading YOLO pose model '{target}'...")
            self._model = YOLO(target)
            print("[PersonTracker] YOLO model loaded successfully.")
        except ImportError:
            print("[PersonTracker] WARNING: 'ultralytics' not yet available. Tracker in standby.")
            self._model = None
        except Exception as e:
            print(f"[PersonTracker] Error loading YOLO model: {e}")
            self._model = None

    def is_ready(self) -> bool:
        return self._model is not None

    def track(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Runs YOLOv8-Pose + ByteTrack on the frame.
        Returns list of structured PersonDetection objects.
        """
        if self._model is None:
            self._init_model()
            if self._model is None:
                return []

        h, w = frame.shape[:2]
        # Run tracker with ByteTrack (optimized 384px size for 2.5x speedup)
        results = self._model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.min_person_conf,
            imgsz=384,
            verbose=False,
        )

        detections: List[PersonDetection] = []
        if not results or len(results) == 0:
            return detections

        res = results[0]
        if res.boxes is None or res.keypoints is None:
            return detections

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        track_ids = (
            res.boxes.id.cpu().numpy().astype(int)
            if res.boxes.id is not None
            else list(range(len(boxes)))
        )
        kpts_data = res.keypoints.data.cpu().numpy()  # shape (N, 17, 3)

        for i, (box, conf, tid) in enumerate(zip(boxes, confs, track_ids)):
            x1, y1, x2, y2 = [int(v) for v in box]
            kpts = kpts_data[i]  # shape (17, 3)

            # 1. Posture Classification (Standing vs Sitting)
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            aspect_ratio = box_h / float(box_w)
            is_standing = aspect_ratio >= 1.35

            # 2. Distance Estimation
            dist_info = self.distance_estimator.estimate_distance(kpts, (h, w))

            # 3. Torso / Shirt Crop Bounding Box
            torso_bbox = None
            if (
                kpts[5][2] > 0.25
                and kpts[6][2] > 0.25
                and kpts[11][2] > 0.25
                and kpts[12][2] > 0.25
            ):
                # Full torso available (shoulders to hips)
                tx1 = max(0, int(min(kpts[5][0], kpts[11][0]) - 15))
                ty1 = max(0, int(min(kpts[5][1], kpts[6][1]) - 10))
                tx2 = min(w, int(max(kpts[6][0], kpts[12][0]) + 15))
                ty2 = min(h, int(max(kpts[11][1], kpts[12][1]) + 15))
            elif kpts[5][2] > 0.25 or kpts[6][2] > 0.25:
                # Close-up: shoulders visible, hips off-screen
                sh_y = min(
                    kpts[5][1] if kpts[5][2] > 0.25 else y2,
                    kpts[6][1] if kpts[6][2] > 0.25 else y2,
                )
                tx1 = max(0, int(x1 + box_w * 0.10))
                ty1 = max(0, int(sh_y))
                tx2 = min(w, int(x2 - box_w * 0.10))
                ty2 = min(h, y2)
            else:
                # Fallback to middle-to-lower region of bounding box
                tx1 = max(0, int(x1 + box_w * 0.15))
                ty1 = max(0, int(y1 + box_h * 0.40))
                tx2 = min(w, int(x2 - box_w * 0.15))
                ty2 = min(h, y2)

            if tx2 > tx1 + 10 and ty2 > ty1 + 10:
                torso_bbox = (tx1, ty1, tx2, ty2)

            # 4. Face Crop Bounding Box
            face_bbox = None
            if kpts[0][2] > 0.3:  # Nose detected
                nose_x, nose_y = kpts[0][0], kpts[0][1]
                head_radius = max(30, int(box_w * 0.22))
                fx1 = max(0, int(nose_x - head_radius))
                fy1 = max(0, int(nose_y - head_radius * 1.2))
                fx2 = min(w, int(nose_x + head_radius))
                fy2 = min(h, int(nose_y + head_radius * 1.2))
            else:
                # Fallback to top 35% of person box
                fx1 = max(0, int(x1 + box_w * 0.20))
                fy1 = max(0, y1)
                fx2 = min(w, int(x2 - box_w * 0.20))
                fy2 = min(h, int(y1 + box_h * 0.35))

            if fx2 > fx1 + 15 and fy2 > fy1 + 15:
                face_bbox = (fx1, fy1, fx2, fy2)

            # 5. Wrist Keypoints for Far Mode
            rw = (float(kpts[10][0]), float(kpts[10][1]), float(kpts[10][2]))  # Right wrist
            lw = (float(kpts[9][0]), float(kpts[9][1]), float(kpts[9][2]))    # Left wrist

            detections.append(
                PersonDetection(
                    track_id=int(tid),
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    keypoints=kpts,
                    is_standing=is_standing,
                    distance_info=dist_info,
                    torso_bbox=torso_bbox,
                    face_bbox=face_bbox,
                    right_wrist=rw,
                    left_wrist=lw,
                )
            )

        return detections

    def draw_visuals(
        self,
        frame: np.ndarray,
        persons: List[PersonDetection],
        teacher_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Renders sleek skeleton overlay, bounding boxes, distance badges,
        and posture status on top of the frame.
        """
        for p in persons:
            is_teacher = teacher_id is not None and p.track_id == teacher_id
            primary_color = (0, 255, 102) if is_teacher else (255, 180, 0)  # Green or Cyan/Orange
            kpt_color = (0, 255, 255) if is_teacher else (200, 200, 200)

            x1, y1, x2, y2 = p.bbox

            # Draw Person Bounding Box
            box_thick = 2 if is_teacher else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), primary_color, box_thick, cv2.LINE_AA)

            # Draw 17 Keypoint Skeleton
            for a, b in SKELETON_CONNECTIONS:
                if p.keypoints[a][2] > 0.35 and p.keypoints[b][2] > 0.35:
                    pt_a = (int(p.keypoints[a][0]), int(p.keypoints[a][1]))
                    pt_b = (int(p.keypoints[b][0]), int(p.keypoints[b][1]))
                    cv2.line(frame, pt_a, pt_b, primary_color, 2, cv2.LINE_AA)

            # Draw Keypoint Nodes
            for kx, ky, kc in p.keypoints:
                if kc > 0.35:
                    cv2.circle(frame, (int(kx), int(ky)), 3, kpt_color, -1, cv2.LINE_AA)

            # Label Badge Header
            posture_str = "STAND" if p.is_standing else "SIT"
            dist_str = f"{p.distance_info.get('distance_m', 0.0):.1f}m ({p.distance_info.get('mode', 'NEAR')})"
            role_tag = "[TEACHER] " if is_teacher else f"ID #{p.track_id} "
            badge_text = f"{role_tag}| {dist_str} | {posture_str}"

            # Badge background
            tw, th = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            by1 = max(0, y1 - 22)
            cv2.rectangle(frame, (x1, by1), (x1 + tw + 10, by1 + 20), (25, 25, 30), -1)
            cv2.putText(
                frame,
                badge_text,
                (x1 + 5, by1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                primary_color,
                1,
                cv2.LINE_AA,
            )

        return frame
