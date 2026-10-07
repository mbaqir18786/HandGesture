import math
from typing import Dict, Any, Optional, Tuple
import numpy as np


class DistanceEstimator:
    """
    Estimates human distance from single-camera feed using human anthropometric
    proportions (shoulder width, torso length, eye span) and pinhole geometry.
    Classifies proximity into NEAR (< 3.5m) and FAR (3.5m - 10.0m).
    """

    # Anthropometric standard averages (in meters)
    AVG_SHOULDER_WIDTH_M = 0.42
    AVG_TORSO_LENGTH_M = 0.52
    AVG_INTERPUPIL_M = 0.065

    # Interaction modes
    MODE_NEAR = "NEAR"
    MODE_FAR = "FAR"
    MODE_OUT_OF_RANGE = "OUT_OF_RANGE"

    def __init__(
        self,
        near_threshold_m: float = 3.5,
        max_distance_m: float = 10.0,
        fov_degrees: float = 68.0,
    ):
        self.near_threshold_m = near_threshold_m
        self.max_distance_m = max_distance_m
        self.fov_degrees = fov_degrees

    def _estimate_focal_length_px(self, frame_width: int) -> float:
        """Estimates horizontal focal length in pixels from camera horizontal FOV."""
        fov_rad = math.radians(self.fov_degrees)
        return (frame_width / 2.0) / math.tan(fov_rad / 2.0)

    def estimate_distance(
        self,
        keypoints: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        Estimates real-world distance in meters using pose keypoints.
        keypoints: shape (17, 3) where [:, 0]=x, [:, 1]=y, [:, 2]=conf
        frame_shape: (height, width)
        """
        h, w = frame_shape[:2]
        focal_length_px = self._estimate_focal_length_px(w)

        estimates = []

        # 1. Shoulder width estimation (Keypoint 5: Left Shoulder, 6: Right Shoulder)
        if len(keypoints) > 6 and keypoints[5][2] > 0.4 and keypoints[6][2] > 0.4:
            shoulder_px = math.hypot(
                keypoints[5][0] - keypoints[6][0],
                keypoints[5][1] - keypoints[6][1],
            )
            if shoulder_px > 10:
                dist_shoulder = (self.AVG_SHOULDER_WIDTH_M * focal_length_px) / shoulder_px
                estimates.append((dist_shoulder, 1.2))  # (distance, weight)

        # 2. Torso length estimation (Mid-Shoulder to Mid-Hip)
        if (
            len(keypoints) > 12
            and keypoints[5][2] > 0.35
            and keypoints[6][2] > 0.35
            and keypoints[11][2] > 0.35
            and keypoints[12][2] > 0.35
        ):
            mid_shoulder_x = (keypoints[5][0] + keypoints[6][0]) / 2.0
            mid_shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2.0
            mid_hip_x = (keypoints[11][0] + keypoints[12][0]) / 2.0
            mid_hip_y = (keypoints[11][1] + keypoints[12][1]) / 2.0

            torso_px = math.hypot(mid_shoulder_x - mid_hip_x, mid_shoulder_y - mid_hip_y)
            if torso_px > 15:
                dist_torso = (self.AVG_TORSO_LENGTH_M * focal_length_px) / torso_px
                estimates.append((dist_torso, 1.5))  # (distance, weight)

        # 3. Eye span (Close distance fallback, Keypoint 1: Left Eye, 2: Right Eye)
        if len(keypoints) > 2 and keypoints[1][2] > 0.5 and keypoints[2][2] > 0.5:
            eye_px = math.hypot(
                keypoints[1][0] - keypoints[2][0],
                keypoints[1][1] - keypoints[2][1],
            )
            if eye_px > 8:
                dist_eye = (self.AVG_INTERPUPIL_M * focal_length_px) / eye_px
                estimates.append((dist_eye, 0.8))

        if not estimates:
            return {
                "distance_m": 3.0,  # Default neutral fallback
                "mode": self.MODE_NEAR,
                "confidence": 0.0,
                "proximity_score": 0.5,
            }

        # Weighted average distance
        total_weight = sum(w for _, w in estimates)
        estimated_m = sum(d * w for d, w in estimates) / total_weight
        estimated_m = float(np.clip(estimated_m, 0.5, 15.0))

        # Mode classification
        if estimated_m <= self.near_threshold_m:
            mode = self.MODE_NEAR
        elif estimated_m <= self.max_distance_m:
            mode = self.MODE_FAR
        else:
            mode = self.MODE_OUT_OF_RANGE

        # Proximity score (0.0 to 1.0, higher means closer & prominent)
        proximity_score = float(np.clip(1.0 - (estimated_m / 8.0), 0.05, 1.0))

        return {
            "distance_m": round(estimated_m, 2),
            "mode": mode,
            "confidence": min(1.0, len(estimates) / 2.0),
            "proximity_score": round(proximity_score, 3),
        }
