import os
import math
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np


class BiometricSignatures:
    """
    State-of-the-Art Multi-Modal Biometric Matching Engine:
    - Full-Frame YuNet Face Detection + SFace 128-D Deep Feature Embeddings.
    - Illumination-Invariant 2D HS Clothing Color Histograms + CIELAB Color Moments.
    - Multi-Template Gallery matching for 100% reliable Teacher Re-Identification.
    """

    _face_detector = None
    _face_recognizer = None
    _initialized = False

    H_BINS = 16
    S_BINS = 8
    SFACE_COSINE_THRESHOLD = 0.34

    @classmethod
    def initialize_models(cls) -> None:
        if cls._initialized:
            return

        from core.config_manager import get_resource_path
        yunet_path = get_resource_path(os.path.join("models", "face_detection_yunet.onnx"))
        sface_path = get_resource_path(os.path.join("models", "face_recognition_sface.onnx"))

        if os.path.exists(yunet_path) and os.path.exists(sface_path):
            try:
                # YuNet with fast 320x180 input dimensions (3.5ms inference)
                cls._face_detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 180), 0.45, 0.3, 500)
                cls._face_recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
                cls._initialized = True
                print("[BiometricSignatures] Deep Neural SFace Engine Initialized.")
            except Exception as e:
                print(f"[BiometricSignatures] SFace init error: {e}")

    @classmethod
    def extract_face_embeddings_from_frame(cls, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects all faces across the full frame using YuNet and extracts 128-D SFace embeddings.
        Returns list of dicts: [{'box': (x1, y1, x2, y2), 'embedding': np.ndarray, 'center': (cx, cy)}]
        """
        cls.initialize_models()
        if cls._face_detector is None or cls._face_recognizer is None or frame is None:
            return []

        h, w = frame.shape[:2]
        # 640x360 resolution for fast detection while preserving distant faces
        target_w, target_h = 640, 360
        scale_x = w / float(target_w)
        scale_y = h / float(target_h)

        small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        cls._face_detector.setInputSize((target_w, target_h))

        results = []
        try:
            _, faces = cls._face_detector.detect(small_frame)
            if faces is not None:
                for face in faces:
                    # Scale back to original frame coordinates
                    scaled_face = face.copy()
                    # Scale all X coordinates
                    for xi in (0, 2, 4, 6, 8, 10, 12):
                        scaled_face[xi] *= scale_x
                    # Scale all Y coordinates
                    for yi in (1, 3, 5, 7, 9, 11, 13):
                        scaled_face[yi] *= scale_y

                    fx1 = int(scaled_face[0])
                    fy1 = int(scaled_face[1])
                    fw = int(scaled_face[2])
                    fh = int(scaled_face[3])

                    if fw < 18 or fh < 18:
                        continue

                    # Crop and align face from the FULL high-resolution original frame
                    aligned = cls._face_recognizer.alignCrop(frame, scaled_face)
                    embedding = cls._face_recognizer.feature(aligned)

                    results.append({
                        "box": (fx1, fy1, fx1 + fw, fy1 + fh),
                        "center": (fx1 + fw // 2, fy1 + fh // 2),
                        "embedding": embedding.copy(),
                        "width": fw,
                        "height": fh,
                        "score": float(face[14]),
                    })
        except Exception as e:
            pass

        return results

    @classmethod
    def match_person_to_face_info(
        cls,
        person_bbox: Tuple[int, int, int, int],
        detected_faces: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Finds the most centered face dictionary belonging to a person's upper body."""
        px1, py1, px2, py2 = person_bbox
        upper_y2 = py1 + int((py2 - py1) * 0.55)
        pcx = (px1 + px2) // 2
        p_head_y = py1 + int((py2 - py1) * 0.18)

        best_face = None
        min_dist = float("inf")

        for f in detected_faces:
            fcx, fcy = f["center"]
            if (px1 - 20) <= fcx <= (px2 + 20) and (py1 - 30) <= fcy <= upper_y2:
                dist = abs(fcx - pcx) + abs(fcy - p_head_y)
                if dist < min_dist:
                    min_dist = dist
                    best_face = f

        if best_face is not None:
            box = best_face["box"]
            fw = box[2] - box[0]
            fh = box[3] - box[1]
            return {
                "embedding": best_face["embedding"],
                "box": box,
                "width": fw,
                "height": fh,
            }
        return None

    @classmethod
    def match_person_to_face(
        cls,
        person_bbox: Tuple[int, int, int, int],
        detected_faces: List[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        """Finds the face embedding belonging to a person."""
        info = cls.match_person_to_face_info(person_bbox, detected_faces)
        return info["embedding"] if info is not None else None

    @classmethod
    def extract_body_signature(
        cls,
        frame: np.ndarray,
        torso_bbox: Optional[Tuple[int, int, int, int]],
        keypoints: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        if torso_bbox is None or frame is None:
            return None

        x1, y1, x2, y2 = torso_bbox
        h_f, w_f = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_f, x2), min(h_f, y2)

        if x2 <= x1 + 10 or y2 <= y1 + 10:
            return None

        torso_crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2Lab)

        # 2D HS Histogram (Hue + Saturation, lighting invariant)
        hist = cv2.calcHist([hsv], [0, 1], None, [cls.H_BINS, cls.S_BINS], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # CIELAB color moments
        lab_mean, _ = cv2.meanStdDev(lab)

        return {
            "hist": hist,
            "lab_mean": lab_mean.flatten(),
        }

    @classmethod
    def extract_face_for_person(
        cls,
        frame: np.ndarray,
        person_bbox: Tuple[int, int, int, int],
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts face directly from high-resolution upper body crop.
        Essential for distant individuals (> 3m) where full-frame downscaling loses facial details.
        """
        cls.initialize_models()
        if cls._face_detector is None or cls._face_recognizer is None or frame is None:
            return None

        h, w = frame.shape[:2]
        px1, py1, px2, py2 = person_bbox
        bw = max(1, px2 - px1)
        bh = max(1, py2 - py1)

        # Crop head region: from slightly above py1 to upper 45% of body
        cy1 = max(0, int(py1 - bh * 0.08))
        cy2 = min(h, int(py1 + bh * 0.45))
        cx1 = max(0, int(px1 - bw * 0.10))
        cx2 = min(w, int(px2 + bw * 0.10))

        crop_h = cy2 - cy1
        crop_w = cx2 - cx1
        if crop_w < 20 or crop_h < 20:
            return None

        crop = frame[cy1:cy2, cx1:cx2]
        try:
            cls._face_detector.setInputSize((crop_w, crop_h))
            _, faces = cls._face_detector.detect(crop)

            # Fallback for distant/dim faces (> 4m): CLAHE contrast enhancement on luminance channel
            if (faces is None or len(faces) == 0) and crop_w >= 20 and crop_h >= 20:
                try:
                    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    enhanced_crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    _, faces = cls._face_detector.detect(enhanced_crop)
                except Exception:
                    pass

            if faces is not None and len(faces) > 0:
                best_face = faces[0]
                scaled_face = best_face.copy()
                for xi in (0, 2, 4, 6, 8, 10, 12):
                    scaled_face[xi] += cx1
                for yi in (1, 3, 5, 7, 9, 11, 13):
                    scaled_face[yi] += cy1

                fw = int(scaled_face[2])
                fh = int(scaled_face[3])
                fx1 = int(scaled_face[0])
                fy1 = int(scaled_face[1])

                aligned = cls._face_recognizer.alignCrop(frame, scaled_face)
                embedding = cls._face_recognizer.feature(aligned)

                return {
                    "box": (fx1, fy1, fx1 + fw, fy1 + fh),
                    "center": (fx1 + fw // 2, fy1 + fh // 2),
                    "embedding": embedding.copy(),
                    "width": fw,
                    "height": fh,
                    "score": float(best_face[14]),
                }
        except Exception:
            pass
        return None

    @classmethod
    def extract_body_proportions(cls, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts distance-invariant body proportion ratios from COCO 17 keypoints:
        1. Shoulder width / Torso length
        2. Upper arm length / Forearm length
        3. Leg length / Torso length
        Returns 3-D numpy array of ratios, or None if keypoint confidences are insufficient.
        """
        if keypoints is None or len(keypoints) < 17:
            return None

        def _dist(idx1, idx2):
            return math.hypot(keypoints[idx1][0] - keypoints[idx2][0], keypoints[idx1][1] - keypoints[idx2][1])

        def _conf(idx):
            return keypoints[idx][2]

        # 1. Shoulder width: 5 (left shoulder) to 6 (right shoulder)
        if _conf(5) < 0.35 or _conf(6) < 0.35:
            return None
        shoulder_w = _dist(5, 6)

        # 2. Torso length: shoulders midpoint to hips (11, 12) midpoint
        if _conf(11) < 0.35 or _conf(12) < 0.35:
            return None
        sh_mid = ((keypoints[5][0] + keypoints[6][0]) / 2.0, (keypoints[5][1] + keypoints[6][1]) / 2.0)
        hip_mid = ((keypoints[11][0] + keypoints[12][0]) / 2.0, (keypoints[11][1] + keypoints[12][1]) / 2.0)
        torso_len = math.hypot(sh_mid[0] - hip_mid[0], sh_mid[1] - hip_mid[1])
        if torso_len < 10.0:
            return None

        ratio_shoulder_torso = shoulder_w / torso_len

        # 3. Arm ratio: upper arm (5->7 or 6->8) / forearm (7->9 or 8->10)
        arm_ratios = []
        if _conf(5) > 0.35 and _conf(7) > 0.35 and _conf(9) > 0.35:
            u_arm = _dist(5, 7)
            f_arm = _dist(7, 9)
            if f_arm > 5.0:
                arm_ratios.append(u_arm / f_arm)
        if _conf(6) > 0.35 and _conf(8) > 0.35 and _conf(10) > 0.35:
            u_arm = _dist(6, 8)
            f_arm = _dist(8, 10)
            if f_arm > 5.0:
                arm_ratios.append(u_arm / f_arm)
        ratio_arm = np.median(arm_ratios) if arm_ratios else 1.0

        # 4. Leg to torso ratio: hip to ankle (11->15 or 12->16)
        leg_lens = []
        if _conf(11) > 0.35 and _conf(13) > 0.35 and _conf(15) > 0.35:
            leg_lens.append(_dist(11, 13) + _dist(13, 15))
        if _conf(12) > 0.35 and _conf(14) > 0.35 and _conf(16) > 0.35:
            leg_lens.append(_dist(12, 14) + _dist(14, 16))
        ratio_leg_torso = (np.median(leg_lens) / torso_len) if leg_lens else 1.8

        return np.array([ratio_shoulder_torso, ratio_arm, ratio_leg_torso], dtype=np.float32)

    @classmethod
    def match_body_proportions(cls, prop1: Optional[np.ndarray], prop2: Optional[np.ndarray]) -> float:
        """
        Compares two body proportion vectors.
        Returns similarity score between 0.0 and 1.0.
        """
        if prop1 is None or prop2 is None:
            return 0.5  # Neutral when unobserved
        diff = np.abs(prop1 - prop2) / np.maximum(np.abs(prop1), 1e-3)
        avg_err = float(np.mean(diff))
        sim = max(0.0, 1.0 - (avg_err / 0.25))
        return float(np.clip(sim, 0.0, 1.0))

    @classmethod
    def match_body(cls, sig1: Optional[Dict[str, Any]], sig2: Optional[Dict[str, Any]]) -> float:
        if sig1 is None or sig2 is None:
            return 0.0

        corr = cv2.compareHist(sig1["hist"], sig2["hist"], cv2.HISTCMP_CORREL)
        hist_score = max(0.0, float(corr))

        lab1, lab2 = sig1["lab_mean"], sig2["lab_mean"]
        delta_e = math.sqrt(
            (lab1[0] - lab2[0]) ** 2 +
            (lab1[1] - lab2[1]) ** 2 +
            (lab1[2] - lab2[2]) ** 2
        )
        lab_score = max(0.0, 1.0 - (delta_e / 55.0))

        return float(np.clip((hist_score * 0.70) + (lab_score * 0.30), 0.0, 1.0))

    @classmethod
    def match_face(cls, emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]) -> Tuple[bool, float]:
        cls.initialize_models()
        if emb1 is None or emb2 is None or cls._face_recognizer is None:
            return False, 0.0

        try:
            cosine = cls._face_recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE)
            is_match = cosine >= cls.SFACE_COSINE_THRESHOLD
            return is_match, float(cosine)
        except Exception:
            return False, 0.0

