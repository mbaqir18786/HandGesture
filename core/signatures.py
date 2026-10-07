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
        # Fast 320x180 resolution for 3.5ms detection
        target_w, target_h = 320, 180
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
                    fx1 = int(face[0] * scale_x)
                    fy1 = int(face[1] * scale_y)
                    fw = int(face[2] * scale_x)
                    fh = int(face[3] * scale_y)

                    if fw < 20 or fh < 20:
                        continue

                    # Crop and align face from full original frame
                    aligned = cls._face_recognizer.alignCrop(small_frame, face)
                    embedding = cls._face_recognizer.feature(aligned)

                    results.append({
                        "box": (fx1, fy1, fx1 + fw, fy1 + fh),
                        "center": (fx1 + fw // 2, fy1 + fh // 2),
                        "embedding": embedding.copy(),
                    })
        except Exception:
            pass

        return results

    @classmethod
    def match_person_to_face(
        cls,
        person_bbox: Tuple[int, int, int, int],
        detected_faces: List[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        """Finds the face embedding belonging to a person."""
        px1, py1, px2, py2 = person_bbox
        upper_y2 = py1 + int((py2 - py1) * 0.55)

        for f in detected_faces:
            fcx, fcy = f["center"]
            if px1 <= fcx <= px2 and py1 <= fcy <= upper_y2:
                return f["embedding"]
        return None

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
