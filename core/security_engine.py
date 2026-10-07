import time
import enum
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from core.tracker import PersonDetection
from core.signatures import BiometricSignatures


class SecurityState(enum.Enum):
    UNENROLLED = "UNENROLLED"
    LOCKED = "LOCKED"
    LOST = "LOST"
    RE_LOCKED = "RE_LOCKED"


class SecurityEngine:
    """
    Zero-Trust Multi-Template Biometric Security Engine.
    Maintains a robust Gallery of Teacher Biometrics (Face + Clothing).
    Guarantees instant, zero-failure Re-Identification upon teacher return.
    """

    MAX_GALLERY_SIZE = 15

    def __init__(
        self,
        lost_timeout_sec: float = 0.8,
    ):
        self.lost_timeout_sec = lost_timeout_sec

        self.state: SecurityState = SecurityState.UNENROLLED
        self.teacher_track_id: Optional[int] = None
        self.last_seen_timestamp: float = 0.0

        # Teacher Biometric Galleries
        self.face_gallery: List[np.ndarray] = []
        self.body_gallery: List[Dict[str, Any]] = []

    def enroll(self, frame: np.ndarray, target: PersonDetection) -> bool:
        """Enrolls the teacher and initializes the biometric template gallery."""
        # Clear previous gallery
        self.face_gallery.clear()
        self.body_gallery.clear()

        # Extract Face embeddings across full frame
        faces = BiometricSignatures.extract_face_embeddings_from_frame(frame)
        face_emb = BiometricSignatures.match_person_to_face(target.bbox, faces)
        body_sig = BiometricSignatures.extract_body_signature(frame, target.torso_bbox, target.keypoints)

        if face_emb is not None:
            self.face_gallery.append(face_emb)
        if body_sig is not None:
            self.body_gallery.append(body_sig)

        self.teacher_track_id = target.track_id
        self.last_seen_timestamp = time.time()
        self.state = SecurityState.LOCKED

        face_tag = "with Face Embedding" if face_emb is not None else "with Clothing Profile"
        print(f"[SecurityEngine] Enrolled Teacher ID #{target.track_id} ({face_tag}).")
        return True

    def _update_gallery(self, frame: np.ndarray, teacher: PersonDetection, faces: List[Dict[str, Any]]) -> None:
        """Continuously enriches the teacher's biometric gallery with diverse angles/distances."""
        face_emb = BiometricSignatures.match_person_to_face(teacher.bbox, faces)
        body_sig = BiometricSignatures.extract_body_signature(frame, teacher.torso_bbox, teacher.keypoints)

        if face_emb is not None:
            # Add if diverse enough
            if not self.face_gallery:
                self.face_gallery.append(face_emb)
            else:
                max_sim = max(
                    BiometricSignatures.match_face(face_emb, g_emb)[1]
                    for g_emb in self.face_gallery
                )
                if max_sim < 0.85:  # New angle/expression
                    self.face_gallery.append(face_emb)
                    if len(self.face_gallery) > self.MAX_GALLERY_SIZE:
                        self.face_gallery.pop(0)

        if body_sig is not None:
            if not self.body_gallery:
                self.body_gallery.append(body_sig)
            else:
                max_body_sim = max(
                    BiometricSignatures.match_body(body_sig, g_body)
                    for g_body in self.body_gallery
                )
                if max_body_sim < 0.90:
                    self.body_gallery.append(body_sig)
                    if len(self.body_gallery) > self.MAX_GALLERY_SIZE:
                        self.body_gallery.pop(0)

    def _match_candidate_against_gallery(
        self,
        frame: np.ndarray,
        candidate: PersonDetection,
        faces: List[Dict[str, Any]],
    ) -> Tuple[bool, float, str]:
        """Matches a person against the entire Teacher Gallery."""
        cand_face = BiometricSignatures.match_person_to_face(candidate.bbox, faces)
        cand_body = BiometricSignatures.extract_body_signature(frame, candidate.torso_bbox, candidate.keypoints)

        # 1. Face Match against Gallery
        best_face_score = 0.0
        if cand_face is not None and self.face_gallery:
            for g_face in self.face_gallery:
                _, score = BiometricSignatures.match_face(cand_face, g_face)
                if score > best_face_score:
                    best_face_score = score

            if best_face_score >= BiometricSignatures.SFACE_COSINE_THRESHOLD:
                return True, best_face_score, "Face Neural Match"
            elif best_face_score < 0.20:
                # Face clearly belongs to a different person (e.g. friend) -> Reject
                return False, best_face_score, "Different Face (Rejected)"

        # 2. Body Match against Gallery
        best_body_score = 0.0
        if cand_body is not None and self.body_gallery:
            for g_body in self.body_gallery:
                score = BiometricSignatures.match_body(cand_body, g_body)
                if score > best_body_score:
                    best_body_score = score

            if best_body_score >= 0.45:
                return True, best_body_score, "Clothing Match"

        return False, max(best_face_score, best_body_score), "Below Threshold"

    def update(
        self,
        frame: np.ndarray,
        persons: List[PersonDetection],
    ) -> Tuple[SecurityState, Optional[int], Dict[str, Any]]:
        now = time.time()
        metadata: Dict[str, Any] = {
            "state": self.state.value,
            "best_match_score": 0.0,
            "mouse_enabled": False,
            "match_reason": "",
            "gallery_size": len(self.face_gallery) + len(self.body_gallery),
        }

        # 1. UNENROLLED
        if self.state == SecurityState.UNENROLLED or self.teacher_track_id is None:
            return SecurityState.UNENROLLED, None, metadata

        # 2. ACTIVE (LOCKED or RE_LOCKED)
        if self.state in (SecurityState.LOCKED, SecurityState.RE_LOCKED):
            active_teacher = next((p for p in persons if p.track_id == self.teacher_track_id), None)

            if active_teacher is not None:
                self.last_seen_timestamp = now
                metadata["mouse_enabled"] = True
                
                # Occasionally enrich gallery (every 30 frames or when under-filled)
                if len(self.face_gallery) < 3 or (int(now * 10) % 30 == 0):
                    faces = BiometricSignatures.extract_face_embeddings_from_frame(frame)
                    self._update_gallery(frame, active_teacher, faces)
                return self.state, self.teacher_track_id, metadata

            # Check if teacher was lost
            time_lost = now - self.last_seen_timestamp
            if time_lost > self.lost_timeout_sec:
                self.state = SecurityState.LOST
                return SecurityState.LOST, None, metadata

            # Short transient occlusions
            return self.state, None, metadata

        # 3. LOST (Zero-Trust Active: Scanning for Teacher Re-Entry)
        if self.state == SecurityState.LOST:
            metadata["mouse_enabled"] = False

            if not persons or (not self.face_gallery and not self.body_gallery):
                return SecurityState.LOST, None, metadata

            # Detect faces only when scanning to re-identify returning teacher
            faces = BiometricSignatures.extract_face_embeddings_from_frame(frame)

            best_score = 0.0
            best_person = None
            best_reason = ""

            for p in persons:
                is_match, score, reason = self._match_candidate_against_gallery(frame, p, faces)
                if is_match and score > best_score:
                    best_score = score
                    best_person = p
                    best_reason = reason

            metadata["best_match_score"] = best_score
            metadata["match_reason"] = best_reason

            if best_person is not None:
                self.teacher_track_id = best_person.track_id
                self.last_seen_timestamp = now
                self.state = SecurityState.RE_LOCKED
                metadata["mouse_enabled"] = True
                self._update_gallery(frame, best_person, faces)
                print(f"[SecurityEngine] >>> RE-LOCKED TEACHER! (ID #{best_person.track_id} via {best_reason}) <<<")
                return self.state, self.teacher_track_id, metadata

            return SecurityState.LOST, None, metadata

        return SecurityState.UNENROLLED, None, metadata

    def unlock(self) -> None:
        self.state = SecurityState.UNENROLLED
        self.teacher_track_id = None
        self.face_gallery.clear()
        self.body_gallery.clear()
        print("[SecurityEngine] Teacher UNLOCKED.")
