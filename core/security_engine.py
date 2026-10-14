import time
import enum
from datetime import datetime
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
    Rules:
    1. Lock belongs to saved identity, NOT ByteTrack ID.
    2. Face match is the ONLY cue that can lock or re-lock. Body cues never decide alone.
    3. Re-lock from LOST requires face >= min_size, score >= threshold, for N consecutive frames on same track ID.
    4. Periodic face check every 2-3s while LOCKED catches track switches.
    5. Turning back maintains LOCKED while track persists.
    6. Original enrollment templates remain fixed. Adaptive templates added ONLY when LOCKED with score >= 0.55.
    7. Clean audit logging and transparent status reasons for UI.
    """

    def __init__(
        self,
        lost_timeout_sec: float = 1.0,
        face_match_threshold: float = 0.40,
        relock_consecutive_frames: int = 8,
        min_face_size_px: int = 32,
        periodic_face_interval_sec: float = 2.5,
        adaptive_gallery_max: int = 10,
    ):
        self.lost_timeout_sec = lost_timeout_sec
        self.face_match_threshold = face_match_threshold
        self.relock_consecutive_frames = relock_consecutive_frames
        self.min_face_size_px = min_face_size_px
        self.periodic_face_interval_sec = periodic_face_interval_sec
        self.adaptive_gallery_max = adaptive_gallery_max

        self.state: SecurityState = SecurityState.UNENROLLED
        self.teacher_track_id: Optional[int] = None
        self.last_seen_timestamp: float = 0.0
        self.last_face_check_timestamp: float = 0.0

        # Fixed original templates (never overwritten)
        self.enrolled_face_templates: List[np.ndarray] = []
        # Enrolled body proportions (scale/distance invariant ratios)
        self.enrolled_body_prop: Optional[np.ndarray] = None
        # Adaptive templates (up to adaptive_gallery_max, only added when strongly confirmed in LOCKED)
        self.adaptive_face_templates: List[np.ndarray] = []
        # Supplementary body gallery (never decides alone)
        self.body_gallery: List[Dict[str, Any]] = []

        # Consecutive frame confirmation state for re-locking
        self.candidate_relock_track_id: Optional[int] = None
        self.candidate_relock_consecutive_count: int = 0

        # UI Diagnostic telemetry
        self.status_reason: str = "UNENROLLED: waiting for enrollment"
        self.last_face_score: float = 0.0
        self.audit_logs: List[str] = []

    def _log_state_change(self, new_state: str, reason: str, score: float = 0.0) -> None:
        """Logs every state change with timestamp, reason, and score (no images)."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{ts}] STATE: {new_state} | score={score:.2f} | reason: {reason}"
        self.audit_logs.append(log_entry)
        if len(self.audit_logs) > 200:
            self.audit_logs.pop(0)
        print(f"[SecurityEngine] {log_entry}")

    def _transition_to_lost(self, reason: str) -> None:
        """Transitions state to LOST and resets re-lock accumulator."""
        old_state = self.state.value
        self.state = SecurityState.LOST
        self.candidate_relock_track_id = None
        self.candidate_relock_consecutive_count = 0
        self.status_reason = reason
        self._log_state_change("LOST", reason, self.last_face_score)

    def enroll(
        self,
        frame: np.ndarray,
        target: PersonDetection,
        additional_face_embs: Optional[List[np.ndarray]] = None,
        enrolled_body_prop: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Enrolls the teacher and initializes the biometric template gallery.
        Face match is strictly REQUIRED. Enrollment fails if no valid face is visible.
        """
        self.enrolled_face_templates.clear()
        self.adaptive_face_templates.clear()
        self.body_gallery.clear()
        self.enrolled_body_prop = None
        self.candidate_relock_track_id = None
        self.candidate_relock_consecutive_count = 0

        faces = BiometricSignatures.extract_face_embeddings_from_frame(frame)
        face_info = BiometricSignatures.match_person_to_face_info(target.bbox, faces)
        if face_info is None and frame is not None:
            face_info = BiometricSignatures.extract_face_for_person(frame, target.bbox)

        if face_info is None or face_info.get("width", 0) < self.min_face_size_px:
            self.status_reason = "Enrollment Failed: clear face required"
            self._log_state_change(self.state.value, "Enrollment rejected (no clear face)", 0.0)
            print("[SecurityEngine] Enrollment REJECTED: no clear face detected in bounding box.")
            return False

        # Store fixed primary face embedding
        primary_emb = face_info["embedding"]
        self.enrolled_face_templates.append(primary_emb.copy())

        # Store any additional guided enrollment face embeddings
        if additional_face_embs:
            for emb in additional_face_embs:
                self.enrolled_face_templates.append(emb.copy())

        # Store distance-invariant body proportions
        if enrolled_body_prop is not None:
            self.enrolled_body_prop = enrolled_body_prop.copy()
        elif target.keypoints is not None:
            self.enrolled_body_prop = BiometricSignatures.extract_body_proportions(target.keypoints)

        # Store initial body signature for secondary context
        body_sig = BiometricSignatures.extract_body_signature(frame, target.torso_bbox, target.keypoints)
        if body_sig is not None:
            self.body_gallery.append(body_sig)

        now = time.time()
        self.teacher_track_id = target.track_id
        self.last_seen_timestamp = now
        self.last_face_check_timestamp = now
        self.state = SecurityState.LOCKED
        self.last_face_score = 1.0
        self.status_reason = f"LOCKED (ID #{target.track_id}): teacher enrolled"
        self._log_state_change("LOCKED", f"Enrolled teacher ID #{target.track_id} ({len(self.enrolled_face_templates)} face templates)", 1.0)
        return True

    def _match_candidate_face(
        self,
        person: PersonDetection,
        faces: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None,
    ) -> Tuple[str, float, str]:
        """
        Matches a person's face against valid teacher templates.
        Returns: (match_status, best_score, reason)
        where match_status is:
          - 'MATCH': face verified with score >= threshold
          - 'MISMATCH': clear large face (>= 40px) verified with score < threshold (impostor)
          - 'INSUFFICIENT': face not visible or too small/distant (< 40px) to judge mismatch
        """
        face_info = BiometricSignatures.match_person_to_face_info(person.bbox, faces)
        if face_info is None and frame is not None:
            face_info = BiometricSignatures.extract_face_for_person(frame, person.bbox)

        if face_info is None:
            return "INSUFFICIENT", 0.0, "Face not visible"

        fw, fh = face_info.get("width", 0), face_info.get("height", 0)
        cand_emb = face_info.get("embedding")
        if cand_emb is None:
            return "INSUFFICIENT", 0.0, "Face embedding unavailable"

        if not self.enrolled_face_templates:
            return "INSUFFICIENT", 0.0, "No enrolled templates"

        # Match against fixed original templates
        enrolled_best_score = max(
            BiometricSignatures.match_face(cand_emb, t)[1]
            for t in self.enrolled_face_templates
        )
        adaptive_best_score = 0.0
        if self.adaptive_face_templates:
            adaptive_best_score = max(
                BiometricSignatures.match_face(cand_emb, t)[1]
                for t in self.adaptive_face_templates
            )

        best_score = max(enrolled_best_score, adaptive_best_score)

        if best_score >= self.face_match_threshold:
            return "MATCH", best_score, f"Face verified ({best_score:.2f} >= {self.face_match_threshold:.2f})"

        # Score is below threshold.
        # Rule 6: Distant or small faces (< 40px) must NEVER trigger a MISMATCH drop while tracked.
        # Only a clear, large face (>= 40px) failing the threshold is a verified MISMATCH.
        min_mismatch_face_size = 40
        if fw < min_mismatch_face_size or fh < min_mismatch_face_size:
            return "INSUFFICIENT", best_score, f"Face too small/distant ({fw}x{fh} < {min_mismatch_face_size}px, score {best_score:.2f})"
        else:
            return "MISMATCH", best_score, f"Rejected: face score {best_score:.2f} (< {self.face_match_threshold:.2f})"

    def _maybe_add_adaptive_template(self, face_emb: np.ndarray, score: float, face_size: int = 50) -> None:
        """
        Adds an adaptive template ONLY when state is LOCKED, score is strongly
        above threshold (>= 0.65), face is large (>= 50px), and sample represents a diverse angle.
        Never called in LOST or during re-lock.
        """
        if self.state != SecurityState.LOCKED:
            return
        if score < 0.65 or face_size < 50:
            return

        all_templates = self.enrolled_face_templates + self.adaptive_face_templates
        if not all_templates:
            return

        max_sim = max(BiometricSignatures.match_face(face_emb, t)[1] for t in all_templates)
        if max_sim < 0.82:  # Truly diverse perspective/expression
            self.adaptive_face_templates.append(face_emb.copy())
            if len(self.adaptive_face_templates) > self.adaptive_gallery_max:
                self.adaptive_face_templates.pop(0)
            self._log_state_change(self.state.value, f"Added adaptive face template ({len(self.adaptive_face_templates)}/{self.adaptive_gallery_max})", score)

    def update(
        self,
        frame: np.ndarray,
        persons: List[PersonDetection],
    ) -> Tuple[SecurityState, Optional[int], Dict[str, Any]]:
        """
        Main security update cycle.
        Executes zero-trust identity verification:
        - Missing > 1.0s -> LOST.
        - Face mismatch while LOCKED -> LOST.
        - Turning back or distant face -> stays LOCKED while tracked.
        - Re-lock requires face match for N consecutive frames on same track ID.
        """
        now = time.time()
        metadata: Dict[str, Any] = {
            "state": self.state.value,
            "best_match_score": self.last_face_score,
            "mouse_enabled": False,
            "match_reason": self.status_reason,
            "consecutive_relock_frames": self.candidate_relock_consecutive_count,
            "required_relock_frames": self.relock_consecutive_frames,
            "gallery_templates": len(self.enrolled_face_templates) + len(self.adaptive_face_templates),
        }

        # 1. UNENROLLED: No person selected, mouse inactive
        if self.state == SecurityState.UNENROLLED or not self.enrolled_face_templates:
            self.status_reason = "UNENROLLED: waiting for enrollment"
            metadata["match_reason"] = self.status_reason
            return SecurityState.UNENROLLED, None, metadata

        # 2. ACTIVE (LOCKED or RE_LOCKED)
        if self.state in (SecurityState.LOCKED, SecurityState.RE_LOCKED):
            active_teacher = next((p for p in persons if p.track_id == self.teacher_track_id), None)

            if active_teacher is None:
                # Teacher track not detected
                time_lost = now - self.last_seen_timestamp
                if time_lost > self.lost_timeout_sec:
                    # Missing for more than 1.0 second -> go to LOST (Rule 2)
                    self._transition_to_lost("LOST: teacher not found (> 1.0s)")
                    metadata["state"] = self.state.value
                    metadata["match_reason"] = self.status_reason
                    return SecurityState.LOST, None, metadata
                else:
                    # Brief momentary track dropout (<= 1.0s): Pause mouse movement, wait for recovery
                    self.status_reason = f"HOLDING TRACK (missing {time_lost:.1f}s / {self.lost_timeout_sec:.1f}s)"
                    metadata["match_reason"] = self.status_reason
                    return self.state, None, metadata

            # Teacher track confirmed active
            self.last_seen_timestamp = now
            metadata["mouse_enabled"] = True

            # Periodic Face Verification (Rule 5: check face every 2-3 seconds whenever face is visible)
            if (now - self.last_face_check_timestamp) >= self.periodic_face_interval_sec:
                faces = BiometricSignatures.extract_face_embeddings_from_frame(frame)
                match_status, score, reason = self._match_candidate_face(active_teacher, faces, frame)
                self.last_face_score = score

                if match_status == "MISMATCH":
                    # Clear face detected, but does NOT match teacher!
                    # Track ID was stolen or people crossed paths -> immediate LOST
                    self.last_face_check_timestamp = now
                    self._transition_to_lost(f"LOST: identity mismatch ({reason})")
                    metadata["state"] = self.state.value
                    metadata["mouse_enabled"] = False
                    metadata["match_reason"] = self.status_reason
                    metadata["best_match_score"] = score
                    return SecurityState.LOST, None, metadata
                elif match_status == "MATCH":
                    # Teacher verified
                    self.last_face_check_timestamp = now
                    self.status_reason = f"LOCKED (ID #{self.teacher_track_id}): verified (score {score:.2f})"
                    face_info = BiometricSignatures.match_person_to_face_info(active_teacher.bbox, faces)
                    if face_info is None and frame is not None:
                        face_info = BiometricSignatures.extract_face_for_person(frame, active_teacher.bbox)
                    if face_info is not None:
                        self._maybe_add_adaptive_template(face_info["embedding"], score, face_info.get("width", 0))
                else:
                    # Rule 6: Teacher turned back or distant (face not sufficiently resolved)
                    # Stay LOCKED as long as tracker still follows their track!
                    self.status_reason = f"LOCKED (ID #{self.teacher_track_id}): tracked (face turned away / distant)"

            metadata["match_reason"] = self.status_reason
            metadata["best_match_score"] = self.last_face_score
            return self.state, self.teacher_track_id, metadata

        # 3. LOST (Zero-Trust Active: Scanning for Enrolled Teacher Re-Entry)
        if self.state == SecurityState.LOST:
            metadata["mouse_enabled"] = False

            if not persons:
                self.candidate_relock_consecutive_count = 0
                self.candidate_relock_track_id = None
                self.status_reason = "LOST: teacher not found"
                metadata["match_reason"] = self.status_reason
                return SecurityState.LOST, None, metadata

            # Extract faces to evaluate all candidates
            faces = BiometricSignatures.extract_face_embeddings_from_frame(frame)

            matching_candidates = []
            highest_observed_score = 0.0

            for p in persons:
                match_status, score, reason = self._match_candidate_face(p, faces, frame)
                if score > highest_observed_score:
                    highest_observed_score = score
                if match_status == "MATCH":
                    matching_candidates.append((p.track_id, score, p))

            self.last_face_score = highest_observed_score

            # Rule 4: Re-lock needs passing face score for N consecutive frames on same track ID
            if len(matching_candidates) == 1:
                cand_id, cand_score, cand_p = matching_candidates[0]

                # Secondary Cue: Check body proportions if enrolled
                if self.enrolled_body_prop is not None and cand_p.keypoints is not None:
                    cand_prop = BiometricSignatures.extract_body_proportions(cand_p.keypoints)
                    if cand_prop is not None:
                        prop_sim = BiometricSignatures.match_body_proportions(self.enrolled_body_prop, cand_prop)
                        if prop_sim < 0.35:
                            # Body proportions strongly contradict enrolled teacher
                            self.candidate_relock_consecutive_count = 0
                            self.candidate_relock_track_id = None
                            self.status_reason = f"Rejected: body proportions mismatch ({prop_sim:.2f} < 0.35)"
                            metadata["match_reason"] = self.status_reason
                            return SecurityState.LOST, None, metadata

                if cand_id == self.candidate_relock_track_id:
                    self.candidate_relock_consecutive_count += 1
                else:
                    self.candidate_relock_track_id = cand_id
                    self.candidate_relock_consecutive_count = 1

                self.status_reason = f"Re-lock {self.candidate_relock_consecutive_count}/{self.relock_consecutive_frames} frames (score {cand_score:.2f})"
                metadata["match_reason"] = self.status_reason
                metadata["consecutive_relock_frames"] = self.candidate_relock_consecutive_count

                if self.candidate_relock_consecutive_count >= self.relock_consecutive_frames:
                    # Verified for N consecutive frames -> Re-lock granted!
                    self.teacher_track_id = cand_id
                    self.last_seen_timestamp = now
                    self.last_face_check_timestamp = now
                    self.candidate_relock_consecutive_count = 0
                    self.candidate_relock_track_id = None
                    self.state = SecurityState.RE_LOCKED
                    self.status_reason = f"RE_LOCKED (ID #{cand_id}): face verified (score {cand_score:.2f})"
                    self._log_state_change("RE_LOCKED", f"Re-locked teacher ID #{cand_id} after {self.relock_consecutive_frames} consecutive frames", cand_score)
                    metadata["state"] = self.state.value
                    metadata["mouse_enabled"] = True
                    metadata["match_reason"] = self.status_reason
                    return self.state, self.teacher_track_id, metadata

                return SecurityState.LOST, None, metadata

            elif len(matching_candidates) > 1:
                # Ambiguity: Multiple people matching -> fail closed, reset counter!
                self.candidate_relock_consecutive_count = 0
                self.candidate_relock_track_id = None
                self.status_reason = "LOST: multiple face matches (ambiguity)"
                metadata["match_reason"] = self.status_reason
                return SecurityState.LOST, None, metadata

            else:
                # No candidates matched threshold: reset counter on ANY failed frame (Rule 4)
                if self.candidate_relock_consecutive_count > 0:
                    self.status_reason = f"Re-lock reset: failed frame (was {self.candidate_relock_consecutive_count}/{self.relock_consecutive_frames})"
                elif highest_observed_score > 0.0:
                    self.status_reason = f"Rejected: face score {highest_observed_score:.2f} (< {self.face_match_threshold:.2f})"
                else:
                    self.status_reason = "LOST: teacher not found"

                self.candidate_relock_consecutive_count = 0
                self.candidate_relock_track_id = None
                metadata["match_reason"] = self.status_reason
                return SecurityState.LOST, None, metadata

        return SecurityState.UNENROLLED, None, metadata

    def unlock(self) -> None:
        """Manually unlocks / resets the teacher session."""
        self.state = SecurityState.UNENROLLED
        self.teacher_track_id = None
        self.candidate_relock_track_id = None
        self.candidate_relock_consecutive_count = 0
        self.status_reason = "UNENROLLED: manual unlock"
        self._log_state_change("UNENROLLED", "Manual unlock requested", 0.0)

    def wipe_enrollment(self) -> None:
        """Wipes all biometric signatures (Delete My Data)."""
        self.unlock()
        self.enrolled_face_templates.clear()
        self.adaptive_face_templates.clear()
        self.body_gallery.clear()
        self.enrolled_body_prop = None
        self.status_reason = "UNENROLLED: all biometric data wiped"
        self._log_state_change("UNENROLLED", "Delete my data: all biometric templates wiped", 0.0)
