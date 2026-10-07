"""
ClassroomGestureMouse Core Engine Modules
"""
from core.config_manager import ConfigManager
from core.camera_stream import CameraStream
from core.distance_estimator import DistanceEstimator
from core.tracker import PersonTracker, PersonDetection
from core.signatures import BiometricSignatures
from core.security_engine import SecurityEngine, SecurityState
from core.one_euro_filter import OneEuroFilter, OneEuroFilter2D
from core.mouse_controller import DualModeMouseController

__all__ = [
    "ConfigManager",
    "CameraStream",
    "DistanceEstimator",
    "PersonTracker",
    "PersonDetection",
    "BiometricSignatures",
    "SecurityEngine",
    "SecurityState",
    "OneEuroFilter",
    "OneEuroFilter2D",
    "DualModeMouseController",
]



