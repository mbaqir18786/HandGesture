import threading
import queue
from typing import Optional, Dict, Any


class VoiceFeedback:
    """
    Threaded, non-blocking Text-to-Speech audio feedback engine.
    Ensures voice announcements never stall or block computer vision loops.
    """

    def __init__(
        self,
        enabled: bool = True,
        speech_rate: int = 180,
        volume: float = 0.9,
        alerts: Optional[Dict[str, str]] = None,
    ):
        self.enabled = enabled
        self.speech_rate = speech_rate
        self.volume = volume
        self.alerts = alerts or {
            "locked": "Teacher locked. Gesture control active.",
            "lost": "Teacher lost. Mouse control frozen.",
            "relocked": "Welcome back teacher. Control restored.",
            "unauthorized": "Access denied.",
        }

        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_alert_type = ""

        if self.enabled:
            self._start_worker()

    def _start_worker(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="VoiceFeedbackWorker")
        self._thread.start()

    def _worker_loop(self) -> None:
        """Background thread handling text-to-speech without blocking vision loops."""
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.speech_rate)
            engine.setProperty("volume", self.volume)
        except Exception as e:
            print(f"[VoiceFeedback] Audio engine initialization error: {e}")
            return

        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
                if text is None:
                    break

                if engine is not None and self.enabled:
                    engine.say(text)
                    engine.runAndWait()

                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VoiceFeedback] Speech error: {e}")

    def speak(self, text: str) -> None:
        """Queues custom speech text non-blockingly."""
        if self.enabled and self._running:
            self._queue.put(text)

    def trigger_alert(self, alert_key: str, force: bool = False) -> None:
        """Triggers predefined alert if state changed."""
        if not self.enabled:
            return

        if alert_key == self._last_alert_type and not force:
            return  # Prevent repetitive speech spam

        text = self.alerts.get(alert_key)
        if text:
            self._last_alert_type = alert_key
            self.speak(text)

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
