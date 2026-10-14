import os
import sys
import time
import threading
import subprocess
import urllib.parse
from typing import Optional, Callable, Dict, Any
import pyautogui

# Try importing speech_recognition
try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False


class VoiceControlEngine:
    """
    Multimodal Voice Control Engine for Touch-Free Classroom Interaction:
    - Mouse Actions: "left click", "right click", "double click", "scroll down", "scroll up", "zoom in", "zoom out"
    - Cursor Stability: "stop here" / "freeze", "resume" / "continue"
    - App Launching: "open chrome", "open notepad", "open calculator", "open edge"
    - Web Search: "search <query>" (searches Google in active browser or default browser)
    - Real-Time Dictation: "type <text>" / "write <text>" (types directly into active app e.g. Notepad)
    - Text Editing: "new line", "enter", "backspace", "delete", "select all", "copy", "paste"
    - Presentation Mode: "next slide", "previous slide", "fullscreen", "exit presentation"
    """

    def __init__(
        self,
        on_command_callback: Optional[Callable[[str, str], None]] = None,
        voice_feedback: Optional[Any] = None,
    ):
        self.on_command_callback = on_command_callback
        self.voice_feedback = voice_feedback
        self.is_running = False
        self.is_frozen = False
        self._thread: Optional[threading.Thread] = None
        self._recognizer: Optional[sr.Recognizer] = None
        self._microphone: Optional[sr.Microphone] = None
        self.mode = "DESKTOP"  # "DESKTOP" or "PRESENTATION"

        if HAS_SR:
            try:
                self._recognizer = sr.Recognizer()
                # Static energy threshold with high noise-floor prevents picking up breathing / fan noise
                self._recognizer.energy_threshold = 650
                self._recognizer.dynamic_energy_threshold = False
                self._recognizer.pause_threshold = 0.4
                self._recognizer.non_speaking_duration = 0.3
                self._microphone = sr.Microphone()
            except Exception as e:
                print(f"[VoiceControlEngine] Initialization error: {e}")

    def start(self) -> bool:
        """Starts background voice command listening thread."""
        if not HAS_SR or self._microphone is None:
            print("[VoiceControlEngine] Speech recognition not available.")
            return False

        if self.is_running:
            return True

        self.is_running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[VoiceControlEngine] Noise-gated voice listener active in background.")
        return True

    def stop(self) -> None:
        """Stops the voice command listener."""
        self.is_running = False

    def toggle_freeze(self) -> bool:
        """Toggles cursor freezing via voice."""
        self.is_frozen = not self.is_frozen
        return self.is_frozen

    def _listen_loop(self) -> None:
        """Continuous background listening loop with ambient noise filtering."""
        with self._microphone as source:
            try:
                # Calibrate ambient noise floor once for 0.6s
                self._recognizer.adjust_for_ambient_noise(source, duration=0.6)
                # Keep threshold safely above room noise
                self._recognizer.energy_threshold = max(self._recognizer.energy_threshold, 600)
            except Exception:
                self._recognizer.energy_threshold = 600

            while self.is_running:
                try:
                    # Snappy 2.5s phrase limit for instant command detection
                    audio = self._recognizer.listen(source, timeout=3.0, phrase_time_limit=2.5)
                    try:
                        text = self._recognizer.recognize_google(audio).strip()
                        if text:
                            self._dispatch_command(text)
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError:
                        pass
                except sr.WaitTimeoutError:
                    continue
                except Exception:
                    time.sleep(0.1)

    def _dispatch_command(self, phrase: str) -> None:
        """Parses recognized voice phrase using fuzzy intent matching and executes actions."""
        p = phrase.lower().strip().replace(".", "").replace(",", "").replace("!", "").replace("?", "")
        print(f"[VoiceControl] Heard: '{p}'")

        # -------------------------------------------------------------
        # 1. Dictation & Writing ("type <text>" / "write <text>")
        # -------------------------------------------------------------
        if p.startswith("write ") or p.startswith("type "):
            content = phrase.split(" ", 1)[1].strip()
            self._type_text(content)
            self._notify("DICTATION", f"Typed: '{content}'")
            return

        # -------------------------------------------------------------
        # 2. Web Search ("search <query>" / "google <query>")
        # -------------------------------------------------------------
        if p.startswith("search ") or p.startswith("google "):
            query = phrase.split(" ", 1)[1].strip()
            encoded_query = urllib.parse.quote_plus(query)
            subprocess.Popen(["cmd", "/c", "start", f"https://www.google.com/search?q={encoded_query}"], shell=True)
            self._notify("SEARCH", f"Searching for '{query}'")
            if self.voice_feedback:
                self.voice_feedback.speak_text(f"Searching {query}")
            return

        # -------------------------------------------------------------
        # 3. Application Closers ("close notepad", "close chrome", etc.)
        # -------------------------------------------------------------
        if ("close" in p or "exit" in p or "quit" in p) and "notepad" in p and "tab" not in p:
            subprocess.Popen(["taskkill", "/IM", "notepad.exe"], shell=True)
            self._notify("APP_CLOSE", "Closing Notepad")
            if self.voice_feedback:
                self.voice_feedback.speak_text("Closing Notepad")
            return

        if ("close" in p or "exit" in p or "quit" in p) and ("chrome" in p or "browser" in p) and "tab" not in p:
            subprocess.Popen(["taskkill", "/IM", "chrome.exe"], shell=True)
            self._notify("APP_CLOSE", "Closing Google Chrome")
            if self.voice_feedback:
                self.voice_feedback.speak_text("Closing Chrome")
            return

        if ("close" in p or "exit" in p) and ("calc" in p or "calculator" in p):
            subprocess.Popen(["taskkill", "/IM", "CalculatorApp.exe"], shell=True)
            self._notify("APP_CLOSE", "Closing Calculator")
            if self.voice_feedback:
                self.voice_feedback.speak_text("Closing Calculator")
            return

        if ("close" in p or "exit" in p) and ("window" in p or "app" in p) and "tab" not in p:
            pyautogui.hotkey("alt", "f4")
            self._notify("APP_CLOSE", "Closed Active Window")
            if self.voice_feedback:
                self.voice_feedback.speak_text("Window closed")
            return

        # -------------------------------------------------------------
        # 4. Application Launchers ("open notepad", "open chrome", etc.)
        # -------------------------------------------------------------
        if ("open" in p or "launch" in p) and "notepad" in p:
            subprocess.Popen(["cmd", "/c", "start", "notepad"], shell=True)
            self._notify("APP_LAUNCH", "Opening Notepad")
            if self.voice_feedback:
                self.voice_feedback.speak_text("Opening Notepad")
            return

        if ("open" in p or "launch" in p) and ("chrome" in p or "browser" in p):
            subprocess.Popen(["cmd", "/c", "start", "chrome"], shell=True)
            self._notify("APP_LAUNCH", "Opening Google Chrome")
            if self.voice_feedback:
                self.voice_feedback.speak_text("Opening Chrome")
            return

        if ("open" in p or "launch" in p) and ("calc" in p or "calculator" in p):
            subprocess.Popen(["cmd", "/c", "start", "calc"], shell=True)
            self._notify("APP_LAUNCH", "Opening Calculator")
            if self.voice_feedback:
                self.voice_feedback.speak_text("Opening Calculator")
            return

        if ("open" in p or "launch" in p) and "edge" in p:
            subprocess.Popen(["cmd", "/c", "start", "msedge"], shell=True)
            self._notify("APP_LAUNCH", "Opening Microsoft Edge")
            if self.voice_feedback:
                self.voice_feedback.speak_text("Opening Edge")
            return

        # -------------------------------------------------------------
        # 5. Tab & Window Management (Notepad Tabs, Chrome Tabs)
        # -------------------------------------------------------------
        if "new tab" in p or "open tab" in p or "another tab" in p:
            pyautogui.hotkey("ctrl", "t")
            self._notify("TAB", "Opened New Tab")
            if self.voice_feedback:
                self.voice_feedback.speak_text("New tab")
            return

        if "close tab" in p or "exit tab" in p:
            pyautogui.hotkey("ctrl", "w")
            self._notify("TAB", "Closed Tab")
            if self.voice_feedback:
                self.voice_feedback.speak_text("Tab closed")
            return

        if "next tab" in p or "switch tab" in p:
            pyautogui.hotkey("ctrl", "tab")
            self._notify("TAB", "Next Tab")
            return

        if "previous tab" in p or "prev tab" in p:
            pyautogui.hotkey("ctrl", "shift", "tab")
            self._notify("TAB", "Previous Tab")
            return

        if "save" in p or "save file" in p:
            pyautogui.hotkey("ctrl", "s")
            self._notify("EDIT", "Saved Document")
            if self.voice_feedback:
                self.voice_feedback.speak_text("Saved")
            return

        if "new file" in p or "new document" in p or "new window" in p:
            pyautogui.hotkey("ctrl", "n")
            self._notify("WINDOW", "Opened New Window")
            return

        # -------------------------------------------------------------
        # 6. Mouse Clicks & Freezing
        # -------------------------------------------------------------
        if "right click" in p or "context menu" in p:
            pyautogui.rightClick()
            self._notify("RIGHT_CLICK", "Right Click")
            if self.voice_feedback:
                self.voice_feedback.speak_text("Right click")
            return

        if "double click" in p or "open this" in p:
            pyautogui.doubleClick()
            self._notify("DOUBLE_CLICK", "Double Click")
            return

        words = p.split()
        if "click" in words or "tap" in words or p == "left click":
            pyautogui.click()
            self._notify("CLICK", "Left Click")
            return

        if "stop here" in p or "freeze" in p or "pause" in p or "stop mouse" in p or p == "stop" or "hold on" in p:
            self.is_frozen = True
            self._notify("FREEZE", "Gesture Mouse Paused")
            if self.voice_feedback:
                self.voice_feedback.speak_text("Mouse paused")
            return

        if "resume" in p or "continue" in p or "unfreeze" in p or "start mouse" in p or "resume mouse" in p or "track" in p:
            self.is_frozen = False
            self._notify("RESUME", "Gesture Mouse Resumed")
            if self.voice_feedback:
                self.voice_feedback.speak_text("Mouse resumed")
            return

        # -------------------------------------------------------------
        # 7. Scrolling & Zooming
        # -------------------------------------------------------------
        if "scroll down" in p or "page down" in p or ("scroll" in p and "down" in p):
            pyautogui.scroll(-350)
            self._notify("SCROLL", "Scrolled Down")
            return

        if "scroll up" in p or "page up" in p or ("scroll" in p and "up" in p):
            pyautogui.scroll(350)
            self._notify("SCROLL", "Scrolled Up")
            return

        if "zoom in" in p or "magnify" in p:
            pyautogui.hotkey("ctrl", "+")
            self._notify("ZOOM", "Zoomed In")
            return

        if "zoom out" in p or "shrink" in p:
            pyautogui.hotkey("ctrl", "-")
            self._notify("ZOOM", "Zoomed Out")
            return

        # -------------------------------------------------------------
        # 8. Presentation Commands (Classroom PPT Mode)
        # -------------------------------------------------------------
        if "next slide" in p or "next page" in p:
            pyautogui.press("right")
            self._notify("SLIDE", "Next Slide")
            return

        if "previous slide" in p or "previous page" in p or "go back" in p:
            pyautogui.press("left")
            self._notify("SLIDE", "Previous Slide")
            return

        if "start presentation" in p or "slideshow" in p or "fullscreen" in p:
            pyautogui.press("f5")
            self._notify("PRESENTATION", "Fullscreen Slideshow")
            return

        if "exit presentation" in p or "escape" in p or "exit fullscreen" in p:
            pyautogui.press("esc")
            self._notify("PRESENTATION", "Exited Presentation")
            return

        # -------------------------------------------------------------
        # 9. Editing Keys
        # -------------------------------------------------------------
        if "enter" in words or "new line" in p:
            pyautogui.press("enter")
            self._notify("KEY", "Enter Pressed")
            return

        if "backspace" in words or "delete" in words or "erase" in words:
            pyautogui.press("backspace")
            self._notify("KEY", "Backspace")
            return

        if "select all" in p:
            pyautogui.hotkey("ctrl", "a")
            self._notify("EDIT", "Selected All")
            return

        if "copy" in words or "copy this" in p:
            pyautogui.hotkey("ctrl", "c")
            self._notify("EDIT", "Copied")
            return

        if "paste" in words or "paste this" in p:
            pyautogui.hotkey("ctrl", "v")
            self._notify("EDIT", "Pasted")
            return

        # Reject any ambient chatter or random background noise
        print(f"[VoiceControl] Ignored ambient noise: '{p}'")

    def _type_text(self, text: str) -> None:
        """Types text accurately into active application via clipboard for Unicode safety."""
        try:
            import win32clipboard
            import win32con

            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(text + " ", win32con.CF_UNICODETEXT)
            win32clipboard.CloseClipboard()
            pyautogui.hotkey("ctrl", "v")
        except Exception:
            pyautogui.write(text + " ", interval=0.01)

    def _notify(self, cmd_type: str, message: str) -> None:
        """Dispatches notification callback to GUI and floating status pill."""
        if self.on_command_callback:
            try:
                self.on_command_callback(cmd_type, message)
            except Exception:
                pass
