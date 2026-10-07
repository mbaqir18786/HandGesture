import os
import sys
import time
import subprocess
import threading
import tempfile
import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, Tuple, Callable


def parse_semver(ver_str: str) -> Tuple[int, ...]:
    """Converts version strings like '1.2.0' to comparable tuple (1, 2, 0)."""
    try:
        clean = ver_str.strip().lstrip("vV")
        parts = [int(p) for p in clean.split(".") if p.isdigit()]
        return tuple(parts) if parts else (0, 0, 0)
    except Exception:
        return (0, 0, 0)


class UpdateManager:
    """
    Asynchronous, non-blocking In-App Update Checker & Self-Updater.
    - Checks remote version endpoint (GitHub raw version.json or Releases API).
    - Downloads updated binary with real-time percentage progress.
    - Spawns self-replacing restart script on Windows without requiring manual re-extraction.
    """

    def __init__(
        self,
        current_version: str = "1.0.0",
        update_url: str = "",
        timeout_sec: float = 5.0,
    ):
        self.current_version = current_version
        self.update_url = update_url
        self.timeout_sec = timeout_sec

    def check_for_updates_sync(self) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Synchronous update check.
        Returns: (is_update_available, update_info_dict, message)
        """
        if not self.update_url:
            return False, None, "No update URL configured."

        try:
            req = urllib.request.Request(
                self.update_url,
                headers={"User-Agent": "ClassroomGestureMouse-Updater/1.0"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                if response.status != 200:
                    return False, None, f"HTTP Error {response.status}"

                data = json.loads(response.read().decode("utf-8"))

            latest_ver_str = data.get("latest_version", "0.0.0")
            current_tuple = parse_semver(self.current_version)
            latest_tuple = parse_semver(latest_ver_str)

            if latest_tuple > current_tuple:
                return True, data, f"New version {latest_ver_str} available!"
            else:
                return False, data, "You are using the latest version."

        except urllib.error.URLError as e:
            return False, None, f"Network error: {e.reason}"
        except json.JSONDecodeError:
            return False, None, "Invalid version metadata from server."
        except Exception as e:
            return False, None, f"Check failed: {e}"

    def check_for_updates_async(self, callback: Callable[[bool, Optional[Dict[str, Any]], str], None]) -> None:
        """
        Non-blocking background thread update check.
        Invokes callback(is_available, data, message) on completion.
        """
        def _worker():
            is_avail, data, msg = self.check_for_updates_sync()
            if callback:
                try:
                    callback(is_avail, data, msg)
                except Exception as e:
                    print(f"[UpdateManager] Callback error: {e}")

        thread = threading.Thread(target=_worker, daemon=True, name="UpdateCheckerWorker")
        thread.start()

    def download_file(
        self,
        url: str,
        dest_path: str,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> bool:
        """
        Downloads a file with streaming progress updates.
        progress_callback(bytes_downloaded, total_bytes, percentage_float)
        """
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ClassroomGestureMouse-Updater/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30.0) as response:
                total_bytes = int(response.headers.get("Content-Length", 0))
                bytes_downloaded = 0
                chunk_size = 1024 * 64  # 64 KB chunks

                with open(dest_path, "wb") as out_file:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        bytes_downloaded += len(chunk)
                        if progress_callback and total_bytes > 0:
                            pct = (bytes_downloaded / total_bytes) * 100.0
                            progress_callback(bytes_downloaded, total_bytes, pct)

            if progress_callback:
                progress_callback(bytes_downloaded, bytes_downloaded, 100.0)
            return True
        except Exception as e:
            print(f"[UpdateManager] Download failed: {e}")
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except Exception:
                    pass
            return False

    def download_and_apply_update_async(
        self,
        download_url: str,
        progress_callback: Callable[[float], None],
        completion_callback: Callable[[bool, str], None],
    ) -> None:
        """
        Downloads the latest executable and triggers self-update on background thread.
        """
        def _worker():
            try:
                # Target path in temp directory
                temp_dir = tempfile.gettempdir()
                temp_exe = os.path.join(temp_dir, f"ClassroomGestureMouse_update_{int(time.time())}.exe")

                def _on_chunk(downloaded, total, pct):
                    if progress_callback:
                        progress_callback(pct)

                success = self.download_file(download_url, temp_exe, _on_chunk)
                if not success:
                    if completion_callback:
                        completion_callback(False, "Failed to download update file.")
                    return

                # If running frozen executable, prepare restart script
                if completion_callback:
                    completion_callback(True, temp_exe)

            except Exception as e:
                if completion_callback:
                    completion_callback(False, str(e))

        thread = threading.Thread(target=_worker, daemon=True, name="UpdateDownloaderWorker")
        thread.start()

    @staticmethod
    def apply_update_and_restart(new_exe_path: str) -> bool:
        """
        Spawns a self-replacing Windows batch updater and terminates the current process.
        """
        if not getattr(sys, "frozen", False):
            # Development mode: Cannot self-replace script
            print("[UpdateManager] In dev mode, downloaded new exe to:", new_exe_path)
            return True

        current_exe = os.path.abspath(sys.executable)
        updater_bat = os.path.join(tempfile.gettempdir(), f"update_runner_{os.getpid()}.bat")

        bat_script = f"""@echo off
timeout /t 2 /nobreak > nul
:retry
move /y "{new_exe_path}" "{current_exe}" > nul 2>&1
if errorlevel 1 (
    timeout /t 1 /nobreak > nul
    goto retry
)
start "" "{current_exe}"
del "%~f0"
"""
        with open(updater_bat, "w", encoding="utf-8") as f:
            f.write(bat_script)

        try:
            # Spawn detached batch runner
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            DETACHED_PROCESS = 0x00000008
            subprocess.Popen(
                ["cmd.exe", "/c", updater_bat],
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                close_fds=True,
            )
            # Exit current application
            time.sleep(0.3)
            os._exit(0)
        except Exception as e:
            print(f"[UpdateManager] Failed to launch update runner: {e}")
            return False
