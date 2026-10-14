# ClassroomGestureMouse 🎓🖱️
> **Intelligent Touch-Free Vision Mouse & Presentation Controller for Classroom Smart Boards & PCs**

[![Release](https://img.shields.io/github/v/release/mbaqir18786/HandGesture?color=00FF66&label=Latest%20Version)](https://github.com/mbaqir18786/HandGesture/releases/latest)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%20%2F%2011-blue?logo=windows)](https://github.com/mbaqir18786/HandGesture)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-yellow?logo=python)](https://github.com/mbaqir18786/HandGesture)
[![AI Engine](https://img.shields.io/badge/AI-YOLOv8%20%2B%20MediaPipe%20%2B%20SFace-blueviolet)](https://github.com/mbaqir18786/HandGesture)

---

## ⚡ Quick Download & Run (No Python Required!)

You do **NOT** need Python, Git, or any coding libraries to run this on your Windows PC or Classroom Smart Board.

### 📥 Step 1: Download the Standalone App
Click the link below to download the direct `.exe` file:

👉 **[Download ClassroomGestureMouse.exe (Latest Release)](https://github.com/mbaqir18786/HandGesture/releases/latest/download/ClassroomGestureMouse.exe)**

*(Alternatively, go to the [Releases Page](https://github.com/mbaqir18786/HandGesture/releases/latest) and download `ClassroomGestureMouse.exe` under Assets).*

---

### 🚀 Step 2: Run the App
1. Move `ClassroomGestureMouse.exe` to your **Desktop** or any folder (or copy it to other PCs via USB Pen Drive).
2. **Double-click `ClassroomGestureMouse.exe`** to launch.
   > **Note:** If Windows Defender SmartScreen shows *"Windows protected your PC"*, click **More info** ➔ **Run anyway**.

---

### 🎯 Step 3: Teacher Enrollment & Gesture Control
1. **Teacher Enrollment:**
   - Stand in front of your camera.
   - Click **"Lock / Enroll Teacher"** on the dashboard.
   - Enter the Default PIN: **`1234`** (configurable in settings).
   - Look at the camera for ~3 seconds while the system scans your facial biometric signature and clothing colors.
2. **Gesture Controls:**
   - **Near Mode (< 3.5 meters):**
     - **Move Cursor:** Move your hand naturally (tracked via Index Knuckle).
     - **Left Click:** Pinch your Thumb and Index finger together briefly.
     - **Double Click:** Double pinch within 0.4 seconds.
   - **Far Mode (3.5 – 10 meters):**
     - **Move Cursor:** Raise your right arm/wrist and move it across the room.
     - **Dwell Click:** Hold your wrist still over an icon for 1.0 second. A circular countdown ring appears and automatically clicks!
3. **Classroom Privacy:**
   - Click **"👁️ Hide Feed (Classroom Mode)"** during lectures to blank out the webcam window while gesture control stays 100% active.

---

## 🎙️ Multimodal Voice Commands & Smart Dictation

Control apps, search the web, manage browser/Notepad tabs, and dictate text completely hands-free while your hands guide the pointer:

| Category | Voice Commands | Action Performed |
| :--- | :--- | :--- |
| **App Launching & Exit** | `"Open Notepad"`, `"Close Notepad"` | Launches or closes Notepad window cleanly |
| | `"Open Chrome"`, `"Close Chrome"` | Launches or closes Google Chrome |
| | `"Open Calculator"`, `"Close Calculator"` | Opens or closes Windows Calculator |
| | `"Close App"` / `"Close Window"` | Closes active application (`Alt+F4`) |
| **Tab Management** | `"New Tab"` / `"Open New Tab"` | Opens a new tab in Notepad / Chrome (`Ctrl+T`) |
| *(Notepad & Chrome)* | `"Close Tab"` | Closes current tab (`Ctrl+W`) |
| | `"Next Tab"` / `"Previous Tab"` | Cycles between active tabs (`Ctrl+Tab` / `Ctrl+Shift+Tab`) |
| | `"New Window"` / `"Save File"` | Opens new document (`Ctrl+N`) / Saves file (`Ctrl+S`) |
| **Voice Dictation** | `"Write <text>"` / `"Type <text>"` | Dictates text directly into Notepad, Word, or search boxes |
| | `"New line"` / `"Press enter"` | Presses Enter key |
| | `"Backspace"` / `"Delete"` | Erases last character / word |
| | `"Select all"`, `"Copy"`, `"Paste"` | Standard clipboard shortcuts |
| **Web Search** | `"Search <query>"` (e.g. `"Search AI"`) | Opens Google search in Chrome/browser for the query |
| **Mouse Actions** | `"Click"` / `"Left Click"` | Fires left mouse click at current pointer position |
| | `"Right Click"` / `"Double Click"` | Opens context menu / double clicks |
| | `"Scroll Down"` / `"Scroll Up"` | Scrolls active document or page |
| | `"Zoom In"` / `"Zoom Out"` | Zooms web page or slide (`Ctrl +/-`) |
| **Gesture Mouse Pause** | Click `⏸️` button on Floating Pill | Stops hand gesture cursor movement (Voice-Only Mode) |
| *(Voice-Only Mode)* | Click `⏸️ Pause Mouse` in Control Hub | Freely talk & dictate without hand gestures moving cursor |
| | Press **`F8`** or **`Ctrl+Shift+M`** | Instant keyboard shortcut to toggle pause/resume |
| | `"Pause mouse"` / `"Stop mouse"` | Voice command to pause gesture cursor |
| | `"Resume mouse"` / `"Start mouse"` | Voice command to resume gesture tracking |
| **Presentation Mode** | `"Next slide"` / `"Previous slide"` | Advances or goes back in PowerPoint/PDFs |
| | `"Fullscreen"` / `"Exit presentation"` | Starts slideshow (`F5`) or exits (`Esc`) |

---

## 🔄 Automatic In-App Updates

Whenever a new version or update is released:
1. Open the app and click the **"🔄 Updates"** button in the top menu bar.
2. If an update is available, a prompt will appear:
   ```text
   🚀 New Update Available: v1.0.1
   [⚡ Update Now]    [Later]
   ```
3. Click **"⚡ Update Now"**. The app will download the update with a real-time progress bar, replace itself, and restart automatically. **No pen drives needed!**

---

## 🛡️ Zero-Trust Security Policy
- **Anti-Impostor Protection:** The system locks exclusively onto the enrolled teacher using **128-D SFace Deep Metric Face Embeddings** + **HSV Clothing Signatures**.
- **Student Rejection:** If students or other individuals move in front of the board, their gestures are completely ignored.
- **Auto-Freeze:** If the teacher leaves the camera view, mouse control immediately freezes to prevent accidental clicks. Control restores instantly the second the teacher steps back into frame.

---

## 🛠️ Running from Source (For Developers)

If you want to run or modify the Python source code directly:

### 1. Clone the repository
```bash
git clone https://github.com/mbaqir18786/HandGesture.git
cd HandGesture
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
*(Or install core packages: `pip install opencv-python numpy mediapipe ultralytics pyttsx3 PyQt6 pyautogui`)*

### 3. Run the application
```bash
python app.py
```

### 4. Build Standalone Application
- **🚀 Fast-Launch Mode (Recommended — Opens in 1–2 seconds)**:
  ```bash
  python build_app.py --onedir
  ```
  Generates `dist/ClassroomGestureMouse/`. Double-click `ClassroomGestureMouse.exe` inside to run instantly with zero decompression lag! Copy the folder to any pen drive.

- **📦 Single Portable `.exe` Mode**:
  ```bash
  python build_app.py --onefile
  ```
  Generates a single self-extracting file at `dist/ClassroomGestureMouse.exe`.

---

## 📄 License
This project is open-source and free for educational and personal use.
