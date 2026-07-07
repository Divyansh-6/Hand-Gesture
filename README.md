# 🖐️ Gesture-Controlled Desktop

Control your mouse and keyboard using only your webcam and hand gestures. Built with **MediaPipe Hands**, **OpenCV**, and **PyAutoGUI**.

---

## ✨ Features

- Real-time hand tracking at ~30 FPS
- 7 gesture vocabulary (pointer, pinch, fist, scroll, right-click, double-click, pause)
- Jitter smoothing via rolling-average filter
- Hold-frame confirmation — no accidental misfires
- Per-action cooldown timer
- Live debug HUD overlay
- Cross-platform: Windows, macOS, Linux

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Linux note:** You may also need:
> ```bash
> sudo apt-get install python3-tk python3-dev scrot
> ```

> **macOS note:** Grant Terminal (or your IDE) permission under
> System Preferences → Security & Privacy → Accessibility and Screen Recording.

### 2. Run

```bash
python main.py
```

### Options

```
python main.py --help

  --camera INT        Camera device index (default: 0)
  --debug             Start with debug HUD enabled
  --smoothing INT     Rolling-average window size (default: 5)
  --confidence FLOAT  MediaPipe detection confidence (default: 0.7)
  --cooldown FLOAT    Seconds between discrete actions (default: 0.4)
```

---

## 🤌 Gesture Reference

| Gesture | Hand shape | Action |
|---------|-----------|--------|
| ☝ **Pointer** | Index finger up, others curled | Move mouse |
| 🤌 **Pinch** | Thumb + index tips touching | Left click |
| 🤞 **Scissors** | Index + middle tips touching | Double click |
| 👍 **Thumbs up** | Thumb up, fingers curled | Right click |
| ✊ **Fist** | All fingers curled | Click and drag |
| ✌ **Two fingers** | Index + middle up, move hand up/down | Scroll |
| 🖐 **Open palm** | All 5 fingers extended | Pause input |

---

## ⌨️ Keyboard Shortcuts (while running)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause / unpause gesture control |
| `d` | Toggle debug HUD overlay |
| `h` | Print gesture help to terminal |

**Emergency stop:** Move mouse to **top-left corner** of screen — PyAutoGUI FAILSAFE will terminate the script immediately.

---

## 🏗️ Project Structure

```
gesture_desktop/
├── main.py                 # Entry point, webcam loop
├── hand_tracker.py         # MediaPipe wrapper → 21 landmark points
├── gesture_classifier.py   # Landmark geometry → gesture label
├── action_mapper.py        # Gesture label → PyAutoGUI OS action
├── utils.py                # Smoothing, debug overlay, FPS limiter
├── requirements.txt
└── README.md
```

### How it flows

```
Webcam frame
    → HandTracker         (MediaPipe: 21 (x,y,z) landmarks)
    → SmoothingFilter     (rolling average over 5 frames)
    → GestureClassifier   (geometry rules + hold-frame confirm)
    → ActionMapper        (PyAutoGUI mouse/keyboard commands)
```

---

## 🔧 Tuning Tips

### Gesture sensitivity

Edit the constants at the top of `gesture_classifier.py`:

```python
PINCH_THRESHOLD        = 0.055   # Lower = harder to trigger pinch
DOUBLE_CLICK_THRESHOLD = 0.055   # Lower = harder to trigger double-click
```

### Scroll speed

Edit `action_mapper.py`:

```python
SCROLL_SENSITIVITY = 60   # Higher = faster scroll
```

### Jitter / smoothness tradeoff

```bash
python main.py --smoothing 8   # Smoother but slightly more lag
python main.py --smoothing 2   # More responsive but jittery
```

### Reduce accidental clicks

```bash
python main.py --cooldown 0.8   # 800 ms between clicks
```

### Confidence threshold

```bash
python main.py --confidence 0.8   # Stricter detection (better lighting needed)
python main.py --confidence 0.5   # More permissive (lower light ok, more noise)
```

---

## 💡 Extending the Project

### Add a new gesture

1. Add a detection rule in `gesture_classifier.py` → `_classify_once()`
2. Add an action in `action_mapper.py` → `execute()`

Example — "rock on" (index + pinky up) → mute/unmute:

```python
# gesture_classifier.py
if fi[0] == 1 and fi[1] == 0 and fi[2] == 0 and fi[3] == 1:
    return "rock_on"

# action_mapper.py
elif gesture == "rock_on":
    if self._cooled_down("rock_on"):
        pyautogui.hotkey("volumemute")
        self._stamp("rock_on")
```

### Train a custom ML classifier

Replace the rule-based `_classify_once()` with a trained model:

```python
import pickle
import numpy as np

model = pickle.load(open("gesture_model.pkl", "rb"))

def _classify_once(lm):
    flat = np.array(lm).flatten().reshape(1, -1)
    return model.predict(flat)[0]
```

Collect training data by logging landmarks per gesture label into a CSV,
then train a simple sklearn `RandomForestClassifier` or small MLP.

### Two-hand mode

Change `HandTracker(max_hands=2)` and update `get_landmarks()` to return
a list of two landmark sets. Map one hand to mouse movement and the
other to discrete actions.

### Volume control (macOS)

```python
import subprocess
# Map thumb-index spread to volume
spread = dist(lm[4], lm[8])
volume = int(spread * 200)  # scale to 0-100
subprocess.run(["osascript", "-e", f"set volume output volume {volume}"])
```

### Window-specific profiles

```python
import subprocess

def get_active_app():
    # macOS
    result = subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to get name of first application process whose frontmost is true'],
        capture_output=True, text=True
    )
    return result.stdout.strip()
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera not found | Try `--camera 1` or `--camera 2` |
| Mouse jumps around | Increase `--smoothing` (try 8–10) |
| Clicks firing accidentally | Increase `--cooldown` (try 0.6–1.0) |
| Gestures not recognised | Improve lighting; plain background helps |
| High CPU usage | Reduce resolution: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` |
| macOS permission error | Allow accessibility + screen recording in System Preferences |
| Linux: `_tkinter` error | `sudo apt-get install python3-tk` |
| Script won't stop | Move mouse to **top-left corner** |

---

## 📐 How MediaPipe Landmark Indices Work

```
         8   12  16  20
         |   |   |   |
         7   11  15  19
         |   |   |   |
         6   10  14  18
          \   |   |   /
    4      5   9  13  17
    |       \  |  |  /
    3        \ | / \/
    |          0 (wrist)
    2
    |
    1
```

- Index 0 = Wrist
- Indices 1–4 = Thumb (base → tip)
- Indices 5–8 = Index finger (base → tip)
- And so on for middle (9–12), ring (13–16), pinky (17–20)

A finger is **extended** when its tip (e.g. index=8) has a **lower y value**
than its PIP joint (e.g. index PIP=6) — because y=0 is the top of the frame.

---

## 📄 License

MIT — free to use, modify, and distribute.
