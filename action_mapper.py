"""
action_mapper.py
================
Translates confirmed gesture labels into OS-level actions
using PyAutoGUI.

Gesture → Action mapping:
  pointer      → Move mouse (smooth, mirrored x-axis)
  pinch        → Left click  (with cooldown)
  thumbs_up    → Right click (with cooldown)
  double_click → Double click (with cooldown)
  fist         → Click-and-drag
  two_fingers  → Scroll (up/down based on vertical movement delta)
  open_palm    → No-op (pause mode — hand detected but ignored)
  none/unknown → No action

Design notes:
  - Mouse position is mapped from normalised webcam coords [0,1]
    to screen pixels, with x mirrored so movement feels natural.
  - Discrete actions (clicks) use a per-gesture cooldown timer to
    avoid repeat-firing from a held gesture.
  - Dragging uses pyautogui.mouseDown / mouseUp to maintain state
    across frames.
  - Scroll delta is scaled by a sensitivity constant.
"""

import pyautogui
import time
import os
import subprocess
from typing import Optional

try:
    import screen_brightness_control as sbc
except ImportError:
    sbc = None

# ── Landmark indices ──────────────────────────────────────────────────────────
INDEX_TIP = 8   # We track the index fingertip for mouse position


class ActionMapper:
    SCROLL_SENSITIVITY = 60     # Pixels of scroll per unit of hand movement
    DRAG_MOVE_DURATION = 0      # Instant drag (no easing)

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        cam_w: int,
        cam_h: int,
        cooldown: float = 0.4,
    ):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cam_w    = cam_w
        self.cam_h    = cam_h
        self.cooldown = cooldown

        # Per-gesture last-fired timestamps
        self._last: dict[str, float] = {}

        # Scroll state
        self._prev_scroll_y: Optional[float] = None

        # Drag state
        self._dragging = False

        # Last action description (for debug overlay)
        self.last_action = "—"

    # ── Public ────────────────────────────────────────────────────────────────

    def execute(self, gesture: str, landmarks: Optional[list]) -> None:
        if landmarks is None or gesture in ("none", "unknown", "open_palm"):
            self._reset_states(gesture)
            return

        tip = landmarks[INDEX_TIP]

        if gesture == "pointer":
            self._move_mouse(tip)

        elif gesture == "pinch":
            self._stop_drag()
            if self._cooled_down("pinch"):
                pyautogui.click()
                self.last_action = "Left click"
                self._stamp("pinch")

        elif gesture == "thumbs_up":
            self._stop_drag()
            if self._cooled_down("thumbs_up"):
                pyautogui.rightClick()
                self.last_action = "Right click"
                self._stamp("thumbs_up")

        elif gesture == "double_click":
            self._stop_drag()
            if self._cooled_down("double_click"):
                pyautogui.doubleClick()
                self.last_action = "Double click"
                self._stamp("double_click")

        elif gesture == "fist":
            self._drag(tip)

        elif gesture == "two_fingers":
            self._stop_drag()
            self._scroll(landmarks)

        elif gesture == "pinky_up":
            self._stop_drag()
            if self._cooled_down("pinky_up"):
                subprocess.Popen("rundll32.exe user32.dll,LockWorkStation", shell=True)
                self.last_action = "Lock PC"
                self._stamp("pinky_up")
                self._last["pinky_up"] += 2.0  # extra cooldown

        elif gesture == "three_fingers":
            self._stop_drag()
            if self._cooled_down("three_fingers"):
                subprocess.Popen("start chrome", shell=True)
                self.last_action = "Open Chrome"
                self._stamp("three_fingers")
                self._last["three_fingers"] += 2.0

        elif gesture == "four_fingers":
            self._stop_drag()
            if self._cooled_down("four_fingers"):
                subprocess.Popen("code", shell=True)
                self.last_action = "Open VS Code"
                self._stamp("four_fingers")
                self._last["four_fingers"] += 2.0

        elif gesture == "rock_sign":
            self._stop_drag()
            if self._cooled_down("rock_sign"):
                pyautogui.press('volumeup')
                self.last_action = "Volume Up"
                self._stamp("rock_sign")
                self._last["rock_sign"] -= self.cooldown - 0.1  # fast repeat

        elif gesture == "shaka_sign":
            self._stop_drag()
            if self._cooled_down("shaka_sign"):
                pyautogui.press('volumedown')
                self.last_action = "Volume Down"
                self._stamp("shaka_sign")
                self._last["shaka_sign"] -= self.cooldown - 0.1

        elif gesture == "l_sign":
            self._stop_drag()
            if self._cooled_down("l_sign"):
                if sbc:
                    try:
                        curr = sbc.get_brightness(display=0)[0]
                        sbc.set_brightness(min(100, curr + 10), display=0)
                    except Exception:
                        pass
                self.last_action = "Brightness Up"
                self._stamp("l_sign")
                self._last["l_sign"] -= self.cooldown - 0.2

        elif gesture == "gun_sign":
            self._stop_drag()
            if self._cooled_down("gun_sign"):
                if sbc:
                    try:
                        curr = sbc.get_brightness(display=0)[0]
                        sbc.set_brightness(max(0, curr - 10), display=0)
                    except Exception:
                        pass
                self.last_action = "Brightness Down"
                self._stamp("gun_sign")
                self._last["gun_sign"] -= self.cooldown - 0.2

        else:
            self._reset_states(gesture)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _map_coords(self, x: float, y: float) -> tuple[int, int]:
        """
        Convert normalised hand coords → screen pixels.
        x is mirrored so the user's right hand moves cursor rightward.
        Clamp to screen bounds.
        """
        sx = int((1.0 - x) * self.screen_w)
        sy = int(y * self.screen_h)
        sx = max(0, min(sx, self.screen_w - 1))
        sy = max(0, min(sy, self.screen_h - 1))
        return sx, sy

    def _move_mouse(self, tip: tuple) -> None:
        sx, sy = self._map_coords(tip[0], tip[1])
        pyautogui.moveTo(sx, sy, duration=0)
        self.last_action = f"Move ({sx}, {sy})"
        self._stop_drag()
        self._prev_scroll_y = None

    def _scroll(self, landmarks: list) -> None:
        curr_y = landmarks[INDEX_TIP][1]
        if self._prev_scroll_y is not None:
            delta = curr_y - self._prev_scroll_y
            # Positive delta = hand moved down = scroll down (negative clicks)
            scroll_amount = int(-delta * self.SCROLL_SENSITIVITY)
            if scroll_amount != 0:
                pyautogui.scroll(scroll_amount)
                direction = "↑" if scroll_amount > 0 else "↓"
                self.last_action = f"Scroll {direction} ({abs(scroll_amount)})"
        self._prev_scroll_y = curr_y

    def _drag(self, tip: tuple) -> None:
        sx, sy = self._map_coords(tip[0], tip[1])
        if not self._dragging:
            pyautogui.mouseDown(button="left")
            self._dragging = True
            self.last_action = "Drag start"
        else:
            pyautogui.moveTo(sx, sy, duration=self.DRAG_MOVE_DURATION)
            self.last_action = f"Dragging ({sx}, {sy})"
        self._prev_scroll_y = None

    def _stop_drag(self) -> None:
        if self._dragging:
            pyautogui.mouseUp(button="left")
            self._dragging = False

    def _reset_states(self, gesture: str) -> None:
        self._stop_drag()
        self._prev_scroll_y = None
        if gesture == "open_palm":
            self.last_action = "Paused (palm)"
        elif gesture == "none":
            self.last_action = "No hand"
        else:
            self.last_action = "—"

    def _cooled_down(self, key: str) -> bool:
        return time.time() - self._last.get(key, 0) >= self.cooldown

    def _stamp(self, key: str) -> None:
        self._last[key] = time.time()
