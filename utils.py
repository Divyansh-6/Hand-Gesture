"""
utils.py
========
Utility classes:

  SmoothingFilter  — Rolling-average over N frames to reduce hand jitter.
  DebugOverlay     — Draws gesture info and landmark coords on the frame.
  FrameRateLimiter — Caps the processing loop at a target FPS.
"""

import cv2
import time
import numpy as np
from collections import deque
from typing import Optional


# ── Colours (BGR for OpenCV) ──────────────────────────────────────────────────
GREEN   = (50, 220, 100)
CYAN    = (220, 220, 50)
YELLOW  = (30, 220, 240)
RED     = (60, 60, 230)
WHITE   = (240, 240, 240)
DARK    = (30, 30, 30)
ORANGE  = (30, 140, 255)

# Gesture → colour mapping for the HUD badge
GESTURE_COLORS = {
    "pointer":      GREEN,
    "pinch":        CYAN,
    "thumbs_up":    YELLOW,
    "double_click": ORANGE,
    "fist":         RED,
    "two_fingers":  (200, 180, 50),
    "open_palm":    (180, 180, 180),
    "none":         (100, 100, 100),
    "unknown":      (80, 80, 80),
}

GESTURE_EMOJI = {
    "pointer":      "POINTER  ☝",
    "pinch":        "PINCH    🤌",
    "thumbs_up":    "THUMB UP 👍",
    "double_click": "DBL CLCK 🤞",
    "fist":         "FIST     ✊",
    "two_fingers":  "SCROLL   ✌",
    "open_palm":    "PALM     🖐",
    "none":         "NO HAND   ",
    "unknown":      "UNKNOWN   ",
}


class SmoothingFilter:
    """
    Rolling-average landmark smoother.

    Stores the last `window` frames of landmarks and returns the
    element-wise mean. When no hand is detected, clears the history
    so stale positions don't bleed into the next detection.
    """

    def __init__(self, window: int = 5):
        self.window  = window
        self.history: deque = deque(maxlen=window)

    def smooth(self, landmarks: Optional[list]) -> Optional[list]:
        if landmarks is None:
            self.history.clear()
            return None
        self.history.append(landmarks)
        arr = np.array(self.history)       # shape: (frames, 21, 3)
        return arr.mean(axis=0).tolist()


class DebugOverlay:
    """
    Renders a HUD on the frame showing:
      • Current gesture name + colour badge
      • Last OS action taken
      • Current FPS
      • Index fingertip coordinates (normalised)
      • Finger extension state (bitmap)
    """

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SMALL = 0.5
    FONT_MED   = 0.65
    FONT_LG    = 0.9
    THICK      = 1
    THICK_LG   = 2
    PAD        = 12

    def draw(
        self,
        frame,
        gesture: str,
        landmarks: Optional[list],
        last_action: str,
        fps: float,
    ):
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # ── Background panel (top-left) ──────────────────────────────────────
        panel_w, panel_h = 280, 140
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), DARK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # ── Gesture badge ────────────────────────────────────────────────────
        color  = GESTURE_COLORS.get(gesture, WHITE)
        label  = GESTURE_EMOJI.get(gesture, gesture.upper())
        cv2.putText(frame, label, (self.PAD, 32),
                    self.FONT, self.FONT_LG, color, self.THICK_LG, cv2.LINE_AA)

        # ── Last action ──────────────────────────────────────────────────────
        action_text = f"Action: {last_action}"
        cv2.putText(frame, action_text, (self.PAD, 60),
                    self.FONT, self.FONT_SMALL, WHITE, self.THICK, cv2.LINE_AA)

        # ── FPS counter ──────────────────────────────────────────────────────
        cv2.putText(frame, f"FPS: {fps:.0f}", (self.PAD, 85),
                    self.FONT, self.FONT_SMALL, (150, 220, 150), self.THICK, cv2.LINE_AA)

        # ── Finger state (if hand detected) ──────────────────────────────────
        if landmarks:
            tip  = landmarks[8]
            tip_text = f"Tip: ({tip[0]:.2f}, {tip[1]:.2f})"
            cv2.putText(frame, tip_text, (self.PAD, 108),
                        self.FONT, self.FONT_SMALL, (180, 180, 180), self.THICK, cv2.LINE_AA)

            # Finger up/down visualiser
            fingers_label = "Fingers: "
            bits = self._finger_bits(landmarks)
            for i, (name, val) in enumerate(zip(["I","M","R","P"], bits)):
                col = GREEN if val else RED
                x = self.PAD + 75 + i * 22
                cv2.putText(frame, name, (x, 128),
                            self.FONT, self.FONT_SMALL, col, self.THICK, cv2.LINE_AA)
            cv2.putText(frame, fingers_label, (self.PAD, 128),
                        self.FONT, self.FONT_SMALL, (180, 180, 180), self.THICK, cv2.LINE_AA)

        # ── Corner reminder ──────────────────────────────────────────────────
        reminder = "Move mouse to corner to quit"
        tw, _ = cv2.getTextSize(reminder, self.FONT, 0.4, 1)[0]
        cv2.putText(frame, reminder, (w - tw - 8, h - 8),
                    self.FONT, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

        return frame

    def _finger_bits(self, lm: list) -> list:
        TIPS  = [8, 12, 16, 20]
        PIPS  = [6, 10, 14, 18]
        return [1 if lm[t][1] < lm[p][1] else 0 for t, p in zip(TIPS, PIPS)]


class FrameRateLimiter:
    """
    Keeps the loop at a target FPS by sleeping when frames process too fast,
    and tracks actual FPS via an exponential moving average.
    """

    def __init__(self, target_fps: int = 30):
        self.target_interval = 1.0 / target_fps
        self._last_tick = time.perf_counter()
        self._fps_ema   = float(target_fps)
        self._alpha     = 0.1          # EMA smoothing factor

    def tick(self) -> None:
        now     = time.perf_counter()
        elapsed = now - self._last_tick
        sleep   = self.target_interval - elapsed
        if sleep > 0:
            time.sleep(sleep)
        actual_interval = time.perf_counter() - self._last_tick
        if actual_interval > 0:
            self._fps_ema = (1 - self._alpha) * self._fps_ema + self._alpha * (1.0 / actual_interval)
        self._last_tick = time.perf_counter()

    @property
    def fps(self) -> float:
        return self._fps_ema
