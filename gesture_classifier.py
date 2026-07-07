"""
gesture_classifier.py
=====================
Converts 21 MediaPipe hand landmarks into a named gesture label.

Strategy:
  1. Determine which fingers are extended (tip y < PIP y).
  2. Measure thumb-index pinch distance for click detection.
  3. Require the same gesture for N consecutive frames before
     confirming — this prevents single-frame misfires.

Gesture vocabulary:
  "none"         - No hand detected
  "pointer"      - Index finger only extended → mouse move
  "pinch"        - Thumb and index tips close together → left click
  "fist"         - All fingers curled → drag
  "two_fingers"  - Index + middle extended → scroll
  "open_palm"    - All 5 fingers extended → pause / mode switch
  "thumbs_up"    - Thumb up, all others curled → right click
  "double_click" - Index + middle touching tips (scissors pinch)
  "unknown"      - Hand detected but no pattern matched
"""

import math
from collections import deque
from typing import Optional


# ── Landmark indices ──────────────────────────────────────────────────────────
WRIST = 0
THUMB_TIP,  THUMB_IP,  THUMB_MCP,  THUMB_CMC  = 4, 3, 2, 1
INDEX_TIP,  INDEX_DIP, INDEX_PIP,  INDEX_MCP  = 8, 7, 6, 5
MIDDLE_TIP, MIDDLE_DIP,MIDDLE_PIP, MIDDLE_MCP = 12,11,10, 9
RING_TIP,   RING_DIP,  RING_PIP,   RING_MCP   = 16,15,14,13
PINKY_TIP,  PINKY_DIP, PINKY_PIP,  PINKY_MCP  = 20,19,18,17

# Finger (tip, pip) pairs for extension test
FINGERS = [
    (INDEX_TIP,  INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP,   RING_PIP),
    (PINKY_TIP,  PINKY_PIP),
]

PINCH_THRESHOLD        = 0.055   # Normalised distance for pinch/click
DOUBLE_CLICK_THRESHOLD = 0.055   # Index-middle tip distance for double click


def _dist(a, b) -> float:
    """Euclidean distance between two (x,y) or (x,y,z) landmarks."""
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(min(len(a), len(b)))))


def _fingers_extended(lm: list) -> list:
    """
    Returns [index, middle, ring, pinky] as 1 (up) or 0 (down).
    Uses distance from wrist to determine extension (rotation invariant).
    """
    return [1 if _dist(lm[WRIST], lm[tip]) > _dist(lm[WRIST], lm[pip]) else 0
            for tip, pip in FINGERS]


def _thumb_up(lm: list) -> bool:
    """Thumb is extended when it is far from the pinky MCP."""
    return _dist(lm[PINKY_MCP], lm[THUMB_TIP]) > _dist(lm[PINKY_MCP], lm[THUMB_MCP])


def _classify_once(lm: list) -> str:
    """Single-frame classification — raw, no debounce."""
    if lm is None:
        return "none"

    fi = _fingers_extended(lm)           # [index, middle, ring, pinky]
    thumb = _thumb_up(lm)
    pinch_d = _dist(lm[THUMB_TIP], lm[INDEX_TIP])

    # ── Pinch (thumb + index close) ──────────────────────────────────────────
    if pinch_d < PINCH_THRESHOLD and fi[1] == 0 and fi[2] == 0 and fi[3] == 0:
        return "pinch"

    # ── Double click & Two fingers ───────────────────────────────────────────
    if fi[0] == 1 and fi[1] == 1 and fi[2] == 0 and fi[3] == 0:
        scissors_d = _dist(lm[INDEX_TIP], lm[MIDDLE_TIP])
        if scissors_d < DOUBLE_CLICK_THRESHOLD:
            return "double_click"
        if not thumb:
            return "two_fingers"

    # ── Fist (all curled) ────────────────────────────────────────────────────
    if fi == [0, 0, 0, 0] and not thumb:
        return "fist"

    # ── Open palm (all extended) ─────────────────────────────────────────────
    if fi == [1, 1, 1, 1] and thumb:
        return "open_palm"

    # ── Thumbs up ────────────────────────────────────────────────────────────
    if thumb and fi == [0, 0, 0, 0]:
        return "thumbs_up"

    # ── Pointer (index only) ─────────────────────────────────────────────────
    if fi[0] == 1 and fi[1] == 0 and fi[2] == 0 and fi[3] == 0 and not thumb:
        return "pointer"

    # ── New Hot Gestures ─────────────────────────────────────────────────────
    if fi == [0, 0, 0, 1] and not thumb:
        return "pinky_up"

    if fi == [1, 1, 1, 0] and not thumb:
        return "three_fingers"

    if fi == [1, 1, 1, 1] and not thumb:
        return "four_fingers"

    if fi == [1, 0, 0, 1] and not thumb:
        return "rock_sign"

    if fi == [0, 0, 0, 1] and thumb:
        return "shaka_sign"

    if fi == [1, 0, 0, 0] and thumb:
        return "l_sign"

    if fi == [1, 1, 0, 0] and thumb:
        return "gun_sign"

    return "unknown"


class GestureClassifier:
    """
    Wraps single-frame classification with a hold-frame requirement.
    A gesture is only confirmed after appearing in N consecutive frames.
    This eliminates accidental triggers from brief hand transitions.
    """

    def __init__(self, hold_frames: int = 3):
        self.hold_frames = hold_frames
        self.history: deque = deque(maxlen=hold_frames)
        self.confirmed: str = "none"

    def classify(self, lm: Optional[list]) -> str:
        raw = _classify_once(lm)
        self.history.append(raw)

        # Confirm only if the last N frames all agree
        if len(self.history) == self.hold_frames and len(set(self.history)) == 1:
            self.confirmed = raw

        return self.confirmed

    @property
    def raw_gesture(self) -> str:
        """Latest single-frame classification (no debounce)."""
        return self.history[-1] if self.history else "none"
