"""
hand_tracker.py
===============
Wraps MediaPipe Hands to extract 21 3D landmark coordinates
from a webcam frame. Returns normalized (x, y, z) tuples where
x and y are 0–1 relative to frame dimensions.

Landmark indices (MediaPipe convention):
  0  = Wrist
  1-4  = Thumb  (CMC, MCP, IP, TIP)
  5-8  = Index  (MCP, PIP, DIP, TIP)
  9-12 = Middle (MCP, PIP, DIP, TIP)
  13-16= Ring   (MCP, PIP, DIP, TIP)
  17-20= Pinky  (MCP, PIP, DIP, TIP)
"""

import cv2
import mediapipe as mp
from typing import Optional


class HandTracker:
    # Landmark indices for quick reference
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    def __init__(self, max_hands: int = 1, confidence: float = 0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw  = mp.solutions.drawing_utils
        self.mp_style = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence,
            model_complexity=1,         # 0 = fast, 1 = accurate
        )

    def get_landmarks(self, frame) -> tuple[Optional[list], any]:
        """
        Process a BGR frame and return (landmarks, annotated_frame).

        landmarks: list of 21 (x, y, z) tuples, or None if no hand detected.
        annotated_frame: frame with hand skeleton drawn on it.
        """
        annotated = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        rgb.flags.writeable = True

        if not result.multi_hand_landmarks:
            return None, annotated

        hand_lm = result.multi_hand_landmarks[0]

        # Draw skeleton
        self.mp_draw.draw_landmarks(
            annotated,
            hand_lm,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_style.get_default_hand_landmarks_style(),
            self.mp_style.get_default_hand_connections_style(),
        )

        # Extract normalized coordinates
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_lm.landmark]
        return landmarks, annotated

    def __del__(self):
        if hasattr(self, "hands"):
            self.hands.close()
