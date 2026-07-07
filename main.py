import time
import logging
import pyautogui
import cv2
from core.settings import Settings
from core.profile_manager import ProfileManager
from core.macro_recorder import MacroRecorder
from vision.camera_thread import CameraThread
from vision.hand_tracker import HandTracker
from vision.gesture_classifier import GestureClassifier
from vision.kalman_smoother import KalmanSmoother
from vision.hud import JarvisHUD
from actions.action_mapper import ActionMapper
from core.voice_assistant import VoiceAssistant
from core.action_queue import get_action

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/app.log'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def main():
    logging.info("Starting Jarvis Gesture OS...")
    settings = Settings.load()

    WIN = "J.A.R.V.I.S  //  GESTURE OS"
    W, H = settings.cam_width, settings.cam_height

    # Init all components
    profile_mgr = ProfileManager()
    macro_rec   = MacroRecorder()
    cam         = CameraThread(settings.camera_index, W, H)
    if not cam.start():
        logging.error("Camera failed to start.")
        return

    tracker    = HandTracker(max_hands=2, model_complexity=settings.model_complexity)
    classifier = GestureClassifier(
        hold_frames=settings.hold_frames,
        confidence_threshold=settings.confidence_threshold
    )
    smoother = KalmanSmoother()
    hud      = JarvisHUD(W, H)

    voice = VoiceAssistant(model_size="tiny.en")
    if settings.voice_enabled:
        voice.start()

    screen_w, screen_h = pyautogui.size()
    mapper = ActionMapper(screen_w, screen_h, W, H,
                          settings.cooldown, settings.dead_zone)

    pyautogui.PAUSE    = 0
    pyautogui.FAILSAFE = False

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    paused = False

    try:
        while True:
            # 1. Voice / background actions
            bg_action = get_action()
            if bg_action:
                act = bg_action["action"]
                if act == "pause_system":
                    paused = True
                    mapper.last_action = "Voice: Paused"
                elif act == "resume_system":
                    paused = False
                    mapper.last_action = "Voice: Resumed"
                else:
                    mapper.execute(act, (0, 0), None)
                    mapper.last_action = f"Voice: {mapper.last_action}"

            frame = cam.read()
            if frame is None:
                continue

            fps_start = time.time()
            hud_data = {
                'fps':            0,
                'profile':        profile_mgr.active_profile,
                'paused':         paused,
                'action':         mapper.last_action or "",
                'primary_hand':   None,
                'secondary_hand': None,
            }

            if not paused:
                # 2. Track hands
                hands, results = tracker.process(frame)
                frame = tracker.draw(frame, results)

                primary_hand = secondary_hand = None
                for hand in hands:
                    if hand["label"] == settings.dominant_hand:
                        primary_hand = hand
                    else:
                        secondary_hand = hand

                if primary_hand is None and secondary_hand is not None:
                    primary_hand, secondary_hand = secondary_hand, None

                # 3. Primary hand
                if primary_hand:
                    lm = primary_hand["landmarks"]
                    sx, sy = smoother.smooth(lm[8][0], lm[8][1])
                    lm[8] = (sx, sy, lm[8][2])

                    gesture, conf = classifier.classify(lm)
                    macro_rec.add_gesture(gesture)
                    macro_action = macro_rec.match(
                        profile_mgr.profiles.get(
                            profile_mgr.active_profile, {}).get("macros", []))

                    if macro_action:
                        mapper.execute(macro_action, lm[8], lm)
                    else:
                        action = profile_mgr.get_action(gesture)
                        if gesture == "pointer":
                            action = "pointer"
                        mapper.execute(action, lm[8], lm)

                    hud_data['primary_hand'] = {
                        'gesture': gesture,
                        'conf':    conf,
                        'x':       lm[8][0] / W,
                        'y':       lm[8][1] / H,
                    }
                else:
                    smoother.reset()

                # 4. Secondary hand
                if secondary_hand:
                    sec_lm = secondary_hand["landmarks"]
                    sec_gesture, _ = classifier.classify(sec_lm)
                    hud_data['secondary_hand'] = {
                        'gesture': sec_gesture,
                        'x':       sec_lm[8][0] / W,
                        'y':       sec_lm[8][1] / H,
                    }
                    if sec_gesture == "open_palm":
                        paused = True
                        logging.info("System paused via secondary hand.")
            else:
                # Check for resume gesture
                hands, _ = tracker.process(frame)
                for hand in hands:
                    gest, _ = classifier.classify(hand["landmarks"])
                    if gest == "thumbs_up":
                        paused = False
                        logging.info("System resumed.")
                        break

            # 5. FPS & HUD
            hud_data['fps'] = 1.0 / (time.time() - fps_start + 1e-5)
            frame = hud.draw(frame, hud_data)

            cv2.imshow(WIN, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
            elif key == ord('1'):
                profile_mgr.switch_profile("Default")
            elif key == ord('2'):
                profile_mgr.switch_profile("Browser")

    except KeyboardInterrupt:
        pass
    finally:
        if settings.voice_enabled:
            voice.stop()
        cam.stop()
        cv2.destroyAllWindows()
        logging.info("Jarvis OS stopped.")


if __name__ == "__main__":
    main()
