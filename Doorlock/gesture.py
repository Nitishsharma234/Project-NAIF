"""
gesture.py - Gesture Detection Module
======================================
Gestures:
  TWO_FINGERS  (index + middle up) → UNLOCK
  FIST         (all curled)        → LOCK
  THUMBS_UP                        → CONFIRM
  OPEN_PALM    (all 5 extended)    → (informational)
  UNKNOWN      → wrong gesture counter → alarm at 3
"""

import cv2, time, threading

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[WARNING] MediaPipe not installed. Gesture detection disabled.")


class GestureDetector:
    FINGER_TIPS = [4, 8, 12, 16, 20]

    def __init__(self, min_detection_confidence=0.75,
                 min_tracking_confidence=0.6,
                 wrong_gesture_limit=3,
                 alarm_callback=None):
        self.enabled             = MEDIAPIPE_AVAILABLE
        self.wrong_gesture_limit = wrong_gesture_limit
        self.alarm_callback      = alarm_callback
        self._wrong_count        = 0
        self._last_gesture       = None
        self._alarm_active       = False
        self._cooldown_until     = 0
        self._hands = None

        if self.enabled:
            self._mp_hands  = mp.solutions.hands
            self._mp_draw   = mp.solutions.drawing_utils
            self._mp_styles = mp.solutions.drawing_styles
            self._hands = self._mp_hands.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence)
            print("[INFO] Gesture detector ready.  Unlock = ✌ TWO FINGERS")

    def detect_gesture(self, frame):
        if not self.enabled or self._hands is None:
            return frame, None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True
        if not results.multi_hand_landmarks:
            return frame, None
        hlm = results.multi_hand_landmarks[0]
        self._mp_draw.draw_landmarks(frame, hlm,
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_styles.get_default_hand_landmarks_style(),
            self._mp_styles.get_default_hand_connections_style())
        gesture = self._classify(hlm)
        self._update_wrong(gesture)
        self._draw_label(frame, gesture)
        return frame, gesture

    def _classify(self, hlm):
        lm = hlm.landmark
        # THUMBS UP
        if lm[4].y < lm[3].y < lm[2].y and all(lm[t].y > lm[t-2].y for t in self.FINGER_TIPS[1:]):
            return "THUMBS_UP"
        ext = [False]*5
        ext[0] = abs(lm[4].x - lm[2].x) > 0.06
        for i, tip in enumerate(self.FINGER_TIPS[1:], 1):
            ext[i] = lm[tip].y < lm[tip-2].y
        n = sum(ext)
        # TWO_FINGERS: index+middle up, ring+pinky down
        if ext[1] and ext[2] and not ext[3] and not ext[4]:
            return "TWO_FINGERS"
        if n <= 1:   return "FIST"
        if n >= 4:   return "OPEN_PALM"
        return "UNKNOWN"

    def _update_wrong(self, gesture):
        if time.time() < self._cooldown_until or gesture == self._last_gesture:
            return
        self._last_gesture = gesture
        good = {"TWO_FINGERS","THUMBS_UP","FIST",None}
        if gesture not in good:
            self._wrong_count += 1
            print(f"[GESTURE] ⚠ Wrong gesture #{self._wrong_count}: {gesture}")
            if self._wrong_count >= self.wrong_gesture_limit and not self._alarm_active:
                self._alarm_active   = True
                self._wrong_count    = 0
                self._cooldown_until = time.time() + 12
                print("[GESTURE] 🚨 Alarm triggered by wrong gestures!")
                if self.alarm_callback:
                    threading.Thread(target=self.alarm_callback, daemon=True).start()
        else:
            self._wrong_count  = 0
            self._alarm_active = False

    def _draw_label(self, frame, gesture):
        h, w = frame.shape[:2]
        COLORS = {"TWO_FINGERS":(0,255,100),"THUMBS_UP":(0,215,255),
                  "FIST":(0,80,255),"OPEN_PALM":(200,200,0),"UNKNOWN":(50,50,220)}
        LABELS = {"TWO_FINGERS":"✌ TWO FINGERS — UNLOCK","THUMBS_UP":"👍 THUMBS UP",
                  "FIST":"✊ FIST — LOCK","OPEN_PALM":"✋ OPEN PALM",
                  "UNKNOWN":f"? WRONG GESTURE  {self._wrong_count}/{self.wrong_gesture_limit}"}
        color = COLORS.get(gesture,(180,180,180))
        label = LABELS.get(gesture, gesture or "")
        cv2.rectangle(frame,(0,h-44),(w,h),(0,0,0),-1)
        cv2.putText(frame, label,(12,h-12),cv2.FONT_HERSHEY_DUPLEX,0.72,color,2)

    def reset_alarm(self):
        self._alarm_active=False; self._wrong_count=0; self._cooldown_until=0

    def release(self):
        if self._hands: self._hands.close()
