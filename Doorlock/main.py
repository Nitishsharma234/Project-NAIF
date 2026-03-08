"""
main.py - Smart AI Door Security System  v3
=============================================
  Q  – quit          T  – enroll new face
  R  – record        U  – unlock   L  – lock
  A  – stop alarm    F  – toggle face scan ON/OFF
"""

import cv2, os, sys, time, threading, random
import numpy as np
from datetime import datetime

# winsound is Windows-only — import safely
try:
    import winsound
    WINSOUND = True
except ImportError:
    WINSOUND = False

from face_auth import FaceAuthenticator
from voice    import VoiceEngine
from gesture  import GestureDetector
import server as web_server

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════
CONFIG = {
    "faces_dir":           "faces",
    "alerts_dir":          "alerts",
    "recordings_dir":      "recordings",
    "voice_password":      "Activate Jarvis",
    "max_failed_attempts": 3,
    "access_cooldown_sec": 4,
    "camera_index":        0,
    "frame_width":         1280,
    "frame_height":        720,
    "fps":                 20,
    "motion_threshold":    900,
    "unknown_linger_sec":  5,
    "face_scan_enabled":   True,   # toggled with F key or web button
    "gesture_enabled":     True,
    "web_server_port":     5000,
    "show_fps":            True,
    "show_motion_box":     True,
    # Face must be detected in N consecutive scans before triggering auth
    "face_confirm_frames": 2,
}

POLITE_QUESTIONS = [
    "Do you need something, sir?",
    "Are you waiting for someone?",
    "Can I help you with something?",
    "Hello! Is there anything I can assist you with?",
    "You have been standing here a while. May I ask why?",
    "Is everything alright out there?",
    "Are you looking for someone?",
]

# ══════════════════════════════════════════════════════════════
#  GLOBAL ALARM STATE  (can be stopped from web or keyboard)
# ══════════════════════════════════════════════════════════════
alarm_active   = False   # True while alarm is sounding
_alarm_stop    = threading.Event()   # set this to stop the alarm loop

def trigger_alarm_sound(repeats=10):
    """Non-blocking alarm. Runs in a thread. Stops when _alarm_stop is set."""
    global alarm_active
    alarm_active = True
    _alarm_stop.clear()
    print("[ALARM] 🔔🔔🔔  ALARM ACTIVE — press A or web button to stop  🔔🔔🔔")
    web_server.door_state["alarm_active"] = True
    count = 0
    while not _alarm_stop.is_set() and count < repeats * 2:
        if WINSOUND:
            try:
                winsound.Beep(1200 if count % 2 == 0 else 800, 280)
            except Exception:
                pass
        else:
            print("\a", end="", flush=True)
        _alarm_stop.wait(0.18)
        count += 1
    alarm_active = False
    web_server.door_state["alarm_active"] = False
    print("[ALARM] ■ Alarm stopped.")

def stop_alarm():
    """Stop the alarm from anywhere — keyboard A, web button, or voice."""
    global alarm_active
    _alarm_stop.set()
    alarm_active = False
    web_server.door_state["alarm_active"] = False
    web_server.add_log("🔕 Alarm stopped manually")
    print("[ALARM] ■ Alarm stopped by user.")

# ══════════════════════════════════════════════════════════════
#  DOOR STATE
# ══════════════════════════════════════════════════════════════
door_locked = True

def set_door(locked: bool, reason: str = "system"):
    global door_locked
    door_locked = locked
    web_server.set_door_locked(locked, reason)
    icon = "🔒" if locked else "🔓"
    print(f"\n[DOOR] {icon} {'LOCKED' if locked else 'UNLOCKED'} ({reason})\n")

# ══════════════════════════════════════════════════════════════
#  MOTION DETECTOR
# ══════════════════════════════════════════════════════════════
class MotionDetector:
    def __init__(self):
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False)

    def detect(self, frame):
        mask    = self._bg.apply(frame)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_a, best_r = 0, None
        for c in cnts:
            a = cv2.contourArea(c)
            if a > best_a:
                best_a, best_r = a, cv2.boundingRect(c)
        return best_a > CONFIG["motion_threshold"], best_r

# ══════════════════════════════════════════════════════════════
#  VIDEO RECORDER
# ══════════════════════════════════════════════════════════════
class VideoRecorder:
    def __init__(self):
        self._writer    = None
        self._recording = False
        os.makedirs(CONFIG["recordings_dir"], exist_ok=True)

    def start(self, label="manual"):
        if self._recording:
            return
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        path   = os.path.join(CONFIG["recordings_dir"], f"{label}_{ts}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._writer    = cv2.VideoWriter(
            path, fourcc, CONFIG["fps"],
            (CONFIG["frame_width"], CONFIG["frame_height"]))
        self._recording = True
        print(f"[REC] ● Recording → {path}")

    def write(self, frame):
        if self._recording and self._writer:
            self._writer.write(
                cv2.resize(frame, (CONFIG["frame_width"], CONFIG["frame_height"])))

    def stop(self):
        if self._recording and self._writer:
            self._writer.release()
            self._writer    = None
            self._recording = False
            print("[REC] ■ Recording saved.")

    def toggle(self, label="manual"):
        if self._recording: self.stop()
        else:               self.start(label)

    @property
    def is_recording(self): return self._recording

# ══════════════════════════════════════════════════════════════
#  HUD / OSD
# ══════════════════════════════════════════════════════════════
def draw_hud(frame, fps, is_recording, alert_count,
             enrolled_people=None, face_scan_on=True):
    h, w = frame.shape[:2]
    now  = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    # Top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, now, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 1)

    if CONFIG["show_fps"]:
        cv2.putText(frame, f"FPS:{fps:.0f}", (w - 100, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (150, 255, 150), 1)

    # Door badge
    locked = web_server.get_door_locked()
    btext  = "LOCKED" if locked else "UNLOCKED"
    bcolor = (0, 0, 180) if locked else (0, 180, 0)
    cv2.rectangle(frame, (w//2 - 70, 5), (w//2 + 70, 38), bcolor, -1)
    cv2.putText(frame, btext, (w//2 - 55, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # REC dot
    if is_recording:
        cv2.circle(frame, (w - 22, 62), 9, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 62, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

    # Face scan status badge
    fscan_color = (0, 200, 80) if face_scan_on else (60, 60, 60)
    fscan_text  = "FACE:ON" if face_scan_on else "FACE:OFF"
    cv2.rectangle(frame, (w - 105, 55), (w - 5, 80), fscan_color, -1)
    cv2.putText(frame, fscan_text, (w - 100, 73),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Alarm indicator
    if alarm_active:
        # Flashing red border when alarm is active
        t_mod = int(time.time() * 4) % 2
        if t_mod:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        cv2.putText(frame, "🚨 ALARM ACTIVE — press A to stop",
                    (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Alert count
    if alert_count:
        cv2.putText(frame, f"ALERTS:{alert_count}",
                    (10, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # Enrolled people
    if enrolled_people:
        label = "Known: " + ", ".join(enrolled_people)
        cv2.putText(frame, label, (10, h - 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 255, 100), 1)

    return frame


def draw_banner(frame, text, color=(0, 200, 0)):
    h, w = frame.shape[:2]
    y_offset = 60
    ov = frame.copy()
    cv2.rectangle(ov, (0, h - y_offset), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    scale = min(0.8, w / (max(len(text), 1) * 13 + 20))
    scale = max(0.42, scale)
    tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)[0]
    x = max(8, (w - tw) // 2)
    y = h - y_offset + th + 8
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, 2)
    return frame

# ══════════════════════════════════════════════════════════════
#  SECURITY ALERT
# ══════════════════════════════════════════════════════════════
def _fire_security_alert(snapshot, voice_eng, status, alerts):
    print("\n" + "!" * 58)
    print("!!!       🚨  SECURITY ALERT TRIGGERED  🚨          !!!")
    print("!" * 58 + "\n")
    if snapshot is not None:
        FaceAuthenticator.capture_snapshot(snapshot, CONFIG["alerts_dir"], "intruder")
    voice_eng.speak("Security alert! Too many failed attempts. Alarm activated.")
    print("[NOTIFY] 📧 Email alert — intruder at door!")
    print("[NOTIFY] 📱 SMS alert   — multiple failed attempts!")
    alerts["count"] += 1
    web_server.door_state["alert_count"] = alerts["count"]
    web_server.add_log(f"🚨 SECURITY ALERT #{alerts['count']} — snapshot saved")
    status["text"]  = "🚨 SECURITY ALERT — ALARM ACTIVE"
    status["color"] = (0, 0, 255)
    threading.Thread(target=trigger_alarm_sound, args=(12,), daemon=True).start()

# ══════════════════════════════════════════════════════════════
#  VOICE AUTH  (background thread)
# ══════════════════════════════════════════════════════════════
def handle_voice_auth(voice_eng, snapshot, status, attempts, alerts, auth_active_ref):
    voice_eng.speak("Unknown person detected. Please say the password to enter.")
    print("\n" + "─" * 52)
    print("  🎤  Say:  ACTIVATE JARVIS  to unlock")
    print("─" * 52)

    spoken = voice_eng.listen(timeout=10, phrase_limit=7)

    if voice_eng.verify_password(spoken):
        voice_eng.speak("Password accepted. Welcome. Door is now unlocked.")
        set_door(False, "voice password")
        status["text"]  = "✅ Activate Jarvis — Access Granted!"
        status["color"] = (0, 220, 0)
        attempts["count"]         = 0
        attempts["gesture_warned"] = False
        web_server.add_log("✅ Voice password correct → door unlocked")
        time.sleep(7)
        set_door(True, "auto re-lock")
    else:
        attempts["count"] += 1
        wrong = attempts["count"]
        left  = CONFIG["max_failed_attempts"] - wrong
        if wrong < CONFIG["max_failed_attempts"]:
            voice_eng.speak(
                f"Wrong password. {left} attempt{'s' if left != 1 else ''} remaining.")
            status["text"]  = f"❌ Wrong Password  ({wrong}/{CONFIG['max_failed_attempts']})"
            status["color"] = (0, 80, 255)
            web_server.add_log(f"❌ Wrong voice password #{wrong}")
        else:
            attempts["count"] = 0
            _fire_security_alert(snapshot, voice_eng, status, alerts)

    auth_active_ref["active"] = False

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    global alarm_active

    print("=" * 60)
    print("   Smart AI Door Security System  —  Starting Up")
    print("=" * 60)

    # ── Init modules ─────────────────────────────────────────
    face_auth  = FaceAuthenticator(faces_dir=CONFIG["faces_dir"])
    voice_eng  = VoiceEngine(password=CONFIG["voice_password"])
    motion_det = MotionDetector()
    recorder   = VideoRecorder()

    def gesture_alarm_cb():
        voice_eng.speak("Alarm! Wrong gesture detected multiple times.")
        _fire_security_alert(
            face_snapshot["frame"], voice_eng, status, alerts)
        gesture_det.reset_alarm()

    gesture_det = GestureDetector(
        wrong_gesture_limit=3,
        alarm_callback=gesture_alarm_cb
    ) if CONFIG["gesture_enabled"] else None

    # ── Web server ────────────────────────────────────────────
    threading.Thread(
        target=web_server.start_server,
        kwargs={"port": CONFIG["web_server_port"]},
        daemon=True).start()

    # Register web-side alarm stop + face scan toggle callbacks
    web_server.register_stop_alarm_callback(stop_alarm)
    web_server.register_face_scan_callback(
        lambda enabled: CONFIG.update({"face_scan_enabled": enabled})
    )

    # ── Camera ────────────────────────────────────────────────
    print(f"[INFO] Opening camera {CONFIG['camera_index']}…")
    cap = cv2.VideoCapture(CONFIG["camera_index"], cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[WARN] Index 0 failed — trying index 1…")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] No webcam found."); sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
    cap.set(cv2.CAP_PROP_FPS,          CONFIG["fps"])
    for _ in range(5): cap.read()   # warm-up
    print(f"[INFO] Camera: {int(cap.get(3))}×{int(cap.get(4))}")
    print(f"[INFO] Dashboard: http://localhost:{CONFIG['web_server_port']}")
    print("[INFO] Keys: Q=quit  T=enroll  R=record  F=face-scan  A=stop-alarm  U=unlock  L=lock\n")

    voice_eng.speak_async("Smart door security system activated.")

    # ── State ─────────────────────────────────────────────────
    status        = {"text": "System Ready", "color": (130, 130, 130)}
    auth_active   = {"active": False}
    attempts      = {"count": 0, "gesture_warned": False}
    alerts        = {"count": 0}
    face_snapshot = {"frame": None}

    unknown_first_seen   = None
    unknown_question_due = 0
    unknown_captured     = False

    # Consecutive face-seen counter (prevents false positives on empty frames)
    face_seen_streak = 0   # increments each scan that finds faces

    enroll_gen  = None
    enroll_name = ""

    last_face_time    = 0
    last_gesture_time = 0

    enrolled = face_auth.list_enrolled()
    if enrolled:
        status["text"]  = f"Ready — Known: {', '.join(enrolled)}"
        status["color"] = (0, 200, 100)
    else:
        status["text"]  = "No faces enrolled — press T to add a person"
        status["color"] = (0, 140, 255)

    with web_server._state_lock:
        web_server.door_state["enrolled_people"] = enrolled
        web_server.door_state["system_status"]   = "Running"
        web_server.door_state["alarm_active"]     = False
        web_server.door_state["face_scan_enabled"] = CONFIG["face_scan_enabled"]

    fps_count, fps_start, current_fps = 0, time.time(), 0.0

    # ── Main loop ─────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05); continue
        frame = cv2.flip(frame, 1)
        disp  = frame.copy()

        # FPS
        fps_count += 1
        if time.time() - fps_start >= 1.0:
            current_fps = fps_count / (time.time() - fps_start)
            fps_count   = 0
            fps_start   = time.time()

        # ── ENROLLMENT MODE ───────────────────────────────────
        if enroll_gen is not None:
            try:
                prog = next(enroll_gen)
                disp = prog.get("frame", disp)

                # Push progress to web dashboard every frame
                web_server.update_enroll_progress(
                    prog.get("count", 0),
                    prog.get("total", 300),
                    done=prog.get("done", False),
                    cancelled=prog.get("cancelled", False)
                )

                if prog.get("done"):
                    if prog.get("cancelled"):
                        status["text"]  = "Enrollment cancelled"
                        status["color"] = (80, 80, 80)
                        web_server.update_enroll_progress(0, 300, done=True, cancelled=True)
                    else:
                        enrolled = face_auth.list_enrolled()
                        status["text"]  = f"✅ '{enroll_name}' enrolled! Known: {enrolled}"
                        status["color"] = (0, 220, 0)
                        voice_eng.speak_async(
                            f"Training complete. {enroll_name} is now authorised.")
                        with web_server._state_lock:
                            web_server.door_state["enrolled_people"] = enrolled
                        web_server.add_log(f"✅ New face enrolled: {enroll_name}")
                        web_server.update_enroll_progress(
                            prog.get("total", 300), prog.get("total", 300),
                            done=True, cancelled=False)
                    enroll_gen  = None
                    enroll_name = ""
                else:
                    _, jpeg = cv2.imencode(".jpg", disp,
                                          [cv2.IMWRITE_JPEG_QUALITY, 72])
                    web_server.update_frame(jpeg.tobytes())
                    cv2.imshow(
                        "Smart Door Security System — Press Q to Quit", disp)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        prog["cancel"] = True
                    continue

            except StopIteration:
                enroll_gen = None

        # ── MOTION ────────────────────────────────────────────
        motion_on, m_rect = motion_det.detect(frame)
        if motion_on and m_rect and CONFIG["show_motion_box"]:
            x, y, w, h = m_rect
            cv2.rectangle(disp, (x, y), (x + w, y + h), (255, 255, 0), 1)

        recorder.write(frame)

        # ── GESTURE DETECTION ─────────────────────────────────
        current_gesture = None
        if gesture_det:
            disp, current_gesture = gesture_det.detect_gesture(disp)
            now = time.time()
            if current_gesture and now - last_gesture_time > 1.2:

                if current_gesture == "TWO_FINGERS":
                    last_gesture_time          = now
                    attempts["count"]          = 0
                    attempts["gesture_warned"] = False
                    gesture_det.reset_alarm()
                    voice_eng.speak_async(
                        "Two finger gesture recognised. Door unlocked.")
                    set_door(False, "gesture ✌")
                    status["text"]  = "✌ Two Fingers — Door Unlocked"
                    status["color"] = (0, 220, 100)
                    web_server.add_log("✌ Two-finger gesture → unlocked")
                    def _rg():
                        time.sleep(7); set_door(True, "auto re-lock")
                    threading.Thread(target=_rg, daemon=True).start()

                elif current_gesture == "FIST":
                    last_gesture_time = now
                    set_door(True, "gesture ✊")
                    status["text"]  = "✊ Fist — Door Locked"
                    status["color"] = (0, 0, 200)

                elif current_gesture not in (None, "THUMBS_UP", "OPEN_PALM"):
                    # Wrong / unknown gesture → shared counter
                    last_gesture_time = now
                    attempts["count"] += 1
                    wrong = attempts["count"]
                    left  = CONFIG["max_failed_attempts"] - wrong
                    print(f"[GESTURE] ⚠ Wrong #{wrong}: {current_gesture}")
                    web_server.add_log(f"⚠ Wrong gesture #{wrong}")
                    if wrong < CONFIG["max_failed_attempts"]:
                        voice_eng.speak_async(
                            f"Wrong gesture. {left} attempt"
                            f"{'s' if left != 1 else ''} remaining.")
                        status["text"]  = (
                            f"⚠ Wrong Gesture  "
                            f"({wrong}/{CONFIG['max_failed_attempts']})")
                        status["color"] = (0, 100, 255)
                    else:
                        attempts["count"] = 0
                        _fire_security_alert(
                            face_snapshot["frame"],
                            voice_eng, status, alerts)
                        gesture_det.reset_alarm()

        # ── FACE RECOGNITION ──────────────────────────────────
        now   = time.time()
        faces = []

        if (CONFIG["face_scan_enabled"]
                and not auth_active["active"]
                and now - last_face_time > CONFIG["access_cooldown_sec"]):

            last_face_time = now
            raw_faces = face_auth.identify_faces(frame)

            # ── FALSE-POSITIVE GUARD ─────────────────────────
            # Only act if face detected in 2+ consecutive scans
            if raw_faces:
                face_seen_streak += 1
            else:
                face_seen_streak = 0

            # Only use faces once streak is confirmed
            if face_seen_streak >= CONFIG["face_confirm_frames"]:
                faces = raw_faces

            if faces:
                face_snapshot["frame"] = frame.copy()
                has_auth    = any(f["authorized"] for f in faces)
                has_unknown = any(not f["authorized"] for f in faces)

                if has_auth:
                    af   = next(f for f in faces if f["authorized"])
                    name = af["name"]
                    print(f"[ACCESS] ✅ Authorised: {name}")
                    voice_eng.speak_async(f"Access granted. Welcome {name}.")
                    set_door(False, f"face ({name})")
                    status["text"]  = f"✅ Welcome, {name}!"
                    status["color"] = (0, 220, 0)
                    attempts["count"]          = 0
                    attempts["gesture_warned"] = False
                    face_seen_streak           = 0
                    last_face_time = now + CONFIG["access_cooldown_sec"]
                    unknown_first_seen   = None
                    unknown_captured     = False
                    unknown_question_due = 0
                    web_server.add_log(f"✅ Access granted: {name}")
                    def _rl():
                        time.sleep(8); set_door(True, "auto re-lock")
                    threading.Thread(target=_rl, daemon=True).start()

                elif has_unknown:
                    if unknown_first_seen is None:
                        unknown_first_seen   = now
                        unknown_captured     = False
                        unknown_question_due = (
                            now + CONFIG["unknown_linger_sec"] + 2)
                        print("[ACCESS] ❓ Unknown person — watching…")

            else:
                # No confirmed face — reset linger timer
                if unknown_first_seen is not None and face_seen_streak == 0:
                    print("[ACCESS] No face confirmed — timer reset.")
                    unknown_first_seen   = None
                    unknown_captured     = False
                    unknown_question_due = 0

        # ── UNKNOWN LINGER LOGIC (real-time, every frame) ─────
        if unknown_first_seen is not None and not auth_active["active"]:
            linger    = now - unknown_first_seen
            linger_th = CONFIG["unknown_linger_sec"]
            auth_th   = linger_th + 3

            if linger < linger_th:
                remaining = linger_th - linger
                status["text"]  = (
                    f"❓ Unknown person — {remaining:.0f}s until action")
                status["color"] = (0, 120, 255)
                # Countdown ring
                h_f, w_f = disp.shape[:2]
                pct = linger / linger_th
                cv2.ellipse(disp, (w_f - 60, 80), (38, 38), 0,
                            -90, int(-90 + 360 * pct), (0, 100, 255), 4)
                cv2.putText(disp, f"{remaining:.0f}s",
                            (w_f - 72, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 120, 255), 2)
            else:
                if not unknown_captured:
                    unknown_captured = True
                    snap = face_snapshot["frame"]
                    if snap is not None:
                        path = FaceAuthenticator.capture_snapshot(
                            snap, CONFIG["alerts_dir"], "unknown_visitor")
                        print(f"[CAPTURE] 📸 {path}")
                        web_server.add_log("📸 Unknown visitor — photo captured")
                    status["text"]  = "📸 Unknown visitor — photo captured"
                    status["color"] = (0, 150, 255)

                if now >= unknown_question_due:
                    q = random.choice(POLITE_QUESTIONS)
                    print(f"[QUESTION] 💬 {q}")
                    voice_eng.speak_async(q)
                    status["text"]       = f"💬 {q}"
                    status["color"]      = (0, 190, 255)
                    unknown_question_due = now + 13

                if linger >= auth_th and not auth_active["active"]:
                    auth_active["active"] = True
                    last_face_time        = now + 20
                    t = threading.Thread(
                        target=handle_voice_auth,
                        args=(voice_eng, face_snapshot["frame"],
                              status, attempts, alerts, auth_active),
                        daemon=True)
                    t.start()

        disp = face_auth.draw_faces(disp, faces)

        # ── HUD + BANNER ──────────────────────────────────────
        disp = draw_hud(disp, current_fps, recorder.is_recording,
                        alerts["count"],
                        enrolled_people=face_auth.list_enrolled(),
                        face_scan_on=CONFIG["face_scan_enabled"])
        disp = draw_banner(disp, status["text"], status["color"])

        # ── Web stream ────────────────────────────────────────
        _, jpeg = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 72])
        web_server.update_frame(jpeg.tobytes())

        cv2.imshow("Smart Door Security System — Press Q to Quit", disp)

        # ── KEY BINDINGS ──────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quitting…"); break

        elif key == ord("a"):
            stop_alarm()
            status["text"]  = "🔕 Alarm Stopped"
            status["color"] = (100, 100, 100)

        elif key == ord("f"):
            CONFIG["face_scan_enabled"] = not CONFIG["face_scan_enabled"]
            state = "ON" if CONFIG["face_scan_enabled"] else "OFF"
            print(f"[INFO] Face scan turned {state}")
            status["text"]  = f"Face scan: {state}"
            status["color"] = (0, 200, 80) if CONFIG["face_scan_enabled"] \
                              else (60, 60, 60)
            face_seen_streak = 0
            with web_server._state_lock:
                web_server.door_state["face_scan_enabled"] = \
                    CONFIG["face_scan_enabled"]

        elif key == ord("t"):
            if enroll_gen is not None:
                print("[ENROLL] Already enrolling — ESC to cancel first.")
            else:
                # Check if web triggered enrollment
                pending = web_server.get_pending_enroll_name()
                if pending:
                    enroll_name = pending
                else:
                    print("\n" + "─" * 50)
                    print("  FACE ENROLLMENT — type name then Enter (15s timeout):")
                    print("─" * 50)
                    name_holder = [None]
                    ev = threading.Event()
                    def _ask():
                        try: name_holder[0] = input("  Name ▶ ").strip()
                        except Exception: pass
                        ev.set()
                    threading.Thread(target=_ask, daemon=True).start()
                    ev.wait(timeout=15)
                    enroll_name = (name_holder[0] or "").replace(" ", "_")

                if enroll_name:
                    enroll_gen = face_auth.enroll_live(cap, enroll_name)
                    voice_eng.speak_async(
                        f"Starting enrollment for {enroll_name}. "
                        "Look at the camera.")
                    status["text"]  = f"📸 Enrolling {enroll_name}…"
                    status["color"] = (0, 200, 255)
                else:
                    print("[ENROLL] No name — cancelled.")

        elif key == ord("r"):
            recorder.toggle("manual")
            if recorder.is_recording:
                voice_eng.speak_async("Recording started.")
                status["text"]  = "⏺ Recording"
                status["color"] = (0, 0, 220)
            else:
                voice_eng.speak_async("Recording stopped.")
                status["text"]  = "⏹ Recording saved"
                status["color"] = (130, 130, 130)

        elif key == ord("u"):
            set_door(False, "keyboard")
            status["text"]  = "🔓 Unlocked (keyboard)"
            status["color"] = (0, 200, 0)
        elif key == ord("l"):
            set_door(True, "keyboard")
            status["text"]  = "🔒 Locked (keyboard)"
            status["color"] = (0, 0, 200)

        # ── Poll web-triggered actions ─────────────────────────
        web_action = web_server.pop_pending_action()
        if web_action == "stop_alarm":
            stop_alarm()
            status["text"]  = "🔕 Alarm Stopped"
            status["color"] = (100, 100, 100)
        elif web_action == "face_scan_on":
            CONFIG["face_scan_enabled"] = True
            face_seen_streak = 0
        elif web_action == "face_scan_off":
            CONFIG["face_scan_enabled"] = False
            face_seen_streak = 0
        elif web_action and web_action.startswith("enroll:"):
            name = web_action.split(":", 1)[1].strip().replace(" ", "_")
            if name and enroll_gen is None:
                enroll_name = name
                enroll_gen  = face_auth.enroll_live(cap, enroll_name)
                voice_eng.speak_async(
                    f"Starting enrollment for {enroll_name}.")
                status["text"]  = f"📸 Enrolling {enroll_name}…"
                status["color"] = (0, 200, 255)

    # ── Cleanup ───────────────────────────────────────────────
    stop_alarm()
    recorder.stop()
    cap.release()
    cv2.destroyAllWindows()
    if gesture_det: gesture_det.release()
    print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    main()
