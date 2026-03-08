from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import threading
import time
import json
import os
import numpy as np
import pyautogui
import mediapipe as mp
import math
from prior import unified_chat, load_index, load_memory_embeddings, load_current_persons
from tts import speak, set_voice_enabled, is_voice_enabled
from vosk_listener import listen as stt_listen

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Global State ───────────────────────────────────────────────────────
mic_active          = False
stt_active          = False
mic_thread_started  = False
messages     = [{"role": "system", "content": "You are NAIF, created by Nitish Sharma."}]
index        = load_index()
memory_data  = load_memory_embeddings()
memory_lock  = threading.Lock()

# ── Face System ────────────────────────────────────────────────────────
FACE_DB_PATH         = "face_veri/face_db"
FACE_MODEL_PATH      = "face_veri/face_model.yml"
LABEL_MAP_PATH       = "face_veri/label_map.json"
CURRENT_PERSONS_FILE = "face_veri/current_persons.json"
CONFIDENCE_THRESHOLD = 100

os.makedirs(FACE_DB_PATH, exist_ok=True)

face_cascade        = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face_recognizer     = cv2.face.LBPHFaceRecognizer_create()
face_reverse_labels = {}

# Registration state
face_register_name   = ""
face_register_count  = 0
face_register_target = 50
face_register_active = False

# Debounce — camera runs at ~30fps
# 10 seconds to confirm presence, 10 seconds to confirm gone
detected_persons = []
_seen_counts     = {}
_absent_counts   = {}
CONFIRM_FRAMES   = 15    # ~0.5s to appear
LOSE_FRAMES      = 300   # ~10s at 30fps before removed


# ── Label map helpers ──────────────────────────────────────────────────

def _save_label_map(label_map):
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[Face] Label map saved: {label_map}")


def _load_label_map():
    global face_reverse_labels
    if not os.path.exists(LABEL_MAP_PATH):
        face_reverse_labels = {}
        return {}
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    face_reverse_labels = {int(v): k for k, v in label_map.items()}
    print(f"[Face] Labels loaded: {face_reverse_labels}")
    return label_map


def _train_face_model():
    persons = sorted([
        p for p in os.listdir(FACE_DB_PATH)
        if os.path.isdir(os.path.join(FACE_DB_PATH, p))
    ])
    if not persons:
        return False, "No person folders found."

    label_map = {name: idx for idx, name in enumerate(persons)}
    _save_label_map(label_map)
    _load_label_map()

    faces, labels = [], []
    for name, label in label_map.items():
        person_path = os.path.join(FACE_DB_PATH, name)
        for img_name in os.listdir(person_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = cv2.imread(os.path.join(person_path, img_name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(cv2.resize(img, (200, 200)))
                labels.append(label)

    if not faces:
        return False, "No images found."

    face_recognizer.train(faces, np.array(labels, dtype=np.int32))
    face_recognizer.save(FACE_MODEL_PATH)
    print(f"[Face] Trained: {len(faces)} images, persons={persons}")
    return True, f"✅ Trained for: {', '.join(persons)}"


def _save_current_persons(names):
    """Write to BOTH locations so prior.py always finds it regardless of cwd."""
    data = json.dumps({"persons": names})
    # Primary path (face_veri subfolder)
    with open(CURRENT_PERSONS_FILE, "w") as f:
        f.write(data)
    # Root path — prior.py reads from here
    with open("current_persons.json", "w") as f:
        f.write(data)
    if names:
        print(f"[Face] Persons saved: {names}")


def get_known_people():
    return sorted([
        p for p in os.listdir(FACE_DB_PATH)
        if os.path.isdir(os.path.join(FACE_DB_PATH, p))
    ])


def _update_debounce(raw_names):
    global detected_persons, _seen_counts, _absent_counts
    changed = False

    for name in raw_names:
        _seen_counts[name]   = _seen_counts.get(name, 0) + 1
        _absent_counts[name] = 0
        if name not in detected_persons and _seen_counts[name] >= CONFIRM_FRAMES:
            detected_persons.append(name)
            changed = True

    for name in list(_seen_counts.keys()):
        if name not in raw_names:
            _absent_counts[name] = _absent_counts.get(name, 0) + 1
            _seen_counts[name]   = 0
            if name in detected_persons and _absent_counts[name] >= LOSE_FRAMES:
                detected_persons.remove(name)
                changed = True

    return changed


# ── Load model at startup ──────────────────────────────────────────────
if os.path.exists(FACE_MODEL_PATH) and os.path.exists(LABEL_MAP_PATH):
    face_recognizer.read(FACE_MODEL_PATH)
    _load_label_map()
    print("[Face] Model and label map ready.")
elif os.path.exists(FACE_MODEL_PATH):
    print("[Face] Model found but no label map — retraining...")
    ok, msg = _train_face_model()
    if ok:
        face_recognizer.read(FACE_MODEL_PATH)
    print(f"[Face] {msg}")
else:
    print("[Face] No model yet. Register a person first.")

# Write empty persons file at startup so prior.py never crashes on missing file
_save_current_persons(detected_persons)


# ── Hand Detector ──────────────────────────────────────────────────────
hand_detector = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
drawing_utils = mp.solutions.drawing_utils
prev_x, prev_y = 0, 0
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# ── UNIFIED CAMERA THREAD ──────────────────────────────────────────────
def camera_thread():
    global prev_x, prev_y, face_register_count, face_register_active

    screen_w, screen_h = pyautogui.size()
    smoothening   = 7
    click_thresh  = 40
    scroll_thresh = 60

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)

        # ── REGISTER MODE ─────────────────────────────────────────────
        if face_register_active:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                if face_register_count < face_register_target:
                    face_img    = gray[y:y+h, x:x+w]
                    face_img    = cv2.resize(face_img, (200, 200))
                    person_path = os.path.join(FACE_DB_PATH, face_register_name)
                    os.makedirs(person_path, exist_ok=True)
                    cv2.imwrite(os.path.join(person_path, f"{face_register_count}.jpg"), face_img)
                    face_register_count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            pct   = face_register_count / face_register_target
            bar_w = int(frame.shape[1] * pct)
            cv2.rectangle(frame, (0, frame.shape[0]-18), (bar_w, frame.shape[0]), (0,255,100), -1)
            cv2.putText(frame,
                f"Registering '{face_register_name}': {face_register_count}/{face_register_target}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)

            socketio.emit('face_status', {
                "status": f"Capturing {face_register_count}/{face_register_target}",
                "register_count": face_register_count,
                "register_done": False
            })

            if face_register_count >= face_register_target:
                face_register_active = False
                socketio.emit('face_status', {
                    "status": "Capture complete! Training...",
                    "register_count": face_register_count,
                    "register_done": False
                })
                threading.Thread(target=_do_train, daemon=True).start()

        # ── ALWAYS-ON RECOGNITION ─────────────────────────────────────
        elif os.path.exists(FACE_MODEL_PATH) and face_reverse_labels:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            raw_names = []

            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                try:
                    label, confidence = face_recognizer.predict(face_img)
                    name = face_reverse_labels.get(label, "Unknown") if confidence < CONFIDENCE_THRESHOLD else "Unknown"
                except Exception as e:
                    print(f"[Face] Predict error: {e}")
                    name = "Unknown"

                if name != "Unknown":
                    raw_names.append(name)

                color = (0, 255, 80) if name != "Unknown" else (0, 60, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

            changed = _update_debounce(raw_names)
            if changed:
                _save_current_persons(detected_persons)
                socketio.emit('nearby_persons', {"persons": detected_persons})

        # ── HAND CONTROL ──────────────────────────────────────────────
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            drawing_utils.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
            thumb   = lm.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            index_f = lm.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            middle  = lm.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]

            tx    = int(thumb.x * screen_w)
            ty    = int(thumb.y * screen_h)
            cur_x = prev_x + (tx - prev_x) / smoothening
            cur_y = prev_y + (ty - prev_y) / smoothening
            pyautogui.moveTo(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y

            if math.hypot(index_f.x-thumb.x, index_f.y-thumb.y)*screen_w < click_thresh:
                pyautogui.click()
            if math.hypot(middle.x-index_f.x, middle.y-index_f.y)*screen_w < scroll_thresh:
                pyautogui.scroll(20)

        ret2, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret2:
            socketio.emit('camera_frame', buf.tobytes())

        time.sleep(0.033)


def _do_train():
    ok, msg = _train_face_model()
    if ok and os.path.exists(FACE_MODEL_PATH):
        face_recognizer.read(FACE_MODEL_PATH)
    socketio.emit('face_status', {
        "status": msg,
        "register_done": True,
        "known_people": get_known_people()
    })
    print(f"[Face] {msg}")


threading.Thread(target=camera_thread, daemon=True).start()


# ── STT Loop ───────────────────────────────────────────────────────────
def mic_stt_loop():
    global memory_data
    while True:
        if mic_active and stt_active:
            try:
                text = stt_listen()
                if text:
                    socketio.emit('receive_message', {"text": text, "role": "user"})
                    # Always read fresh from file
                    nearby_names = load_current_persons()
                    with memory_lock:
                        reply, memory_data = unified_chat(
                            text, index, messages, memory_data, nearby_names
                        )
                    messages.append({"role": "assistant", "content": reply})
                    socketio.emit('receive_message', {"text": reply, "role": "assistant"})
                    if is_voice_enabled():
                        speak(reply)
            except Exception as e:
                print(f"[STT Error] {e}")
                time.sleep(1)
        else:
            time.sleep(0.2)


# ── Socket Events ──────────────────────────────────────────────────────

@socketio.on('send_message')
def handle_message(data):
    global memory_data
    user_input = data.get('text', '').strip()
    if not user_input:
        return
    messages.append({"role": "user", "content": user_input})
    # Read directly from in-memory list (most up to date)
    nearby_names = list(detected_persons)
    print(f"[NAIF] Sending to LLM with nearby_names={nearby_names}")
    with memory_lock:
        reply, memory_data = unified_chat(
            user_input, index, messages, memory_data, nearby_names
        )
    messages.append({"role": "assistant", "content": reply})
    socketio.emit('receive_message', {"text": reply, "role": "assistant"})
    if is_voice_enabled():
        speak(reply)


@socketio.on('toggle_mic')
def handle_mic(data):
    global mic_active, stt_active, mic_thread_started
    mic_active = data.get('active', False)
    stt_active = mic_active
    socketio.emit('mic_status', {"active": mic_active})
    socketio.emit('stt_status', {"active": stt_active})
    if not mic_thread_started:
        mic_thread_started = True
        threading.Thread(target=mic_stt_loop, daemon=True).start()


@socketio.on('toggle_stt')
def handle_stt(data):
    global stt_active, mic_thread_started
    stt_active = data.get('active', False)
    socketio.emit('stt_status', {"active": stt_active})
    if not mic_thread_started:
        mic_thread_started = True
        threading.Thread(target=mic_stt_loop, daemon=True).start()


@socketio.on('toggle_tts')
def handle_tts(data):
    set_voice_enabled(data.get('active', True))
    socketio.emit('tts_status', {"active": data.get('active', True)})


@socketio.on('start_register')
def handle_start_register(data):
    global face_register_name, face_register_count, face_register_active
    name = data.get('name', '').strip()
    if not name:
        socketio.emit('face_status', {"status": "Enter a name first."})
        return
    face_register_name   = name
    face_register_count  = 0
    face_register_active = True
    socketio.emit('face_status', {
        "status": f"Capturing for '{name}'... look at camera.",
        "register_count": 0,
        "register_done": False
    })


@socketio.on('get_face_info')
def handle_get_face_info():
    socketio.emit('face_info', {
        "known_people": get_known_people(),
        "model_exists": os.path.exists(FACE_MODEL_PATH)
    })


@socketio.on("delete_person")
def handle_delete_person(data):
    import shutil
    name = data.get("name","").strip()
    person_path = os.path.join(FACE_DB_PATH, name)
    try:
        if os.path.exists(person_path):
            shutil.rmtree(person_path)
        # Retrain without this person
        people = get_known_people()
        if people:
            ok, msg = _train_face_model()
            if ok and os.path.exists(FACE_MODEL_PATH):
                face_recognizer.read(FACE_MODEL_PATH)
        else:
            # No one left — delete model and label map
            for f in [FACE_MODEL_PATH, LABEL_MAP_PATH]:
                if os.path.exists(f): os.remove(f)
            global face_reverse_labels
            face_reverse_labels = {}
            msg = f"Deleted {name}. No persons remain."
        socketio.emit("delete_result", {"success": True, "message": f"Deleted {name}.", "known_people": get_known_people()})
    except Exception as e:
        socketio.emit("delete_result", {"success": False, "message": str(e), "known_people": get_known_people()})


@socketio.on("delete_all_faces")
def handle_delete_all_faces():
    import shutil
    global face_reverse_labels
    try:
        if os.path.exists(FACE_DB_PATH): shutil.rmtree(FACE_DB_PATH)
        os.makedirs(FACE_DB_PATH, exist_ok=True)
        for f in [FACE_MODEL_PATH, LABEL_MAP_PATH, CURRENT_PERSONS_FILE, "current_persons.json"]:
            if os.path.exists(f): os.remove(f)
        face_reverse_labels = {}
        _save_current_persons([])
        socketio.emit("nearby_persons", {"persons": []})
        socketio.emit("delete_result", {"success": True, "message": "All face data deleted.", "known_people": []})
    except Exception as e:
        socketio.emit("delete_result", {"success": False, "message": str(e), "known_people": get_known_people()})


@socketio.on('delete_person')
def handle_delete_person(data):
    import shutil
    global detected_persons, face_reverse_labels
    name = data.get('name', '').strip()
    person_path = os.path.join(FACE_DB_PATH, name)
    try:
        if os.path.exists(person_path):
            shutil.rmtree(person_path)
        # Retrain without this person
        remaining = get_known_people()
        if remaining:
            ok, msg = _train_face_model()
            if ok:
                face_recognizer.read(FACE_MODEL_PATH)
            result_msg = f'Deleted {name}. Model retrained.'
        else:
            # No one left — remove model and label map
            for f in [FACE_MODEL_PATH, LABEL_MAP_PATH]:
                if os.path.exists(f): os.remove(f)
            face_reverse_labels = {}
            result_msg = f'Deleted {name}. No persons remain.'
        detected_persons = [p for p in detected_persons if p != name]
        _save_current_persons(detected_persons)
        socketio.emit('delete_result', {'success': True, 'message': result_msg, 'known_people': get_known_people()})
    except Exception as e:
        socketio.emit('delete_result', {'success': False, 'message': str(e), 'known_people': get_known_people()})


@socketio.on('delete_all_faces')
def handle_delete_all():
    import shutil
    global detected_persons, face_reverse_labels
    try:
        if os.path.exists(FACE_DB_PATH): shutil.rmtree(FACE_DB_PATH)
        os.makedirs(FACE_DB_PATH, exist_ok=True)
        for f in [FACE_MODEL_PATH, LABEL_MAP_PATH, CURRENT_PERSONS_FILE, 'current_persons.json']:
            if os.path.exists(f): os.remove(f)
        face_reverse_labels = {}
        detected_persons = []
        socketio.emit('delete_result', {'success': True, 'message': '✅ All face data deleted.', 'known_people': []})
        socketio.emit('nearby_persons', {'persons': []})
    except Exception as e:
        socketio.emit('delete_result', {'success': False, 'message': str(e), 'known_people': get_known_people()})


@app.route("/")
def index_page():
    return render_template("index.html")


if __name__ == "__main__":
    print("NAIF → http://localhost:6500")
    socketio.run(app, host="0.0.0.0", port=6500, debug=False)
