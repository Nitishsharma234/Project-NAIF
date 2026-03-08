"""
naif_server.py — NAIF Flask-SocketIO server.

Handles:
- Chat (send_message / receive_message)
- Camera frames (camera_frame)
- Face register / delete
- nearby_persons pushed every second from background thread
- Mic / STT / TTS toggles
"""

import os
import time
import shutil
import threading
import base64

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from face_system import NAIFFaceSystem
from prior import unified_chat, load_index, load_memory_embeddings, load_current_persons

# ── App setup ──────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "naif-secret"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── NAIF state ─────────────────────────────────────────────────────────
index       = load_index()
memory_data = load_memory_embeddings()
messages    = [{"role": "system", "content": "You are NAIF, an AI assistant created by Nitish Sharma."}]

# ── Face system ────────────────────────────────────────────────────────
face_ai = NAIFFaceSystem()
face_ai.start_camera()
face_ai.start_recognize()   # start recognizing immediately on boot

# ── Control flags ──────────────────────────────────────────────────────
mic_on  = False
stt_on  = False
tts_on  = True


# ═══════════════════════════════════════════════════════════════════════
# BACKGROUND THREADS
# ═══════════════════════════════════════════════════════════════════════

def camera_push_thread():
    """Push JPEG camera frames to all clients over Socket.IO."""
    while True:
        frame = face_ai.get_frame_bytes()
        if frame:
            socketio.emit("camera_frame", frame)
        time.sleep(0.04)   # ~25 fps


def nearby_push_thread():
    """
    Push nearby_persons to all clients every second.
    Reads directly from the live face_ai object — most reliable source.
    Also writes current_persons.json so prior.py always has fresh data.
    """
    last_sent = None
    while True:
        time.sleep(1)
        try:
            if face_ai.mode == "recognize" and face_ai.detected_persons is not None:
                persons = [n for n in face_ai.detected_persons if n != "Unknown"]
            else:
                persons = load_current_persons()

            # Always emit — even if unchanged — so new clients catch up
            # Only log when it changes to avoid spam
            if persons != last_sent:
                print(f"[Nearby] {persons}")
                last_sent = persons

            socketio.emit("nearby_persons", {"persons": persons})

            # Keep JSON file fresh for prior.py
            face_ai._save_current_persons(persons)

        except Exception as e:
            print(f"[Nearby thread] Error: {e}")


# ═══════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")


# ═══════════════════════════════════════════════════════════════════════
# SOCKET.IO EVENTS
# ═══════════════════════════════════════════════════════════════════════

@socketio.on("connect")
def on_connect():
    print("[Socket] Client connected")
    # Send face info immediately so UI shows known people on load
    emit("face_info", {
        "known_people": face_ai.get_known_people()
    })
    # Send current persons immediately
    if face_ai.mode == "recognize":
        persons = [n for n in face_ai.detected_persons if n != "Unknown"]
    else:
        persons = load_current_persons()
    emit("nearby_persons", {"persons": persons})


@socketio.on("disconnect")
def on_disconnect():
    print("[Socket] Client disconnected")


# ── CHAT ───────────────────────────────────────────────────────────────

@socketio.on("send_message")
def on_send_message(data):
    global memory_data

    user_input = (data.get("text") or "").strip()
    if not user_input:
        return

    # Echo user message back so UI renders it
    emit("receive_message", {"text": user_input, "role": "user"})

    # Get current detected persons
    if face_ai.mode == "recognize" and face_ai.detected_persons is not None:
        nearby_names = [n for n in face_ai.detected_persons if n != "Unknown"]
    else:
        nearby_names = load_current_persons()

    memory_data = load_memory_embeddings()

    try:
        reply, memory_data = unified_chat(
            user_input,
            index,
            messages,
            memory_data,
            nearby_names
        )
    except Exception as e:
        print(f"[Chat] Error: {e}")
        reply = "System error. Please try again."

    emit("receive_message", {"text": reply, "role": "naif"})


# ── FACE REGISTER ──────────────────────────────────────────────────────

@socketio.on("start_register")
def on_start_register(data):
    name = (data.get("name") or "").strip()
    if not name:
        emit("face_status", {"status": "⚠ Enter a name first."})
        return

    def _register_and_report():
        face_ai.start_register(name)

        # Poll until done
        while face_ai.mode == "register":
            count = face_ai.register_count
            socketio.emit("face_status", {
                "register_count": count,
                "status": f"Capturing '{name}'... {count}/300"
            })
            time.sleep(0.3)

        # Training happens in background inside face_system — wait for it
        time.sleep(1.5)
        socketio.emit("face_status", {
            "status": f"✅ {name} registered and trained.",
            "register_done": True,
            "known_people": face_ai.get_known_people()
        })
        # Restart recognition with new model
        face_ai.start_recognize()

    threading.Thread(target=_register_and_report, daemon=True).start()


@socketio.on("get_face_info")
def on_get_face_info():
    emit("face_info", {"known_people": face_ai.get_known_people()})


# ── DELETE PERSON ──────────────────────────────────────────────────────

@socketio.on("delete_person")
def on_delete_person(data):
    name = (data.get("name") or "").strip()
    if not name:
        emit("delete_result", {"success": False, "message": "No name provided.", "known_people": face_ai.get_known_people()})
        return

    person_path = os.path.join(face_ai.db_path, name)
    if not os.path.exists(person_path):
        emit("delete_result", {"success": False, "message": f"{name} not found.", "known_people": face_ai.get_known_people()})
        return

    shutil.rmtree(person_path)

    # Remove from detected list
    face_ai.detected_persons = [n for n in face_ai.detected_persons if n != name]
    face_ai._save_current_persons(face_ai.detected_persons)

    # Retrain
    remaining = face_ai.get_known_people()
    if remaining:
        face_ai.train_model()
        face_ai.start_recognize()
    else:
        # No one left — clear model
        if os.path.exists(face_ai.model_path):
            os.remove(face_ai.model_path)
        face_ai.mode = "idle"

    emit("delete_result", {
        "success": True,
        "message": f"✅ {name} deleted.",
        "known_people": face_ai.get_known_people()
    })


@socketio.on("delete_all_faces")
def on_delete_all_faces():
    # Remove all person folders
    for entry in os.listdir(face_ai.db_path):
        full = os.path.join(face_ai.db_path, entry)
        if os.path.isdir(full):
            shutil.rmtree(full)

    # Remove model
    if os.path.exists(face_ai.model_path):
        os.remove(face_ai.model_path)

    face_ai.detected_persons = []
    face_ai._save_current_persons([])
    face_ai.labels = {}
    face_ai.reverse_labels = {}
    face_ai.mode = "idle"

    emit("delete_result", {
        "success": True,
        "message": "✅ All face data deleted.",
        "known_people": []
    })


# ── MIC / STT / TTS ────────────────────────────────────────────────────

@socketio.on("toggle_mic")
def on_toggle_mic(data):
    global mic_on, stt_on
    mic_on = data.get("active", False)
    if not mic_on:
        stt_on = False
    emit("mic_status", {"active": mic_on})

@socketio.on("toggle_stt")
def on_toggle_stt(data):
    global stt_on
    stt_on = data.get("active", False)
    emit("stt_status", {"active": stt_on})

@socketio.on("toggle_tts")
def on_toggle_tts(data):
    global tts_on
    tts_on = data.get("active", False)
    emit("tts_status", {"active": tts_on})


# ═══════════════════════════════════════════════════════════════════════
# START
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Starting NAIF server...")
    print("Face system: ON")
    print("Recognition: AUTO-START")

    # Start background threads
    threading.Thread(target=camera_push_thread, daemon=True).start()
    threading.Thread(target=nearby_push_thread, daemon=True).start()

    socketio.run(
        app,
        host="0.0.0.0",
        port=6500,
        debug=False,        # NEVER True — kills background threads
        use_reloader=False  # NEVER True — same reason
    )
