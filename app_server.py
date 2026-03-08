from flask import Flask, Response, request, jsonify, render_template_string
import time
import os
from face_system import NAIFFaceSystem

app = Flask(__name__)
face_ai = NAIFFaceSystem()

# ── Load HTML ──────────────────────────────────────────────────────────────────
with open("index.html", "r", encoding="utf-8") as f:
    HTML_PAGE = f.read()


# ── MJPEG stream generator ─────────────────────────────────────────────────────
def generate_stream():
    while True:
        frame_bytes = face_ai.get_frame_bytes()
        if frame_bytes is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
        )
        time.sleep(0.033)  # ~30fps


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/start_camera", methods=["POST"])
def start_camera():
    face_ai.start_camera()
    return jsonify({"ok": True})


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    face_ai.stop_camera()
    return jsonify({"ok": True})


@app.route("/start_register", methods=["POST"])
def start_register():
    data = request.json
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "message": "Name is required."})
    face_ai.start_register(name)
    return jsonify({"ok": True, "message": f"Capturing for '{name}'..."})


@app.route("/start_recognize", methods=["POST"])
def start_recognize():
    face_ai.start_recognize()
    return jsonify({"ok": True, "message": face_ai.status_message})


@app.route("/stop_action", methods=["POST"])
def stop_action():
    face_ai.mode = "idle"
    face_ai.status_message = "Stopped."
    return jsonify({"ok": True})


@app.route("/pause_camera", methods=["POST"])
def pause_camera():
    face_ai._paused_mode = face_ai.mode   # remember current mode
    face_ai.camera_paused = True          # blocks core camera loop
    face_ai.mode = "idle"
    face_ai.status_message = "Camera paused."
    return jsonify({"ok": True})


@app.route("/resume_camera", methods=["POST"])
def resume_camera():
    face_ai.camera_paused = False         # unblocks core camera loop
    prev = getattr(face_ai, "_paused_mode", "idle")
    face_ai.mode = prev
    face_ai._paused_mode = None
    face_ai.status_message = "Camera resumed."
    return jsonify({"ok": True})


@app.route("/status")
def status():
    """Polled every second by browser for live updates."""
    return jsonify({
        "mode": face_ai.mode,
        "status": face_ai.status_message,
        "register_count": face_ai.register_count,
        "register_target": face_ai.register_target,
        "register_done": face_ai.register_done,
        "detected_persons": face_ai.detected_persons,
        "known_people": face_ai.get_known_people(),
    })


if __name__ == "__main__":
    face_ai.start_camera()          # open camera on server start
    print("NAIF Face Server running → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)