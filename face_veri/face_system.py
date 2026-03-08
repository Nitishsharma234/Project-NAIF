import cv2
import os
import numpy as np
import json
import threading
import time

CURRENT_PERSONS_FILE = "current_persons.json"


class NAIFFaceSystem:
    def __init__(self, db_path="face_db", model_path="face_model.yml"):
        self.db_path = db_path
        self.model_path = model_path
        os.makedirs(self.db_path, exist_ok=True)

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # Requires: pip install opencv-contrib-python
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)

        self.labels = {}
        self.reverse_labels = {}

        self._lock = threading.Lock()
        self._current_frame = None       # latest raw frame from camera
        self._output_frame = None        # latest annotated frame for streaming
        self._cap = None
        self._thread = None
        self._running = False

        # State machine
        self.mode = "idle"               # "idle" | "register" | "recognize"
        self.register_name = ""
        self.register_count = 0
        self.register_target = 300
        self.register_done = False
        self.status_message = "System ready."
        self.detected_persons = []
        self.camera_paused = False       # True = freeze capture + recognition

        if os.path.exists(self.model_path):
            self.recognizer.read(self.model_path)
            self._generate_labels()

    # ─────────────────────────────────────────
    # LABEL MANAGEMENT
    # ─────────────────────────────────────────

    def _generate_labels(self):
        persons = sorted([
            p for p in os.listdir(self.db_path)
            if os.path.isdir(os.path.join(self.db_path, p))
        ])
        self.labels = {name: idx for idx, name in enumerate(persons)}
        self.reverse_labels = {v: k for k, v in self.labels.items()}

    def get_known_people(self):
        return sorted([
            p for p in os.listdir(self.db_path)
            if os.path.isdir(os.path.join(self.db_path, p))
        ])

    # ─────────────────────────────────────────
    # CAMERA THREAD
    # ─────────────────────────────────────────

    def start_camera(self):
        if self._running:
            return
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self.status_message = "ERROR: Cannot open camera."
            return
        self._running = True
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

    def stop_camera(self):
        self._running = False
        self.mode = "idle"
        time.sleep(0.3)
        if self._cap:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._output_frame = None

    def _camera_loop(self):
        while self._running:
            if not self._cap or not self._cap.isOpened():
                break

            # ── CORE PAUSE: skip read + processing entirely ──
            if self.camera_paused:
                time.sleep(0.1)
                continue

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)  # mirror view

            if self.mode == "register":
                frame = self._handle_register(frame)
            elif self.mode == "recognize":
                frame = self._handle_recognize(frame)
            else:
                cv2.putText(frame, "NAIF Face System - Idle",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 200, 255), 2)

            with self._lock:
                self._output_frame = frame.copy()

            time.sleep(0.03)  # ~30fps

    # ─────────────────────────────────────────
    # REGISTER MODE
    # ─────────────────────────────────────────

    def start_register(self, name):
        person_path = os.path.join(self.db_path, name)
        os.makedirs(person_path, exist_ok=True)
        self.register_name = name
        self.register_count = 0
        self.register_done = False
        self.status_message = f"Capturing samples for '{name}'..."
        self.mode = "register"

    def _handle_register(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalisation so lighting variation is captured across 300 frames
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 6)

        for (x, y, w, h) in faces:
            if self.register_count < self.register_target:
                face = gray[y:y+h, x:x+w]
                face = cv2.equalizeHist(cv2.resize(face, (200, 200)))
                person_path = os.path.join(self.db_path, self.register_name)
                cv2.imwrite(
                    os.path.join(person_path, f"{self.register_count}.jpg"),
                    face
                )
                self.register_count += 1

            color = (0, 255, 0) if self.register_count < self.register_target else (0, 200, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Progress bar
        pct = self.register_count / self.register_target
        bar_w = int(frame.shape[1] * pct)
        cv2.rectangle(frame, (0, frame.shape[0]-20),
                      (bar_w, frame.shape[0]), (0, 255, 100), -1)
        cv2.rectangle(frame, (0, frame.shape[0]-20),
                      (frame.shape[1], frame.shape[0]), (60, 60, 60), 1)

        label = f"Capturing '{self.register_name}': {self.register_count}/{self.register_target}"
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        if self.register_count >= self.register_target and not self.register_done:
            self.register_done = True
            self.mode = "idle"
            self.status_message = f"Capture complete for '{self.register_name}'! Now training..."
            # Train in background so camera doesn't freeze
            t = threading.Thread(target=self._train_background, daemon=True)
            t.start()

        return frame

    # ─────────────────────────────────────────
    # TRAIN MODEL
    # ─────────────────────────────────────────

    def _train_background(self):
        success, msg = self.train_model()
        self.status_message = msg

    def train_model(self):
        faces, labels = [], []
        self._generate_labels()

        for name, label in self.labels.items():
            person_path = os.path.join(self.db_path, name)
            for img_name in os.listdir(person_path):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img = cv2.imread(
                    os.path.join(person_path, img_name),
                    cv2.IMREAD_GRAYSCALE
                )
                if img is not None:
                    faces.append(cv2.resize(img, (200, 200)))
                    labels.append(label)

        if faces:
            self.recognizer.train(faces, np.array(labels, dtype=np.int32))
            self.recognizer.save(self.model_path)
            return True, f"✅ Trained on {len(faces)} images for {len(self.labels)} person(s). Ready!"
        return False, "⚠️ No images found to train."

    # ─────────────────────────────────────────
    # RECOGNIZE MODE
    # ─────────────────────────────────────────

    def start_recognize(self):
        if not os.path.exists(self.model_path):
            self.status_message = "⚠️ No model found. Register someone first."
            return
        if not self.reverse_labels:
            self._generate_labels()
        self.detected_persons = []
        self.mode = "recognize"
        self.status_message = "Recognizing faces..."

    def _handle_recognize(self, frame, confidence_threshold=75):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Equalise before recognition so it matches training conditions
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 6)

        found_names = []

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            try:
                label, confidence = self.recognizer.predict(face)
                name = (self.reverse_labels.get(label, "Unknown")
                        if confidence < confidence_threshold else "Unknown")
            except Exception:
                name = "Unknown"
                confidence = 999.0

            found_names.append(name)

            color = (0, 255, 80) if name != "Unknown" else (0, 60, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Update detected persons + write JSON for prior.py
        unique = list(set(found_names))
        if unique != self.detected_persons:
            self.detected_persons = unique
            self._save_current_persons(unique)
            if unique:
                self.status_message = "Detected: " + ", ".join(unique)

        cv2.putText(frame, "NAIF Recognition | press Stop to exit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return frame

    # ─────────────────────────────────────────
    # JSON FOR PRIOR.PY
    # ─────────────────────────────────────────

    def _save_current_persons(self, names):
        with open(CURRENT_PERSONS_FILE, "w") as f:
            json.dump({"persons": names}, f)

    def load_current_persons(self):
        try:
            with open(CURRENT_PERSONS_FILE, "r") as f:
                return json.load(f).get("persons", [])
        except Exception:
            return []

    # ─────────────────────────────────────────
    # FRAME GENERATOR (for Flask MJPEG stream)
    # ─────────────────────────────────────────

    def get_frame_bytes(self):
        with self._lock:
            frame = self._output_frame
        if frame is None:
            return None
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes()