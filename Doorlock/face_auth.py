"""
face_auth.py  —  Pure OpenCV Face Recognition  (NO dlib / NO cmake)
=====================================================================
Method:
  Detection  : Haar Cascade (haarcascade_frontalface_default.xml)
               + LBP Cascade (lbpcascade_frontalface_improved.xml) fallback
  Recognition: OpenCV LBPH (Local Binary Pattern Histogram) Recogniser
               cv2.face.LBPHFaceRecognizer  — ships inside opencv-contrib-python

Enrollment flow:
  1. Press T in main window → type person name
  2. Camera captures 300 live face images automatically
  3. Each image is saved to  faces/<name>/
  4. Data augmentation (flip, brightness, contrast) triples the dataset
  5. LBPH model is trained immediately on ALL enrolled people
  6. Model saved to  faces/lbph_model.xml  (auto-loaded on next run)

Prediction output on screen:
  ✅ GREEN box  →  "John  (conf:42)"   authorised, confidence shown
  ❌ RED box    →  "UNKNOWN (conf:95)" unrecognised
  Confidence:  LBPH lower = better.  < 65 = strong match, < 80 = accepted

Install (no cmake/dlib needed):
  pip install opencv-python opencv-contrib-python numpy
"""

import os, cv2, pickle, time
import numpy as np
from datetime import datetime


# ════════════════════════════════════════════════════════════════
class FaceAuthenticator:

    ENROLL_TARGET  = 300          # live frames to capture per person
    FACE_IMG_SIZE  = (200, 200)   # all faces resized to this before training
    CONF_THRESHOLD = 80           # LBPH confidence below this → authorised
    SCAN_INTERVAL  = 3            # seconds between recognition scans

    def __init__(self, faces_dir="faces"):
        self.faces_dir        = faces_dir
        self._model_path      = os.path.join(faces_dir, "lbph_model.xml")
        self._labelmap_path   = os.path.join(faces_dir, "label_map.pkl")
        self._lbph            = None
        self._label_map       = {}   # {int_label: "name"}
        self._next_label      = 0
        self._trained         = False

        os.makedirs(faces_dir, exist_ok=True)

        # ── Face detector (Haar cascade, pure OpenCV) ─────────────────────────
        haar = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(haar)
        if self.detector.empty():
            raise RuntimeError("Haar cascade XML not found — reinstall opencv-python")

        self._load_model()

    # ─────────────────────────────────────────────────────────────────────────
    # Model persistence
    # ─────────────────────────────────────────────────────────────────────────
    def _load_model(self):
        """Load saved LBPH model + label map if they exist."""
        if os.path.exists(self._model_path) and os.path.exists(self._labelmap_path):
            try:
                self._lbph = cv2.face.LBPHFaceRecognizer_create()
                self._lbph.read(self._model_path)
                with open(self._labelmap_path, "rb") as f:
                    self._label_map = pickle.load(f)
                self._next_label = max(self._label_map.keys(), default=-1) + 1
                self._trained    = True
                people = list(self._label_map.values())
                print(f"[FACE] ✅ LBPH model loaded.  Authorised people: {people}")
            except Exception as e:
                print(f"[FACE] ⚠ Could not load model: {e}  — will retrain on next enroll.")
                self._lbph   = None
                self._trained = False
        else:
            print("[FACE] No model yet. Press T to enroll a face.")

    def _save_model(self):
        """Persist trained LBPH model + label map to disk."""
        self._lbph.save(self._model_path)
        with open(self._labelmap_path, "wb") as f:
            pickle.dump(self._label_map, f)
        print(f"[FACE] 💾 Model saved → {self._model_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Training (LBPH)
    # ─────────────────────────────────────────────────────────────────────────
    def _collect_all_training_data(self):
        """
        Walk faces/<name>/ subfolders and collect ALL saved images + labels.
        Returns (images: list[np.ndarray], labels: list[int])
        """
        images, labels = [], []
        supported = (".jpg", ".jpeg", ".png")

        for label, name in self._label_map.items():
            folder = os.path.join(self.faces_dir, name)
            if not os.path.isdir(folder):
                continue
            loaded = 0
            for fname in os.listdir(folder):
                if not fname.lower().endswith(supported):
                    continue
                path = os.path.join(folder, fname)
                img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, self.FACE_IMG_SIZE)
                images.append(img)
                labels.append(label)
                loaded += 1
            print(f"  [TRAIN] Loaded {loaded} images for '{name}'")

        return images, labels

    def _augment(self, gray_face):
        """
        Return a list of augmented versions of one grayscale face crop.
        Augmentations: horizontal flip, 2 brightness levels, 2 contrast levels.
        Gives 6× more training data from each real capture.
        """
        variants = [gray_face]
        # Flip
        variants.append(cv2.flip(gray_face, 1))
        # Brightness ±30
        for delta in (+30, -30):
            variants.append(cv2.convertScaleAbs(gray_face, alpha=1.0, beta=delta))
        # Contrast ×1.3 and ×0.8
        for alpha in (1.3, 0.8):
            variants.append(cv2.convertScaleAbs(gray_face, alpha=alpha, beta=0))
        return variants

    def train_model(self, new_images=None, new_labels=None):
        """
        (Re-)train LBPH on all enrolled data.
        Optionally pass freshly captured images/labels to skip disk reload.
        """
        print("\n[TRAIN] ═══════════════════════════════")
        print("[TRAIN]  Training LBPH recogniser…")
        print("[TRAIN] ═══════════════════════════════")

        # Collect everything from disk
        all_imgs, all_lbls = self._collect_all_training_data()

        # Add newly captured images passed directly (avoids re-reading from disk)
        if new_images and new_labels:
            all_imgs.extend(new_images)
            all_lbls.extend(new_labels)

        if not all_imgs:
            print("[TRAIN] ⚠ No training data found.")
            return False

        # Apply augmentation
        aug_imgs, aug_lbls = [], []
        for img, lbl in zip(all_imgs, all_lbls):
            for variant in self._augment(img):
                aug_imgs.append(variant)
                aug_lbls.append(lbl)

        print(f"[TRAIN] Real images : {len(all_imgs)}")
        print(f"[TRAIN] After augment: {len(aug_imgs)} (×{len(aug_imgs)//max(len(all_imgs),1)})")

        self._lbph = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=8, grid_x=8, grid_y=8
        )
        self._lbph.train(aug_imgs, np.array(aug_lbls, dtype=np.int32))
        self._trained = True
        self._save_model()

        people = list(self._label_map.values())
        print(f"[TRAIN] ✅ Done!  Model knows: {people}\n")
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # LIVE ENROLLMENT  (call from main loop — returns generator for progress)
    # ─────────────────────────────────────────────────────────────────────────
    def enroll_live(self, cap, name, window_name="Smart Door Security System — Press Q to Quit"):
        """
        Capture ENROLL_TARGET face images live from `cap`.
        This is a GENERATOR — yields progress dicts so the caller
        can update the display on each frame without blocking.

        Usage (in main loop):
            for prog in face_auth.enroll_live(cap, "Alice"):
                # prog = {"count":N, "total":300, "done":False, "cancelled":False}
                show_frame(prog["frame"])
                if cv2.waitKey(1) & 0xFF == 27:
                    prog["cancel"] = True   # set flag to abort

        After the generator exhausts, training has already completed.
        """
        save_dir = os.path.join(self.faces_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        # Reserve a label for this person
        if name in self._label_map.values():
            label = next(k for k, v in self._label_map.items() if v == name)
            print(f"[ENROLL] Re-enrolling existing person '{name}' (label {label})")
        else:
            label = self._next_label
            self._label_map[label] = name
            self._next_label += 1

        count      = 0
        cancel_ref = {"flag": False}
        new_imgs   = []
        new_lbls   = []

        print(f"\n[ENROLL] ══════════════════════════════════════")
        print(f"[ENROLL]  Enrolling: {name.upper()}")
        print(f"[ENROLL]  Target   : {self.ENROLL_TARGET} images")
        print(f"[ENROLL]  Tips     : face the camera, move slightly")
        print(f"[ENROLL]  ESC      : cancel")
        print(f"[ENROLL] ══════════════════════════════════════\n")

        while count < self.ENROLL_TARGET:
            ret, raw = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.flip(raw, 1)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray  = cv2.equalizeHist(gray)          # improve contrast

            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=6,
                minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE
            )

            face_found = len(faces) > 0
            for (x, y, w, h) in faces[:1]:          # one face per frame
                # Pad the crop slightly for better context
                pad  = int(w * 0.12)
                x1   = max(0, x - pad)
                y1   = max(0, y - pad)
                x2   = min(frame.shape[1], x + w + pad)
                y2   = min(frame.shape[0], y + h + pad)
                crop = gray[y1:y2, x1:x2]
                face_resized = cv2.resize(crop, self.FACE_IMG_SIZE)

                # Save raw image
                ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                fpath = os.path.join(save_dir, f"{count+1:04d}_{ts}.jpg")
                cv2.imwrite(fpath, face_resized)

                new_imgs.append(face_resized)
                new_lbls.append(label)
                count += 1

                # ── Draw on frame ─────────────────────────────────────────────
                pct    = count / self.ENROLL_TARGET
                bar_w  = int((frame.shape[1] - 20) * pct)
                bar_y1, bar_y2 = frame.shape[0] - 30, frame.shape[0] - 10

                # Face box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
                # Progress bar background
                cv2.rectangle(frame, (10, bar_y1), (frame.shape[1]-10, bar_y2), (40,40,40), -1)
                # Progress bar fill
                bar_color = (0, int(180*pct), int(255*(1-pct)))
                cv2.rectangle(frame, (10, bar_y1), (10 + bar_w, bar_y2), bar_color, -1)
                # Labels
                cv2.putText(frame,
                    f"ENROLLING: {name.upper()}   {count}/{self.ENROLL_TARGET}  ({int(pct*100)}%)",
                    (10, bar_y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 180), 2)

                # Milestone announcements
                if count in (100, 200, 250, 300):
                    print(f"[ENROLL] {count} images captured…")
                break

            # Header overlay
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
            cv2.putText(frame, "FACE ENROLLMENT MODE  |  ESC = cancel",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 255), 2)
            if not face_found:
                cv2.putText(frame, "⚠ No face detected — move closer / improve lighting",
                            (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)

            cv2.imshow(window_name, frame)

            prog = {
                "frame":     frame,
                "count":     count,
                "total":     self.ENROLL_TARGET,
                "done":      False,
                "cancelled": False,
                "cancel":    False,      # caller sets this to True to abort
            }
            yield prog

            if prog["cancel"] or (cv2.waitKey(1) & 0xFF == 27):
                print("[ENROLL] Cancelled.")
                yield {"done": True, "cancelled": True, "count": count,
                       "total": self.ENROLL_TARGET, "frame": frame, "cancel": False}
                return

        # ── Capture complete → train ──────────────────────────────────────────
        print(f"\n[ENROLL] ✅ Captured {count} images for '{name}'")
        print("[ENROLL] Training model — please wait…")

        # Show "Training…" screen
        training_frame = np.zeros((200, 520, 3), dtype=np.uint8)
        cv2.putText(training_frame, "Training model — please wait…",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 210, 255), 2)
        cv2.putText(training_frame, f"Enrolled: {name}",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
        cv2.imshow(window_name, training_frame)
        cv2.waitKey(1)

        self.train_model(new_images=new_imgs, new_labels=new_lbls)

        yield {"done": True, "cancelled": False, "count": count,
               "total": self.ENROLL_TARGET, "frame": training_frame, "cancel": False}

    # ─────────────────────────────────────────────────────────────────────────
    # Detection helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_raw(self, gray):
        """Return list of (x, y, w, h) from Haar cascade."""
        return self.detector.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=6,
            minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Identification
    # ─────────────────────────────────────────────────────────────────────────
    def identify_faces(self, frame):
        """
        Detect and recognise all faces in `frame`.

        Returns list of dicts:
          {
            "name":       str,             # recognised name or "Unknown"
            "confidence": float,           # LBPH confidence (lower = better)
            "authorized": bool,
            "location":   (top,right,bottom,left)  # compatible with draw_faces
          }
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        raw  = self._detect_raw(gray)
        results = []

        for (x, y, w, h) in raw:
            # Convert (x,y,w,h) → (top,right,bottom,left) for location tuple
            top, right, bottom, left = y, x+w, y+h, x
            loc = (top, right, bottom, left)

            if not self._trained or self._lbph is None:
                results.append({
                    "name": "Unknown", "confidence": 999.0,
                    "authorized": False, "location": loc
                })
                continue

            face_crop    = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, self.FACE_IMG_SIZE)

            try:
                label, conf = self._lbph.predict(face_resized)
            except cv2.error:
                results.append({
                    "name": "Unknown", "confidence": 999.0,
                    "authorized": False, "location": loc
                })
                continue

            if conf < self.CONF_THRESHOLD and label in self._label_map:
                name = self._label_map[label]
                auth = True
            else:
                name = "Unknown"
                auth = False

            results.append({
                "name": name, "confidence": round(conf, 1),
                "authorized": auth, "location": loc
            })

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Drawing
    # ─────────────────────────────────────────────────────────────────────────
    def draw_faces(self, frame, face_results):
        """
        Draw bounding boxes + name + confidence on `frame`.
        Green = authorised, Red = unknown.
        """
        for f in face_results:
            top, right, bottom, left = f["location"]
            auth  = f["authorized"]
            name  = f["name"]
            conf  = f["confidence"]
            color = (0, 210, 0) if auth else (0, 0, 210)

            # Bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Label background
            label_h = 34
            cv2.rectangle(frame, (left, bottom), (right, bottom + label_h), color, -1)

            # Name + confidence
            if auth:
                label_text = f"{name}  conf:{conf:.0f}"
            else:
                label_text = f"UNKNOWN  conf:{conf:.0f}"

            cv2.putText(frame, label_text,
                        (left + 5, bottom + 22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55,
                        (255, 255, 255), 1)

            # Confidence bar inside box
            bar_range = right - left
            bar_fill  = int(bar_range * max(0, 1 - conf / 100))
            cv2.rectangle(frame, (left, top), (left + bar_fill, top + 5), color, -1)

        return frame

    # ─────────────────────────────────────────────────────────────────────────
    # Snapshot utility
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def capture_snapshot(frame, save_dir="alerts", prefix="visitor"):
        os.makedirs(save_dir, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f"{prefix}_{ts}.jpg")
        cv2.imwrite(path, frame)
        print(f"[SNAP] 📸 Saved: {path}")
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────
    def list_enrolled(self):
        return list(self._label_map.values())

    def is_trained(self):
        return self._trained
