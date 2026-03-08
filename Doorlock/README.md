# 🔐 Smart AI Door Security System — v2

Full Python door security system running on any PC with a webcam + microphone.

---

## ✨ What's New in v2

| Feature | Detail |
|---|---|
| **Unknown face > 5 s → auto photo** | Timestamped snapshot saved to `alerts/` automatically |
| **Random polite questions** | "Are you waiting for someone?" etc. spoken every ~12 s |
| **✌ TWO FINGERS = unlock** | Only index + middle finger up triggers door unlock |
| **3 wrong gestures = alarm** | UNKNOWN gestures counted; buzzer fires on 3rd |
| **Manual record button** | Press **R** to start/stop recording anytime |
| **Live face enrollment** | Press **T**, type a name → captures 200 images → trains immediately |
| **Alarm buzzer** | `winsound.Beep` on Windows (alternating tones) |

---

## 📁 Folder Structure

```
smart_door_security/
├── faces/
│   ├── john/          ← auto-created during enrollment (200 images)
│   │   ├── 0001_....jpg
│   │   └── ...
│   ├── lbph_model.xml ← trained model (auto-generated)
│   └── label_map.pkl  ← name→label map (auto-generated)
├── alerts/            ← intruder snapshots + visitor photos saved here
├── recordings/        ← manual recordings saved here
├── main.py
├── face_auth.py
├── voice.py
├── gesture.py
├── server.py
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python main.py
```

---

## 🎮 Controls

### Keyboard (in camera window)
| Key | Action |
|-----|--------|
| **T** | Enroll new face — type name in terminal, look at camera |
| **R** | Toggle manual video recording on/off |
| **Q** | Quit |
| **U** | Unlock door |
| **L** | Lock door |

### Gestures (show to camera)
| Gesture | Action |
|---------|--------|
| ✌ **Two Fingers** (index + middle up) | Unlock door |
| ✊ **Fist** | Lock door |
| 👍 **Thumbs Up** | Confirm / OK |
| ❓ **Any other gesture × 3** | Triggers alarm buzzer |

---

## 🔄 System Flow

```
Webcam Frame
  │
  ├─► Gesture Detection
  │     ├─► ✌ TWO FINGERS  → Unlock door
  │     ├─► ✊ FIST         → Lock door
  │     └─► UNKNOWN × 3    → 🔔 Alarm buzzer
  │
  └─► Face Detection
        ├─► Authorized face   → "Welcome [name]" + Unlock
        └─► Unknown face
              ├─► First seen  → Start 5-second timer
              ├─► After 5 s   → 📸 Auto-capture timestamped photo
              │                  💬 Ask polite question every 12 s
              └─► Voice auth prompt
                    ├─► Correct password → Unlock
                    └─► Wrong × 3        → 🚨 Alert + alarm
```

---

## 🧑 Enrolling a New Face

1. Run the system: `python main.py`
2. Press **T** in the camera window
3. Type the person's name in the terminal and press Enter
4. Look straight at the camera — system captures **200 photos automatically**
5. Model trains immediately after capture — person is authorized right away

No restart needed. Works with both `face_recognition` and LBPH fallback.

---

## 📷 Alert Photos

All photos saved to `alerts/` are timestamped:
- `unknown_visitor_20260305_131200.jpg` — unknown person lingering > 5 s
- `intruder_20260305_131500.jpg` — failed password × 3
- `gesture_alarm_20260305_132000.jpg` — wrong gesture alarm

---

## ⚙️ Key Settings (`main.py` CONFIG)

```python
"voice_password":      "open sesame",  # change your secret phrase
"unknown_linger_sec":  5,              # seconds before auto-photo + question
"max_failed_attempts": 3,              # wrong passwords before alert
"record_on_motion":    False,          # True = auto-record; False = manual R key
"gesture_enabled":     True,           # False = disable MediaPipe
```
