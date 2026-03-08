"""
voice.py - Voice Authentication & Text-to-Speech
==================================================
  • TTS  : pyttsx3 (fully offline)
  • ASR  : SpeechRecognition + Google (online) or Vosk (offline)
  • Fallback when no mic/SR: clean console prompt that doesn't
    fight with Flask log output
"""

import threading, time

# ── TTS ──────────────────────────────────────────────────────────────────────
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[WARNING] pyttsx3 not installed. TTS will print only.")

# ── SpeechRecognition ────────────────────────────────────────────────────────
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("[WARNING] SpeechRecognition not installed. Using keyboard fallback.")

# ── Vosk (fully offline ASR, optional) ───────────────────────────────────────
try:
    from vosk import Model, KaldiRecognizer
    import json as _json
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False


class VoiceEngine:
    def __init__(self, password="Activate Jarvis", vosk_model_path=None):
        self.password        = password.lower().strip()
        self.vosk_model_path = vosk_model_path
        self._tts            = None
        self._tts_lock       = threading.Lock()
        self._use_vosk       = False
        self._init_tts()
        self._init_asr()

    # ── TTS init ─────────────────────────────────────────────────────────────
    def _init_tts(self):
        if not TTS_AVAILABLE:
            return
        try:
            self._tts = pyttsx3.init()
            self._tts.setProperty("rate",   155)
            self._tts.setProperty("volume", 0.95)
            # Prefer a clear English voice
            for v in self._tts.getProperty("voices"):
                if "english" in v.name.lower() or "en_" in v.id.lower():
                    self._tts.setProperty("voice", v.id)
                    break
            print("[INFO] TTS engine initialised.")
        except Exception as e:
            print(f"[WARNING] TTS init failed: {e}")
            self._tts = None

    # ── ASR init ─────────────────────────────────────────────────────────────
    def _init_asr(self):
        if VOSK_AVAILABLE and self.vosk_model_path:
            try:
                self._vosk_model = Model(self.vosk_model_path)
                self._use_vosk   = True
                print("[INFO] Vosk offline ASR loaded.")
                return
            except Exception as e:
                print(f"[WARNING] Vosk load failed: {e}")
        if SR_AVAILABLE:
            self._rec = sr.Recognizer()
            self._rec.pause_threshold  = 0.8
            self._rec.energy_threshold = 300
            print("[INFO] SpeechRecognition (Google) initialised.")

    # ── Speak (blocking) ─────────────────────────────────────────────────────
    def speak(self, text):
        print(f"[TTS] 🔊 {text}")
        if self._tts:
            with self._tts_lock:
                try:
                    self._tts.say(text)
                    self._tts.runAndWait()
                except Exception as e:
                    print(f"[WARNING] TTS error: {e}")

    # ── Speak (non-blocking) ─────────────────────────────────────────────────
    def speak_async(self, text):
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()

    # ── Listen ───────────────────────────────────────────────────────────────
    def listen(self, timeout=9, phrase_limit=6):
        """
        Returns recognised text (lowercase str) or None.
        • If SpeechRecognition is available: uses microphone.
        • Otherwise: opens a clean console prompt with a countdown timer.
        """
        if not SR_AVAILABLE:
            return self._keyboard_fallback(timeout)

        try:
            with sr.Microphone() as src:
                print("[ASR] Adjusting for noise…")
                self._rec.adjust_for_ambient_noise(src, duration=0.6)
                print("[ASR] 🎤 Listening…")
                audio = self._rec.listen(src,
                                         timeout=timeout,
                                         phrase_time_limit=phrase_limit)
        except sr.WaitTimeoutError:
            print("[ASR] No speech detected.")
            return None
        except Exception as e:
            print(f"[ASR] Mic error: {e}")
            return self._keyboard_fallback(timeout)

        return self._transcribe_vosk(audio) if self._use_vosk \
               else self._transcribe_google(audio)

    # ── Keyboard fallback — clean, timed ─────────────────────────────────────
    def _keyboard_fallback(self, timeout):
        """
        Prints a clear prompt AFTER a short delay so Flask log lines
        don't overwrite it, then waits up to `timeout` seconds.
        """
        time.sleep(0.3)                        # let any in-flight prints finish
        print("\n" + "─"*50)
        print(f"  🔑 TYPE PASSWORD  (you have {timeout}s, then Enter)")
        print("─"*50)

        result    = [None]
        done_evt  = threading.Event()

        def _read():
            try:
                result[0] = input("  Password ▶ ").strip().lower()
            except Exception:
                result[0] = None
            finally:
                done_evt.set()

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        got_input = done_evt.wait(timeout=timeout)

        if not got_input or not result[0]:
            print("  [No input received — treating as wrong password]")
            return None

        print("─"*50 + "\n")
        return result[0]

    # ── Google transcription ──────────────────────────────────────────────────
    def _transcribe_google(self, audio):
        try:
            text = self._rec.recognize_google(audio)
            print(f"[ASR] Heard: '{text}'")
            return text.lower().strip()
        except sr.UnknownValueError:
            print("[ASR] Could not understand audio.")
            return None
        except sr.RequestError as e:
            print(f"[ASR] Google unavailable: {e}")
            return self._keyboard_fallback(9)

    # ── Vosk transcription ────────────────────────────────────────────────────
    def _transcribe_vosk(self, audio):
        try:
            raw  = audio.get_raw_data(convert_rate=16000, convert_width=2)
            rec  = KaldiRecognizer(self._vosk_model, 16000)
            rec.AcceptWaveform(raw)
            text = _json.loads(rec.Result()).get("text","").strip()
            print(f"[ASR] Vosk: '{text}'")
            return text.lower() or None
        except Exception as e:
            print(f"[ASR] Vosk error: {e}")
            return None

    # ── Password check ────────────────────────────────────────────────────────
    def verify_password(self, spoken):
        if spoken is None:
            return False
        a = spoken.lower().strip().replace(" ","")
        b = self.password.replace(" ","")
        ok = (a == b)
        print(f"[AUTH] Heard: '{a}'  |  Expected: '{b}'  |  Match: {ok}")
        return ok
