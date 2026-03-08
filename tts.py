import pyttsx3
import threading

_voice_enabled = True
_lock = threading.Lock()


def set_voice_enabled(state: bool):
    global _voice_enabled
    with _lock:
        _voice_enabled = state
    print(f"[TTS {'ENABLED' if state else 'DISABLED'}]")


def is_voice_enabled() -> bool:
    with _lock:
        return _voice_enabled


def speak(text: str):
    if not text:
        return
    with _lock:
        if not _voice_enabled:
            return

    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 165)
            engine.setProperty('volume', 1.0)
            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[0].id)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[TTS Error] {e}")

    threading.Thread(target=_speak, daemon=True).start()