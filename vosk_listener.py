import queue
import sounddevice as sd
import json
import time
from vosk import Model, KaldiRecognizer

MODEL_PATH = "vosk-model-en-us-0.22"

_q = queue.Queue()
_model = Model(MODEL_PATH)


def _callback(indata, frames, time_info, status):
    _q.put(bytes(indata))


def listen(silence_timeout: float = 3.0) -> str:
    """
    Listen for speech. Returns text when the user stops talking.
    Auto-stops after `silence_timeout` seconds of silence.
    """
    # Drain stale audio from queue before starting
    while not _q.empty():
        try:
            _q.get_nowait()
        except:
            break

    recognizer = KaldiRecognizer(_model, 16000)
    print("[STT] Listening...")

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype='int16',
        channels=1,
        callback=_callback
    ):
        last_speech_time = time.time()

        while True:
            try:
                data = _q.get(timeout=0.5)
            except queue.Empty:
                if time.time() - last_speech_time > silence_timeout:
                    partial = json.loads(recognizer.PartialResult())
                    text = partial.get("partial", "").strip()
                    if text:
                        print(f"[STT] Captured (timeout): {text}")
                        return text
                    # Reset timer and keep waiting
                    last_speech_time = time.time()
                continue

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    print(f"[STT] Captured: {text}")
                    return text
            else:
                partial = json.loads(recognizer.PartialResult())
                if partial.get("partial", "").strip():
                    last_speech_time = time.time()