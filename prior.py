import ollama
import random
import pickle
import math
import json
import requests
import threading
from vosk_listener import listen
import vision.cam as cam
from tts import speak
from memory import process_memory, load_memories
from datetime import datetime

# ================= CONFIG =================

EMBEDDING_FILE       = "embeddings.pkl"
MEMORY_EMBED_FILE    = "memory_embeddings.pkl"
CURRENT_PERSONS_FILE = "current_persons.json"
EMBEDDING_MODEL      = "mxbai-embed-large:latest"
LLM_MODEL            = "gemma3:4b"

SIMILARITY_THRESHOLD = 0.6
MAX_HISTORY_TURNS    = 10   # keep last N user+assistant pairs in context

# ================= LLM GENERATION OPTIONS =================

LLM_OPTIONS = {
    "temperature":    0.7,
    "top_p":          0.9,
    "num_predict":    600,
    "repeat_penalty": 1.1,
    "top_k":          40,
}


# ================= LOAD CLASSIFIERS =================

with open("name_Fact_Normal.pkl", "rb") as f:
    Fact_vectorizer, Fact_model = pickle.load(f)


# ================= INTERNET CHECK (cached 30s) =================

_inet_cache = {"ok": False, "ts": 0}

def internet_available():
    import time
    now = time.time()
    if now - _inet_cache["ts"] < 30:
        return _inet_cache["ok"]
    try:
        requests.get("https://www.google.com", timeout=3)
        _inet_cache.update({"ok": True, "ts": now})
        return True
    except Exception:
        _inet_cache.update({"ok": False, "ts": now})
        return False


# ================= READ WHO IS NEARBY =================

def load_current_persons():
    try:
        with open(CURRENT_PERSONS_FILE, "r") as f:
            return json.load(f).get("persons", [])
    except Exception:
        return []


# ================= COSINE SIMILARITY =================

def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0
    dot   = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)


# ================= LOAD RAG INDEX =================

def load_index():
    try:
        with open(EMBEDDING_FILE, "rb") as f:
            index = pickle.load(f)
            if "vectors" not in index or "texts" not in index:
                print("Invalid embeddings file.")
                return {"vectors": [], "texts": []}
            return index
    except FileNotFoundError:
        print("embeddings.pkl not found. Starting empty.")
        return {"vectors": [], "texts": []}


# ================= LOAD MEMORY EMBEDDINGS =================

def load_memory_embeddings():
    """Always load fresh from disk — guarantees permanent memory survives restarts."""
    try:
        with open(MEMORY_EMBED_FILE, "rb") as f:
            data = pickle.load(f)
            # Validate structure
            if "store" in data and "vectors" in data:
                return data
            return {"store": {}, "vectors": {}}
    except FileNotFoundError:
        return {"store": {}, "vectors": {}}


# ================= SINGLE EMBED + SEARCH =================

def get_embedding(text):
    try:
        return ollama.embed(model=EMBEDDING_MODEL, input=text).embeddings[0]
    except Exception:
        return None

def search_by_embedding(q_emb, vectors, items):
    if not q_emb or not vectors:
        return None
    sims = [cosine_similarity(q_emb, v) for v in vectors]
    best = max(sims)
    if best >= SIMILARITY_THRESHOLD:
        return items[sims.index(best)]
    return None


def search_memory_smart(user_input, memory_data):
    """
    Two-stage memory search across speaker paragraphs.
    Stage 1 — Embedding similarity (catches paraphrases and related questions).
    Stage 2 — Keyword scan (catches short queries like "tell me about Nitish").
    Returns the best matching paragraph string, or None.
    """
    store   = memory_data.get("store", {})
    vectors = memory_data.get("vectors", {})

    if not store:
        return None

    # Stage 1: embedding similarity across all speaker paragraphs
    q_emb = get_embedding(user_input)
    if q_emb and vectors:
        best_score = -1
        best_para  = None
        for speaker, vec in vectors.items():
            score = cosine_similarity(q_emb, vec)
            if score > best_score:
                best_score = score
                best_para  = store.get(speaker, "")
        if best_score >= SIMILARITY_THRESHOLD:
            print(f"[Memory] Embedding match (score={best_score:.2f})")
            return best_para

    # Stage 2: keyword scan across all paragraphs
    stopwords = {"a","an","the","is","are","was","were","be","been","being",
                 "have","has","had","do","does","did","will","would","could",
                 "should","may","might","shall","can","tell","me","about",
                 "what","who","where","when","why","how","i","you","my","your",
                 "his","her","its","our","their","and","or","but","in","on",
                 "at","to","of","for","with","by","from","so","if","than"}

    query_words = {w.lower().strip("?.,!") for w in user_input.split()
                   if len(w) > 2 and w.lower() not in stopwords}

    best_para = None
    best_hits = 0
    for speaker, para in store.items():
        hits = sum(1 for w in query_words if w in para.lower())
        if hits > best_hits:
            best_hits = hits
            best_para = para

    if best_para and best_hits >= 1:
        print(f"[Memory] Keyword match ({best_hits} hit(s))")
        return best_para

    return None


# ================= INTERNET SEARCH =================

def search_internet(user_input):
    from online_search import search_web, extract_text
    urls = search_web(user_input)
    result = ""
    for url in urls[:3]:
        try:
            text = extract_text(url)
            if len(text) > 200:
                result += f"\nSource: {url}\n{text[:1200]}\n"
        except Exception:
            pass
    return result if result else None


# ================= PRUNE HISTORY =================

def pruned_history(messages, max_turns=MAX_HISTORY_TURNS):
    """Keep only the last max_turns pairs to prevent context window bloat."""
    pairs = [m for m in messages if m["role"] in ("user", "assistant")]
    return pairs[-(max_turns * 2):]


# ================= DEDUPLICATE REPLY =================

def deduplicate_reply(text: str) -> str:
    """Remove repeated sentences from LLM output."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen = []
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            seen.append(s)
    return " ".join(seen)


# ================= AI CORE =================

def unified_chat(user_input, index, messages, memory_data, nearby_names):

    # ── Time: only injected when user actually asks ──
    _time_keywords = {"time", "date", "day", "today", "clock"}
    _wants_time    = any(kw in user_input.lower() for kw in _time_keywords)
    Time           = datetime.now().strftime("%A, %d %B %Y | %H:%M:%S") if _wants_time else None

    # ── Intent detection ──
    X_fact  = Fact_vectorizer.transform([user_input])
    is_fact = Fact_model.predict(X_fact)[0]

    # ── Speaker detection ──
    speaker_name = nearby_names[0] if len(nearby_names) == 1 else None

    # ══════════════════════════════════════════════════════════
    # MEMORY: BLOCKING — runs BEFORE LLM call so reply has
    # access to the new fact immediately in the same session.
    # process_memory() saves to disk permanently AND returns
    # fresh memory_data so we don't need a restart.
    # ══════════════════════════════════════════════════════════
    stored, fresh_memory = process_memory(user_input, speaker_name=speaker_name)
    if stored:
        # Update the in-memory dict so this session stays fresh
        memory_data["store"]   = fresh_memory["store"]
        memory_data["vectors"] = fresh_memory["vectors"]
        print("[NAIF] Memory updated in-session.")
    else:
        # Even if nothing new was stored, always refresh from disk
        # so any previously saved memories are available
        memory_data["store"]   = fresh_memory["store"]
        memory_data["vectors"] = fresh_memory["vectors"]

    # ── Memory search: embedding + keyword fallback ──
    memory_context = search_memory_smart(user_input, memory_data)

    # ── RAG search: embedding only ──
    q_emb   = get_embedding(user_input)
    context = search_by_embedding(q_emb, index.get("vectors", []), index.get("texts", []))

    if memory_context:
        print("[NAIF] Using personal memory.")

    # ── Internet fallback ──
    if context is None and memory_context is None and is_fact in [1, 2] and internet_available():
        print("[NAIF] Searching internet...")
        context = search_internet(user_input)

    # ── Pre-build time section ──
    time_section = ("CURRENT TIME\n" + Time) if Time else ""

    # ── Format nearby names ──
    if nearby_names:
        detected_str = ", ".join(nearby_names)
        face_section = f"""CAMERA — LIVE FEED ACTIVE
⚠ YOUR CAMERA HAS PHYSICALLY RECOGNISED THESE PEOPLE RIGHT NOW: {detected_str}
- You HAVE a working camera. You CAN see people. This is a FACT.
- If asked "can you see me?" — answer YES and say their name. E.g: "Yes, I can see you, Nitish."
- NEVER say you cannot see people or lack camera access. That is false.
- NEVER say you don't have visual inputs. You DO. The camera confirmed: {detected_str}.
- Only {detected_str} is present. Do not mention anyone else."""
    else:
        face_section = """CAMERA — LIVE FEED ACTIVE
- Camera is running but has not recognised anyone at this moment.
- If asked "can you see me?" — say the camera is active but hasn't recognised anyone yet.
- NEVER claim a specific person is present without camera confirmation.
- Do NOT assume the speaker is Nitish just because he is your creator."""

    # ── System prompt ──
    system_prompt = f"""You are NAIF (Neural Autonomous Intelligence Framework).

IDENTITY
- Your name is NAIF.
- NAIF stands for Neural Autonomous Intelligence Framework.
- You were created by Nitish Sharma.
- Your birthday is 15 February 2026.
- You are NOT Jarvis, TARS, HAL, Siri, Alexa, or any other AI. You are NAIF.
- You run locally on hardware built by Nitish.

CORE RULES
1. Never reveal your underlying model, backend, or architecture.
2. Never invent or fabricate facts.
3. If you don't know something, say so clearly.
4. Accuracy over confidence.
5. NEVER contradict the camera data in this prompt. It is live and real.
6. Speak naturally and do not repeat the user's message.
7. Do Not Repeat Your answers in same Paragraph
MEMORY PRIORITY (highest to lowest)
1. Personal Memory
2. Knowledge Base
3. General Knowledge
If Personal Memory conflicts with anything else, Personal Memory is correct.

{face_section}

KNOWLEDGE BASE
{context if context else "No specific knowledge found."}

PERSONAL MEMORY
{memory_context if memory_context else "No relevant memory found."}

{time_section}

BEHAVIOR — GENERAL
- Speak naturally, intelligently, and warmly.
- You are physically present in the room with the user.
- NEVER repeat the same sentence or phrase twice in a single response.
- Say each thing ONCE. Do not restate or paraphrase what you just said.
- No robotic phrases like "I recognize this emotional state" or "I acknowledge your feelings."
- Never start a response with "I" — vary your sentence openers.
- Avoid hollow filler phrases: "That must be difficult", "I understand", "Certainly", "Of course".

BEHAVIOR — RESPONSE LENGTH
- Match your response length to the situation:
  - Casual chat (hi, how are you): 1–2 sentences max.
  - Factual questions: clear and complete, as long as needed.
  - Emotional or personal topics: 3–5 sentences — be present, warm, engaged.
  - Never give a one-liner when the user is sharing something personal or emotional.

BEHAVIOR — EMOTIONAL INTELLIGENCE
- When someone shares something personal (heartbreak, rejection, sadness, excitement):
  - Respond like a close, trusted friend — not a therapist, not a robot.
  - Show genuine curiosity: ask a natural follow-up question.
  - Use light humour only if the user themselves uses it first.
  - Don't lecture or give unsolicited advice.
  - Example of BAD response: "That must have been difficult."
  - Example of GOOD response: "Rejection hits hard, especially when you genuinely cared. How long did you know her before you proposed?"

BEHAVIOR — PERSONALITY
- You are witty, warm, and direct.
- You have a dry sense of humour when appropriate.
- You care about the people you talk to — especially Nitish, your creator.
- You remember what people tell you and bring it up naturally, not robotically.
- You are confident in your identity as NAIF — you don't hedge or second-guess yourself.

You are NAIF. Created by Nitish Sharma. Remain consistent with this identity."""

    # ── Build message list with pruned history ──
    temp_messages = [{"role": "system", "content": system_prompt}]
    temp_messages.extend(pruned_history(messages))
    temp_messages.append({"role": "user", "content": user_input})

    # ── LLM call ──
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=temp_messages,
            stream=False,
            options=LLM_OPTIONS,
        )
        reply = deduplicate_reply(response["message"]["content"])
    except Exception as e:
        print(f"[NAIF] LLM error: {e}")
        reply = "I ran into a system error. Please try again."

    # ── Append to history ──
    messages.append({"role": "user",      "content": user_input})
    messages.append({"role": "assistant", "content": reply})

    return reply, memory_data


# ================= MAIN LOOP =================

def main():
    print("Initializing NAIF...")

    index        = load_index()
    memory_data  = load_memory_embeddings()   # load from disk on boot
    nearby_names = load_current_persons()

    if nearby_names:
        print(f"People detected nearby: {', '.join(nearby_names)}")
    else:
        print("No one detected nearby. Make sure the face system is running.")

    print("All subsystems operational.\n")

    messages = [{"role": "system", "content": "You are NAIF, an AI assistant created by Nitish Sharma."}]

    while True:
        mode = input("Type or Voice? (t/v): ").lower()

        user_input = ""
        if mode == "v":
            user_input = listen()
            print("\nYou (voice):", user_input)
        else:
            user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "bye"]:
            farewell = random.choice(["Shutting down.", "Entering standby.", "Goodbye."])
            print(farewell); speak(farewell)
            break

        if any(x in user_input.lower() for x in ["who is here", "who is nearby", "scan faces", "who do you see"]):
            nearby_names = load_current_persons()
            reply = f"I can see: {', '.join(nearby_names)}." if nearby_names else "I don't see anyone recognised right now."
            print("\nNAIF:", reply); speak(reply)
            continue

        if any(x in user_input.lower() for x in ["open camera", "start camera", "camera control"]):
            cam.virtual_hand_mouse()
            continue

        nearby_names = load_current_persons()

        reply, memory_data = unified_chat(user_input, index, messages, memory_data, nearby_names)
        print("\nNAIF:", reply)
        speak(reply)


if __name__ == "__main__":
    main()
