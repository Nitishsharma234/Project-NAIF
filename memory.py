"""
memory.py — Permanent smart memory system for NAIF.

HOW IT WORKS:
- Every personal fact is saved to memorize.json PERMANENTLY (never wiped on restart).
- Each person gets a growing paragraph that gets smarter over time.
- If you say something new → it's added.
- If you correct something → old value is replaced.
- Memory is available INSTANTLY in the same conversation (no restart needed).
- Embeddings are also saved to memory_embeddings.pkl for fast semantic search.
"""

import ollama
import pickle
import json
import os

MEMORY_FILE       = "memorize.json"        # { "Nitish": "paragraph...", ... }
MEMORY_EMBED_FILE = "memory_embeddings.pkl"
EMBEDDING_MODEL   = "mxbai-embed-large:latest"
DECISION_MODEL    = "gemma3:4b"


# ═══════════════════════════════════════════════════════════
# LOAD / SAVE  —  Always reads/writes to disk (permanent)
# ═══════════════════════════════════════════════════════════

def load_memories() -> dict:
    """Load memory store from disk. Returns {} if file missing or broken."""
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Validate it's a proper dict
            if isinstance(data, dict):
                return data
            return {}
    except (json.JSONDecodeError, Exception):
        return {}


def save_memories(store: dict):
    """Save memory store to disk atomically (write to temp then rename)."""
    tmp = MEMORY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    os.replace(tmp, MEMORY_FILE)  # atomic — no partial writes


# ═══════════════════════════════════════════════════════════
# EXTRACT — Does this message have anything worth remembering?
# ═══════════════════════════════════════════════════════════

def extract_memory(user_input: str, speaker_name: str = None) -> tuple[bool, str]:
    """
    Ask the LLM if there's a personal fact worth storing.
    Returns (should_store: bool, clean_fact: str)
    """
    first_person_words = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd"}
    tokens = set(user_input.lower().split())
    uses_first_person = bool(tokens & first_person_words)

    if uses_first_person and speaker_name:
        speaker_context = f"The speaker is: {speaker_name}. Replace I/me/my/mine with '{speaker_name}'s / '{speaker_name}'."
    elif uses_first_person and not speaker_name:
        speaker_context = "Speaker is UNKNOWN. Prefix the memory with 'Unknown user:'."
    else:
        speaker_context = "No first-person pronouns. Write memory exactly as stated — no pronoun changes needed."

    prompt = f"""You are the memory extractor for NAIF, a personal AI assistant.

Your job: decide if this message contains any personal information worth remembering long-term.

EXTRACT (store: true) if the message contains:
- Personal facts: name, age, birthday, location, job, school
- Relationships: family members, friends, crush, partner, colleagues
- Preferences: hobbies, favourite food/colour/music/sport
- Goals, dreams, projects the person is working on
- Life events: achievements, failures, things that happened to them
- Any named person mentioned in a personal context
- Corrections to previously stated facts ("actually my birthday is...")
- Anything the user explicitly says to remember

DO NOT EXTRACT (store: false):
- Pure greetings: hi, hello, bye, thanks, good morning
- Generic questions with no personal info: "what is AI", "what time is it"
- Small talk with zero personal content
- Commands or instructions to NAIF with no personal info

Speaker context: {speaker_context}

Message: "{user_input}"

Reply ONLY with valid JSON. No markdown fences. No explanation. Just the JSON object:
{{"store": true, "memory": "Nitish's birthday is 7th November"}}
or
{{"store": false, "memory": ""}}

RULES for the memory string:
- Never use I / me / my / the user.
- Write as a clean third-person factual sentence.
- If correcting a fact, include "UPDATED:" prefix so the merger knows to replace the old value."""

    try:
        response = ollama.chat(
            model=DECISION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        raw = response["message"]["content"].strip()

        # Strip markdown fences if LLM added them anyway
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(l for l in lines if not l.startswith("```")).strip()

        result     = json.loads(raw)
        should     = bool(result.get("store", False))
        fact       = result.get("memory", "").strip()
        return should, fact

    except Exception as e:
        print(f"[Memory] Extraction error: {e}")
        return False, ""


# ═══════════════════════════════════════════════════════════
# MERGE — Add new fact into existing paragraph intelligently
# ═══════════════════════════════════════════════════════════

def merge_into_paragraph(existing: str, new_fact: str, speaker_name: str) -> str:
    """
    LLM merges new_fact into the existing paragraph.
    - If it's new info → adds it as a new sentence.
    - If it updates old info → replaces the old value.
    - Never duplicates.
    - Never loses existing info unless directly contradicted.
    """
    if not existing.strip():
        # First ever memory for this person — clean up the UPDATED: prefix if present
        fact = new_fact.replace("UPDATED:", "").strip()
        return fact

    prompt = f"""You maintain a permanent memory paragraph about a person called "{speaker_name}" for an AI assistant.

Current paragraph:
\"\"\"{existing}\"\"\"

New fact to incorporate:
\"{new_fact}\"

Instructions:
- If the new fact starts with "UPDATED:" it means the user is correcting something. Find the old value in the paragraph and REPLACE it with the new one.
- If the new fact is completely new information not mentioned anywhere in the paragraph, add it as a new sentence at the end.
- If the new fact is essentially the same as something already in the paragraph (duplicate), do NOT add it again.
- Keep the paragraph clean, concise and factual.
- Never use "I", "me", "my" — always use "{speaker_name}'s" or "{speaker_name}".
- Return ONLY the updated paragraph. No explanation, no markdown, no intro sentence."""

    try:
        response = ollama.chat(
            model=DECISION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[Memory] Merge error: {e}")
        # Safe fallback: just append
        clean_fact = new_fact.replace("UPDATED:", "").strip()
        return existing.rstrip(". ") + ". " + clean_fact + "."


# ═══════════════════════════════════════════════════════════
# EMBED — Save semantic vectors for fast memory search
# ═══════════════════════════════════════════════════════════

def embed_and_save(store: dict):
    """Embed all speaker paragraphs and save vectors to disk."""
    # Load existing vectors so we don't re-embed unchanged paragraphs
    try:
        with open(MEMORY_EMBED_FILE, "rb") as f:
            existing_pkl = pickle.load(f)
            old_vectors = existing_pkl.get("vectors", {})
            old_store   = existing_pkl.get("store", {})
    except Exception:
        old_vectors = {}
        old_store   = {}

    vectors = dict(old_vectors)  # start from existing

    for speaker, paragraph in store.items():
        # Only re-embed if paragraph changed
        if old_store.get(speaker) == paragraph and speaker in old_vectors:
            continue
        try:
            resp = ollama.embed(model=EMBEDDING_MODEL, input=paragraph)
            vectors[speaker] = resp["embeddings"][0]
            print(f"[Memory] Re-embedded '{speaker}'")
        except Exception as e:
            print(f"[Memory] Embed error for '{speaker}': {e}")
            vectors[speaker] = []

    with open(MEMORY_EMBED_FILE, "wb") as f:
        pickle.dump({"store": store, "vectors": vectors}, f)


# ═══════════════════════════════════════════════════════════
# MAIN ENTRY — Called from prior.py for every user message
# ═══════════════════════════════════════════════════════════

def process_memory(user_input: str, speaker_name: str = None) -> tuple[bool, dict]:
    """
    Full pipeline — BLOCKING (must complete before NAIF replies so memory is fresh).

    Steps:
    1. Extract fact from message.
    2. Load existing memory from disk.
    3. Merge new fact into the speaker's paragraph.
    4. Save updated paragraph to disk (permanent).
    5. Re-embed updated paragraph.
    6. Return (stored: bool, updated_memory_data: dict)

    Returns:
        (True, fresh_memory_data)  if something was stored
        (False, fresh_memory_data) if nothing was stored (but data still fresh)
    """
    key = speaker_name if speaker_name else "Unknown"

    # Step 1: Extract
    should_store, new_fact = extract_memory(user_input, speaker_name)

    if not should_store or not new_fact:
        print("[Memory] Nothing to store.")
        # Still return fresh data from disk
        memory_store = load_memories()
        try:
            with open(MEMORY_EMBED_FILE, "rb") as f:
                pkl = pickle.load(f)
        except Exception:
            pkl = {"store": memory_store, "vectors": {}}
        return False, pkl

    print(f"[Memory] Extracted: {new_fact}")

    # Step 2: Load existing
    memory_store = load_memories()
    existing = memory_store.get(key, "")

    # Step 3: Merge
    updated = merge_into_paragraph(existing, new_fact, key)
    print(f"[Memory] Paragraph for '{key}': {updated}")

    # Step 4: Save to disk PERMANENTLY
    memory_store[key] = updated
    save_memories(memory_store)
    print(f"[Memory] ✅ Saved to disk.")

    # Step 5: Re-embed
    embed_and_save(memory_store)

    # Step 6: Return fresh data for immediate in-session use
    try:
        with open(MEMORY_EMBED_FILE, "rb") as f:
            fresh_pkl = pickle.load(f)
    except Exception:
        fresh_pkl = {"store": memory_store, "vectors": {}}

    return True, fresh_pkl


# ═══════════════════════════════════════════════════════════
# RETRIEVE — Get memory paragraph for a speaker
# ═══════════════════════════════════════════════════════════

def get_memory(speaker_name: str = None) -> str:
    """Retrieve the current memory paragraph for a speaker."""
    key = speaker_name if speaker_name else "Unknown"
    return load_memories().get(key, "")


def get_all_memories() -> dict:
    """Retrieve all stored memory paragraphs."""
    return load_memories()


# ═══════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cases = [
        ("Nitish", "My birthday is on 7th November"),
        (None,     "Hi there"),
        ("Nitish", "I am working on the NAIF project"),
        (None,     "Tell me a joke"),
        ("Priya",  "My favourite colour is blue"),
        ("Nitish", "I live in Patna, India"),
        (None,     "What is the capital of France"),
        ("Nitish", "Actually my birthday is 8th November"),  # Should UPDATE, not duplicate
        ("Muskan", "I am Nitish's crush"),
        ("Parth",  "I am one of the NAIF team members"),
        (None,     "Thanks bye"),
    ]

    print("=" * 50)
    print("RUNNING MEMORY TESTS")
    print("=" * 50)

    for speaker, text in test_cases:
        print(f"\n→ Speaker: {speaker or 'UNKNOWN'} | Input: \"{text}\"")
        stored, _ = process_memory(text, speaker_name=speaker)
        print(f"  Stored: {stored}")

    print("\n\n===== FINAL MEMORY STORE =====")
    for speaker, para in load_memories().items():
        print(f"\n[{speaker}]\n  {para}")
