from flask import Flask, request, render_template, Response, stream_with_context, jsonify
import pandas as pd
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance, ImageFilter
import torch
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import os
import json
from online_search import search_and_extract, internet_available

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------- LOAD MEDICINES ----------------
df = pd.read_csv("medicines.csv")
df.columns = df.columns.str.strip()
df["Name_lower"] = df["Name"].str.lower().str.strip()
medicine_list = df["Name_lower"].tolist()

# ---------------- LOAD TrOCR ----------------
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.eval()

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image.thumbnail((768, 768), Image.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = ImageEnhance.Sharpness(image).enhance(2.0)
    image = ImageEnhance.Brightness(image).enhance(1.2)
    image = image.filter(ImageFilter.SHARPEN)
    return image

def extract_text(image_file):
    image = preprocess_image(image_file)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values, max_length=64, num_beams=8,
            no_repeat_ngram_size=2, early_stopping=True
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# ---------------- FUZZY MATCH ----------------
def clean_and_match(extracted_text):
    candidates = []
    full_clean = re.sub(r'[^a-zA-Z ]', '', extracted_text).lower().strip()
    if len(full_clean) >= 4:
        candidates.append(full_clean)
    for word in extracted_text.split():
        cleaned = re.sub(r'[^a-zA-Z]', '', word).lower()
        if len(cleaned) >= 4:
            candidates.append(cleaned)

    best_match, best_score = None, 0
    for candidate in candidates:
        match = process.extractOne(candidate, medicine_list, scorer=fuzz.token_set_ratio)
        if match:
            name, fuzzy_score, _ = match
            partial = fuzz.partial_ratio(candidate, name)
            prefix = fuzz.ratio(candidate[:4], name[:4])
            lev_distance = Levenshtein.distance(candidate, name)
            lev_similarity = (1 - lev_distance / max(len(candidate), len(name))) * 100
            final_score = fuzzy_score*0.35 + partial*0.25 + prefix*0.15 + lev_similarity*0.25
            if final_score > best_score:
                best_score = final_score
                best_match = name
    return (best_match, round(best_score, 2)) if best_score >= 45 else (None, round(best_score, 2))

# ---------------- OLLAMA (non-streaming, for prescription) ----------------
def ollama_generate(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma3:4b", "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "No response.")
    except Exception as e:
        return f"⚠️ Ollama not running or error: {str(e)}"

# ---------------- OLLAMA STREAMING ----------------
def ollama_stream(prompt):
    """Generator: yields SSE chunks from Ollama's streaming API."""
    try:
        with requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma3:4b", "prompt": prompt, "stream": True},
            stream=True,
            timeout=120
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    chunk = data.get("response", "")
                    if chunk:
                        yield f"data: {json.dumps({'token': chunk})}\n\n"
                    if data.get("done"):
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        return
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

# ---------------- RAG ----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
datas_folder = Path("datas")
documents = []

for file in datas_folder.glob("*.txt"):
    text = file.read_text(encoding="utf-8")
    documents.append({"name": file.stem, "text": text})

embeddings = embed_model.encode([d["text"] for d in documents], convert_to_numpy=True)

def retrieve_docs(query, top_k=3):
    q_vec = embed_model.encode([query], convert_to_numpy=True)[0]
    sims = np.dot(embeddings, q_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec))
    idxs = sims.argsort()[::-1][:top_k]
    return [documents[i] for i in idxs], [sims[i] for i in idxs]

# ---------------- ROUTES ----------------
chat_history = []

@app.route("/", methods=["GET", "POST"])
def index():
    global chat_history
    prescription_result = None
    medicine_info = None
    extracted_text = None

    if request.method == "POST":
        # Chat is now handled by /chat_stream SSE — only prescription here
        if "prescription" in request.files:
            file = request.files["prescription"]
            if file.filename != "":
                path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(path)
                extracted_text = extract_text(path)
                match, score = clean_and_match(extracted_text)
                prescription_result = {"match": match, "score": score}
                if match:
                    row = df[df["Name_lower"] == match].iloc[0]
                    medicine_info = {
                        col: row[col]
                        for col in ["Name", "Category", "Dosage Form", "Strength",
                                    "Manufacturer", "Indication", "Classification"]
                        if col in row
                    }
                    explanation = ollama_generate(f"Explain medicine info: {medicine_info}")
                    prescription_result["explanation"] = explanation

    return render_template(
        "index.html",
        prescription_result=prescription_result,
        medicine_info=medicine_info,
        extracted_text=extracted_text,
        chat_history=chat_history,
    )


@app.route("/internet_status")
def internet_status():
    """Returns whether internet is currently available."""
    return jsonify({"online": internet_available()})


@app.route("/chat_stream")
def chat_stream():
    """SSE endpoint — streams Ollama response token by token.
    Uses live web search when internet is available, falls back to local RAG.
    """
    global chat_history
    user_input = request.args.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    online = internet_available()

    if online:
        # --- Online: DuckDuckGo search + extract ---
        web_context = search_and_extract(user_input, max_results=4)
        if web_context:
            context = web_context
            source_mode = "web"
        else:
            # Web returned nothing — fall back silently to RAG
            retrieved_docs, _ = retrieve_docs(user_input)
            context = "\n\n".join([d["text"] for d in retrieved_docs])
            source_mode = "rag_fallback"
    else:
        # --- Offline: local RAG ---
        retrieved_docs, _ = retrieve_docs(user_input)
        context = "\n\n".join([d["text"] for d in retrieved_docs])
        source_mode = "rag"

    prompt = (
        "You are NAIF, a professional medical assistant. "
        "Use the following context to answer the question concisely and accurately. "
        "If the context is insufficient, say so honestly.\n\n"
        f"Context:\n{context}\n\nQuestion: {user_input}"
    )

    collected = []

    def generate():
        # First event: tell the client which source mode is active
        yield f"data: {json.dumps({'source_mode': source_mode})}\n\n"
        for chunk in ollama_stream(prompt):
            try:
                payload = json.loads(chunk.replace("data: ", "").strip())
                if "token" in payload:
                    collected.append(payload["token"])
                if payload.get("done"):
                    chat_history.append({
                        "user": user_input,
                        "bot": "".join(collected),
                        "source_mode": source_mode
                    })
            except Exception:
                pass
            yield chunk

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    print("🚀 NAIF server running → http://localhost:6002")
    app.run(port=6002, debug=True)