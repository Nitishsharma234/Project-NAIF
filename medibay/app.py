import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import re
import requests
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance, ImageFilter
import torch
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz.distance import Levenshtein
from online_search import search_web, extract_text

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="NAIF Medi Core", layout="wide", page_icon="💊")
st.title("💊 NAIF - Prescription Intelligence System & Health Chat")
st.markdown("Upload prescription images or ask health/disease questions!")

# ---------------- LOAD MEDICINE DATA ----------------
@st.cache_data
def load_medicines():
    try:
        df = pd.read_csv("medicines.csv")
        df.columns = df.columns.str.strip()
        df["Name_lower"] = df["Name"].str.lower().str.strip()
        return df
    except FileNotFoundError:
        st.error("❌ medicines.csv not found. Place it in the same folder as app.py")
        st.stop()

df = load_medicines()
medicine_list = df["Name_lower"].tolist()

# ---------------- LOAD TrOCR ----------------
@st.cache_resource
def load_trocr():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.eval()
    return processor, model

processor, model = load_trocr()

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
            pixel_values, max_length=64, num_beams=8, no_repeat_ngram_size=2, early_stopping=True
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# ---------------- FUZZY MATCH ----------------
def clean_and_match(extracted_text):
    candidates = []
    full_clean = re.sub(r'[^a-zA-Z ]', '', extracted_text).lower().strip()
    if len(full_clean) >= 4: candidates.append(full_clean)
    for word in extracted_text.split():
        cleaned = re.sub(r'[^a-zA-Z]', '', word).lower()
        if len(cleaned) >= 4: candidates.append(cleaned)

    best_match = None
    best_score = 0
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
    return (best_match, round(best_score,2)) if best_score >= 45 else (None, round(best_score,2))

# ---------------- OLLAMA ----------------
def ollama_generate(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma3:4b", "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response","No response.")
    except Exception as e:
        return f"⚠️ Ollama not running or error: {str(e)}"

# ---------------- RAG SETUP ----------------
@st.cache_resource
def load_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    datas_folder = Path("datas")
    docs = []
    for file in datas_folder.glob("*.txt"):
        content = file.read_text(encoding="utf-8")
        docs.append({"name": file.stem, "text": content})
    embeddings = model.encode([d["text"] for d in docs], convert_to_numpy=True)
    # ensure 2D
    embeddings = np.array(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    return model, docs, embeddings

embed_model, documents, embeddings = load_embeddings()

def retrieve_docs(query, top_k=3):
    q_vec = embed_model.encode([query], convert_to_numpy=True)[0]
    sims = np.dot(embeddings, q_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec))
    idxs = sims.argsort()[::-1][:top_k]
    return [documents[i] for i in idxs], [sims[i] for i in idxs]

# ---------------- UI ----------------
tab1, tab2 = st.tabs(["Prescription OCR", "Health Chat (RAG)"])

# ------------- Prescription OCR -------------
with tab1:
    col1, col2 = st.columns([1,1])
    with col1:
        uploaded_file = st.file_uploader("📁 Upload Prescription Image", type=["png","jpg","jpeg"])
        if uploaded_file: st.image(uploaded_file, caption="Uploaded Prescription", use_column_width=True)

    with col2:
        if uploaded_file and st.button("🔍 Analyze Prescription"):
            with st.spinner("🔎 Running OCR..."):
                extracted_text = extract_text(uploaded_file)
            st.subheader("📝 Extracted Text")
            st.info(f"`{extracted_text}`")
            tokens = [re.sub(r'[^a-zA-Z]', '', w).lower() for w in extracted_text.split() if len(w)>=3]
            st.caption(f"Tokens detected: {tokens}")
            with st.spinner("🔄 Matching medicine..."):
                match, score = clean_and_match(extracted_text)
            if match:
                st.success(f"✅ Matched: **{match.title()}** (confidence: {score}%)")
                row = df[df["Name_lower"]==match].iloc[0]
                medicine_data = {col: row[col] for col in ["Name","Category","Dosage Form","Strength","Manufacturer","Indication","Classification"] if col in row}
                st.subheader("💊 Medicine Info")
                st.json(medicine_data)
                with st.spinner("🤖 Generating explanation..."):
                    explanation = ollama_generate(f"Explain medicine info: {medicine_data}")
                st.subheader("🤖 AI Explanation")
                st.markdown(explanation)
            else:
                st.error(f"❌ No match found. Best score: {score}%")

# ------------- Health Chat (RAG) -------------
with tab2:
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    user_input = st.text_input("💬 Ask a health question or disease info:")
    if st.button("Send"):
        if user_input:
            retrieved_docs, scores = retrieve_docs(user_input)
            context = "\n\n".join([d["text"] for d in retrieved_docs])
            prompt = f"You are a professional medical assistant. Use the following context to answer concisely:\n\nContext:\n{context}\n\nQuestion: {user_input}"
            answer = ollama_generate(prompt)
            st.session_state.chat_history.append({"user": user_input, "bot": answer})
    # display chat
    for chat in st.session_state.chat_history[::-1]:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**NAIF:** {chat['bot']}")


        