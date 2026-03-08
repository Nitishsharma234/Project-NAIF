import os
import pickle
import ollama

DATA_DIR = r"D:\dream\data"
EMBEDDING_FILE = r"D:\dream\embeddings.pkl"
EMBEDDING_MODEL = "mxbai-embed-large:latest"


# ===============================
# Load all .txt files recursively
# ===============================
def load_documents(folder_path):
    docs = []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".txt"):
                full_path = os.path.join(root, filename)
                with open(full_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    docs.append(text)

    return docs


# ===============================
# Chunk text to avoid token limit
# ===============================
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ===============================
# Main ingestion
# ===============================
def main():

    documents = load_documents(DATA_DIR)

    if not documents:
        print("No documents found.")
        return

    print(f"Loaded {len(documents)} documents.")

    all_chunks = []

    # Split documents into chunks
    for doc in documents:
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)

    print(f"Generating embeddings for {len(all_chunks)} chunks...")

    vectors = []

    # Embed chunk by chunk (safe way)
    for chunk in all_chunks:
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=chunk
        )

        vector = response["embeddings"][0]
        vectors.append(vector)

    # Save
    data_to_save = {
        "texts": all_chunks,
        "vectors": vectors
    }

    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(data_to_save, f)

    print("Embedding completed successfully.")


if __name__ == "__main__":
    main()