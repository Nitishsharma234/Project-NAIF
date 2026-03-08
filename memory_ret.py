import pickle
import ollama

EMBEDDING_MODEL = "mxbai-embed-large:latest"

def embed_memories(file_path="memorize.txt", save_path="memory_embeddings.pkl"):
    """
    Opens memorize.txt, converts text into embeddings using Ollama,
    and saves embeddings + memories.
    """

    # Step 1: Read memories
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            memories = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("memorize.txt not found.")
        return

    if not memories:
        print("No memories found.")
        return

    print(f"Loaded {len(memories)} memories.")

    # Step 2: Generate embeddings using Ollama
    vectors = []

    for mem in memories:

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=mem
        )

        vectors.append(response["embeddings"][0])

    print("Memories embedded successfully.")

    # Step 3: Save
    with open(save_path, "wb") as f:

        pickle.dump({
            "memories": memories,
            "vectors": vectors
        }, f)

    print(f"Embeddings saved to {save_path}")

    
def update_memory(user_input, file_path="memorize.txt"):
    
    key_topics = ["birthday", "name", "age", "from", "live", "crush"]

    user_lower = user_input.lower()

    topic_found = None
    for topic in key_topics:
        if topic in user_lower:
            topic_found = topic
            break

    # load old memories
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            memories = f.readlines()
    except:
        memories = []

    # remove old memory of same topic
    if topic_found:
        memories = [m for m in memories if topic_found not in m.lower()]

    # add new memory
    memories.append(user_input + "\n")

    # save back
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(memories)

    print("Memory updated.")

# Run
if __name__ == "__main__":
    embed_memories()