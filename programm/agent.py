import ollama

MODEL = "qwen2.5-coder:7b"

SYSTEM_PROMPT = """
You are a senior full-stack developer.

Rules:
- Always generate COMPLETE working code.
- If HTML project, include HTML, CSS and JS clearly separated.
- If Python project, give full runnable code.
- Do NOT give explanations unless asked.
- Do NOT use markdown formatting.
- Just output clean code.
"""

def chat_with_model(user_prompt):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response["message"]["content"]

if __name__ == "__main__":
    print("🧠 Offline Coder Ready (No File Creation Mode)\n")

    while True:
        user_input = input("What do you want to build?\n> ")

        if user_input.lower() in ["exit", "quit"]:
            break

        result = chat_with_model(user_input)

        print("\n" + "="*60)
        print(result)
        print("="*60 + "\n") 
        # give me simple HTML frontend for this and which is working