# name_intent_model.py
# ML model to detect if user is setting or asking the bot's name

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import re

# ----------------------------
# Dataset for intent training
# ----------------------------
training_sentences = [
    # Set name intents
    "Your name is Max",
    "Call yourself Alex",
    "I want to name you Bella",
    "Please set your name to Luna",
    "From now your name is NAIF",
    "Set your name as Charlie",
    "From now you are NAIF",
    
    # Get name intents
    "What is your name?",
    "Can you tell me your name?",
    "Who are you?",
    "your name?",
    "Tell me your name",
    "May I know your name?"
    
]

training_labels = [
    "set_name", "set_name", "set_name", "set_name", "set_name", "set_name","set_name",
    "get_name", "get_name", "get_name", "get_name", "get_name", "get_name"
]

# ----------------------------
# Train the model
# ----------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)

model = LogisticRegression()
model.fit(X, training_labels)

# Save the model
with open("name_intent_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("Name intent model trained and saved!")

# ----------------------------
# Helper function to extract name from user input
# ----------------------------
def extract_name(text):
    text = text.lower()
    patterns = [
        r"name is (\w+)",
        r"name to (\w+)",
        r"call yourself (\w+)",
        r"name as (\w+)",
        r"You are (\w+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).capitalize()
    return None
