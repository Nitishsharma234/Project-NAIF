# name_binary_model.py
# Detects if user input is about NAIF's name (1) or not (0)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# ================= DATASET =================

sentences = [

    # NAME SET → 1
    "Your name is Max",
    "Call yourself Alex",
    "I want to name you Bella",
    "Set your name to Luna",
    "From now your name is NAIF",
    "I will call you Jarvis",
    "You are now called Nova",

    # NAME GET → 1
    "What is your name",
    "Tell me your name",
    "your name",
    "Who are you",
    "Identify yourself",
    "What should I call you",
    "Do you have a name",

    # NOT NAME → 0
    "Search internet about Bhagavad Gita",
    "Tell me about Krishna",
    "Explain black hole",
    "Search SpaceX rocket",
    "Find information about NASA",
    "Explain Mahabharata",
    "Tell me about universe",
    "Search GitHub repository",
    "What is gravity",
    "Explain AI",
    "Do you Know my Name",
    "What is the meaning of Name",
    "My Name is Nitish",
    "My name is Jalandhar",
    "When Is your Birthday?",
    'SO are your Excited?',
    "YOu dont have to mention That you are NAIF every time",
    "Can you check your own code ?",


    # NORMAL CHAT → 0
    "Hi",
    "Hello",
    "How are you",
    "Tell me a joke",
    "Good morning",
    "Bye",
    "Do you want to conquer the world if you become concious?"
]

labels = [
    1,1,1,1,1,1,1,   # set
    1,1,1,1,1,1,1,   # get
    0,0,0,0,0,0,0,0,0,0,0,0,0,   # factual but not name
    0,0,0,0,0,0,0,0,0,0,0,0      # chat
]

# ================= TRAIN =================

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

model = LogisticRegression()
model.fit(X, labels)

# ================= SAVE =================

with open("name_binary_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("Binary name detection model trained and saved successfully!")