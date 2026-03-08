from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from collections import Counter

sentences = [

#    ================================Search==============================================

    "Can you search on internet about RVS College of engineering",
    "Find information about Python",
    "Search final year project github",
    "Find information about galaxies",
    "Search about NASA",
    "Search about ISRO",
#    ================================Search==============================================

    "Who is PM of india in 2026?",
    "Which Are the new Anime Released",
    "What is solar system",
    "What is Earth",
    "What is black hole",
    "What is supernova",
    "What is neutron star",
    "What is gravity",
    "What is quantum mechanics",
    "Who is Elon Musk",
    "Who discovered gravity",
    "Explain Big Bang",
    "Explain dark matter",
    "Explain relativity",
    "Tell me about Mars",
    "Tell me about Jupiter",
    
    "Define artificial intelligence",
    "Find information about github",
    "Search github repository",
    "Find github project",
    "Find machine learning projects",
    "Search information about supernova",
    "Explain neutron stars",
    "Tell me about black holes",
    "Find information about universe",
    "Search about space",
    "Explain time dilation",
    "Search about artificial intelligence",
    "Explain how gravity works",
    "Tell me about quantum physics",
    "Search about galaxies",
    "Find information about NASA missions",
    "please Search Bhagwad Geeta On internet",
    "Please can you find About SpaceX",
    "2026 Space Events",
    "Tell me about Michael jackson",
    "Elon musk Sons name",


    # ================= NORMAL CHAT =================

    "Hi",
    "Hello",
    "How are you",
    "Tell me a joke",
    "What are you doing",
    "Do you like music",
    "Nice to meet you",
    "Good morning",
    "Good night",
    "Bye",
    "Thanks",
    "Cool",
    "Okay",
    "Let's talk",
    "That is interesting",
    "You are smart",
    "I like you",
    "Can we talk",
    "What is your name",
    "Who are you",
    "Are you real",
    "Tell me something",
    "I am bored",
    "Talk to me",
    "You are cool",
    "That is nice",
    "Haha",
    "Lol",
    "Nice",
    "Great",
    "Full form of NAIF",
    "When Is Your Birthday",
    "Tell me about yourself",

    # ================= CONFUSING / EDGE CASES =================

    "What is Git hub",
    "Git hub is cool",
    "I like github",
    "Github is interesting",
    "Do you know github",
    "Github sounds nice",
    "Python is cool",
    "Space is beautiful",
    "AI is amazing",
    "That black hole sounds scary",
    "Definetly you have birthday bro you are born in 2026 febuary"
]


labels = [
    #search = 1
    2,2,2,2,2,2,
    
    # FACT = 1
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,

    # CHAT = 0
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

    # EDGE CHAT = 0
    0,0,0,0,0,0,0,0,0,0,0,0
]

print("Length of sentence is :",len(sentences))
print("Lenght of Label is :",len(labels))

# print("the numbers of sentences: ", sentences)
# print("the numbers of labels: ", labels)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

model = LogisticRegression()
model.fit(X, labels)


with open("name_Fact_Normal.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)


print("Fact classifier trained and saved.")