import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('stopwords')

# -------------------------------
# STEP 1: LOAD DATASET
# -------------------------------
data = []

with open("./Sarcasm.json", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
df = df[['headline', 'is_sarcastic']]

print("✅ Dataset loaded")

# -------------------------------
# STEP 2: PREPROCESSING
# -------------------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['headline'] = df['headline'].apply(preprocess)

# -------------------------------
# STEP 3: TRAIN SARCASM MODEL
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['headline'])
y = df['is_sarcastic']

model = LogisticRegression()
model.fit(X, y)

print("✅ Sarcasm model trained")

# -------------------------------
# STEP 4: KNOWLEDGE BASE (REAL CHATBOT)
# -------------------------------
knowledge = [
    {"q": "what is nlp", "a": "NLP stands for Natural Language Processing."},
    {"q": "what is ai", "a": "AI is the simulation of human intelligence in machines."},
    {"q": "applications of ai", "a": "AI is used in healthcare, chatbots, robotics, and more."},
    {"q": "what is machine learning", "a": "Machine learning allows systems to learn from data."},
    {"q": "what is sarcasm", "a": "Sarcasm means saying something but meaning the opposite."}
]

questions = [item["q"] for item in knowledge]
answers = [item["a"] for item in knowledge]

chat_vectorizer = TfidfVectorizer()
chat_X = chat_vectorizer.fit_transform(questions)

# -------------------------------
# STEP 5: CHATBOT RESPONSE (SMART)
# -------------------------------
def get_response(user_input):
    user_vec = chat_vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, chat_X)

    idx = np.argmax(similarity)

    if similarity[0][idx] < 0.3:
        return "I’m not sure, but I can learn that!"

    return answers[idx]

# -------------------------------
# STEP 6: SARCASM DETECTION
# -------------------------------
def detect_sarcasm(text):
    processed = preprocess(text)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]

    return "⚠️ Sarcastic" if pred == 1 else "😊 Not Sarcastic"

# -------------------------------
# STEP 7: MAIN CHATBOT LOOP
# -------------------------------
print("\n🤖 Smart Chatbot Ready! (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    response = get_response(user_input)
    tone = detect_sarcasm(user_input)

    print("Bot:", response)
    print("Tone:", tone)
    print()