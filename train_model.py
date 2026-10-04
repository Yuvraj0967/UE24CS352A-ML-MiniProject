import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Sample Gaming Sentiment Dataset
messages = [
    "gg EZ trash team uninstall", "you are terrible at this game", "report feeder bot",
    "great job team well played", "nice play dude gg", "good game everyone",
    "going mid now", "defend top tower", "buying item wait", "be right back",
    "stop dying feed team", "uninstall game useless player", "clutch win nice work"
]

labels = [
    "Derogatory / Toxic", "Derogatory / Toxic", "Derogatory / Toxic",
    "Positive", "Positive", "Positive",
    "Neutral", "Neutral", "Neutral", "Neutral",
    "Derogatory / Toxic", "Derogatory / Toxic", "Positive"
]

# 2. Vectorize Text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(messages)

# 3. Train Classifier
model = MultinomialNB()
model.fit(X, labels)

# 4. Save Model Artifacts
os.makedirs("models", exist_ok=True)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("SUCCESS: Model and Vectorizer saved inside 'models/' folder!")