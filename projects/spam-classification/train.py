from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

texts = [
    "win free money now", "cheap loans available", "free prize waiting",
    "meeting tomorrow at office", "project deadline update", "lunch with team"
]
labels = [1, 1, 1, 0, 0, 0]
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(texts, labels)
print(model.predict(["free cash prize", "team meeting schedule"]))
