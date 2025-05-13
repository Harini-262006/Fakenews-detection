
import pandas as pd
import string
import nltk
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Download stopwords
nltk.download("stopwords")

# Load and clean data
df = pd.read_csv("news.csv", usecols=["title", "text", "label"], low_memory=False)
df.dropna(inplace=True)
df["content"] = df["title"] + " " + df["text"]

def clean_text(text):
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    return text

df["content"] = df["content"].apply(clean_text)

# Check class balance
print(df["label"].value_counts())

# Features and labels
X = df["content"]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF with more features
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model with class balancing
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
print("Model and vectorizer saved successfully.")

