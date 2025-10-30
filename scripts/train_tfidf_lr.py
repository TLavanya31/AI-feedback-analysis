# scripts/train_tfidf_lr.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

df = pd.read_csv("data/processed/feedbacks_cleaned.csv")
def map_label(r):
    if r >= 4: return 2
    if r == 3: return 1
    return 0
df['label'] = df['rating'].apply(map_label)

X_train, X_val, y_train, y_val = train_test_split(df['processed'], df['label'], test_size=0.15, random_state=42, stratify=df['label'])

vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_train_t = vec.fit_transform(X_train)
X_val_t = vec.transform(X_val)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_t, y_train)
preds = clf.predict(X_val_t)
print(classification_report(y_val, preds, target_names=['neg','neu','pos']))

os.makedirs("models", exist_ok=True)
joblib.dump(vec, "models/tfidf_vectorizer.joblib")
joblib.dump(clf, "models/sentiment_model.pkl")
print("Saved TF-IDF vectorizer and model to models/")

