# scripts/data_preprocessing.py
import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

infile = os.path.join("data","raw","feedbacks.csv")
outfile = os.path.join("data","processed","feedbacks_cleaned.csv")

df = pd.read_csv(infile)

# Drop duplicates and rows with empty text
df = df.drop_duplicates(subset=['text']).dropna(subset=['text']).reset_index(drop=True)

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r'http\S+',' ', s)            # remove urls
    s = re.sub(r'[^a-z0-9\s]', ' ', s)      # remove special chars
    s = re.sub(r'\s+', ' ', s).strip()
    return s

df['clean_text'] = df['text'].apply(clean_text)

stop = set(stopwords.words('english'))
lemm = WordNetLemmatizer()

def preprocess(s):
    tokens = nltk.word_tokenize(s)
    tokens = [lemm.lemmatize(t) for t in tokens if t not in stop and len(t) > 1]
    return " ".join(tokens)

df['processed'] = df['clean_text'].apply(preprocess)

os.makedirs(os.path.dirname(outfile), exist_ok=True)
df.to_csv(outfile, index=False)
print("Saved cleaned file to", outfile)
