# scripts/extractive_summary.py
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

nltk.download('punkt')

def extractive_summary(text, top_n=3):
    sents = nltk.sent_tokenize(text)
    if len(sents) <= top_n:
        return " ".join(sents)
    tfidf = TfidfVectorizer().fit_transform(sents)
    centroid = tfidf.mean(axis=0)
    scores = linear_kernel(tfidf, centroid).flatten()
    top_idx = scores.argsort()[-top_n:][::-1]
    top_idx_sorted = sorted(top_idx)
    return " ".join([sents[i] for i in top_idx_sorted])

# example
if __name__ == "__main__":
    txt = "Your long text here..."
    print(extractive_summary(txt, top_n=3))
