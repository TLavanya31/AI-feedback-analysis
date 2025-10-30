# app/app.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from scripts.summarizer import summarize_feedback

st.set_page_config(page_title="AI Customer Feedback Analysis", layout="wide")
st.title("🧠 AI Customer Feedback Analysis System")

st.subheader("📝 Summarize a Single Feedback")
st.write("Enter customer feedback below to generate a summary using an AI model.")

feedback_text = st.text_area("✍️ Enter customer feedback:", height=200)

if st.button("Generate Summary"):
    with st.spinner("Summarizing... Please wait..."):
        try:
            summary = summarize_feedback(feedback_text)
            st.success(summary)
        except Exception as e:
            st.error(f"Summarization failed: {e}")

st.markdown("---")

st.subheader("📊 Analyze Sentiment from CSV File")

uploaded = st.file_uploader("Upload CSV (must contain 'text' column)", type="csv")

MODEL_PATH = Path("models")
tfidf_exists = (MODEL_PATH / "tfidf_vectorizer.joblib").exists()
hf_exists = (MODEL_PATH / "sentiment_model").exists()

@st.cache_resource
def load_models():
    vec = None
    clf = None
    hf_pipe = None

    try:
        if tfidf_exists:
            vec = joblib.load(str(MODEL_PATH / "tfidf_vectorizer.joblib"))
            clf = joblib.load(str(MODEL_PATH / "sentiment_model.pkl"))
        else:
            from transformers import pipeline
            hf_pipe = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1
            )
    except Exception as e:
        st.warning(f"⚠️ Could not load model: {e}")
        hf_pipe = None

    return vec, clf, hf_pipe


vec, clf, hf_pipe = load_models()

if uploaded:
    df = pd.read_csv(uploaded)

    if 'text' not in df.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        st.write("📄 Sample rows:")
        st.dataframe(df.head())

        if st.button("Run Sentiment Analysis"):
            with st.spinner("Running Sentiment Analysis..."):
                if vec is not None and clf is not None:
                    X = vec.transform(df['text'].fillna("").astype(str))
                    preds = clf.predict(X)
                    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
                    df['sentiment'] = [mapping.get(p, "Unknown") for p in preds]
                elif hf_pipe is not None:
                    results = [hf_pipe(t)[0] for t in df['text'].fillna("").astype(str)]
                    df['sentiment'] = [r['label'] for r in results]
                else:
                    st.warning("No model found. Please train and place model files in models/.")
                    df['sentiment'] = "N/A"

                st.write("✅ Sentiment Results:")
                st.dataframe(df[['text', 'sentiment']].head(50))
                st.write("📈 Sentiment Distribution:")
                st.bar_chart(df['sentiment'].value_counts())

        if st.button("Summarize Longest Feedback"):
            st.info("Finding the longest feedback entry...")
            long_text = df['text'].astype(str).loc[df['text'].astype(str).str.len().idxmax()]
            st.write("🗒️ Longest Feedback:")
            st.write(long_text)

            try:
                if os.path.exists("scripts/extractive_summary.py"):
                    from scripts.extractive_summary import extractive_summary
                    summary = extractive_summary(long_text, top_n=3)
                else:
                    summary = long_text[:300] + "..."
                st.subheader("📄 Extractive Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"Summarization failed: {e}")

        if 'sentiment' in df.columns:
            st.markdown("---")
            st.download_button(
                "⬇️ Download Processed CSV",
                data=df.to_csv(index=False),
                file_name="feedbacks_with_sentiment.csv",
                mime="text/csv"
            )
