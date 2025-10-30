# scripts/summarizer.py

from transformers import pipeline

# Load model only once
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_feedback(text):
    """
    Summarizes customer feedback text using DistilBART.
    Returns a short summary string.
    """
    if not text or text.strip() == "":
        return "No feedback provided."

    try:
        result = summarizer(text, max_length=60, min_length=20, do_sample=False)
        summary = result[0]['summary_text']
        return summary
    except Exception as e:
        return f"Summarization failed: {e}"
