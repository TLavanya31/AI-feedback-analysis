# scripts/train_distilbert.py
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("data/processed/feedbacks_cleaned.csv")

# map rating to labels
def map_label(r):
    if r >= 4: return 2
    if r == 3: return 1
    return 0

df['label'] = df['rating'].apply(map_label)
df = df[['processed','label']].dropna().reset_index(drop=True)
train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])

train_ds = Dataset.from_pandas(train_df.rename(columns={'processed':'text'}))
val_ds = Dataset.from_pandas(val_df.rename(columns={'processed':'text'}))

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds = train_ds.remove_columns(["text","__index_level_0__"])
val_ds = val_ds.remove_columns(["text","__index_level_0__"])

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

args = TrainingArguments(
    output_dir="models/distilbert-sentiment",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(labels, preds)
    p_, r_, f_, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy": acc, "precision": p_, "recall": r_, "f1": f_}

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
trainer.train()
trainer.save_model("models/sentiment_model")
tokenizer.save_pretrained("models/sentiment_model")
print("Saved model in models/sentiment_model")
