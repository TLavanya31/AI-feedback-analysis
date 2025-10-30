# scripts/simulate_data.py
import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta
import os

os.makedirs("../data/raw", exist_ok=True)  # when run from scripts folder, adjust if needed

faker = Faker()

templates = [
    "The product quality is great, I loved the packaging.",
    "Delivery was late and support didn't help — very disappointed.",
    "Okay experience, nothing special.",
    "Amazing service and quick response from customer care!",
    "I had repeated issues with login and checkout.",
    "Product stopped working after a week, need replacement.",
    "Excellent experience. Will buy again!",
    "Customer service was rude and unhelpful.",
    "The UI is confusing and checkout failed multiple times.",
    "Received damaged product; refund process was slow."
]

rows = []
for i in range(1200):
    base = random.choice(templates)
    extra = " ".join(faker.sentences(nb=random.randint(0,3)))
    rating = random.choices([1,2,3,4,5], weights=[10,15,30,25,20])[0]
    created_at = faker.date_time_between(start_date='-6M', end_date='now')
    source = random.choice(['email','chat','twitter','facebook','survey'])
    text = f"{base} {extra}"
    rows.append({
        "id": i,
        "source": source,
        "text": text,
        "rating": rating,
        "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S")
    })

df = pd.DataFrame(rows)
os.makedirs("../data/raw", exist_ok=True)
df.to_csv("../data/raw/feedbacks.csv", index=False)
print("Saved ../data/raw/feedbacks.csv with", len(df), "rows")
