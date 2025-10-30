# scripts/generate_insights.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from prophet import Prophet

df = pd.read_csv("data/processed/feedbacks_cleaned.csv")
# Topic modeling
cv = CountVectorizer(max_features=1000, stop_words='english')
X = cv.fit_transform(df['processed'])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)
terms = cv.get_feature_names_out()
topics = []
for i, comp in enumerate(lda.components_):
    terms_idx = comp.argsort()[-8:]
    topics.append([terms[t] for t in terms_idx])
    print(f"Topic {i}: ", [terms[t] for t in terms_idx])

# Forecast monthly avg rating using Prophet
df['created_at'] = pd.to_datetime(df['created_at'])
monthly = df.set_index('created_at').resample('M').rating.mean().reset_index()
monthly.columns = ['ds','y']

m = Prophet()
m.fit(monthly)
future = m.make_future_dataframe(periods=30)  # days; adjust if months needed
forecast = m.predict(future)

# plot
plt.figure(figsize=(8,4))
plt.plot(monthly['ds'], monthly['y'], label='actual')
plt.plot(forecast['ds'], forecast['yhat'], label='forecast')
plt.legend()
plt.tight_layout()
plt.savefig("reports/forecast.png")
print("Saved reports/forecast.png")
