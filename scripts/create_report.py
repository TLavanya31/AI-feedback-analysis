# scripts/create_report.py
import matplotlib.pyplot as plt
from PIL import Image
import os

os.makedirs("reports", exist_ok=True)

fig = plt.figure(figsize=(8.27, 11.69))  # A4 inches
plt.axis('off')
plt.text(0.05, 0.95, "AI Insights Report", fontsize=16, weight='bold')
plt.text(0.05, 0.90, "Summary of findings and recommendations:", fontsize=12)
plt.text(0.05, 0.85, "- Top recurring issues: login failures, delivery delay, damaged items", fontsize=10)
plt.text(0.05, 0.82, "- Forecast: next month rating expected to be around X (from Prophet)", fontsize=10)
if os.path.exists("reports/forecast.png"):
    img = Image.open("reports/forecast.png")
    plt.imshow(img, aspect='auto', extent=(0.05,0.95,0.2,0.7))
plt.savefig("reports/AI_insights_report.pdf", bbox_inches='tight')
print("Saved reports/AI_insights_report.pdf")
