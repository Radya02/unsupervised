import json, numpy as np, joblib

model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
features = json.load(open('features.json'))

LABELS = {
    0: "Mild / Asymptomatic",
    1: "Moderate symptoms",
    2: "Severe symptoms",
    3: "High-risk contact",
}

print("COVID-19 Cluster Predictor — enter 0 or 1 for each feature\n")
values = []
for f in features:
    v = input(f"  {f:<32} [0/1]: ").strip() or "0"
    values.append(int(v))

x = scaler.transform([values])
cluster = int(model.predict(x)[0])
print(f"\n✅  Cluster: {cluster} — {LABELS[cluster]}")
