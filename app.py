import json, numpy as np, pandas as pd, joblib
from flask import Flask, render_template_string, request

model    = joblib.load('model.pkl')
scaler   = joblib.load('scaler.pkl')
features = json.load(open('features.json'))
samples  = pd.read_csv('Cleaned-Data.csv').drop(
    columns=['Country','Gender_Female','Gender_Male','Gender_Transgender']
).head(10).to_html(index=False, border=0, classes='table')

LABELS = {0: "Mild / Asymptomatic", 1: "Moderate symptoms",
          2: "Severe symptoms",     3: "High-risk contact"}

# Symptom features (plain 0/1 checkboxes)
SYMPTOMS = ['Fever','Tiredness','Dry-Cough','Difficulty-in-Breathing',
            'Sore-Throat','None_Sympton','Pains','Nasal-Congestion','Runny-Nose','Diarrhea',
            'None_Experiencing']
AGE_OPTS  = ['Age_0-9','Age_10-19','Age_20-24','Age_25-59','Age_60+']
SEV_OPTS  = ['Severity_Mild','Severity_Moderate','Severity_None','Severity_Severe']
CON_OPTS  = ['Contact_Dont-Know','Contact_No','Contact_Yes']

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>COVID-19 Clustering</title>
  <style>
    body  { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #222; }
    h1    { color: #2c3e50; }
    h2    { color: #2980b9; border-bottom: 2px solid #eee; padding-bottom: 6px; margin-top: 30px; }
    p     { line-height: 1.7; }
    img   { max-width: 100%; border-radius: 8px; margin: 10px 0; }
    .table { border-collapse: collapse; width: 100%; font-size: .82rem; overflow-x: auto; display: block; }
    .table th { background: #2c3e50; color: white; padding: 8px; text-align: left; }
    .table td { padding: 6px 8px; border-bottom: 1px solid #ddd; }
    .checks { display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0; }
    .checks label { background: #f4f4f4; border-radius: 6px; padding: 6px 12px;
                    cursor: pointer; font-size: .88rem; user-select: none; }
    .checks input { margin-right: 5px; }
    .row  { display: flex; gap: 20px; flex-wrap: wrap; margin: 12px 0; }
    .row select { padding: 7px 10px; border-radius: 6px; border: 1px solid #ccc;
                  font-size: .9rem; flex: 1; min-width: 160px; }
    button { background: #2980b9; color: white; border: none; padding: 10px 28px;
             border-radius: 6px; font-size: 1rem; cursor: pointer; margin-top: 10px; }
    button:hover { background: #1a6fa0; }
    .result { margin-top: 16px; padding: 14px 18px; background: #eaf6ea;
              border-left: 4px solid #27ae60; border-radius: 6px; font-size: 1.05rem; }
  </style>
</head>
<body>
  <h1>🦠 COVID-19 Symptoms — K-Means Clustering</h1>

  <h2>📂 Dataset</h2>
  <p><strong>COVID-19 Symptoms &amp; Presence of Disease in Details</strong> — Kaggle<br>
  316 800 rows, 23 features: symptoms, age group, severity, contact history.</p>

  <h2>🤖 Unsupervised Learning</h2>
  <p>Unsupervised learning finds hidden patterns in data <strong>without labels</strong>.
  Here we use <strong>K-Means Clustering</strong> to group patients by symptom profile.
  The optimal K=4 was selected with the Elbow Method.</p>

  <h2>📋 Project</h2>
  <p>Data was one-hot encoded, scaled with StandardScaler, then clustered into 4 groups.
  PCA reduced the space to 2D for visualization. Model saved with joblib.</p>

  <h2>📊 Figure</h2>
  <img src="/figure" alt="Clustering figure"/>

  <h2>🔢 10 Sample Rows</h2>
  {{ samples | safe }}

  <h2>🔬 Predict Cluster</h2>
  <form method="POST" action="/predict">

    <p><strong>Symptoms:</strong></p>
    <div class="checks">
      {% for s in symptoms %}
      <label><input type="checkbox" name="{{ s }}" value="1"> {{ s }}</label>
      {% endfor %}
    </div>

    <p><strong>Age / Severity / Contact:</strong></p>
    <div class="row">
      <select name="age">
        {% for a in age_opts %}<option value="{{ a }}">{{ a }}</option>{% endfor %}
      </select>
      <select name="severity">
        {% for s in sev_opts %}<option value="{{ s }}">{{ s }}</option>{% endfor %}
      </select>
      <select name="contact">
        {% for c in con_opts %}<option value="{{ c }}">{{ c }}</option>{% endfor %}
      </select>
    </div>

    <button type="submit">Predict</button>
  </form>

  {% if result is not none %}
  <div class="result">✅ Cluster <strong>{{ result.cluster }}</strong> — {{ result.label }}</div>
  {% endif %}
</body>
</html>
"""

app = Flask(__name__)

def build_vector(form):
    vec = {}
    for f in features:
        vec[f] = 0
    for s in SYMPTOMS:
        if form.get(s):
            vec[s] = 1
    vec[form.get('age', AGE_OPTS[0])]  = 1
    vec[form.get('severity', SEV_OPTS[0])] = 1
    vec[form.get('contact', CON_OPTS[0])]  = 1
    return [vec[f] for f in features]

@app.route('/')
def index():
    return render_template_string(HTML, samples=samples, symptoms=SYMPTOMS,
        age_opts=AGE_OPTS, sev_opts=SEV_OPTS, con_opts=CON_OPTS, result=None)

@app.route('/predict', methods=['POST'])
def predict():
    values = build_vector(request.form)
    cluster = int(model.predict(scaler.transform([values]))[0])
    result = {'cluster': cluster, 'label': LABELS[cluster]}
    return render_template_string(HTML, samples=samples, symptoms=SYMPTOMS,
        age_opts=AGE_OPTS, sev_opts=SEV_OPTS, con_opts=CON_OPTS, result=result)

@app.route('/figure')
def figure():
    from flask import send_file
    return send_file('figure.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
