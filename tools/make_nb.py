import json

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# Food Adulteration Analysis using Machine Learning

This notebook covers the whole project in one place: data loading, exploratory analysis, model training and comparison, evaluation, saving the models, and running a **Flask web app** inside Google Colab.

**Targets:** `severity`, `health_risk`, `action_taken`
**Features:** product, brand, category, adulterant, detection method, detection date (month and day-of-week)

> **Heads-up:** this dataset looks randomly generated, so the models are expected to perform at roughly the level of random guessing. The notebook shows how to check that properly.
""")

md("## 1. Setup")
code("""
!pip install -q flask scikit-learn pandas matplotlib joblib scipy
""")
code("""
import json, warnings, threading
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
Path("models").mkdir(exist_ok=True)
""")

md("## 2. Load the data\nUpload `food_adulteration_data.csv` when asked (in Colab). If the file is already in the working folder, it is used directly.")
code("""
from pathlib import Path

CSV = "food_adulteration_data.csv"
if not Path(CSV).exists() and Path("data", CSV).exists():
    CSV = str(Path("data", CSV))

if not Path(CSV).exists():
    try:
        from google.colab import files
        up = files.upload()               # choose food_adulteration_data.csv
        CSV = next(iter(up))
    except ImportError:
        raise FileNotFoundError("Put food_adulteration_data.csv next to this notebook.")

df = pd.read_csv(CSV)
print(df.shape)
df.head()
""")
code("""
print(df.dtypes, "\\n")
print("Missing values:\\n", df.isna().sum(), "\\n")
print("Duplicate IDs:", df["adulteration_id"].duplicated().sum())
""")

md("## 3. Cleaning and feature engineering")
code("""
df = df.drop_duplicates(subset="adulteration_id").dropna().copy()
df["detection_date"] = pd.to_datetime(df["detection_date"], format="%m/%d/%Y")
df["month"] = df["detection_date"].dt.month
df["day_of_week"] = df["detection_date"].dt.dayofweek

CAT_FEATURES = ["product_name", "brand", "category", "adulterant", "detection_method"]
NUM_FEATURES = ["month", "day_of_week"]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGETS = ["severity", "health_risk", "action_taken"]

print("Date range:", df.detection_date.min().date(), "to", df.detection_date.max().date())
df[FEATURES + TARGETS].head()
""")

md("## 4. Exploratory data analysis")
code("""
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), ["adulterant", "category", "severity", "health_risk", "action_taken", "detection_method"]):
    df[col].value_counts().plot(kind="bar", ax=ax, color="#2f7d4f")
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.show()
""")
code("""
fig, ax = plt.subplots(figsize=(7, 3))
df.groupby("month").size().plot(kind="bar", ax=ax, color="#2563eb")
ax.set_title("Records per month"); plt.show()
""")

md("""
### 4.1 Is there any signal in the data?
A chi-square test of independence checks whether a feature is related to a target. A **p-value below 0.05** would suggest a real relationship.
""")
code("""
rows = []
for t in TARGETS:
    for f in CAT_FEATURES:
        p = chi2_contingency(pd.crosstab(df[f], df[t]))[1]
        rows.append({"target": t, "feature": f, "p_value": round(p, 3)})
pvals = pd.DataFrame(rows).pivot(index="feature", columns="target", values="p_value")
print(pvals)
print("\\nSmallest p-value:", pvals.values.min())
print("Severity vs health_risk p-value:", round(chi2_contingency(pd.crosstab(df.severity, df.health_risk))[1], 3))
""")
code("""
# Logical consistency check: which products fall under which category?
pd.crosstab(df["product_name"], df["category"])
""")
md("If products such as Butter or Wine are spread across Meat, Dairy and Beverages, the data is internally inconsistent, which is another sign it was generated randomly.")

md("## 5. Build the preprocessing pipeline and candidate models")
code("""
def make_pipeline(model):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])
    return Pipeline([("pre", pre), ("clf", model)])

def candidates():
    return {
        "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=2,
                                                        random_state=RANDOM_STATE),
    }
""")

md("## 6. Train, cross-validate and evaluate\nFor each target we compare the models with 5-fold cross-validation, score the best one on a held-out 20% test set, and compare it with the majority-class baseline.")
code("""
X = df[FEATURES]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
summary, trained, splits = {}, {}, {}

for target in TARGETS:
    y = df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    splits[target] = (X_te, y_te)

    print(f"\\n=== {target} ===")
    cv_scores = {}
    for name, model in candidates().items():
        s = cross_val_score(make_pipeline(model), X_tr, y_tr, cv=cv, scoring="accuracy")
        cv_scores[name] = float(s.mean())
        print(f"  {name:26s} CV accuracy = {s.mean():.3f} (+/- {s.std():.3f})")

    real = {k: v for k, v in cv_scores.items() if not k.startswith("Baseline")}
    best_name = max(real, key=real.get)
    best = make_pipeline(candidates()[best_name]).fit(X_tr, y_tr)
    base = make_pipeline(candidates()["Baseline (most frequent)"]).fit(X_tr, y_tr)

    pred = best.predict(X_te)
    summary[target] = {
        "best_model": best_name,
        "cv_scores": cv_scores,
        "test_accuracy": round(accuracy_score(y_te, pred), 4),
        "test_macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
        "baseline_accuracy": round(accuracy_score(y_te, base.predict(X_te)), 4),
    }
    print(f"  -> best: {best_name} | test acc = {summary[target]['test_accuracy']} "
          f"| baseline acc = {summary[target]['baseline_accuracy']}")
    print(classification_report(y_te, pred, zero_division=0))

    # refit on all data for the final model
    trained[target] = make_pipeline(candidates()[best_name]).fit(X, y)
""")

md("## 7. Results")
code("""
res = pd.DataFrame(summary).T[["best_model", "test_accuracy", "test_macro_f1", "baseline_accuracy"]]
res
""")
code("""
fig, ax = plt.subplots(figsize=(7, 3.5))
x = np.arange(len(TARGETS))
ax.bar(x - 0.2, [summary[t]["test_accuracy"] for t in TARGETS], 0.4, label="Best model", color="#2f7d4f")
ax.bar(x + 0.2, [summary[t]["baseline_accuracy"] for t in TARGETS], 0.4, label="Majority baseline", color="#9ca3af")
ax.set_xticks(x); ax.set_xticklabels(TARGETS); ax.set_ylabel("Test accuracy"); ax.set_ylim(0, 0.5); ax.legend()
plt.show()
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, t in zip(axes, TARGETS):
    X_te, y_te = splits[t]
    ConfusionMatrixDisplay.from_estimator(trained[t], X_te, y_te, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{t} ({summary[t]['best_model']})")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.show()
""")
md("""
**How to read this:** if the best model is not clearly better than the majority baseline, it has learned nothing useful. With 200 test rows, a difference of 1 percentage point is just 2 predictions, which is noise. Together with the chi-square p-values above, the conclusion is that this dataset has no learnable signal. Better data (lab measurements, adulterant concentration, region, inspection history) is needed for a useful model.
""")

md("## 8. Save the models")
code("""
for t, m in trained.items():
    joblib.dump(m, f"models/{t}_model.joblib")

options = {c: sorted(df[c].unique().tolist()) for c in CAT_FEATURES}
json.dump({"summary": summary, "options": options}, open("models/metadata.json", "w"), indent=2)
print("Saved:", sorted(p.name for p in Path("models").iterdir()))
""")

md("## 9. Predict on a new case")
code("""
def predict_case(product_name, brand, category, adulterant, detection_method, detection_date):
    d = pd.to_datetime(detection_date)
    row = pd.DataFrame([{
        "product_name": product_name, "brand": brand, "category": category,
        "adulterant": adulterant, "detection_method": detection_method,
        "month": d.month, "day_of_week": d.dayofweek,
    }])
    out = {}
    for t, m in trained.items():
        proba = m.predict_proba(row)[0]
        out[t] = {"prediction": str(m.classes_[proba.argmax()]),
                  "probabilities": {str(c): round(float(p), 3) for c, p in zip(m.classes_, proba)}}
    return out

sample = {k: v[0] for k, v in options.items()}
print("Input:", sample)
predict_case(**sample, detection_date="2024-05-10")
""")

md("""
## 10. Flask web app inside Colab
Colab cannot open `localhost` directly, so the app runs in a background thread and is shown through Colab's proxy. `debug=False` and `use_reloader=False` are required.
""")
code("""
from flask import Flask, jsonify, request

app = Flask(__name__)

PAGE = '''
<!doctype html><html><head><meta charset="utf-8"><title>Food Adulteration ML</title>
<style>
 body{font-family:system-ui,sans-serif;background:#f6f7f9;margin:0;color:#1f2933}
 header{background:#2f7d4f;color:#fff;padding:14px 24px;font-weight:600}
 main{max-width:860px;margin:20px auto;padding:0 16px}
 .card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:18px;margin-bottom:16px}
 .notice{background:#fffbeb;border-color:#fcd34d;color:#78350f}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
 label{font-size:12px;color:#6b7280;display:block;margin-bottom:3px}
 select,input,button{width:100%;padding:8px;border:1px solid #e5e7eb;border-radius:6px;font-size:14px}
 button{background:#2f7d4f;color:#fff;border:none;font-weight:600;margin-top:12px;cursor:pointer}
 .bar{height:8px;background:#e5e7eb;border-radius:4px;overflow:hidden}.bar span{display:block;height:100%;background:#2f7d4f}
 td{padding:4px 6px;font-size:14px}.small{font-size:12px;color:#6b7280}
</style></head><body><header>Food Adulteration Analyzer</header><main>
<div class="card notice"><b>Please read:</b> on this dataset the models perform at about the level of random guessing.
Treat results as a demo of the ML pipeline, not as real food-safety advice.</div>
<div class="card"><form id="f"><div class="grid">__FIELDS__
<div><label>Detection date</label><input type="date" name="detection_date" id="d" required></div></div>
<button>Predict</button></form></div><div id="out"></div>
<script>
document.getElementById('d').valueAsDate=new Date();
document.getElementById('f').onsubmit=async e=>{e.preventDefault();
 const r=await fetch('predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(Object.fromEntries(new FormData(e.target)))});
 const j=await r.json();const o=document.getElementById('out');
 if(!r.ok){o.innerHTML='<div class="card">Error: '+j.error+'</div>';return}
 o.innerHTML=Object.entries(j).map(([t,x])=>`<div class="card"><h3 style="margin-top:0">${t.replace('_',' ').toUpperCase()}: ${x.prediction}</h3><table style="width:100%">`+
 Object.entries(x.probabilities).map(([c,p])=>`<tr><td style="width:35%">${c}</td><td><div class="bar"><span style="width:${p*100}%"></span></div></td><td style="width:60px">${(p*100).toFixed(1)}%</td></tr>`).join('')+
 `</table><p class="small">Model: ${x.model} | test accuracy ${(x.test_accuracy*100).toFixed(1)}% | baseline ${(x.baseline_accuracy*100).toFixed(1)}%</p></div>`).join('')}
</script></main></body></html>
'''

def build_page():
    fields = "".join(
        f'<div><label>{k.replace("_"," ").title()}</label><select name="{k}">'
        + "".join(f"<option>{v}</option>" for v in vals) + "</select></div>"
        for k, vals in options.items())
    return PAGE.replace("__FIELDS__", fields)

@app.route("/")
def home():
    return build_page()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    for k, vals in options.items():
        if data.get(k) not in vals:
            return jsonify({"error": f"Invalid or missing value for '{k}'."}), 400
    try:
        out = predict_case(**{k: data[k] for k in options}, detection_date=data.get("detection_date"))
    except Exception:
        return jsonify({"error": "detection_date must be in YYYY-MM-DD format."}), 400
    for t in out:
        out[t].update(model=summary[t]["best_model"], test_accuracy=summary[t]["test_accuracy"],
                      baseline_accuracy=summary[t]["baseline_accuracy"])
    return jsonify(out)

@app.route("/api/metrics")
def api_metrics():
    return jsonify(summary)
""")
code("""
threading.Thread(target=lambda: app.run(port=5000, debug=False, use_reloader=False), daemon=True).start()

try:
    from google.colab import output
    output.serve_kernel_port_as_iframe(5000, height=850)   # shows the app below this cell
    # output.serve_kernel_port_as_window(5000)             # or open it in a new tab
except ImportError:
    print("Running locally: open http://127.0.0.1:5000")
""")
md("### Test the API directly (optional)")
code("""
import time, requests
time.sleep(1)
payload = {**{k: v[0] for k, v in options.items()}, "detection_date": "2024-05-10"}
r = requests.post("http://127.0.0.1:5000/predict", json=payload)
print(r.status_code)
print(json.dumps(r.json(), indent=2)[:900])
""")
md("""
## 11. Conclusion
- A full pipeline was built: cleaning, feature engineering, model comparison, evaluation, saving and a Flask app.
- All models score about the same as the majority-class baseline, and chi-square tests find no relationship between inputs and outputs, so the dataset appears to be randomly generated.
- To get a useful system, replace the CSV with real data that has measurable features and rerun the notebook from the top.
""")

nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(cells):
    c["id"] = f"cell{i:03d}"
json.dump(nb, open("/mnt/user-data/outputs/Food_Adulteration_ML.ipynb", "w"), indent=1)
print(len(cells), "cells")

# --- test: run all code cells as a script (drop shell/magic lines) ---
src = []
for c in cells:
    if c["cell_type"] == "code":
        lines = [l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("!", "%"))]
        src.append("\n".join(lines))
open("/tmp/nb_test.py", "w").write("\n\n".join(src))
