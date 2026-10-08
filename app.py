"""Flask web app that serves the trained food-adulteration models.

Run:  python train.py   (once, creates models/)
      python app.py     (starts server on http://127.0.0.1:5000)
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"
TARGETS = ["severity", "health_risk", "action_taken"]

app = Flask(__name__)

if not (MODELS_DIR / "metadata.json").exists():
    raise SystemExit("Models not found. Run `python train.py` first.")

META = json.loads((MODELS_DIR / "metadata.json").read_text())
MODELS = {t: joblib.load(MODELS_DIR / f"{t}_model.joblib") for t in TARGETS}
CAT_FIELDS = list(META["options"].keys())


def build_row(payload: dict) -> pd.DataFrame:
    """Validate input and convert it into the model's feature frame."""
    row = {}
    for field in CAT_FIELDS:
        value = payload.get(field)
        if value not in META["options"][field]:
            raise ValueError(f"Invalid or missing value for '{field}'.")
        row[field] = value
    date_str = payload.get("detection_date")
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
    except (TypeError, ValueError):
        raise ValueError("detection_date must be in YYYY-MM-DD format.")
    row["month"] = d.month
    row["day_of_week"] = d.weekday()
    return pd.DataFrame([row])


@app.route("/")
def index():
    return render_template("index.html", options=META["options"], summary=META["summary"])


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", eda=META["eda"], summary=META["summary"])


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        X = build_row(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = {}
    for target, model in MODELS.items():
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        result[target] = {
            "prediction": str(classes[proba.argmax()]),
            "probabilities": {str(c): round(float(p), 3) for c, p in zip(classes, proba)},
            "model": META["summary"][target]["best_model"],
            "test_accuracy": META["summary"][target]["test_accuracy"],
            "baseline_accuracy": META["summary"][target]["baseline_accuracy"],
        }
    return jsonify(result)


@app.route("/api/metrics")
def metrics():
    return jsonify(META["summary"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
