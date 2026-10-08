"""Train models for the Food Adulteration project.

Three classification targets are trained independently:
  * severity      (Minor / Moderate / Severe)
  * health_risk   (Low / Medium / High)
  * action_taken  (Investigation Launched / Product Recall / Fine Imposed / Warning Issued)

For each target, several algorithms are compared with stratified 5-fold CV,
the best one is refit on a train split, evaluated on a held-out test set,
and saved with joblib. Run:  python train.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
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
BASE = Path(__file__).parent
DATA = BASE / "data" / "food_adulteration_data.csv"
MODELS = BASE / "models"
STATIC = BASE / "static"
MODELS.mkdir(exist_ok=True)
STATIC.mkdir(exist_ok=True)

CAT_FEATURES = ["product_name", "brand", "category", "adulterant", "detection_method"]
NUM_FEATURES = ["month", "day_of_week"]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGETS = ["severity", "health_risk", "action_taken"]
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df = df.drop_duplicates(subset="adulteration_id").dropna()
    df["detection_date"] = pd.to_datetime(df["detection_date"], format="%m/%d/%Y")
    df["month"] = df["detection_date"].dt.month
    df["day_of_week"] = df["detection_date"].dt.dayofweek
    return df


def make_pipeline(model) -> Pipeline:
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


def main():
    df = load_data()
    X = df[FEATURES]
    summary = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for target in TARGETS:
        y = df[target]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

        print(f"\n=== Target: {target} ===")
        cv_scores = {}
        for name, model in candidates().items():
            s = cross_val_score(make_pipeline(model), X_tr, y_tr, cv=cv, scoring="accuracy")
            cv_scores[name] = float(s.mean())
            print(f"  {name:28s} CV accuracy = {s.mean():.3f} (+/- {s.std():.3f})")

        # pick best non-baseline model
        real = {k: v for k, v in cv_scores.items() if not k.startswith("Baseline")}
        best_name = max(real, key=real.get)
        best = make_pipeline(candidates()[best_name]).fit(X_tr, y_tr)
        base = make_pipeline(candidates()["Baseline (most frequent)"]).fit(X_tr, y_tr)

        pred = best.predict(X_te)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average="macro")
        base_acc = accuracy_score(y_te, base.predict(X_te))
        print(f"  -> best: {best_name} | test acc={acc:.3f} macro-F1={f1:.3f} "
              f"(baseline acc={base_acc:.3f})")
        print(classification_report(y_te, pred, zero_division=0))

        # refit on all data for deployment
        final = make_pipeline(candidates()[best_name]).fit(X, y)
        joblib.dump(final, MODELS / f"{target}_model.joblib")

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{target} - {best_name}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(STATIC / f"cm_{target}.png", dpi=110)
        plt.close(fig)

        summary[target] = {
            "best_model": best_name,
            "cv_scores": cv_scores,
            "test_accuracy": round(acc, 4),
            "test_macro_f1": round(f1, 4),
            "baseline_accuracy": round(base_acc, 4),
            "classes": sorted(y.unique().tolist()),
        }

    # options for the web form dropdowns + EDA stats for the dashboard
    options = {c: sorted(df[c].unique().tolist()) for c in CAT_FEATURES}
    eda = {
        "n_records": int(len(df)),
        "date_range": [str(df.detection_date.min().date()), str(df.detection_date.max().date())],
        "by_adulterant": df["adulterant"].value_counts().to_dict(),
        "by_category": df["category"].value_counts().to_dict(),
        "by_severity": df["severity"].value_counts().to_dict(),
        "by_health_risk": df["health_risk"].value_counts().to_dict(),
        "by_month": df.groupby("month").size().to_dict(),
    }
    (MODELS / "metadata.json").write_text(
        json.dumps({"summary": summary, "options": options, "eda": eda}, indent=2))
    print("\nSaved models + metadata to", MODELS)


if __name__ == "__main__":
    main()
