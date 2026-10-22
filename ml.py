"""Shared ML helpers: field definitions, data loading, training and safe model loading."""
import csv
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model.joblib"
CSV_PATH = BASE / "Breast_cancer_dataset.csv"


def _f(key, column, label, lo, hi, step, typical):
    return {"key": key, "column": column, "label": label,
            "lo": lo, "hi": hi, "step": step, "typical": typical}


# One source of truth: used by training, the API validation and the web form.
FIELDS = [
    _f("radius_mean", "radius_mean", "Radius", 5, 35, 0.01, 14.1),
    _f("texture_mean", "texture_mean", "Texture", 8, 50, 0.01, 19.3),
    _f("perimeter_mean", "perimeter_mean", "Perimeter", 40, 250, 0.01, 92.0),
    _f("area_mean", "area_mean", "Area", 100, 3500, 0.1, 654.9),
    _f("smoothness_mean", "smoothness_mean", "Smoothness", 0.03, 0.2, 0.0001, 0.096),
    _f("compactness_mean", "compactness_mean", "Compactness", 0.01, 0.5, 0.0001, 0.104),
    _f("concavity_mean", "concavity_mean", "Concavity", 0, 0.6, 0.0001, 0.089),
    _f("concave_points_mean", "concave points_mean", "Concave points", 0, 0.3, 0.0001, 0.049),
    _f("symmetry_mean", "symmetry_mean", "Symmetry", 0.08, 0.4, 0.0001, 0.181),
    _f("fractal_dimension_mean", "fractal_dimension_mean", "Fractal dimension", 0.04, 0.12, 0.00001, 0.063),
]
COLUMNS = [f["column"] for f in FIELDS]

# Real rows from the dataset, in the same order as FIELDS.
EXAMPLES = {
    "malignant": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871],
    "benign": [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766],
}


def example_dicts():
    return {name: {f["key"]: v for f, v in zip(FIELDS, vals)} for name, vals in EXAMPLES.items()}


def load_data():
    X, y = [], []
    with open(CSV_PATH, newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get("diagnosis"):
                continue
            X.append([float(row[c]) for c in COLUMNS])
            y.append(1 if row["diagnosis"] == "M" else 0)  # 1 = malignant, 0 = benign
    return np.array(X), np.array(y)


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)),
    ])


def get_model():
    """Load the saved model. If it is missing, broken or built with a different
    scikit-learn version, retrain from the CSV so the app never crashes."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(MODEL_PATH)
        model.predict(np.array([EXAMPLES["benign"]]))  # quick sanity check
        return model
    except Exception:
        X, y = load_data()
        return build_model().fit(X, y)
