"""Shared ML helpers (uses only numpy + scikit-learn, so the app stays small for Vercel)."""
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model.joblib"
CSV_PATH = BASE / "Breast_cancer_dataset.csv"

FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
]


def load_data():
    X, y = [], []
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("diagnosis"):
                continue
            X.append([float(row[c]) for c in FEATURES])
            y.append(1 if row["diagnosis"] == "M" else 0)  # 1 = malignant
    return np.array(X), np.array(y)


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])


def get_model():
    """Load the saved model; if it can't be loaded, retrain from the CSV."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        X, y = load_data()
        return build_model().fit(X, y)
