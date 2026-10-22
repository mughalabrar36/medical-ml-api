"""FastAPI service for real-time breast cancer prediction."""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Serves an ML model that predicts whether a tumor is malignant or benign.",
    version="1.0.0",
)

model = joblib.load("model.joblib")
FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
]


class PatientData(BaseModel):
    radius_mean: float = Field(..., gt=0, example=17.99)
    texture_mean: float = Field(..., gt=0, example=10.38)
    perimeter_mean: float = Field(..., gt=0, example=122.8)
    area_mean: float = Field(..., gt=0, example=1001.0)
    smoothness_mean: float = Field(..., gt=0, lt=1, example=0.1184)
    compactness_mean: float = Field(..., ge=0, example=0.2776)
    concavity_mean: float = Field(..., ge=0, example=0.3001)
    concave_points_mean: float = Field(..., ge=0, example=0.1471)
    symmetry_mean: float = Field(..., gt=0, example=0.2419)
    fractal_dimension_mean: float = Field(..., gt=0, example=0.07871)


@app.get("/")
def home():
    return {"message": "Breast Cancer Prediction API is running. Visit /docs to try it."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(p: PatientData):
    values = [p.radius_mean, p.texture_mean, p.perimeter_mean, p.area_mean,
              p.smoothness_mean, p.compactness_mean, p.concavity_mean,
              p.concave_points_mean, p.symmetry_mean, p.fractal_dimension_mean]
    df = pd.DataFrame([values], columns=FEATURES)
    pred = int(model.predict(df)[0])
    proba = model.predict_proba(df)[0]
    return {
        "prediction": "malignant" if pred == 1 else "benign",
        "confidence": round(float(max(proba)), 4),
        "probabilities": {"benign": round(float(proba[0]), 4),
                          "malignant": round(float(proba[1]), 4)},
        "disclaimer": "Educational project only. Not for medical diagnosis.",
    }
