"""Train a breast-cancer classifier on Breast_cancer_dataset.csv and save model.joblib"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# The 10 "mean" measurements are used as inputs to the API
FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
]

df = pd.read_csv("Breast_cancer_dataset.csv")
df = df.dropna(subset=["diagnosis"])
X = df[FEATURES]
y = (df["diagnosis"] == "M").astype(int)  # 1 = malignant, 0 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Rows used:", len(df))
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, target_names=["benign", "malignant"]))

joblib.dump(model, "model.joblib")
print("Saved model.joblib")
