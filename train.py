"""Train a breast-cancer classifier on Breast_cancer_dataset.csv and save model.joblib"""
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from ml import MODEL_PATH, build_model, load_data

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = build_model().fit(X_train, y_train)

pred = model.predict(X_test)
print("Rows used:", len(y))
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, target_names=["benign", "malignant"]))

joblib.dump(model, MODEL_PATH)
print("Saved model.joblib")
