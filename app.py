"""Flask app: web page + JSON API for real-time breast cancer prediction."""
import math

import numpy as np
from flask import Flask, jsonify, render_template_string, request

from ml import FIELDS, example_dicts, get_model
from page import HTML

app = Flask(__name__)
app.json.sort_keys = False
model = get_model()  # loaded once, reused for every request


def parse_input(data):
    """Validate the request body. Returns (values, errors)."""
    if not isinstance(data, dict):
        return None, ["Send a JSON object with the 10 measurements."]
    values, errors = [], []
    for f in FIELDS:
        try:
            v = float(data.get(f["key"]))
        except (TypeError, ValueError):
            errors.append(f"{f['label']} is required and must be a number.")
            continue
        if math.isnan(v) or not f["lo"] <= v <= f["hi"]:
            errors.append(f"{f['label']} must be between {f['lo']} and {f['hi']}.")
        values.append(v)
    return values, errors


@app.get("/")
def home():
    return render_template_string(HTML, fields=FIELDS, examples=example_dicts())


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    values, errors = parse_input(request.get_json(silent=True))
    if errors:
        return jsonify(error="Invalid input", details=errors), 400
    proba = model.predict_proba(np.array([values]))[0]  # [benign, malignant]
    malignant = bool(proba[1] >= 0.5)
    return jsonify(
        prediction="malignant" if malignant else "benign",
        confidence=round(float(max(proba)), 4),
        probabilities={"benign": round(float(proba[0]), 4),
                       "malignant": round(float(proba[1]), 4)},
        disclaimer="Educational project only. Not for medical diagnosis.",
    )


if __name__ == "__main__":
    app.run(debug=True)
