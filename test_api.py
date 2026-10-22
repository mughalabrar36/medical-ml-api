from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

MALIGNANT = {"radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8,
    "area_mean": 1001.0, "smoothness_mean": 0.1184, "compactness_mean": 0.2776,
    "concavity_mean": 0.3001, "concave_points_mean": 0.1471,
    "symmetry_mean": 0.2419, "fractal_dimension_mean": 0.07871}

def test_health():
    assert client.get("/health").json() == {"status": "ok"}

def test_predict_malignant_example():
    r = client.post("/predict", json=MALIGNANT)
    assert r.status_code == 200
    assert r.json()["prediction"] == "malignant"

def test_invalid_input_rejected():
    bad = dict(MALIGNANT, radius_mean=-5)
    assert client.post("/predict", json=bad).status_code == 422
