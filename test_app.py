import pytest
from app import app
from ml import example_dicts

EX = example_dicts()


@pytest.fixture
def client():
    return app.test_client()


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Breast Cancer" in r.data


def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok"}


def test_malignant_example(client):
    assert client.post("/predict", json=EX["malignant"]).get_json()["prediction"] == "malignant"


def test_benign_example(client):
    assert client.post("/predict", json=EX["benign"]).get_json()["prediction"] == "benign"


def test_missing_field(client):
    bad = dict(EX["benign"]); bad.pop("area_mean")
    assert client.post("/predict", json=bad).status_code == 400


def test_out_of_range(client):
    bad = dict(EX["benign"], radius_mean=-5)
    assert client.post("/predict", json=bad).status_code == 400


def test_not_json(client):
    assert client.post("/predict", data="hello").status_code == 400
