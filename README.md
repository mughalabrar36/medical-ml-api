# Breast Cancer Prediction API (FastAPI + scikit-learn)

A REST API that serves a Random Forest model for real-time predictions
(malignant / benign) from tumor measurements.

- **Dataset:** Breast_cancer_dataset.csv (569 patients, 10 mean tumor measurements as inputs)
- **Model:** StandardScaler + RandomForestClassifier (~94% test accuracy)
- **Framework:** FastAPI, validated with Pydantic
- **Deployment:** Vercel

## Run locally
```bash
pip install -r requirements.txt
python train.py                  # creates model.joblib
uvicorn main:app --reload        # open http://127.0.0.1:8000/docs
```

## Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Returns prediction + probabilities |

Example request:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample.json
```

## Run with Docker
```bash
docker build -t medical-api .
docker run -p 8000:8000 medical-api
```

## Deploy on Vercel (free)
1. Push this folder to a GitHub repo.
2. On vercel.com: Add New > Project > import the repo.
3. Framework preset: **FastAPI** (or Other). Click **Deploy**.
4. Open `https://<your-project>.vercel.app/docs`.

*Disclaimer: educational project, not for real medical diagnosis.*
