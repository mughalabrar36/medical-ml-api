# Breast Cancer Prediction (Flask + scikit-learn)

A Flask web app and REST API that serves a Random Forest model for real-time
predictions (malignant / benign) from tumor measurements.

- **Dataset:** Breast_cancer_dataset.csv (569 patients, 10 mean tumor measurements)
- **Model:** StandardScaler + RandomForestClassifier (~94% test accuracy)
- **Backend:** Flask (model loaded once at startup, input validated on every request)
- **Frontend:** single responsive HTML page (`templates/index.html`)
- **Deployment:** Vercel (`vercel.json`)

## Project structure
| File | Purpose |
|------|---------|
| `app.py` | Flask app: web page + API |
| `ml.py` | Field definitions, data loading, model training/loading |
| `train.py` | Trains the model and saves `model.joblib` |
| `templates/index.html` | The web interface |
| `vercel.json` | Vercel deployment config |
| `test_app.py` | Automated tests |

## Run locally
```bash
pip install -r requirements.txt
python train.py        # creates model.joblib
python app.py          # open http://127.0.0.1:5000
```

## API
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web interface |
| GET | `/health` | Health check |
| POST | `/predict` | JSON in, prediction + probabilities out |

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @sample.json
```
Invalid or missing values return HTTP 400 with a list of problems.

## Deploy on Vercel (free)
1. Push this folder to a GitHub repo.
2. On vercel.com: Add New > Project > import the repo > Deploy.
3. Open `https://<your-project>.vercel.app`.

*Disclaimer: educational project, not for real medical diagnosis.*
