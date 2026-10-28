# Aegis — Planetary Defense AI

AEGIS is a FastAPI + vanilla-frontend project for screening near-Earth objects (NEOs) using a trained XGBoost model. The application lets a user estimate asteroid hazard, vary key parameters with a what-if analysis, upload CSV batches, and review model behavior without exposing the underlying training pipeline.

## Project structure

- `main.py` — FastAPI application and API routes
- `static/index.html` — single-page frontend and aerospace UI
- `static/aegis-mark.svg` — local AEGIS logo asset
- `static/favicon.svg` — local favicon
- `artifacts/sentinel_model.joblib` — trained model artifact
- `artifacts/model_card.json` — saved metadata and evaluation summary
- `neo (3).csv` — project dataset used for inference and validation
- `test_app.py` — pytest-compatible API regression tests
- `requirements.txt` — runtime and test dependencies

## Features

- Single-object asteroid classification with live model scoring
- What-if analysis for absolute magnitude, relative velocity, and miss distance
- CSV batch classification with row-level validation and downloadable output
- Model information panel with feature importance and threshold details
- Prediction history saved in the browser for local review
- Dark aerospace interface with responsive layout and accessible controls

## Local setup

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt  # only for local tests and development
python -m uvicorn main:app --reload
```

Then open http://127.0.0.1:8000.

## Model behavior and architecture

The backend loads a real XGBoost model from `artifacts/sentinel_model.joblib` and validates that its required keys are present before serving predictions. The model expects three numeric inputs:

- `absolute_magnitude`
- `relative_velocity`
- `miss_distance`

Those values are transformed using the same feature engineering used during training:

- `log_velocity = log10(relative_velocity)`
- `log_miss_distance = log10(miss_distance)`
- `log_size_dist_ratio = log10((diameter_estimate / miss_distance))`

The real model threshold is read from the artifact and used to label each asteroid as `Hazardous` or `Non-hazardous`.

## Evaluation notes

The saved model card reports the thresholded hold-out metrics for the tuned model, including ROC-AUC and recall. These values are documentation for the model artifact and are not used to replace the real model output.

The app intentionally does not expose or fabricate a more optimistic score than the actual model can produce.

## Limitations

- This project is a screening tool, not a collision-probability forecast.
- A model score should not be interpreted as a guaranteed impact risk.
- The available feature set is limited to magnitude, relative velocity, and miss distance.
- The current app retains browser-local prediction history only; it is not persisted on a backend server.

## Testing

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -v
```

The tests validate artifact loading, valid prediction requests, invalid input handling, batch CSV classification, and basic frontend accessibility.

## Troubleshooting

- If the model reports `Model could not be loaded`, verify the files under `artifacts/` are present and readable.
- If the API returns a 400 while uploading CSV files, confirm the file contains the expected headers: `absolute_magnitude,relative_velocity,miss_distance`.
- If the browser shows a blank page, confirm the local server is running on port 8000 and the frontend is being served from `static/`.

## Deployment

A simple deployment target is a Render web service using:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

This project is intended to be run from the repository root with the model artifact preserved in `artifacts/`.
