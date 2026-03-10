# 🚲 Eco-Bike MLOps Project

End-to-end MLOps project to validate, clean, train, track, and serve a bike demand prediction model using DVC, MLflow, and FastAPI.

## 📌 Project Overview

This repository implements a complete machine learning workflow:

- Data quality validation with Great Expectations.
- Data cleaning and preparation.
- Model training with `GradientBoostingRegressor` and `GridSearchCV`.
- Experiment tracking with MLflow.
- Pipeline orchestration and reproducibility with DVC.
- Online inference API with FastAPI + Uvicorn.

![](/assets/pipeline.png)

## 🧱 Tech Stack

- Python
- DVC
- MLflow
- FastAPI
- scikit-learn
- Great Expectations

## 🗂️ Repository Structure

```text
.
├── data/
│   ├── raw_data.csv.dvc
│   └── cleaned_data.csv
├── models/
│   └── model.joblib
├── src/
│   ├── app.py
│   ├── data_gen.py
│   ├── data_validation.py
│   ├── data_cleaning.py
│   └── train.py
├── dvc.yaml
├── dvc.lock
└── requirements.txt
```

## ⚙️ Pipeline Architecture

The DVC pipeline is defined in `dvc.yaml` with three stages:

1. `validate`: validates `data/raw_data.csv`.
2. `clean`: transforms and cleans data into `data/cleaned_data.csv`.
3. `train`: trains the model and outputs `models/model.joblib`.

Run the pipeline with:

```bash
dvc repro
```

## 🧪 ML Training Details

Training logic is implemented in `src/train.py`:

- Train/test split (`test_size=0.2`, `random_state=42`).
- Pipeline: `SimpleImputer` -> `StandardScaler` -> `GradientBoostingRegressor`.
- Hyperparameter optimization using `GridSearchCV` (`cv=3`).
- MLflow autologging enabled (`mlflow.sklearn.autolog()`).
- Local model artifact saved to `models/model.joblib`.

## 📊 Experiment Tracking with MLflow

All training runs are stored in experiment `eco-bike-mlops`.

Launch the MLflow UI:

```bash
mlflow ui
```

Default URL:

```text
http://127.0.0.1:5000
```

## 🚀 API Inference Service

Inference API is implemented in `src/app.py`.

Behavior:

- On startup, it searches MLflow experiment `eco-bike-mlops`.
- It loads the best finished run ordered by `metrics.mae ASC`.
- It exposes prediction endpoints through FastAPI.

Start API:

```bash
python src/app.py
```

Default URL:

```text
http://127.0.0.1:8000
```

Interactive docs:

```text
http://127.0.0.1:8000/docs
```

### API Endpoints

- `GET /`
  - Health/basic info (`model_ready` status).
- `GET /predict`
  - Query params: `temp`, `humidity`, `holiday`.
  - Example:

```text
GET /predict?temp=22&humidity=55&holiday=0
```

## 🛠️ Installation and Setup

### 1. Clone repository

```bash
git clone <your-repo-url>
cd eco-bike-mlops
```

### 2. Create and activate environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull DVC data (if remote is configured)

```bash
dvc pull
```

### 5. Run workflow

```bash
# Reproduce pipeline
dvc repro

# Inspect experiments
mlflow ui

# Start API
python src/app.py
```

## 🧭 Operational Quick Guide

| Action | Command | Tool |
|---|---|---|
| Run Pipeline | `dvc repro` | DVC |
| View Experiments | `mlflow ui` | MLflow |
| Launch API | `python src/app.py` | FastAPI + Uvicorn |
| Test Model | Open `/docs` | Swagger UI |

## ✅ Reproducibility Notes

- Pipeline dependencies and outputs are versioned through DVC metadata.
- Experiments are tracked in MLflow.
- API always loads the best available MLflow run under the configured criterion.

## 👤 Author

Project developed as an MLOps workflow for eco-bike demand prediction and deployment.
