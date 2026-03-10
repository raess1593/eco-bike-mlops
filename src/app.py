import mlflow
import mlflow.sklearn
from fastapi import FastAPI
import uvicorn

app = FastAPI()

def get_best_model_uri():
    experiment = mlflow.get_experiment_by_name("eco-bike-mlops")
    if not experiment:
        return None
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        max_results=1,
        order_by=["metrics.mae ASC"]
    )

    if runs.empty:
        raise Exception("There are no runs in MLflow yet")
    
    run_id = runs.iloc[0].run_id
    return f"runs:/{run_id}/model"

MODEL_URI = get_best_model_uri()
model = mlflow.sklearn.load_model(MODEL_URI) if MODEL_URI else None

@app.get("/")
def home():
    return {"message": "Bike Demand API", "model_ready": model is not None}

@app.get("/predict")
def predict(temp: int, humidity: int, holiday: int):
    if not model:
        return{"error": "No model has been loaded"}

    pred = model.predict([[temp, humidity, holiday]])

    return {"demand": pred[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)