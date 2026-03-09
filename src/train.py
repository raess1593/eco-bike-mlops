import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import mlflow

#mlflow.create_experiment("eco-bike-mlops")

def train_model():
    root_path = Path(__file__).parent.parent
    data_path = root_path / 'data' / 'cleaned_data.csv'
    model_dir = root_path / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'model.joblib'


    df = pd.read_csv(data_path)
    X = df.drop('demand', axis=1)
    y = df['demand']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('gbr', GradientBoostingRegressor(random_state=42))
    ])
    
    params = {
        'gbr__n_estimators': [100, 200],
        'gbr__learning_rate': [0.05, 0.1],
        'gbr__max_depth': [3],
        'gbr__min_samples_split': [2]
    }

    mlflow.set_experiment("eco-bike-mlops")
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="GradientBoosting_GridSearch"):
        grid = GridSearchCV(pipeline, params, cv=3)
        grid.fit(X_train, y_train)

        mlflow.log_metric("best_score", grid.best_score_)

        best_model = grid.best_estimator_
        joblib.dump(best_model, model_path)

        print(f"Model trained and saved -- {data_path}")

        test_score = grid.score(X_test, y_test)
        mlflow.log_metric('test_r2_score', test_score)
        print(f"R2 score in test: {test_score:.4f}")

if __name__ == "__main__":
    train_model()