import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.mp231.data import (
    load_train, load_later, sha256_of_train, PARAMS
)
from src.mp231.clean import clean_train, clean_later


ROOT = Path(__file__).resolve().parents[2]
CONFIG = PARAMS
PRE = CONFIG["preprocessing"]

FEATURES = PRE["feature_columns"]
NUMERIC = PRE["numeric_columns"]
CATEGORICAL = PRE["categorical_columns"]
TARGET = PRE["target_column"]


def build_pipeline(n_estimators, max_depth, learning_rate):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL)
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        ))
    ])


def train_v1():
    mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])
    mlflow.set_experiment(CONFIG["mlflow"]["experiment_name"])

    train = clean_train(load_train())
    later = clean_later(load_later())

    X = train[FEATURES]
    y = train[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    later_X = later[FEATURES]
    later_y = later[TARGET]

    settings = [
        (100, 3, 0.05),
        (200, 4, 0.10),
        (300, 5, 0.05)
    ]

    best_mae = float("inf")
    best_model = None
    best_run_id = None
    model_name = CONFIG["mlflow"]["registered_model_name"]

    for n, depth, rate in settings:
        model = build_pipeline(n, depth, rate)

        with mlflow.start_run() as run:
            model.fit(X_train, y_train)

            val_predictions = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_predictions)

            later_predictions = model.predict(later_X)
            later_mae = mean_absolute_error(later_y, later_predictions)

            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", depth)
            mlflow.log_param("learning_rate", rate)
            mlflow.log_param("data_sha256", sha256_of_train())
            mlflow.log_param("train_rows", len(train))
            mlflow.log_param("version_note", "v1")

            mlflow.log_metric("mae_val", val_mae)
            mlflow.log_metric("mae_later", later_mae)

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=model_name,
                serialization_format="cloudpickle"
            )

            print(f"Run: {run.info.run_id}")
            print(f"Validation MAE: {val_mae:.3f}")
            print(f"Later MAE: {later_mae:.3f}")

            if val_mae < best_mae:
                best_mae = val_mae
                best_model = model
                best_run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")

    best_version = next(
        int(v.version) for v in versions if v.run_id == best_run_id
    )

    client.set_registered_model_alias(
        model_name, "champion", best_version
    )

    output = ROOT / CONFIG["paths"]["model_output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output)

    print("\nTRAINING COMPLETE")
    print("Best validation MAE:", round(best_mae, 3))
    print("Best run ID:", best_run_id)
    print("Champion version:", best_version)
    print("Model saved:", output)


if __name__ == "__main__":
    train_v1()
