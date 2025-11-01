"""
train_model.py
===============

Train a regression model to predict housing prices and record the results with
MLflow.  This script reads processed training and test sets, fits a simple
model (by default a RandomForestRegressor), evaluates it on the test set and
stores both the model artifact and a metrics file.  Metrics and parameters
are also logged to an MLflow tracking server, which defaults to a local
directory.

The housing dataset contains numeric and categorical features describing
properties such as area, bedrooms, bathrooms, stories and amenities
【948383289304597†L8-L19】.  For demonstration, we perform minimal
feature engineering in `prepare_data.py`, then train using all features.

Example usage:

.. code-block:: bash

   python src/train_model.py \
       --train-path data/processed/train.csv \
       --test-path data/processed/test.csv \
       --model-path models/model.pkl \
       --metrics-path metrics/metrics.json \
       --n-estimators 100 \
       --random-state 42

This script can be invoked from a DVC stage.  MLflow logs will appear in
``mlruns/`` unless a different tracking URI is specified via the
``--mlflow-uri`` argument or the ``MLFLOW_TRACKING_URI`` environment variable.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn


def load_data(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train and test sets."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and target vector y."""
    X = df.drop(columns=["price"])
    y = df["price"]
    return X, y


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train a RandomForestRegressor on the provided data."""
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Evaluate a regression model and return metrics."""
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds, squared=False)
    r2 = r2_score(y, preds)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def save_model(model: Any, model_path: Path) -> None:
    """Persist a trained model to disk using joblib."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def save_metrics(metrics: Dict[str, float], metrics_path: Path) -> None:
    """Write metrics to a JSON file."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a housing price model")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/processed/train.csv"),
        help="Path to processed training data",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/processed/test.csv"),
        help="Path to processed test data",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/model.pkl"),
        help="Path where the trained model will be saved",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/metrics/metrics.json"),
        help="Path where evaluation metrics will be saved",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the RandomForest",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help=(
            "Optional MLflow tracking URI. If provided, this value will be "
            "passed to mlflow.set_tracking_uri() before logging."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="housing-price-experiments",
        help="Name of the MLflow experiment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Configure MLflow tracking URI if supplied
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)

    mlflow.set_experiment(args.experiment_name)

    # Load data and split into features/targets
    train_df, test_df = load_data(args.train_path, args.test_path)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    with mlflow.start_run():
        # Log input parameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("random_state", args.random_state)

        # Train the model
        model = train_model(
            X_train,
            y_train,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
        )

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log model artifact to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="HousingPriceModel",
        )

        # Save model, metrics and feature metadata to local files (for DVC tracking)
        save_model(model, args.model_path)
        save_metrics(metrics, args.metrics_path)

        # Save feature names alongside the model for use by inference code
        feature_metadata_path = args.model_path.with_suffix(".features.json")
        feature_metadata = {"feature_names": list(X_train.columns)}
        with open(feature_metadata_path, "w") as f:
            json.dump(feature_metadata, f, indent=2)

        mlflow.log_artifact(feature_metadata_path)

        # Optionally log the paths as artifacts
        mlflow.log_artifact(args.model_path)
        mlflow.log_artifact(args.metrics_path)


if __name__ == "__main__":
    main()