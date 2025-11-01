"""
api.py
======

Expose a simple REST API using FastAPI to serve predictions from the trained
housing price model.  The API defines a single ``/predict`` endpoint that
accepts the same set of input features as the original dataset and returns
the predicted price.  Categorical variables are converted to numeric codes
consistent with the preprocessing used for training.

To run the API locally, first ensure the model artifact exists (see
``train_model.py``) and then start the server with uvicorn:

.. code-block:: bash

    uvicorn app.api:app --reload

By default the API will look for the model in ``models/model.pkl`` and the
feature metadata in ``models/model.features.json`` relative to the project
root.  These can be overridden via the environment variables
``MODEL_PATH`` and ``FEATURE_METADATA_PATH``.

Note: The input schema uses strings for binary categorical variables (e.g.
``"yes"``/``"no"``).  The backend converts them to numeric codes under the
hood.  The ``furnishingstatus`` field expects one of ``"furnished"``,
``"semi-furnished"`` or ``"unfurnished"``.  See the data dictionary for
details【948383289304597†L8-L19】.
"""

import os
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Pydantic model defining the request schema
class HousingInput(BaseModel):
    area: float = Field(..., description="Total area of the house (square feet)")
    bedrooms: int = Field(..., description="Number of bedrooms")
    bathrooms: int = Field(..., description="Number of bathrooms")
    stories: int = Field(..., description="Number of stories")
    mainroad: str = Field(..., description="yes/no indicating if house faces a main road")
    guestroom: str = Field(..., description="yes/no indicating if there is a guest room")
    basement: str = Field(..., description="yes/no indicating presence of a basement")
    hotwaterheating: str = Field(..., description="yes/no indicating gas hot‑water heating")
    airconditioning: str = Field(..., description="yes/no indicating central air conditioning")
    parking: int = Field(..., ge=0, description="Number of parking spots")
    prefarea: str = Field(..., description="yes/no indicating preferred area")
    furnishingstatus: str = Field(
        ..., description="furnished, semi-furnished or unfurnished"
    )


def load_artifacts() -> tuple:
    """Load the model and feature metadata from disk."""
    model_path = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
    features_path = Path(
        os.getenv("FEATURE_METADATA_PATH", "models/model.features.json")
    )
    if not model_path.exists() or not features_path.exists():
        raise RuntimeError(
            f"Model or feature metadata not found (model={model_path}, features={features_path})"
        )
    model = joblib.load(model_path)
    feature_metadata = pd.read_json(features_path)
    feature_names: List[str] = feature_metadata["feature_names"].tolist()
    return model, feature_names


def preprocess_input(data: HousingInput, feature_names: List[str]) -> pd.DataFrame:
    """Convert a HousingInput instance into a single-row DataFrame matching the training schema."""
    # Map binary yes/no to integer codes (yes=1, no=0)
    def to_code(value: str) -> int:
        value = value.strip().lower()
        if value in {"yes", "y", "1", "true"}:
            return 1
        elif value in {"no", "n", "0", "false"}:
            return 0
        else:
            raise ValueError(f"Invalid binary value: {value}")

    # Build base dictionary
    record: dict[str, float] = {
        "area": float(data.area),
        "bedrooms": int(data.bedrooms),
        "bathrooms": int(data.bathrooms),
        "stories": int(data.stories),
        "mainroad": to_code(data.mainroad),
        "guestroom": to_code(data.guestroom),
        "basement": to_code(data.basement),
        "hotwaterheating": to_code(data.hotwaterheating),
        "airconditioning": to_code(data.airconditioning),
        "parking": int(data.parking),
        "prefarea": to_code(data.prefarea),
    }
    # Furnishing status one‑hot encoding: baseline is "furnished"
    furn = data.furnishingstatus.strip().lower().replace(" ", "-")
    # Add dummies in the order used during training
    record["furnishingstatus_semi-furnished"] = 1 if furn == "semi-furnished" else 0
    record["furnishingstatus_unfurnished"] = 1 if furn == "unfurnished" else 0
    # Create DataFrame with all feature columns; any missing columns default to 0
    row = {name: record.get(name, 0) for name in feature_names}
    df = pd.DataFrame([row], columns=feature_names)
    return df


# Instantiate FastAPI
app = FastAPI(title="Housing Price Prediction API")


@app.on_event("startup")
async def startup_event() -> None:
    """Load model and feature metadata when the application starts."""
    try:
        global _MODEL, _FEATURE_NAMES
        _MODEL, _FEATURE_NAMES = load_artifacts()
    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts: {e}")


@app.post("/predict")
async def predict(input_data: HousingInput) -> dict:
    """Predict the housing price for the given input features."""
    try:
        df = preprocess_input(input_data, _FEATURE_NAMES)
        pred = _MODEL.predict(df)[0]
        return {"prediction": float(pred)}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")