"""
streamlit_app.py
=================

Standalone Streamlit app for housing price prediction.

This version includes the same preprocessing logic that was in the FastAPI backend:
 - binary yes/no → 0/1 encoding
 - one-hot encoding for 'furnishingstatus'
 - uses model.features.json to align feature order
"""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Load model and feature metadata
# -----------------------------
@st.cache_resource
def load_artifacts():
    model_path = Path(os.getenv("MODEL_PATH", "artifacts/models/model.pkl"))
    features_path = Path(os.getenv("FEATURE_METADATA_PATH", "artifacts/models/model.features.json"))

    if not model_path.exists() or not features_path.exists():
        st.error(f"❌ Model or feature metadata not found.\nModel: {model_path}\nFeatures: {features_path}")
        return None, None

    model = joblib.load(model_path)
    feature_metadata = pd.read_json(features_path)
    feature_names = feature_metadata["feature_names"].tolist()
    return model, feature_names


# -----------------------------
# Preprocessing (same as in api.py)
# -----------------------------
def preprocess_input(data: dict, feature_names: list[str]) -> pd.DataFrame:
    """Convert raw user input into model-ready DataFrame."""
    def to_code(value: str) -> int:
        v = value.strip().lower()
        if v in {"yes", "y", "1", "true"}:
            return 1
        elif v in {"no", "n", "0", "false"}:
            return 0
        else:
            raise ValueError(f"Invalid binary value: {value}")

    record = {
        "area": float(data["area"]),
        "bedrooms": int(data["bedrooms"]),
        "bathrooms": int(data["bathrooms"]),
        "stories": int(data["stories"]),
        "mainroad": to_code(data["mainroad"]),
        "guestroom": to_code(data["guestroom"]),
        "basement": to_code(data["basement"]),
        "hotwaterheating": to_code(data["hotwaterheating"]),
        "airconditioning": to_code(data["airconditioning"]),
        "parking": int(data["parking"]),
        "prefarea": to_code(data["prefarea"]),
    }

    furn = data["furnishingstatus"].strip().lower().replace(" ", "-")
    record["furnishingstatus_semi-furnished"] = 1 if furn == "semi-furnished" else 0
    record["furnishingstatus_unfurnished"] = 1 if furn == "unfurnished" else 0

    row = {name: record.get(name, 0) for name in feature_names}
    return pd.DataFrame([row], columns=feature_names)


# -----------------------------
# Streamlit app
# -----------------------------
def main():
    st.set_page_config(page_title="🏠 Housing Price Prediction", page_icon="🏠")
    st.title("🏠 Housing Price Prediction")
    st.write("Enter the house details and click **Predict** to estimate the price.")

    # UI inputs
    area = st.number_input("Area (sq ft)", min_value=0.0, value=5000.0)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    stories = st.number_input("Stories", min_value=1, max_value=10, value=2)
    mainroad = st.selectbox("Faces Main Road?", ["yes", "no"])
    guestroom = st.selectbox("Guest Room?", ["yes", "no"])
    basement = st.selectbox("Basement?", ["yes", "no"])
    hotwaterheating = st.selectbox("Gas Hot-Water Heating?", ["yes", "no"])
    airconditioning = st.selectbox("Central Air Conditioning?", ["yes", "no"])
    parking = st.number_input("Parking Spots", min_value=0, max_value=5, value=1)
    prefarea = st.selectbox("Preferred Area?", ["yes", "no"])
    furnishingstatus = st.selectbox(
        "Furnishing Status", ["furnished", "semi-furnished", "unfurnished"]
    )

    model, feature_names = load_artifacts()

    if st.button("🔮 Predict"):
        if model is None or feature_names is None:
            st.error("Model or features not loaded.")
            return

        request_data = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": int(parking),
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus,
        }

        try:
            df = preprocess_input(request_data, feature_names)
            pred = model.predict(df)[0]
            st.success(f"💰 Estimated Price: **{pred:,.0f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
