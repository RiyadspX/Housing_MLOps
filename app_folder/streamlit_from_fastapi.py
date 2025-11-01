"""
streamlit_app.py
=================

Provide a simple front‑end for the housing price prediction model using
Streamlit.  The app collects input features from the user, sends them to
the FastAPI backend for inference, and displays the predicted price.

To launch the Streamlit application locally, make sure the API server is
running (see ``api.py``) and then execute:

.. code-block:: bash

    streamlit run app/streamlit_app.py

"""

import json
from typing import Dict

import requests
import streamlit as st


API_URL = "http://localhost:8000/predict"


def send_prediction_request(data: Dict) -> float:
    """Send a JSON payload to the FastAPI server and return the prediction."""
    response = requests.post(API_URL, json=data)
    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")


def main() -> None:
    st.title("Housing Price Prediction")
    st.write(
        "Enter the house details below and click **Predict** to estimate the price."
    )

    area = st.number_input("Area (square feet)", min_value=0.0, value=5000.0)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    stories = st.number_input("Stories", min_value=1, max_value=10, value=2)
    mainroad = st.selectbox("Faces Main Road?", ["yes", "no"])
    guestroom = st.selectbox("Guest Room?", ["yes", "no"])
    basement = st.selectbox("Basement?", ["yes", "no"])
    hotwaterheating = st.selectbox("Gas Hot‑Water Heating?", ["yes", "no"])
    airconditioning = st.selectbox("Central Air Conditioning?", ["yes", "no"])
    parking = st.number_input("Parking Spots", min_value=0, max_value=5, value=1)
    prefarea = st.selectbox("Preferred Area?", ["yes", "no"])
    furnishingstatus = st.selectbox(
        "Furnishing Status", ["furnished", "semi-furnished", "unfurnished"]
    )

    if st.button("Predict"):
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
            prediction = send_prediction_request(request_data)
            st.success(f"Estimated Price is: {prediction:,.0f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()