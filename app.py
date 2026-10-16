import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("fcaw_model.joblib")

st.set_page_config(
    page_title="FCAW Parameter Demo",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ FCAW Parameter Recommendation")
st.caption("Machine Learning Project — Synthetic Data Demo")

st.warning(
    "Educational demonstration only. "
    "These predictions are not validated for real welding."
)

st.write(
    "Enter the workpiece thickness and wire diameter "
    "to generate predicted welding parameters."
)

thickness = st.number_input(
    "Workpiece thickness (mm)",
    min_value=2.0,
    max_value=8.0,
    value=6.0,
    step=0.5
)

wire_diameter = st.selectbox(
    "Wire diameter (mm)",
    options=[1.2, 1.6]
)

if st.button("Recommend Parameters"):
    input_data = pd.DataFrame([{
        "thickness_mm": thickness,
        "wire_diameter_mm": wire_diameter
    }])

    prediction = model.predict(input_data)[0]

    results = pd.DataFrame({
        "Parameter": [
            "Current",
            "Voltage",
            "Wire feed speed",
            "Travel speed"
        ],
        "Predicted value": [
            f"{prediction[0]:.1f} A",
            f"{prediction[1]:.1f} V",
            f"{prediction[2]:.2f} m/min",
            f"{prediction[3]:.2f} cm/min"
        ]
    })

    st.subheader("Model Predictions")
    st.table(results)

    st.info(
        "Predictions come from a model trained on "
        "synthetically generated data, not validated "
        "experimental welding results."
    )