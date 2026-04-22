# ============================================================
# Imports
# ============================================================
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download


# ============================================================
# Model Loading
# ============================================================
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="praveenchugh/engine-condition-gbm-model",
        filename="gbm_model.joblib",
    )
    return joblib.load(model_path)


model = load_model()


# ============================================================
# App UI
# ============================================================
st.title("Engine Condition Predictor")

st.write(
    "Provide engine sensor values to predict whether the engine condition "
    "is normal or anomalous."
)


# ============================================================
# Input Collection
# ============================================================
def get_user_inputs():
    engine_rpm = st.number_input("Engine RPM", value=800.0)
    lub_oil_pressure = st.number_input("Lub Oil Pressure", value=3.0)
    fuel_pressure = st.number_input("Fuel Pressure", value=2.0)
    coolant_pressure = st.number_input("Coolant Pressure", value=2.0)
    lub_oil_temp = st.number_input("Lub Oil Temp", value=75.0)
    coolant_temp = st.number_input("Coolant Temp", value=85.0)

    # IMPORTANT: Match EXACT training column names
    input_df = pd.DataFrame([{
        "Engine rpm": engine_rpm,
        "Lub oil pressure": lub_oil_pressure,
        "Fuel pressure": fuel_pressure,
        "Coolant pressure": coolant_pressure,
        "lub oil temp": lub_oil_temp, 
        "Coolant temp": coolant_temp
    }])

    return input_df


# ============================================================
# Prediction Logic
# ============================================================
input_df = get_user_inputs()

if st.button("Predict"):

    try:
        # Ensure column order matches training
        if hasattr(model, "feature_names_in_"):
            input_df = input_df[model.feature_names_in_]

        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Engine Condition: Anomalous")
        else:
            st.success("Engine Condition: Normal")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

        # Debug info (very useful)
        st.write("Input columns:", input_df.columns.tolist())

        if hasattr(model, "feature_names_in_"):
            st.write("Model expects:", list(model.feature_names_in_))
