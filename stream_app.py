import streamlit as st
import requests

# --- API Configuration ---
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="🩺 Diabetes Predictor", page_icon="💉", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.markdown("### Enter the patient details below:")

# --- Input fields ---
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=85)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# --- Prediction button ---
if st.button("🔍 Predict Diabetes Risk"):
    payload = {
        "features": [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            label = result["label"]  # Fixed: was "prediction"
            prob = result["probability"]

            if label == 1:
                st.error(f"⚠️ High Risk of Diabetes (Probability: {prob:.2%})")
            else:
                st.success(f"✅ Low Risk of Diabetes (Probability: {prob:.2%})")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Please ensure FastAPI server is running at http://127.0.0.1:8000")
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("Powered by PyTorch + FastAPI + Streamlit")