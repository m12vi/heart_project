import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Failure Prediction", page_icon="💖", layout="centered")

st.title("💖 Heart Failure Prediction App")
st.write("Enter patient details to predict the presence of heart disease (1) or absence (0).")

# Debug line to confirm reloading
import datetime
print("🚀 App reloaded at:", datetime.datetime.now())

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/heart_model.pkl")

model = load_model()

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("This app uses a scikit-learn pipeline trained on the UCI Heart dataset.")
    st.markdown("Features are preprocessed using StandardScaler and OneHotEncoder.")

# Build inputs
st.subheader("Patient Inputs")
col1, col2 = st.columns(2)

age = col1.number_input("Age", min_value=20, max_value=100, value=50, step=1)
trestbps = col1.number_input("Resting Blood Pressure", min_value=80, max_value=220, value=120, step=1)
chol = col1.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=650, value=230, step=1)
thalach = col1.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1)
oldpeak = col1.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")

sex = col2.selectbox("Sex (1=Male, 0=Female)", options=[0, 1], index=1)
cp = col2.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
fbs = col2.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", options=[0, 1], index=0)
restecg = col2.selectbox("Resting ECG Results (0-2)", options=[0, 1, 2])
exang = col2.selectbox("Exercise Induced Angina (1=Yes, 0=No)", options=[0, 1], index=0)
slope = col2.selectbox("Slope of Peak Exercise ST Segment (0-2)", options=[0, 1, 2])
ca = col2.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
thal = col2.selectbox("Thalassemia (categorical)", options=[0, 1, 2, 3])

# Create input row
input_df = pd.DataFrame([{
    "age": age, "trestbps": trestbps, "chol": chol, "thalach": thalach, "oldpeak": oldpeak,
    "sex": sex, "cp": cp, "fbs": fbs, "restecg": restecg, "exang": exang, "slope": slope,
    "ca": ca, "thal": thal
}])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = None
    if hasattr(model.named_steps["model"], "predict_proba"):
        proba = model.predict_proba(input_df)[0][1]
    if pred == 1:
        st.error("⚠️ Prediction: Heart Disease (1)")
    else:
        st.success("✅ Prediction: No Heart Disease (0)")
    if proba is not None:
        st.write(f"Probability of heart disease: {proba:.2%}")
