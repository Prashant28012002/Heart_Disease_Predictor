import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("heart_model.pkl","rb"))

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.markdown("### 🩺 ML-Based Clinical Risk Estimation")

st.info("Fill patient details below based on clinical report")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", 1, 120)

    sex = st.selectbox("Sex", [0,1], 
        format_func=lambda x: "0 = Female" if x==0 else "1 = Male")

    cp = st.selectbox("Chest Pain Type",
        [1,2,3,4],
        format_func=lambda x:
            "1 = Typical Angina" if x==1 else
            "2 = Atypical Angina" if x==2 else
            "3 = Non-anginal Pain" if x==3 else
            "4 = Asymptomatic")

    trestbps = st.number_input("Resting Blood Pressure (mm Hg)")

    chol = st.number_input("Serum Cholesterol (mg/dl)")

    fbs = st.selectbox("Fasting Blood Sugar",
        [0,1],
        format_func=lambda x:
            "0 = <= 120 mg/dl" if x==0 else
            "1 = > 120 mg/dl")

with col2:
    restecg = st.selectbox("Resting ECG",
        [0,1,2],
        format_func=lambda x:
            "0 = Normal" if x==0 else
            "1 = ST-T Abnormality" if x==1 else
            "2 = LV Hypertrophy")

    thalach = st.number_input("Max Heart Rate Achieved")

    exang = st.selectbox("Exercise Induced Angina",
        [0,1],
        format_func=lambda x:
            "0 = No" if x==0 else
            "1 = Yes")

    oldpeak = st.number_input("ST Depression (Oldpeak)")

    slope = st.selectbox("ST Segment Slope",
        [1,2,3],
        format_func=lambda x:
            "1 = Upsloping" if x==1 else
            "2 = Flat" if x==2 else
            "3 = Downsloping")

st.markdown("---")

if st.button("🔍 Predict Heart Risk"):

    input_data = np.array([[age,sex,cp,trestbps,chol,fbs,
                            restecg,thalach,exang,oldpeak,slope]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.metric("Confidence Level", f"{probability[0][prediction[0]]*100:.2f}%")

st.markdown("---")
st.caption("Model trained using Logistic Regression on clinical dataset")
