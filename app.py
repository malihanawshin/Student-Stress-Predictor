# app.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoder
model = joblib.load("models/stress_model.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")


# =======================
# Streamlit UI
# =======================
st.set_page_config(page_title="Student Stress Predictor", layout="centered")
st.title("Student Stress Level Prediction")
st.write("Fill in the factors below to predict stress level:")

# Input fields
anxiety_level = st.slider("Anxiety Level", 0, 10, 5)
self_esteem = st.slider("Self Esteem", 0, 10, 5)
mental_health_history = st.selectbox("Mental Health History (0 = No, 1 = Yes)", [0, 1])
depression = st.slider("Depression", 0, 10, 5)
headache = st.selectbox("Headache (0 = No, 1 = Yes)", [0, 1])
blood_pressure = st.slider("Blood Pressure", 0, 10, 5)
sleep_quality = st.slider("Sleep Quality", 0, 10, 5)
breathing_problem = st.selectbox("Breathing Problem (0 = No, 1 = Yes)", [0, 1])
noise_level = st.slider("Noise Level", 0, 10, 5)
living_conditions = st.slider("Living Conditions", 0, 10, 5)
safety = st.slider("Safety", 0, 10, 5)
basic_needs = st.slider("Basic Needs Satisfaction", 0, 10, 5)
academic_performance = st.slider("Academic Performance", 0, 10, 5)
study_load = st.slider("Study Load", 0, 10, 5)
teacher_student_relationship = st.slider("Teacher-Student Relationship", 0, 10, 5)
future_career_concerns = st.slider("Future Career Concerns", 0, 10, 5)
social_support = st.slider("Social Support", 0, 10, 5)
peer_pressure = st.slider("Peer Pressure", 0, 10, 5)
extracurricular_activities = st.slider("Extracurricular Activities", 0, 10, 5)
bullying = st.selectbox("Bullying (0 = No, 1 = Yes)", [0, 1])

# Collect all features in the correct order
features = pd.DataFrame([[
    anxiety_level, self_esteem, mental_health_history, depression,
    headache, blood_pressure, sleep_quality, breathing_problem,
    noise_level, living_conditions, safety, basic_needs,
    academic_performance, study_load, teacher_student_relationship,
    future_career_concerns, social_support, peer_pressure,
    extracurricular_activities, bullying
]], columns=[
    "anxiety_level", "self_esteem", "mental_health_history", "depression",
    "headache", "blood_pressure", "sleep_quality", "breathing_problem",
    "noise_level", "living_conditions", "safety", "basic_needs",
    "academic_performance", "study_load", "teacher_student_relationship",
    "future_career_concerns", "social_support", "peer_pressure",
    "extracurricular_activities", "bullying"
])


if st.button("Predict Stress Level"):
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.subheader("Predicted Stress Level:")
    st.success(f"**{predicted_label}**")
