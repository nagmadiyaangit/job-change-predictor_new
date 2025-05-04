import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("job_change_rf_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("Job Change Prediction App")

with st.form("prediction_form"):
    st.subheader("Enter Candidate Details")

    city = st.text_input("City", "city_103")
    city_development_index = st.number_input("City Development Index", min_value=0.0, max_value=1.0, value=0.920)

    gender = st.selectbox("Gender", ["Male", "Female", "Other", "Unknown"])
    relevent_experience = st.selectbox("Relevant Experience", ["Has relevent experience", "No relevent experience"])
    enrolled_university = st.selectbox("Enrolled University", ["no_enrollment", "Full time course", "Part time course"])
    education_level = st.selectbox("Education Level", ["Graduate", "Masters", "High School", "Phd", "Primary School", "Unknown"])
    major_discipline = st.selectbox("Major Discipline", ["STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other", "Unknown"])
    experience = st.selectbox("Experience", ["<1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", ">20", "Unknown"])
    company_size = st.selectbox("Company Size", ["<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+", "Unknown"])
    company_type = st.selectbox("Company Type", ["Private", "Public", "NGO", "Early Stage Startup", "Funded Startup", "Unknown"])
    last_new_job = st.selectbox("Last New Job", ["never", "1", "2", "3", "4", ">4"])
    training_hours = st.number_input("Training Hours", min_value=0, max_value=1000, value=40)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        'city': city,
        'city_development_index': city_development_index,
        'gender': gender,
        'relevent_experience': relevent_experience,
        'enrolled_university': enrolled_university,
        'education_level': education_level,
        'major_discipline': major_discipline,
        'experience': experience,
        'company_size': company_size,
        'company_type': company_type,
        'last_new_job': last_new_job,
        'training_hours': training_hours
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical features using saved encoders
    for col in label_encoders:
        if col in input_df.columns:
            le = label_encoders[col]
            value = input_df.at[0, col]
            if value not in le.classes_:
                # Add the unseen label to the encoder classes temporarily
                le.classes_ = np.append(le.classes_, value)
            input_df[col] = le.transform([value])[0]

    # Reorder columns to match model's expected input
    input_df = input_df[model.feature_names_in_]

    # Make prediction
    prediction = model.predict(input_df)[0]
    result = "✅ Will Change Job" if prediction == 1 else "❌ Will Not Change Job"
    st.success(f"Prediction: {result}")