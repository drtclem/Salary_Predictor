import streamlit as st
import joblib
import os
import pandas as pd

# Load model and scaler
base_dir = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_dir, "best_model.pkl"))
scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))

st.title("Predicted Starting Salary Estimator 💼")

st.write("Enter the following details:")

# Input fields
university_ranking = st.number_input("University Ranking", min_value=1.0, max_value=100.0, value=50.0)
internships = st.number_input("Internships Completed", min_value=0.0, max_value=10.0, value=1.0)
certifications = st.number_input("Certifications", min_value=0.0, max_value=10.0, value=2.0)
job_offers = st.number_input("Job Offers", min_value=0.0, max_value=10.0, value=1.0)

if st.button("Predict Salary"):
    try:
        input_data = [[university_ranking, internships, certifications, job_offers]]
        scaled_data = scaler.transform(input_data)
        df = pd.DataFrame(scaled_data, columns=['University_Ranking', 'Internships_Completed', 'Certifications', 'Job_Offers'])
        prediction = model.predict(df)[0]
        st.success(f"Predicted Starting Salary: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")