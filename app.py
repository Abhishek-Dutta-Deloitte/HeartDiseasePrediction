import streamlit as st
import pandas as pd
import numpy as np
import pickle
from src.utils import *
from src.logger import logger
import json




model = load_obj(os.path.join("artifacts", "model.pkl"))
print(os.path.join("artifacts", "model.pkl"))
print(model)
preprocessor = load_obj(os.path.join("artifacts", "preprocessor_obj.pkl"))
print(preprocessor)
# Read categorical_column_mapping.json file
with open(os.path.join("categorical_column_mapping.json"), "r") as f:
    categorical_column_mapping = json.load(f)


####################### Streamlit ############################################

# Setting the page title and layout
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Adding a header and subheader with some styling
st.title("Heart Disease Prediction")
st.markdown("""
This application predicts the risk of heart disease based on multiple factors
""")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age")
    sex = st.selectbox("Sex - Male:1 Female:0 ", [0,1])
    chest_pain_type = st.selectbox("""Chest Pain Type: 
    -- Value 1: typical angina 
    -- Value 2: atypical angina 
    -- Value 3: non-anginal pain 
    -- Value 4: asymptomatic """,[1,2,3,4])
    resting_bp_s = st.number_input("Resting BP")
    cholesterol = st.number_input("Cholesterol")
    fasting_blood_sugar = st.selectbox("""Fasting Blood Sugar 
                                       (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false): """, [1,0])
    resting_ecg = st.selectbox("""Resting ECG 
                               -- Value 0: normal 
-- Value 1: having ST-T wave abnormality (T wave inversions 
and/or ST elevation or depression of > 0.05 mV) 
-- Value 2: showing probable or definite left ventricular 
hypertrophy by Estes' criteria""", [0,1,2])

with col2:
    max_heart_rate = st.number_input("Max Heart Rate")
    exercise_angina = st.selectbox("Exercise Angina - 1 = yes; 0 = no: ", [0,1])
    oldpeak = st.number_input("Oldpeak")
    ST_slope = st.selectbox("""ST Slope -- Value 1: upsloping 
-- Value 2: flat 
-- Value 3: downsloping """, [1,2,3])
    

data = {
    "age": age,
    "sex": sex,
    "chest pain type": chest_pain_type,
    "resting bp s": resting_bp_s,
    "cholesterol": cholesterol,
    "fasting blood sugar": fasting_blood_sugar,
    "resting ecg": resting_ecg,
    "max heart rate": max_heart_rate,
    "exercise angina": exercise_angina,
    "oldpeak": oldpeak,
    "ST slope": ST_slope,
}

data_df = pd.DataFrame(data=data, index=[0])
print(data_df)
logger.info(data_df)
# Prediction button
submit = st.sidebar.button("Predict Heart Disease")

# Displaying the prediction
if submit:
    try:
        new_data = preprocessor.transform(data_df)
        logger.info("transformation done for new data!!!")
        prediction = model.predict(new_data)
        logger.info("prediction done!!!")
        if prediction[0] < 0.5:
            st.write("The prediction is: No Heart Disease")
        else:
            st.write("There is a chance of heart disease. Please visit the doctor")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
