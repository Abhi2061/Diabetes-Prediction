import streamlit as st
import pickle
import numpy as np

# Load the scaling model and trained model
with open("scaling.pkl", "rb") as scaling_file:
    sc = pickle.load(scaling_file)

with open("diabetes_model_knn.pkl", "rb") as model_file:
    model1 = pickle.load(model_file)

with open("diabetes_model_log.pkl", "rb") as model_file:
    model2 = pickle.load(model_file)

# Streamlit UI
st.title("Diabetes Prediction App")

ch = st.radio("Select the ML model", ["KNN", "Logistic Regression"])

st.write("Enter the details below to predict whether a person has diabetes or not.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

# Prediction
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    scaled_input_data = sc.transform(input_data)

    if ch == "KNN":
        st.write("KNN's Prediction")
        prediction = model1.predict(scaled_input_data)
    else:
        st.write("Logistic Regression's Prediction")
        prediction = model2.predict(scaled_input_data)

    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    
    st.subheader("Prediction Result")
    st.success(f"The model predicts: {result}")
