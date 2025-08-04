import streamlit as st
import pickle
import numpy as np

# Load models and scalers manually
import joblib
# diabetes_model = joblib.load('E:/Data_Science/Deployement/Multiple_Diseases_Prediction/Diabetes/Best_model_diabetes_model.pkl')
# diabetes_scaler = joblib.load('E:/Data_Science/Deployement/Multiple_Diseases_Prediction/Diabetes/diabetes_scaler.pkl')

# cancer_model = joblib.load('E:/Data_Science/Deployement/Multiple_Diseases_Prediction/Cancer/Cancer_model.pkl')
# cancer_scaler = joblib.load('E:/Data_Science/Deployement/Multiple_Diseases_Prediction/Cancer/Cancer_scaler.pkl')

# heart_model = joblib.load('E:/Data_Science/Deployement/Multiple_Diseases_Prediction/Heart_Failure/Heart_Failure_model.pkl')
# heart_scaler = joblib.load('E:/Data_Science/Deployement/Multiple_Diseases_Prediction/Heart_Failure/Heart_failure_scaler.pkl')

diabetes_model = joblib.load('Best_model_diabetes_model.pkl')
diabetes_scaler = joblib.load('diabetes_scaler.pkl')

cancer_model = joblib.load('Cancer_model.pkl')
cancer_scaler = joblib.load('Cancer_scaler.pkl')

heart_model = joblib.load('Heart_Failure_model.pkl')
heart_scaler = joblib.load('Heart_failure_scaler.pkl')


# Sidebar
st.sidebar.title("Multiple Disease Prediction System")
app = st.sidebar.radio("Select Disease to Predict", 
                       ["Diabetes Prediction", "Heart Disease Prediction", "Breast Cancer Prediction"])

st.title(f"{app} using ML")

# =================== Diabetes ===================
if app == "Diabetes Prediction":
    pregnancies = st.number_input("Number of Pregnancies", 0, 20)
    glucose = st.number_input("Glucose Level", 0, 300)
    bp = st.number_input("Blood Pressure", 0, 200)
    skin_thickness = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin Level", 0, 900)
    bmi = st.number_input("BMI Value", 0.0, 100.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    age = st.number_input("Age", 1, 120)

    if st.button("Diabetes Test Result"):
        features = diabetes_scaler.transform([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
        result = diabetes_model.predict(features)
        if result[0] == 1:
            st.success("The person is Diabetic")
        else:
            st.success("The person is Not Diabetic")

# =================== Heart Disease ===================
elif app == "Heart Disease Prediction":
    age = st.number_input("Age", 1, 120)
    anaemia = st.selectbox("Anaemia", [0, 1])
    creatinine = st.number_input("Creatinine Phosphokinase", 0, 8000)
    diabetes = st.selectbox("Diabetes", [0, 1])
    ef = st.number_input("Ejection Fraction", 0, 100)
    hbp = st.selectbox("High Blood Pressure", [0, 1])
    platelets = st.number_input("Platelets", 0.0, 1000000.0)
    serum_creatinine = st.number_input("Serum Creatinine", 0.0, 10.0)
    serum_sodium = st.number_input("Serum Sodium", 0, 200)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
    smoking = st.selectbox("Smoking", [0, 1])
    time = st.number_input("Follow-up time (in days)", 0, 300)

    if st.button("Heart Failure Test Result"):
        heart_input = [[age, anaemia, creatinine, diabetes, ef, hbp, platelets, serum_creatinine,
                        serum_sodium, sex, smoking, time]]
        scaled = heart_scaler.transform(heart_input)
        result = heart_model.predict(scaled)
        if result[0] == 1:
            st.success("High Risk of Heart Failure")
        else:
            st.success("Low Risk of Heart Failure")

# =================== Cancer ===================
elif app == "Breast Cancer Prediction":
    st.write("Enter the following features for Breast Cancer Prediction")

    # You can group features as needed — simplified below
    radius_mean = st.number_input("Radius Mean")
    texture_mean = st.number_input("Texture Mean")
    perimeter_mean = st.number_input("Perimeter Mean")
    area_mean = st.number_input("Area Mean")
    smoothness_mean = st.number_input("Smoothness Mean")
    compactness_mean = st.number_input("Compactness Mean")
    concavity_mean = st.number_input("Concavity Mean")
    concave_points_mean = st.number_input("Concave Points Mean")
    symmetry_mean = st.number_input("Symmetry Mean")
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean")
    id = st.number_input("ID", 0, 1000000)

    radius_se = st.number_input("Radius SE")
    texture_se = st.number_input("Texture SE")
    perimeter_se = st.number_input("Perimeter SE")
    area_se = st.number_input("Area SE")
    smoothness_se = st.number_input("Smoothness SE")
    compactness_se = st.number_input("Compactness SE")
    concavity_se = st.number_input("Concavity SE")
    concave_points_se = st.number_input("Concave Points SE")
    symmetry_se = st.number_input("Symmetry SE")
    fractal_dimension_se = st.number_input("Fractal Dimension SE")

    radius_worst = st.number_input("Radius Worst")
    texture_worst = st.number_input("Texture Worst")
    perimeter_worst = st.number_input("Perimeter Worst")
    area_worst = st.number_input("Area Worst")
    smoothness_worst = st.number_input("Smoothness Worst")
    compactness_worst = st.number_input("Compactness Worst")
    concavity_worst = st.number_input("Concavity Worst")
    concave_points_worst = st.number_input("Concave Points Worst")
    symmetry_worst = st.number_input("Symmetry Worst")
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst")

    if st.button("Cancer Prediction Result"):
        input_data = [[id, radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
                       concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
                       radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se,
                       concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
                       radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
                       compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]]

        scaled = cancer_scaler.transform(input_data)
        result = cancer_model.predict(scaled)
        if result[0] == 1 or result[0] == 'M':
            st.success("Malignant Cancer (High Risk)")
        else:
            st.success("Benign Cancer (Low Risk)")