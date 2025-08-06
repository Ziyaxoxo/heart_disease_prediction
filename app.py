import streamlit as st
import pickle
import pandas as pd
import os
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered", # Can be "wide" or "centered"
    initial_sidebar_state="auto"
)

# Function to load the trained model
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model():
    """Loads the pre-trained model from 'model.pkl'."""
    model_filename = 'model.pkl'
    if not os.path.exists(model_filename):
        st.error("Error: Model file 'model.pkl' not found.")
        st.warning("Please run `python train_model.py` first to train and save the model.")
        return None
    try:
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model once
model = load_model()

# --- Header Section ---
col1_header, col2_header = st.columns([0.8, 0.2])
with col1_header:
    st.title("Heart Disease Prediction 🩺")
with col2_header:
    # Streamlit's theme is controlled by browser/system settings or Streamlit's own settings menu.
    # A direct in-app button for theme toggle is not natively supported without complex JS.
    st.markdown(
        """
        <style>
        .stSwitch > label > div {
            transform: scale(0.8); /* Adjust switch size */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.info("Disclaimer: consult a doctor for accurate advice.") # Hint for theme change

st.markdown("---") # Horizontal line for separation

if model is None:
    st.stop() # Stop the app if model isn't loaded

# --- Input Section ---
st.header("Patient Clinical Data")

# Define features and their input types/options based on your heart.csv
# Using more descriptive labels and appropriate Streamlit widgets (sliders, selectboxes)
# Removed 'education'
features_info = {
    'male': {'label': 'Sex:', 'type': 'radio', 'options': {'Female': 0, 'Male': 1}, 'default_key': 'Male'},
    'age': {'label': 'Age:', 'type': 'slider', 'min': 1, 'max': 120, 'step': 1, 'default': 45},
    'currentSmoker': {'label': 'Current Smoker:', 'type': 'radio', 'options': {'No': 0, 'Yes': 1}, 'default_key': 'No'},
    'cigsPerDay': {'label': 'Cigarettes Per Day:', 'type': 'slider', 'min': 0, 'max': 100, 'step': 1, 'default': 0},
    'BPMeds': {'label': 'On Blood Pressure Meds:', 'type': 'radio', 'options': {'No': 0, 'Yes': 1}, 'default_key': 'No'},
    'prevalentStroke': {'label': 'History of Stroke:', 'type': 'radio', 'options': {'No': 0, 'Yes': 1}, 'default_key': 'No'},
    'prevalentHyp': {'label': 'History of Hypertension:', 'type': 'radio', 'options': {'No': 0, 'Yes': 1}, 'default_key': 'No'},
    'diabetes': {'label': 'Has Diabetes:', 'type': 'radio', 'options': {'No': 0, 'Yes': 1}, 'default_key': 'No'},
    'totChol': {'label': 'Total Cholesterol (mg/dL):', 'type': 'slider', 'min': 100, 'max': 700, 'step': 1, 'default': 200},
    'sysBP': {'label': 'Systolic Blood Pressure (mm Hg):', 'type': 'slider', 'min': 80, 'max': 300, 'step': 1, 'default': 120},
    'diaBP': {'label': 'Diastolic Blood Pressure (mm Hg):', 'type': 'slider', 'min': 40, 'max': 200, 'step': 1, 'default': 80},
    'heartRate': {'label': 'Heart Rate (beats/min):', 'type': 'slider', 'min': 40, 'max': 200, 'step': 1, 'default': 70},
    'glucose': {'label': 'Glucose (mg/dL):', 'type': 'slider', 'min': 40, 'max': 500, 'step': 1, 'default': 90},
}

input_data = {}

# Organize inputs into columns for a more compact layout
cols_per_row = 2
current_col_index = 0
columns = st.columns(cols_per_row)

# Place 'male' (Sex) input first as it's needed for BMI context
with columns[current_col_index]:
    selected_option = st.radio(features_info['male']['label'], list(features_info['male']['options'].keys()), index=list(features_info['male']['options'].keys()).index(features_info['male']['default_key']), key='male')
    input_data['male'] = features_info['male']['options'][selected_option]
current_col_index = (current_col_index + 1) % cols_per_row
if current_col_index == 0:
    columns = st.columns(cols_per_row)

# Iterate through other features, excluding 'male' and 'BMI' (which is calculated)
for feature, info in features_info.items():
    if feature in ['male', 'BMI']: # Skip 'male' as it's handled, and 'BMI' is calculated
        continue

    with columns[current_col_index]:
        if info['type'] == 'slider':
            input_data[feature] = st.slider(
                info['label'],
                min_value=info['min'],
                max_value=info['max'],
                value=info['default'],
                step=info['step'],
                key=feature
            )
        elif info['type'] == 'radio':
            selected_option = st.radio(info['label'], list(info['options'].keys()), index=list(info['options'].keys()).index(info['default_key']), key=feature)
            input_data[feature] = info['options'][selected_option]
        elif info['type'] == 'selectbox': # Though no selectboxes are currently defined in features_info
            selected_option = st.selectbox(info['label'], list(info['options'].keys()), index=list(info['options'].keys()).index(info['default_key']), key=feature)
            input_data[feature] = info['options'][selected_option]

    current_col_index = (current_col_index + 1) % cols_per_row
    # If we just filled the last column in a row, start a new row of columns
    if current_col_index == 0:
        columns = st.columns(cols_per_row)

st.markdown("---")

# --- BMI Calculation Section (Moved to last) ---
st.subheader("Body Mass Index (BMI) Calculation")
st.write("Enter your height and weight to calculate your BMI.")
col_height, col_weight = st.columns(2)
with col_height:
    height_cm = st.number_input("Height (cm):", min_value=50.0, max_value=250.0, value=170.0, step=0.1, key="height_cm")
with col_weight:
    weight_kg = st.number_input("Weight (kg):", min_value=20.0, max_value=200.0, value=70.0, step=0.1, key="weight_kg")

bmi = 0.0
if height_cm > 0 and weight_kg > 0:
    height_m = height_cm / 100.0
    bmi = round(weight_kg / (height_m ** 2), 1)
    st.write(f"Calculated BMI: **{bmi}**")
    # BMI interpretation (universal, not sex-specific calculation)
    if bmi < 18.5:
        st.warning("BMI: Underweight. Consider consulting a doctor.")
    elif 18.5 <= bmi < 24.9:
        st.success("BMI: Normal weight. Keep it up!")
    elif 24.9 <= bmi < 29.9:
        st.warning("BMI: Overweight. Consider consulting a doctor.")
    else:
        st.error("BMI: Obesity. Strongly consider consulting a doctor.")
else:
    st.warning("Please enter valid Height and Weight to calculate BMI.")

input_data['BMI'] = bmi # Add calculated BMI to input_data

st.markdown("---") # Horizontal line for separation

# --- Prediction Button and Result Section ---
if st.button("Predict 10-Year CHD Risk", help="Click to get the prediction based on the entered data"):
    # Ensure all required inputs are present, including BMI
    required_features = [
        'male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
        'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
        'diaBP', 'BMI', 'heartRate', 'glucose'
    ]
    # Re-insert a dummy 'education' value as the model expects it, even if not explicitly input by user
    # This is crucial because the model was trained with 'education' as a feature.
    # We'll use a common default or median value, e.g., 2 (High School Grad).
    input_data['education'] = 2 # Defaulting to High School Grad for 'education'

    # Check if BMI is calculated (i.e., height and weight were valid)
    if input_data['BMI'] == 0.0 and (height_cm == 0 or weight_kg == 0):
        st.error("Please enter valid Height and Weight to calculate BMI before predicting.")
        st.stop()

    # Convert input_data to a DataFrame in the correct order for the model pipeline
    try:
        input_df = pd.DataFrame([input_data], columns=required_features)

        # Predict probability and class
        prediction_proba = model.predict_proba(input_df)[0]
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error(f"High Likelihood of 10-Year Coronary Heart Disease ({prediction_proba[1]*100:.2f}%) 💔")
            # Removed balloons animation
        else:
            st.success(f"Low Likelihood of 10-Year Coronary Heart Disease ({prediction_proba[0]*100:.2f}%) ❤️")
            # Removed snow animation

        st.write("---")
        st.info("Disclaimer: This prediction is based on a machine learning model and should not be used as a substitute for professional medical advice. Consult a healthcare professional for diagnosis and treatment.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input fields are correctly filled and the model is trained.")
