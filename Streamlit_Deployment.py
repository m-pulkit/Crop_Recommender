# Set-ExecutionPolicy Unrestricted -Scope Process
# .\venv\Scripts\activate
# Set-ExecutionPolicy Default -Scope Process

# import Libraries
import numpy as np
import streamlit as st
import pickle

# Load your machine learning model
with open('RF_classifier.pkl', 'rb') as file:
    Crop_Mappings = pickle.load(file)
    RF_classifier = pickle.load(file)
# Define a function to make predictions
def predict(input_param):
    # Make predictions using the loaded model
    encoded_crop = RF_classifier.predict(input_param)
    return [i for i in Crop_Mappings if Crop_Mappings[i] == encoded_crop][0]


# Streamlit app starts here
st.title('Crop Recommender App')

# Add user input elements (e.g., sliders, text inputs, etc.)
col1, col2 = st.columns(2)

with col1:
    N = st.slider('Nitrogen Amount', min_value=1.0, max_value=150.0, value=40.0)
    P = st.slider('Phosphorous Amount', min_value=5.0, max_value=150.0, value=50.0)
    K = st.slider('Potassium Amount', min_value=5.0, max_value=210.0, value=35.0)

with col2:
    temperature = st.slider('Temperature in Celsius', min_value=5.0, max_value=60.0, value=25.0)
    humidity = st.slider('Humidity', min_value=100.0, max_value=100.0, value=80.0)
    ph = st.slider('pH', min_value=3.0, max_value=12.0, value=6.5)

rainfall = st.slider('Rainfall', min_value=0.0, max_value=400.0, value=95.0)

# Collect user input into a feature vector
input_features = np.array([N, P, K, temperature, humidity, ph, rainfall]).reshape(1, -1)

# Make predictions and display the result
if st.button('Predict'):
    crop = predict(input_features)
    st.header(f'Prediction: {crop.capitalize()}')
