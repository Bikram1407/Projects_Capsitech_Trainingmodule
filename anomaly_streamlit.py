import streamlit as st
import pickle
import numpy as np

# Load the trained IsolationForest model
def load_model():
    with open("Anomaly_Detection_in_IoT_Sensor_Data/model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Streamlit UI
st.title("Anomaly Detection using Isolation Forest")

st.write("Enter feature values to check if the data is normal or an anomaly.")

# Update to match the correct number of features
feature_input = []
num_features = 11  # Change this based on the dataset
for i in range(num_features):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    feature_input.append(value)

# Convert input to NumPy array
feature_array = np.array(feature_input).reshape(1, -1)

# Load model
model = load_model()

# Predict
if st.button("Check Anomaly"):
    prediction = model.predict(feature_array)
    result = "Anomaly Detected" if prediction[0] == -1 else "Normal Data"
    st.write(f"Prediction: {result}")
