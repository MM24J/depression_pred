import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the scaler, model, and feature names
@st.cache_resource
def load_resources():
    with open("reduced_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("xgboost_reduced.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return scaler, model, feature_names

reduced_scaler, model, feature_names = load_resources()

# Streamlit app
st.title("Mental Health Prediction App")

st.write("""
This app predicts the likelihood of mental health challenges based on user input.
Please fill out the form below to get a prediction.
""")

# Sidebar input
st.sidebar.header("Input Features")
user_inputs = {}
for feature in feature_names:
    if feature == "Have you ever had suicidal thoughts ?":
        user_inputs[feature] = st.sidebar.selectbox(feature, [0, 1], help="0 = No, 1 = Yes")
    elif feature == "Age":
        user_inputs[feature] = st.sidebar.slider(feature, 18, 60, 25, help="Enter your age (18-60)")
    elif feature == "Work_Life_Balance":
        user_inputs[feature] = st.sidebar.slider(feature, -4.0, 4.0, 0.0, step=1.0, help="Rate your work-life balance (0 = Poor, 4 = Excellent)")
    elif feature == "Financial Stress":
        user_inputs[feature] = st.sidebar.slider(feature, 1, 5, 3, help="Rate your financial stress (1 = Low, 5 = High)")
    elif feature == "Work Hours":
        user_inputs[feature] = st.sidebar.slider(feature, 0, 12, 6, help="Enter your average work hours per day (0-12)")


# Convert user inputs into a DataFrame with feature names
input_df = pd.DataFrame([user_inputs], columns=feature_names)

# Predict button
if st.sidebar.button("Predict"):
    # Scale the input
    scaled_input = reduced_scaler.transform(input_df)

    # Predict probability
    prediction = model.predict_proba(scaled_input)[:, 1][0]

    # Add a slider for adjusting the risk threshold
    threshold = st.sidebar.slider("Set Risk Threshold", 0.1, 0.5, 0.2, step=0.05)

    # Determine risk level based on the threshold
    if prediction > threshold:
        st.write(f"#### Risk Level: High (Threshold: {threshold:.2f})")
        st.warning("High risk detected. Please consult a mental health professional.")
    else:
        st.write(f"#### Risk Level: Low (Threshold: {threshold:.2f})")
        st.success("No significant risk detected. Maintain a healthy lifestyle!")
