import streamlit as st
import joblib
import numpy as np

# Set dashboard layout configuration
st.set_page_config(page_title="Smart Farmer Assistant", page_icon="🌾")

# Safely load the saved model artifacts from Step 1
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("crop_recommendation_model.pkl")
        scaler = joblib.load("soil_features_scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_artifacts()

# App Header Interface
st.title("🌾 Smart Farmer Assistant")
st.markdown("### Real-Time Precision Agriculture Insights")
st.write("Provide your field's soil parameters and environmental conditions below to predict the optimal crop.")
st.markdown("---")

if model is None or scaler is None:
    st.error("⚠️ Model files are missing!")
    st.info("Please make sure you run `python train_engine.py` in your terminal first to generate your model files.")
else:
    st.subheader("📊 Enter Field Parameters")
    
    # Split input fields into two clean side-by-side columns
    col1, col2 = st.columns(2)
    
    with col1:
        n = st.number_input("Nitrogen (N) (mg/kg)", min_value=0, max_value=150, value=50)
        p = st.number_input("Phosphorus (P) (mg/kg)", min_value=0, max_value=150, value=50)
        k = st.number_input("Potassium (K) (mg/kg)", min_value=0, max_value=200, value=50)
        ph = st.number_input("Soil pH Level (0 - 14)", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        
    with col2:
        temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.5)
        humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.5)
        rainfall = st.number_input("Average Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)

    st.markdown("---")

    # When the user clicks the submission button
    if st.button("Generate Farming Recommendation", type="primary"):
        # Combine parameters into a structured array
        input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
        
        # Transform using the saved standard scaler
        scaled_data = scaler.transform(input_data)
        
        # Predict the recommended crop
        prediction = model.predict(scaled_data)[0]
        
        # Output the results
        st.balloons()
        st.success(f"🌱 **Recommended Crop for Your Optimal Yield:** {prediction.upper()}")