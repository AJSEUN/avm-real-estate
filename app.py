import streamlit as st
import pandas as pd
import joblib

# ✅ Load trained model and metrics
model = joblib.load("model.pkl")
metrics = joblib.load("metrics.pkl")

# ✅ App Title
st.title("🏠 Automated Property Valuation Tool")

# ✅ ✅ MODEL PERFORMANCE (THIS IS YOUR STEP 3)
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("R² Score", f"{metrics['r2']:.2%}")

with col2:
    st.metric("MAE", f"₦{metrics['mae']:,.0f}")

# ✅ User Inputs
location = st.selectbox(
    "Location",
    ["Lekki", "Ikoyi", "Yaba", "Surulere", "Victoria Island"]
)

property_type = st.selectbox(
    "Property Type",
    ["Apartment", "Duplex"]
)

bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1, 10)
size_sqft = st.number_input("Size (sqft)")
age = st.number_input("Property Age")

# ✅ Prediction
if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "location": location,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "size_sqft": size_sqft,
        "age": age
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Price: ₦{prediction:,.0f}")
``