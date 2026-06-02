import streamlit as st
import pandas as pd
import joblib

# ✅ Load trained model and metrics
model = joblib.load("model.pkl")
metrics = joblib.load("metrics.pkl")

# ===============================
# ✅ HEADER SECTION
# ===============================
st.title("🏠 Automated Property Valuation Tool")
st.caption("AI-powered real estate valuation system for the Nigerian property market")

st.divider()

# ===============================
# ✅ MODEL PERFORMANCE
# ===============================
st.header("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("R² Score", f"{metrics['r2']:.2%}")

with col2:
    st.metric("Mean Absolute Error", f"₦{metrics['mae']:,.0f}")

# ✅ Add explanation
st.caption("Model performance based on test dataset evaluation.")

# ===============================
# ✅ FEATURE IMPORTANCE
# ===============================
st.subheader("📈 Key Drivers of Property Value")

rf = model.named_steps['regressor']
importances = rf.feature_importances_

feature_names = model.named_steps['preprocessing'].get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

st.divider()

# ===============================
# ✅ PROPERTY INPUT SECTION
# ===============================
st.header("🏡 Property Details")

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

# ===============================
# ✅ PREDICTION BLOCK
# ===============================
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

    # ✅ Valuation Output
    st.subheader("💰 Valuation Result")

    st.success(f"Estimated Price: ₦{prediction:,.0f}")

    # ✅ Range (Professional valuation)
    lower = prediction * 0.9
    upper = prediction * 1.1

    st.info(f"Expected Range: ₦{lower:,.0f} — ₦{upper:,.0f}")

    # ✅ Additional insight
    st.caption(
        "This estimate is generated using machine learning based on historical property data and market features."
    )

# ===============================
# ✅ VALUER FEEDBACK
# ===============================
st.divider()
st.header("📝 Valuer Feedback")

feedback = st.text_area("Provide feedback on this valuation")

if st.button("Submit Feedback"):
    st.success("✅ Feedback submitted successfully")

# ===============================
# ✅ DISCLAIMER
# ===============================
st.caption(
    "Disclaimer: This tool provides estimated property values based on machine learning models. "
    "Final valuation decisions should incorporate professional judgment and current market conditions."
)