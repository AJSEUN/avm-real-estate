import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
data = pd.read_csv("real_estate_data.csv")

X = data.drop("price", axis=1)
y = data["price"]

categorical_cols = ["location", "property_type"]
numerical_cols = ["bedrooms", "bathrooms", "size_sqft", "age"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)

# UI
st.title("🏠 Property Valuation Tool")

location = st.text_input("Location")
property_type = st.selectbox("Property Type", ["Apartment", "Duplex"])
bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1, 10)
size_sqft = st.number_input("Size (sqft)")
age = st.number_input("Property Age")

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
    st.success(f"Estimated Price: ₦{prediction:,.2f}")