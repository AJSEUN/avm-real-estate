import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# ===============================
# PAGE CONFIG (NEW UI FEEL)
# ===============================
st.set_page_config(
    page_title="Protine AVM",
    page_icon="🏠",
    layout="wide"
)

# ===============================
# LOAD MODEL & DATA
# ===============================
model = joblib.load("model.pkl")
metrics = joblib.load("metrics.pkl")
data = pd.read_csv("synthetic_real_estate_data.csv")

location_prices = data.groupby("location")["price"].mean().reset_index()
location_prices = location_prices.sort_values(by="price", ascending=False)

# ===============================
# PDF FUNCTION
# ===============================
def create_pdf(data):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)

    y = 750
    p.setFont("Helvetica", 12)

    p.drawString(200, y, "PROPERTY VALUATION REPORT")
    y -= 40

    for key, value in data.items():
        p.drawString(50, y, f"{key}: {value}")
        y -= 20

    p.drawString(50, y - 30, "Protine Ltd | AI Valuation System")
    p.save()

    buffer.seek(0)
    return buffer

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🏢 Protine AVM")

st.sidebar.info(
    "AI-powered property valuation system built for Lagos real estate market."
)

st.sidebar.metric("Model MAE", f"₦{metrics['mae']:,.0f}")

# ===============================
# HEADER
# ===============================
st.title("🏠 Automated Property Valuation Tool")
st.caption("Professional AI-assisted valuation for Lagos real estate")

# ===============================
# HEATMAP SECTION
# ===============================
st.divider()
st.header("🗺️ Lagos Market Overview")

fig, ax = plt.subplots()
ax.bar(location_prices["location"], location_prices["price"])
ax.set_ylabel("Price (₦)")
ax.set_title("Average Property Price by Location")

st.pyplot(fig)

highest = location_prices.iloc[0]
lowest = location_prices.iloc[-1]

st.write(f"📍 Most Expensive Area: **{highest['location']}**")
st.write(f"📉 Most Affordable Area: **{lowest['location']}**")

# ===============================
# INPUT SECTION (2 COLUMN)
# ===============================
st.divider()
st.header("🏡 Property Details")

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Location", ["Lekki", "Ikoyi", "Yaba", "Surulere", "Victoria Island"])
    bedrooms = st.number_input("Bedrooms", 1, 10)
    size_sqft = st.number_input("Size (sqft)", 500, 5000)
    security_rating = st.slider("Security Rating", 1, 5)

with col2:
    property_type = st.selectbox("Property Type", ["Apartment", "Duplex"])
    bathrooms = st.number_input("Bathrooms", 1, 10)
    age = st.number_input("Property Age", 0, 30)
    road_quality = st.slider("Road Quality", 1, 5)

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Predict Property Value", use_container_width=True):

    input_data = pd.DataFrame([{
        "location": location,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "size_sqft": size_sqft,
        "age": age,
        "security_rating": security_rating,
        "road_quality": road_quality
    }])

    prediction = model.predict(input_data)[0]

    location_factor = {
        "Ikoyi": 1.25,
        "Victoria Island": 1.2,
        "Lekki": 1.1,
        "Surulere": 1.0,
        "Yaba": 0.95
    }

    adjusted_price = prediction * location_factor.get(location, 1)

    lower = adjusted_price * 0.9
    upper = adjusted_price * 1.1

    # ===============================
    # RESULT DISPLAY
    # ===============================
    st.markdown("## 💰 Valuation Dashboard")

    colA, colB, colC = st.columns(3)

    colA.metric("Estimated Value", f"₦{adjusted_price:,.0f}")
    colB.metric("Lower Range", f"₦{lower:,.0f}")
    colC.metric("Upper Range", f"₦{upper:,.0f}")

    # ===============================
    # MARKET COMPARISON
    # ===============================
    avg_price = location_prices[
        location_prices["location"] == location
    ]["price"].values[0]

    st.write(f"📊 Market Average in {location}: ₦{avg_price:,.0f}")

    if adjusted_price > avg_price:
        st.success("⚡ Above Market Value")
    else:
        st.info("💡 Below Market Value")

    # ===============================
    # CONFIDENCE
    # ===============================
    mae = metrics["mae"]

    if mae < 5000000:
        confidence = "High"
    elif mae < 10000000:
        confidence = "Medium"
    else:
        confidence = "Low"

    st.write(f"📊 Confidence Level: **{confidence}**")

    # ===============================
    # INSIGHTS
    # ===============================
    st.subheader("📌 Investment Insight")

    if adjusted_price > 70000000:
        st.write("Premium property – high-value investment area")
    elif adjusted_price < 20000000:
        st.write("Affordable property – strong demand segment")
    else:
        st.write("Mid-market property – stable investment zone")

    # ===============================
    # RISK
    # ===============================
    if (upper - lower) > 15000000:
        st.warning("⚠️ High uncertainty — recommend manual review")

    # ===============================
    # PDF DOWNLOAD
    # ===============================
    report_data = {
        "Location": location,
        "Property Type": property_type,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Size": f"{size_sqft} sqft",
        "Estimated Price": f"₦{adjusted_price:,.0f}",
        "Range": f"₦{lower:,.0f} - ₦{upper:,.0f}",
        "Confidence": confidence
    }

    pdf = create_pdf(report_data)

    st.download_button(
        "📄 Download Valuation Report",
        pdf,
        "valuation_report.pdf",
        "application/pdf"
    )

# ===============================
# FEEDBACK
# ===============================
st.divider()
st.header("📝 Valuer Feedback")

feedback = st.text_area("Provide professional feedback")

if st.button("Submit Feedback"):
    st.success("✅ Feedback recorded")

# ===============================
# FOOTER
# ===============================
st.caption(
    "This system provides AI-based valuation estimates. Final decisions should involve expert review."
)