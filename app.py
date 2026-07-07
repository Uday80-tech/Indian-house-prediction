import streamlit as st
import joblib
import pandas as pd
# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    font-size:42px;
    font-weight:bold;
    color:#0F172A;
}

.subtitle{
    font-size:18px;
    color:#475569;
}

.block-container{
    padding-top:2rem;
}

.stButton>button{
    width:100%;
    height:55px;
    font-size:20px;
    font-weight:bold;
    border-radius:10px;
}

.metric-card{
    background:#ffffff;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 5px 15px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown("<div class='title'>🏠 House Price Prediction</div>", unsafe_allow_html=True)

st.markdown(
"<div class='subtitle'>Predict the market price of residential properties using Machine Learning.</div>",
unsafe_allow_html=True
)

st.divider()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🏡 Property Details")

city = st.sidebar.selectbox(
    "City",
    [
        "Hyderabad",
        "Bangalore",
        "Pune",
        "Mumbai",
        "Nagpur"
    ]
)

locality = st.sidebar.selectbox(
    "Locality Tier",
    [
        "Budget",
        "Mid",
        "Premium"
    ]
)

furnishing = st.sidebar.selectbox(
    "Furnishing",
    [
        "Unfurnished",
        "Semi-Furnished",
        "Fully-Furnished"
    ]
)

# ---------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------

left,right = st.columns(2)

# ---------------------------------------------------
# LEFT COLUMN
# ---------------------------------------------------

with left:

    st.subheader("🏠 Property Information")

    bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=10,
        value=2
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    super_area = st.number_input(
        "Super Area (sqft)",
        min_value=100,
        max_value=10000,
        value=1200
    )

    carpet_area = st.number_input(
        "Carpet Area (sqft)",
        min_value=100,
        max_value=10000,
        value=1000
    )

    floor_no = st.number_input(
        "Floor Number",
        min_value=0,
        max_value=100,
        value=2
    )

    total_floors = st.number_input(
        "Total Floors",
        min_value=1,
        max_value=100,
        value=10
    )

    property_age = st.number_input(
        "Property Age (Years)",
        min_value=0,
        max_value=100,
        value=5
    )

# ---------------------------------------------------
# RIGHT COLUMN
# ---------------------------------------------------

with right:

    st.subheader("📍 Location & Amenities")

    parking =st.selectbox(
        "parking",
        [0,1],
        format_func=lambda x:"Yes" if x==1 else "No" 
    )

    lift = st.selectbox(
        "Lift",
        [0,1],
        format_func=lambda x:"Yes" if x==1 else "No"
    )

    gated = st.selectbox(
        "Gated Society",
        [0,1],
        format_func=lambda x:"Yes" if x==1 else "No"
    )

    metro = st.number_input(
        "Distance to Metro (km)",
        min_value=0.0,
        value=2.5,
        step=0.1
    )

    city_center = st.number_input(
        "Distance to City Center (km)",
        min_value=0.0,
        value=8.0,
        step=0.1
    )

    school = st.number_input(
        "Nearby School (km)",
        min_value=0.0,
        value=1.0,
        step=0.1
    )

    hospital = st.number_input(
        "Nearby Hospital (km)",
        min_value=0.0,
        value=1.5,
        step=0.1
    )

    crime = st.number_input(
        "Crime Rate Index",
        min_value=0.0,
        value=20.0,
        step=0.1
    )

st.divider()

# -----------------------------
# LOAD MODEL
# -----------------------------

preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/model.pkl")

predict = st.button("💰 Predict House Price")

if predict:

    input_df = pd.DataFrame({

        "City":[city],
        "Locality_Tier":[locality],
        "BHK":[bhk],
        "Bathrooms":[bathrooms],
        "Super_Area_sqft":[super_area],
        "Carpet_Area_sqft":[carpet_area],
        "Floor_No":[floor_no],
        "Total_Floors":[total_floors],
        "Property_Age_years":[property_age],
        "Parking":[parking],
        "Furnishing":[furnishing],
        "Lift":[lift],
        "Gated_Society":[gated],
        "Distance_to_Metro_km":[metro],
        "Distance_to_CityCenter_km":[city_center],
        "Nearby_School_km":[school],
        "Nearby_Hospital_km":[hospital],
        "Crime_Rate_Index":[crime]

    })

    processed = preprocessor.transform(input_df)

    prediction = model.predict(processed)[0]

    st.success("Prediction Successful ✅")

    st.markdown("---")

    st.markdown(
        f"""
        <div style="
        background:#16a34a;
        padding:25px;
        border-radius:15px;
        text-align:center;
        color:white;
        ">

        <h2>Estimated House Price</h2>

        <h1>₹ {prediction:,.0f}</h1>

        </div>
        """,
        unsafe_allow_html=True
    )

st.info("📊 Model Accuracy (R² Score): 0.91")

st.caption("Developed using Streamlit • Scikit-Learn • Machine Learning")