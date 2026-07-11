import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
import os 


# ==========================================================
# Project Paths
# ==========================================================

MODEL_DIR = "models"
DATASET_DIR = "datasets"
ASSET_DIR = "assets"



# Page configuration
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Modern UI Design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        padding: 1rem 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Container */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #00d4ff 100%);
        padding: 3rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        text-align: center;
    }
    
    .header-container h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .header-container p {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.95;
    }
    
    /* Input Section Styling */
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 200, 255, 0.1);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.8rem;
        border-radius: 12px;
        border-left: 5px solid #00d4ff;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    /* Result Cards */
    .result-card {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.3);
        text-align: center;
    }
    
    .result-card h2 {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .result-card-secondary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.25);
    }
    
    /* Top Predictions Grid */
    .prediction-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .prediction-item {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .prediction-item:hover {
        border-color: #00d4ff;
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.2);
    }
    
    .prediction-rank {
        font-size: 1.2rem;
        font-weight: 700;
        color: #00d4ff;
        margin-bottom: 0.5rem;
    }
    
    .prediction-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 0.5rem;
    }
    
    .prediction-confidence {
        font-size: 1.3rem;
        font-weight: 700;
        color: #00d4ff;
    }
    
    /* Slider Container */
    .slider-container {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #e0f7ff 0%, #b3e5fc 100%);
        border-left: 4px solid #00d4ff;
        padding: 1.2rem;
        border-radius: 10px;
        color: #01579b;
        margin-bottom: 1rem;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 4px solid #4caf50;
        padding: 1.2rem;
        border-radius: 10px;
        color: #1b5e20;
        margin: 1rem 0;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 212, 255, 0.4);
        background: linear-gradient(135deg, #00e5ff 0%, #00acc1 100%);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f5f7fa 0%, #e0e7ff 100%);
        border-radius: 10px !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    .streamlit-expanderContent {
        background: #fafafa;
        border-radius: 0 0 10px 10px;
    }
    
    /* Divider */
    .stDivider {
        border-color: #00d4ff !important;
    }
    
    /* Column Headers */
    h2 {
        color: #1e3c72;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    h3 {
        color: #2a5298;
        font-weight: 600;
    }
    
    /* Sidebar */
    .sidebar-content {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .header-container {
            padding: 2rem 1.5rem;
        }
        
        .header-container h1 {
            font-size: 2rem;
        }
        
        .input-section {
            padding: 1.5rem;
        }
    }
    </style>
""", unsafe_allow_html=True)
# ==========================================================
# Load Machine Learning Resources
# ==========================================================

@st.cache_resource
def load_resources():

    resources = {}

    # -------------------------
    # Models
    # -------------------------

    resources["crop_model"] = joblib.load(
        os.path.join(
            MODEL_DIR,
            "crop_recommendation_model.pkl"
        )
    )

    resources["yield_model"] = joblib.load(
        os.path.join(
            MODEL_DIR,
            "yield_prediction_model.pkl"
        )
    )

    resources["fertilizer_model"] = joblib.load(
        os.path.join(
            MODEL_DIR,
            "fertilizer_recommendation_model.pkl"
        )
    )

    # -------------------------
    # Feature Encoders
    # -------------------------

    resources["crop_feature_encoders"] = joblib.load(
        os.path.join(
            MODEL_DIR,
            "crop_feature_encoders.pkl"
        )
    )

    resources["yield_feature_encoders"] = joblib.load(
        os.path.join(
            MODEL_DIR,
            "yield_feature_encoders.pkl"
        )
    )

    resources["fertilizer_feature_encoders"] = joblib.load(
        os.path.join(
            MODEL_DIR,
            "fertilizer_feature_encoders.pkl"
        )
    )

    # -------------------------
    # Target Encoders
    # -------------------------

    resources["crop_target_encoder"] = joblib.load(
        os.path.join(
            MODEL_DIR,
            "crop_target_encoder.pkl"
        )
    )

    resources["fertilizer_target_encoder"] = joblib.load(
        os.path.join(
            MODEL_DIR,
            "fertilizer_target_encoder.pkl"
        )
    )

    # -------------------------
    # Scalers
    # -------------------------

    resources["crop_scaler"] = joblib.load(
        os.path.join(
            MODEL_DIR,
            "crop_scaler.pkl"
        )
    )

    resources["yield_scaler"] = joblib.load(
        os.path.join(
            MODEL_DIR,
            "yield_scaler.pkl"
        )
    )
    resources["fertilizer_bins"] = joblib.load(
    os.path.join(
        MODEL_DIR,
        "fertilizer_bins.pkl"
    )
)

    return resources


resources = load_resources()
# ==========================================================
# Verify Loaded Resources
# ==========================================================

st.sidebar.success("✅ Models Loaded Successfully")

# Sidebar
with st.sidebar:
    st.title("🌾 Smart Crop System")
    st.markdown("---")
    
    page = st.radio(
        "Select Page",
        ["Home", "Prediction", "Analysis", "About"]
    )

# Main header
st.markdown("""
    <div class="header-container">
        <h1>🌾 Smart Crop Recommendation System</h1>
        <p>Intelligent crop prediction based on soil and climate conditions</p>
    </div>
""", unsafe_allow_html=True)

if page == "Home":
    st.markdown("""
    <div class="header-container">
        <h1>🌾 Smart Crop Recommendation System</h1>
        <p>Intelligent AI-powered farming assistance for maximum yield</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Smart Predictions</h3>
            <p>Get AI-powered crop recommendations based on your soil and climate conditions with confidence scores</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Yield Analysis</h3>
            <p>Predict your crop yield and optimize resources for better productivity and profitability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🧪 Fertilizer Guide</h3>
            <p>Get personalized fertilizer recommendations based on soil health and crop requirements</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("<h2>📖 How to Use</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin-top: 0;">Step 1️⃣</h3>
            <p>Enter your soil parameters</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin-top: 0;">Step 2️⃣</h3>
            <p>Get recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin-top: 0;">Step 3️⃣</h3>
            <p>Plan your farming</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("<h2>⚡ Quick Start</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🌱 What You'll Get:**
        - ✅ Top 3 crop recommendations
        - ✅ Confidence scores for each crop
        - ✅ Expected yield predictions
        - ✅ Fertilizer recommendations
        - ✅ Alternative crop options
        """)
    
    with col2:
        st.markdown("""
        **📋 What You Need:**
        - 🔹 Soil nutrients (N, P, K)
        - 🔹 Soil pH level
        - 🔹 Rainfall data
        - 🔹 Temperature
        - 🔹 Humidity percentage
        """)

elif page == "Prediction":
    st.markdown("""
    <div class="header-container">
        <h1>🌿 Crop Recommendation</h1>
        <p>Enter your soil and climate parameters to get personalized crop recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        📌 <b>Tip:</b> Use accurate soil testing data for better recommendations. Adjust values based on your farm conditions.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3>📊 Enter Soil Parameters</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='slider-container'>", unsafe_allow_html=True)
        nitrogen = st.slider("🔹 Nitrogen (N)", 0, 140, 50, help="Nitrogen content in soil (0-140 mg/kg)")
        potassium = st.slider("🔹 Potassium (K)", 0, 205, 50, help="Potassium content in soil (0-205 mg/kg)")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='slider-container'>", unsafe_allow_html=True)
        phosphorous = st.slider("🔹 Phosphorous (P)", 0, 145, 50, help="Phosphorous content in soil (0-145 mg/kg)")
        ph = st.slider("🔹 pH Value", 3.5, 9.0, 6.5, help="Soil pH level (3.5-9.0)")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='slider-container'>", unsafe_allow_html=True)
        rainfall = st.slider("🔹 Rainfall (mm)", 20, 250, 100, help="Average annual rainfall (20-250 mm)")
        temperature = st.slider("🔹 Temperature (°C)", 8.8, 43.7, 25.0, help="Average temperature (8.8-43.7°C)")
        humidity = st.slider("🔹 Humidity (%)", 0.0, 100.0, 70.0, help="Average humidity (0-100%)")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🌾 Get Recommendation", use_container_width=True, key="crop_recommendation"):
        with st.spinner("🔄 Analyzing soil conditions and climate patterns..."):

            # -----------------------------------------
            # Create Input Data
            # -----------------------------------------

            input_data = pd.DataFrame({
                "n": [nitrogen],
                "p": [phosphorous],
                "k": [potassium],
                "temperature": [temperature],
                "humidity": [humidity],
                "ph": [ph],
                "rainfall": [rainfall]
            })

        # -----------------------------------------
        # Feature Engineering
        # -----------------------------------------

        input_data["total_npk"] = (
            input_data["n"]
            + input_data["p"]
            + input_data["k"]
        )

        input_data["soil_fertility_index"] = (
            input_data["n"]
            + input_data["p"]
            + input_data["k"]
        ) / 3

        input_data["npk_balance_ratio"] = (
            input_data["n"]
            / (
                input_data["p"]
                + input_data["k"]
                + 1
            )
        )

        # -----------------------------------------
        # Create Categories
        # -----------------------------------------

        input_data["temperature_category"] = pd.cut(
            input_data["temperature"],
            bins=[0, 20, 30, 50],
            labels=["Low", "Moderate", "High"]
        )

        input_data["rainfall_category"] = pd.cut(
            input_data["rainfall"],
            bins=[0, 100, 200, 400],
            labels=["Low", "Medium", "High"]
        )

        input_data["ph_category"] = pd.cut(
            input_data["ph"],
            bins=[0, 6.5, 7.5, 14],
            labels=["Acidic", "Neutral", "Alkaline"]
        )

        # -----------------------------------------
        # Encode Categorical Features
        # -----------------------------------------

        for column in [
            "temperature_category",
            "rainfall_category",
            "ph_category"
        ]:
            input_data[column] = resources[
                "crop_feature_encoders"
            ][column].transform(
                input_data[column].astype(str)
            )

        # -----------------------------------------
        # Correct Feature Order
        # -----------------------------------------

        crop_features = [
            "n",
            "p",
            "k",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
            "total_npk",
            "soil_fertility_index",
            "npk_balance_ratio",
            "temperature_category",
            "rainfall_category",
            "ph_category"
        ]

        input_data = input_data[crop_features]

        # -----------------------------------------
        # Scale Input
        # -----------------------------------------

        input_scaled = resources[
            "crop_scaler"
        ].transform(input_data)

        # -----------------------------------------
        # Predict Crop
        # -----------------------------------------

        prediction = resources[
            "crop_model"
        ].predict(input_scaled)

        crop_name = resources[
            "crop_target_encoder"
        ].inverse_transform(prediction)

        # Get prediction probabilities for all classes
        if hasattr(resources["crop_model"], "predict_proba"):
            proba = resources["crop_model"].predict_proba(input_scaled)[0]
            top_3_idx = np.argsort(proba)[-3:][::-1]
            top_3_crops = resources["crop_target_encoder"].inverse_transform(top_3_idx)
            top_3_proba = proba[top_3_idx]
            
            # Display Result with Top 3 predictions
            st.markdown("""
            <div class="result-card">
                <h2>🌾 Top Recommended Crop</h2>
                <h3 style="font-size: 2rem; margin: 0.5rem 0;">""" + crop_name[0] + """</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h3 style='text-align: center; color: #1e3c72; margin-top: 2rem;'>📊 Top 3 Crop Predictions</h3>", unsafe_allow_html=True)
            
            cols = st.columns(3)
            for i, (crop, confidence) in enumerate(zip(top_3_crops, top_3_proba)):
                with cols[i]:
                    st.markdown(f"""
                    <div class="prediction-item">
                        <div class="prediction-rank">#{i+1}</div>
                        <div class="prediction-name">{crop}</div>
                        <div class="prediction-confidence">{confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card">
                <h2>🌾 Recommended Crop</h2>
                <h3 style="font-size: 2rem; margin: 0.5rem 0;">{crop_name[0]}</h3>
            </div>
            """, unsafe_allow_html=True)

        # Add explanation about confidence
        with st.expander("ℹ️ Understanding the Predictions", expanded=True):
            st.markdown("""
            **📌 What This Means:**
            - The model analyzes your soil nutrients, pH, climate, and rainfall
            - It shows the top 3 most suitable crops for your conditions
            - **Confidence %** shows how confident the model is about each recommendation
            
            **💡 Tips for Better Results:**
            - Use soil testing data for accurate N, P, K values
            - Consider your local climate and past crop performance
            - All top 3 options are viable - choose based on market demand
            - If confidence is low, consult agricultural experts
            """)

    # =========================================================
    # Yield Prediction
    # ==========================================================

    st.divider()
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin-top: 0;">📈 Yield Prediction</h2>
        <p>Estimate your crop yield based on agricultural conditions</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔄 Open Yield Prediction", expanded=True):

        yield_encoders = resources["yield_feature_encoders"]

        col1, col2, col3 = st.columns(3)

        with col1:
            region = st.selectbox(
                "Region",
                yield_encoders["region"].classes_
            )

            soil_type = st.selectbox(
                "Soil Type",
                yield_encoders["soil_type"].classes_
            )

            crop = st.selectbox(
                "Crop",
                yield_encoders["crop"].classes_
            )

            rainfall_yield = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                value=200.0
            )

        with col2:
            temperature_yield = st.number_input(
                "Temperature (°C)",
                value=25.0
            )

            fertilizer_used = st.selectbox(
                "Fertilizer Used",
                yield_encoders["fertilizer_used"].classes_
            )

            irrigation_used = st.selectbox(
                "Irrigation Used",
                [False, True]
            )

            weather_condition = st.selectbox(
                "Weather Condition",
                yield_encoders["weather_condition"].classes_
            )

        with col3:
            days_to_harvest = st.number_input(
                "Days to Harvest",
                min_value=1.0,
                value=100.0
            )

            climate_zone = st.selectbox(
                "Climate Zone",
                yield_encoders["climate_zone"].classes_
            )

            growing_season = st.selectbox(
                "Growing Season",
                yield_encoders["growing_season"].classes_
            )

        if st.button(
            "📈 Predict Yield",
            use_container_width=True
        ):

            water_availability_index = (
                rainfall_yield + 100
                if irrigation_used
                else rainfall_yield
            )

            yield_input = pd.DataFrame({
                "region": [region],
                "soil_type": [soil_type],
                "crop": [crop],
                "rainfall_mm": [rainfall_yield],
                "temperature_celsius": [temperature_yield],
                "fertilizer_used": [fertilizer_used],
                "irrigation_used": [irrigation_used],
                "weather_condition": [weather_condition],
                "days_to_harvest": [days_to_harvest],
                "water_availability_index": [
                    water_availability_index
                ],
                "climate_zone": [climate_zone],
                "growing_season": [growing_season]
            })

            categorical_columns = [
                "region",
                "soil_type",
                "crop",
                "fertilizer_used",
                "weather_condition",
                "climate_zone",
                "growing_season"
            ]

            for column in categorical_columns:
                yield_input[column] = (
                    yield_encoders[column]
                    .transform(yield_input[column])
                )

            # -----------------------------------------
            # Scale Input
            # -----------------------------------------

            yield_scaled = resources[
                "yield_scaler"
            ].transform(yield_input)

            # -----------------------------------------
            # Predict Yield
            # -----------------------------------------

            yield_prediction = resources[
                "yield_model"
            ].predict(yield_scaled)

            st.markdown(f"""
            <div class="result-card-secondary">
                <h3 style="margin-top: 0; color: white;">📈 Predicted Yield</h3>
                <h2 style="color: white; font-size: 2.2rem; margin: 0.5rem 0;">{yield_prediction[0]:.2f} <span style="font-size: 1.2rem;">tons/hectare</span></h2>
            </div>
            """, unsafe_allow_html=True)
    # ==========================================================
    # Fertilizer Recommendation
    # ==========================================================

    st.divider()
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin-top: 0;">🧪 Fertilizer Recommendation</h2>
        <p>Get personalized fertilizer recommendations for optimal crop growth</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(
        "🔄 Open Fertilizer Recommendation",
        expanded=True
    ):

        fertilizer_encoders = resources[
            "fertilizer_feature_encoders"
        ]

        fertilizer_bins = resources[
            "fertilizer_bins"
        ]

        col1, col2, col3 = st.columns(3)

        with col1:

            soil_type_f = st.selectbox(
                "Fertilizer Soil Type",
                fertilizer_encoders["soil_type"].classes_
            )

            soil_ph_f = st.number_input(
                "Soil pH",
                min_value=0.1,
                max_value=14.0,
                value=6.5
            )

            soil_moisture_f = st.number_input(
                "Soil Moisture",
                min_value=0.0,
                value=50.0
            )

            organic_carbon_f = st.number_input(
                "Organic Carbon",
                min_value=0.0,
                value=2.0
            )

            electrical_conductivity_f = st.number_input(
                "Electrical Conductivity",
                min_value=0.0,
                value=1.0
            )

            nitrogen_f = st.number_input(
                "Nitrogen Level",
                min_value=0.0,
                value=50.0
            )

            phosphorus_f = st.number_input(
                "Phosphorus Level",
                min_value=0.0,
                value=50.0
            )

        with col2:

            potassium_f = st.number_input(
                "Potassium Level",
                min_value=0.0,
                value=50.0
            )

            temperature_f = st.number_input(
                "Fertilizer Temperature (°C)",
                value=25.0
            )

            humidity_f = st.number_input(
                "Fertilizer Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=70.0
            )

            rainfall_f = st.number_input(
                "Fertilizer Rainfall (mm)",
                min_value=0.0,
                value=100.0
            )

            crop_type_f = st.selectbox(
                "Crop Type",
                fertilizer_encoders["crop_type"].classes_
            )

            growth_stage_f = st.selectbox(
                "Crop Growth Stage",
                fertilizer_encoders[
                    "crop_growth_stage"
                ].classes_
            )

            season_f = st.selectbox(
                "Season",
                fertilizer_encoders["season"].classes_
            )

        with col3:

            irrigation_type_f = st.selectbox(
                "Irrigation Type",
                fertilizer_encoders[
                    "irrigation_type"
                ].classes_
            )

            previous_crop_f = st.selectbox(
                "Previous Crop",
                fertilizer_encoders[
                    "previous_crop"
                ].classes_
            )

            region_f = st.selectbox(
                "Fertilizer Region",
                fertilizer_encoders["region"].classes_
            )

            fertilizer_last_f = st.selectbox(
               "Fertilizer Used Last Season",
                [False, True]
            )

            yield_last_f = st.number_input(
                "Yield Last Season",
                min_value=0.0,
                value=5.0
            )

        if st.button(
            "🧪 Recommend Fertilizer",
            use_container_width=True
        ):

            fertilizer_input = pd.DataFrame({
                "soil_type": [soil_type_f],
                "soil_ph": [soil_ph_f],
                "soil_moisture": [soil_moisture_f],
                "organic_carbon": [organic_carbon_f],
                "electrical_conductivity": [
                    electrical_conductivity_f
                ],
                "nitrogen_level": [nitrogen_f],
                "phosphorus_level": [phosphorus_f],
                "potassium_level": [potassium_f],
                "temperature": [temperature_f],
                "humidity": [humidity_f],
                "rainfall": [rainfall_f],
                "crop_type": [crop_type_f],
                "crop_growth_stage": [growth_stage_f],
                "season": [season_f],
                "irrigation_type": [irrigation_type_f],
                "previous_crop": [previous_crop_f],
                "region": [region_f],
                "fertilizer_used_last_season": [
                    fertilizer_last_f
                ],
                "yield_last_season": [yield_last_f]
            })

            # Feature Engineering

            fertilizer_input["total_nutrients"] = (
                fertilizer_input["nitrogen_level"]
                + fertilizer_input["phosphorus_level"]
                + fertilizer_input["potassium_level"]
            )

            fertilizer_input["soil_health_index"] = (
                fertilizer_input["organic_carbon"] * 0.4
                + fertilizer_input["soil_moisture"] * 0.3
                + fertilizer_input[
                    "electrical_conductivity"
                ] * 0.3
            )

            fertilizer_input["climate_index"] = (
                fertilizer_input["temperature"]
                + fertilizer_input["humidity"]
                + fertilizer_input["rainfall"]
            ) / 3

            # Category Engineering

            fertilizer_input["soil_ph_category"] = pd.cut(
                fertilizer_input["soil_ph"],
                bins=[0, 6.5, 7.5, 14],
                labels=["Acidic", "Neutral", "Alkaline"],
                include_lowest=True
            )

            fertilizer_input["moisture_category"] = pd.cut(
                fertilizer_input["soil_moisture"],
                bins=fertilizer_bins["moisture_bins"],
                labels=["Low", "Medium", "High"],
                include_lowest=True
            )

            fertilizer_input["yield_category"] = pd.cut(
                fertilizer_input["yield_last_season"],
                bins=fertilizer_bins["yield_bins"],
                labels=["Low", "Medium", "High"],
                include_lowest=True
            )

            fertilizer_input["nutrient_richness"] = pd.cut(
                fertilizer_input["total_nutrients"],
                bins=fertilizer_bins["nutrient_bins"],
                labels=["Poor", "Average", "Rich"],
                include_lowest=True
            )

            # Encoding

            for column, encoder in fertilizer_encoders.items():

                if column in fertilizer_input.columns:

                    fertilizer_input[column] = (
                        encoder.transform(
                            fertilizer_input[column].astype(str)
                        )
                    )

            # Exact Feature Order

            fertilizer_features = [
                "soil_type",
                "soil_ph",
                "soil_moisture",
                "organic_carbon",
                "electrical_conductivity",
                "nitrogen_level",
                "phosphorus_level",
                "potassium_level",
                "temperature",
                "humidity",
                "rainfall",
                "crop_type",
                "crop_growth_stage",
                "season",
                "irrigation_type",
                "previous_crop",
                "region",
                "fertilizer_used_last_season",
                "yield_last_season",
                "total_nutrients",
                "soil_health_index",
                "climate_index",
                "soil_ph_category",
                "moisture_category",
                "yield_category",
                "nutrient_richness"
            ]

            fertilizer_input = fertilizer_input[
                fertilizer_features
            ]

            # Prediction

            fertilizer_prediction = resources[
                "fertilizer_model"
            ].predict(fertilizer_input)

            fertilizer_name = resources[
                "fertilizer_target_encoder"
            ].inverse_transform(
                fertilizer_prediction
            )

            # Get prediction probabilities for all classes
            if hasattr(resources["fertilizer_model"], "predict_proba"):
                proba = resources["fertilizer_model"].predict_proba(fertilizer_input)[0]
                top_3_idx = np.argsort(proba)[-3:][::-1]
                top_3_ferts = resources["fertilizer_target_encoder"].inverse_transform(top_3_idx)
                top_3_proba = proba[top_3_idx]
                
                st.markdown(f"""
                <div class="result-card-secondary">
                    <h3 style="margin-top: 0; color: white;">🧪 Recommended Fertilizer</h3>
                    <h2 style="color: white; font-size: 2rem; margin: 0.5rem 0;">{fertilizer_name[0]}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<h3 style='text-align: center; color: #1e3c72; margin-top: 1.5rem;'>📊 Top 3 Options</h3>", unsafe_allow_html=True)
                cols = st.columns(3)
                for i, (fert, confidence) in enumerate(zip(top_3_ferts, top_3_proba)):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="prediction-item">
                            <div class="prediction-rank">#{i+1}</div>
                            <div class="prediction-name">{fert}</div>
                            <div class="prediction-confidence">{confidence:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-secondary">
                    <h3 style="margin-top: 0; color: white;">🧪 Recommended Fertilizer</h3>
                    <h2 style="color: white; font-size: 2rem; margin: 0.5rem 0;">{fertilizer_name[0]}</h2>
                </div>
                """, unsafe_allow_html=True)

elif page == "Analysis":

    st.markdown("""
    <div class="header-container">
        <h1>📊 Analysis & Insights</h1>
        <p>Overview of the smart crop recommendation system</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>📌 System Overview:</b> This section provides insights into the machine learning models 
        and features that power the Smart Crop Recommendation System.
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================
    # System Statistics
    # ==========================================================

    st.markdown("<h3>📊 System Statistics</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.2rem; color: #00d4ff; font-weight: 700;">3</div>
            <div style="color: #666;">ML Models Integrated</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.2rem; color: #00d4ff; font-weight: 700;">3</div>
            <div style="color: #666;">Active Prediction Modules</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.2rem; color: #00d4ff; font-weight: 700;">✅</div>
            <div style="color: #666;">System Status: Operational</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==========================================================
    # Model Overview
    # ==========================================================

    st.markdown("<h3>🤖 Model Overview</h3>", unsafe_allow_html=True)

    model_data = pd.DataFrame({
        "Model Module": [
            "Crop Recommendation",
            "Yield Prediction",
            "Fertilizer Recommendation"
        ],
        "Task Type": [
            "Classification",
            "Regression",
            "Classification"
        ],
        "Status": [
            "Active",
            "Active",
            "Active"
        ]
    })

    st.dataframe(
        model_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==========================================================
    # Feature Engineering
    # ==========================================================

    st.subheader("⚙️ Feature Engineering")

    st.markdown("""
    ### 🌾 Crop Recommendation

    - Total NPK
    - Soil Fertility Index
    - NPK Balance Ratio
    - Temperature Category
    - Rainfall Category
    - Soil pH Category

    ### 📈 Yield Prediction

    - Water Availability Index
    - Climate Zone
    - Growing Season

    ### 🧪 Fertilizer Recommendation

    - Total Nutrients
    - Soil Health Index
    - Climate Index
    - Soil pH Category
    - Moisture Category
    - Previous Yield Category
    - Nutrient Richness Category
    """)

    st.divider()

    # ==========================================================
    # System Workflow
    # ==========================================================

    st.subheader("🔄 Machine Learning Workflow")

    st.markdown("""
    **Data Collection**
    
    ↓
    
    **Data Cleaning & Preprocessing**
    
    ↓
    
    **Exploratory Data Analysis**
    
    ↓
    
    **Feature Engineering**
    
    ↓
    
    **Feature Encoding & Scaling**
    
    ↓
    
    **Model Training & Evaluation**
    
    ↓
    
    **Model Selection**
    
    ↓
    
    **Streamlit Deployment**
    """)

    st.success(
        "✅ All three machine learning modules are "
        "successfully integrated and operational."
    )

elif page == "About":
    st.header("About This Project")
    
    st.markdown("""
    ### Smart Crop Recommendation System
    
    **Version**: 1.0.0  
    **Status**: Development Phase
    
    ### Project Overview
    This system uses machine learning to recommend the most suitable crops
    based on various environmental and soil parameters.
    
    ### Data Sources
    - Soil analysis data
    - Climate records
    - Historical crop data
    - Agricultural research
    
    ### Machine Learning Models
    - Classification algorithms
    - Ensemble methods
    - Neural networks
    
    ### Technologies
    - Python 3.8+
    - Streamlit
    - Scikit-learn
    - TensorFlow
    - Pandas & NumPy
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🌾 Smart Crop Recommendation System | Built with ❤️ | v1.0.0</p>
</div>
""", unsafe_allow_html=True)
