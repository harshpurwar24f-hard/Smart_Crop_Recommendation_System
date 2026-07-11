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

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
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
    st.header("Welcome to Smart Crop Recommendation System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### About This System
        
        This intelligent system helps farmers and agricultural specialists:
        - **Predict** the best crops for their land
        - **Optimize** resource allocation
        - **Maximize** crop yields
        - **Plan** effective farming strategies
        
        ### Key Features
        - 🎯 Accurate crop predictions
        - 📊 Detailed data analysis
        - 📈 Performance metrics
        - 🔄 Real-time recommendations
        """)
    
    with col2:
        st.info("""
        #### How to Use
        
        1. Go to **Prediction** page
        2. Enter soil and climate parameters
        3. Click "Get Recommendation"
        4. View the recommended crops
        5. Check **Analysis** for detailed insights
        """)

elif page == "Prediction":
    st.header("Crop Prediction")
    
    st.info("Enter the soil and climate parameters below to get crop recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nitrogen = st.slider("Nitrogen (N)", 0, 140, 50)
        potassium = st.slider("Potassium (K)", 0, 205, 50)
    
    with col2:
        phosphorous = st.slider("Phosphorous (P)", 0, 145, 50)
        ph = st.slider("pH Value", 3.5, 9.0, 6.5)
    
    with col3:
        rainfall = st.slider("Rainfall (mm)", 20, 250, 100)
        temperature = st.slider("Temperature (°C)", 8.8, 43.7, 25.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 70.0)
    
    if st.button("🌾 Get Recommendation", use_container_width=True):

        st.success("Processing your request...")

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

        # -----------------------------------------
        # Display Result
        # -----------------------------------------

        st.success(
            f"🌾 Recommended Crop: **{crop_name[0]}**"
        )

    # =========================================================
    # Yield Prediction
    # ==========================================================

    st.divider()
    st.header("📈 Yield Prediction")

    with st.expander("Open Yield Prediction", expanded=True):

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

            yield_prediction = resources[
                "yield_model"
            ].predict(yield_input)

            st.success(
                f"📈 Predicted Yield: "
                f"**{yield_prediction[0]:.2f} tons/hectare**"
            )
    # ==========================================================
    # Fertilizer Recommendation
    # ==========================================================

    st.divider()
    st.header("🧪 Fertilizer Recommendation")

    with st.expander(
        "Open Fertilizer Recommendation",
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

            st.success(
                f"🧪 Recommended Fertilizer: "
                f"**{fertilizer_name[0]}**"
            )

elif page == "Analysis":

    st.header("📊 Data Analysis & Model Insights")

    st.info(
        "Overview of the machine learning models integrated "
        "into the Smart Crop Recommendation System."
    )

    # ==========================================================
    # System Statistics
    # ==========================================================

    st.subheader("📌 System Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "ML Models",
            "3",
            "Integrated"
        )

    with col2:
        st.metric(
            "Prediction Modules",
            "3",
            "Active"
        )

    with col3:
        st.metric(
            "System Status",
            "Ready",
            "Operational"
        )

    st.divider()

    # ==========================================================
    # Model Overview
    # ==========================================================

    st.subheader("🤖 Model Overview")

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
