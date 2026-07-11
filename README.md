# 🌾 Smart Crop Recommendation System

A machine learning-based agricultural decision support system developed to provide intelligent crop, yield, and fertilizer predictions using soil, climate, and environmental parameters.

The system integrates multiple machine learning models into an interactive Streamlit web application.

## 🚀 Features

- 🌾 Crop Recommendation
- 📈 Crop Yield Prediction
- 🧪 Fertilizer Recommendation
- 📊 Machine Learning Analysis and Insights
- ⚙️ Feature Engineering
- 🌐 Interactive Streamlit User Interface

## 🤖 Machine Learning Modules

### 1. Crop Recommendation

Recommends a suitable crop based on:

- Nitrogen (N)
- Phosphorous (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

Feature engineering includes:

- Total NPK
- Soil Fertility Index
- NPK Balance Ratio
- Temperature Category
- Rainfall Category
- Soil pH Category

### 2. Yield Prediction

Predicts crop yield in tons per hectare using agricultural and environmental conditions.

Features include:

- Region
- Soil Type
- Crop
- Rainfall
- Temperature
- Fertilizer Usage
- Irrigation Usage
- Weather Condition
- Days to Harvest
- Climate Zone
- Growing Season

### 3. Fertilizer Recommendation

Recommends a suitable fertilizer based on soil nutrients and environmental conditions.

Feature engineering includes:

- Total Nutrients
- Soil Health Index
- Climate Index
- Soil pH Category
- Moisture Category
- Previous Yield Category
- Nutrient Richness Category

## 🔄 Machine Learning Workflow

Data Collection

↓

Data Cleaning and Preprocessing

↓

Exploratory Data Analysis

↓

Feature Engineering

↓

Feature Encoding and Scaling

↓

Model Training and Evaluation

↓

Model Selection

↓

Streamlit Deployment

## 📁 Project Structure

```text
Smart_Crop_Recommendation_System/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── crop_recommendation_model.pkl
│   ├── crop_scaler.pkl
│   ├── crop_target_encoder.pkl
│   ├── fertilizer_recommendation_model.pkl
│   ├── fertilizer_feature_encoders.pkl
│   ├── fertilizer_target_encoder.pkl
│   ├── fertilizer_bins.pkl
│   ├── yield_prediction_model.pkl
│   ├── yield_feature_encoders.pkl
│   └── yield_scaler.pkl
│
├── datasets/
├── notebooks/
├── assets/
└── screenshots/
```

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- VS Code
- Jupyter Notebook / Google Colab

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <repository-url>
```

### 2. Open the project directory

```bash
cd Smart_Crop_Recommendation_System
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 System Modules

| Module | Task Type | Status |
|---|---|---|
| Crop Recommendation | Classification | Active |
| Yield Prediction | Regression | Active |
| Fertilizer Recommendation | Classification | Active |

## 🎯 Project Objective

The objective of this project is to apply machine learning techniques to agricultural data and develop an intelligent decision support system for crop selection, yield estimation, and fertilizer recommendation.

## 👨‍💻 Developer

Developed as a Data Science and Machine Learning project.

## 📄 License

This project is developed for educational and academic purposes.
