"""Train reproducible model artifacts for the Streamlit application."""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATASET_DIR = "datasets"
MODEL_DIR = "models"
RANDOM_STATE = 42


def encode_columns(frame, columns):
    """Fit one LabelEncoder per categorical column and return encoded data."""
    encoded = frame.copy()
    encoders = {}
    for column in columns:
        encoder = LabelEncoder()
        encoded[column] = encoder.fit_transform(encoded[column].astype(str))
        encoders[column] = encoder
    return encoded, encoders


def train_crop_model():
    data = pd.read_csv(os.path.join(DATASET_DIR, "Crop_recommendation_featured.csv"))
    categorical = ["temperature_category", "rainfall_category", "ph_category"]
    data, encoders = encode_columns(data, categorical)
    target_encoder = LabelEncoder()
    target = target_encoder.fit_transform(data["label"].astype(str))
    features = [
        "n", "p", "k", "temperature", "humidity", "ph", "rainfall",
        "total_npk", "soil_fertility_index", "npk_balance_ratio",
        "temperature_category", "rainfall_category", "ph_category",
    ]
    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.2, random_state=RANDOM_STATE, stratify=target
    )
    scaler = StandardScaler().fit(data.loc[train_idx, features])
    model = HistGradientBoostingClassifier(
        max_iter=250, l2_regularization=0.1, random_state=RANDOM_STATE
    ).fit(scaler.transform(data.loc[train_idx, features]), target[train_idx])
    score = accuracy_score(target[test_idx], model.predict(scaler.transform(data.loc[test_idx, features])))

    scaler = StandardScaler().fit(data[features])
    model.fit(scaler.transform(data[features]), target)
    joblib.dump(model, os.path.join(MODEL_DIR, "crop_recommendation_model.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "crop_feature_encoders.pkl"))
    joblib.dump(target_encoder, os.path.join(MODEL_DIR, "crop_target_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "crop_scaler.pkl"))
    return score


def train_yield_model():
    data = pd.read_csv(os.path.join(DATASET_DIR, "crop_yield_featured.csv"), low_memory=False)
    categorical = [
        "region", "soil_type", "crop", "fertilizer_used", "weather_condition",
        "climate_zone", "growing_season",
    ]
    data["fertilizer_used"] = data["fertilizer_used"].fillna("Unknown")
    data, encoders = encode_columns(data, categorical)
    data["irrigation_used"] = data["irrigation_used"].astype(int)
    features = [
        "region", "soil_type", "crop", "rainfall_mm", "temperature_celsius",
        "fertilizer_used", "irrigation_used", "weather_condition", "days_to_harvest",
        "water_availability_index", "climate_zone", "growing_season",
    ]
    target = data["yield_tons_per_hectare"]
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.2, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(data.loc[train_idx, features])
    model = HistGradientBoostingRegressor(
        max_iter=200, max_leaf_nodes=31, l2_regularization=1.0, random_state=RANDOM_STATE
    ).fit(scaler.transform(data.loc[train_idx, features]), target.iloc[train_idx])
    score = r2_score(target.iloc[test_idx], model.predict(scaler.transform(data.loc[test_idx, features])))

    scaler = StandardScaler().fit(data[features])
    model.fit(scaler.transform(data[features]), target)
    joblib.dump(model, os.path.join(MODEL_DIR, "yield_prediction_model.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "yield_feature_encoders.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "yield_scaler.pkl"))
    return score


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    crop_score = train_crop_model()
    yield_score = train_yield_model()
    metrics = {
        "Crop Recommendation": {"metric": "Accuracy", "value": crop_score},
        "Yield Prediction": {"metric": "R² Score", "value": yield_score},
    }
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
