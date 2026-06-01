# =============================================================================
# PRO502 - Assignment 2 & 3
# File: predict.py
# Deployable prediction file — matches actual preprocessing.py exactly
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.data.preprocessing import preprocess, NUMERICAL_COLS, CATEGORICAL_COLS


# =============================================================================
# BUILD AND TRAIN THE MODEL
# =============================================================================

def build_model():
    """
    Trains the final Random Forest on the full training set.
    Call this ONCE when your application starts.

    Returns:
        model         → trained Random Forest
        scaler        → fitted StandardScaler from preprocessing
        feature_names → ordered list of feature columns
    """
    print("Building model — loading and preprocessing data...")

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess(verbose=False)

    model = RandomForestRegressor(
        n_estimators      = 100,
        max_depth         = 12,
        min_samples_split = 10,
        max_features      = "sqrt",
        random_state      = 42,
        n_jobs            = -1
    )
    model.fit(X_train, y_train)

    feature_names = list(X_train.columns)

    print(f"Model ready. Trained on {len(X_train)} samples.")
    print(f"Features ({len(feature_names)}): {feature_names}\n")

    return model, scaler, feature_names


# =============================================================================
# PREPARE A SINGLE INPUT ROW
# =============================================================================

def prepare_input(datetime_str, temp, rain, snow, clouds,
                  weather, holiday, scaler, feature_names):
    """
    Converts raw input into the exact feature format the model expects.
    Mirrors every step in your preprocessing.py:

        add_time_features() → hour, day_of_week, month, is_weekend, rush_hour
        encode_features()   → one-hot weather_main and holiday
        scale_data()        → StandardScaler on numerical columns

    Parameters:
        datetime_str → "2024-03-18 08:00:00"
        temp         → Kelvin (e.g. 285.0)
        rain         → mm (e.g. 0.0)
        snow         → mm (e.g. 0.0)
        clouds       → 0-100 (e.g. 20)
        weather      → Clear / Clouds / Rain / Snow / Mist / Fog /
                       Drizzle / Haze / Thunderstorm / Smoke / Squall
        holiday      → "None" or US holiday name
        scaler       → fitted StandardScaler from build_model()
        feature_names→ column list from build_model()
    """

    # Mirror add_time_features()
    dt          = pd.to_datetime(datetime_str)
    hour        = dt.hour
    day_of_week = dt.dayofweek          # 0=Monday, 6=Sunday
    month       = dt.month
    is_weekend  = int(day_of_week >= 5)
    rush_hour   = int(hour in [7, 8, 9, 16, 17, 18])  # matches your code exactly

    # Base numerical row
    row = {
        "temp"       : temp,
        "rain_1h"    : rain,
        "snow_1h"    : snow,
        "clouds_all" : clouds,
        "hour"       : hour,
        "day_of_week": day_of_week,
        "month"      : month,
        "is_weekend" : is_weekend,
        "rush_hour"  : rush_hour,
    }

    df = pd.DataFrame([row])

    # Mirror encode_features() — add all one-hot columns, default 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Set correct weather one-hot column to 1
    weather_col = f"weather_main_{weather}"
    if weather_col in df.columns:
        df[weather_col] = 1

    # Set correct holiday one-hot column to 1
    if holiday != "None":
        holiday_col = f"holiday_{holiday}"
        if holiday_col in df.columns:
            df[holiday_col] = 1

    # Reorder to match training feature order exactly
    df = df[feature_names]

    # Mirror scale_data()
    df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])

    return df


# =============================================================================
# PREDICT
# =============================================================================

def predict(model, scaler, feature_names,
            datetime_str, temp, rain, snow, clouds, weather, holiday):
    """
    Main prediction function — call this from any application.

    Returns:
        int → predicted traffic volume in vehicles per hour
    """
    input_row  = prepare_input(datetime_str, temp, rain, snow, clouds,
                               weather, holiday, scaler, feature_names)
    prediction = model.predict(input_row)[0]
    return int(max(0, round(prediction)))


# =============================================================================
# TEST — run directly to verify everything works
# =============================================================================

if __name__ == "__main__":

    model, scaler, feature_names = build_model()

    test_cases = [
        {
            "label"       : "Monday Morning Rush Hour",
            "datetime_str": "2024-03-18 08:00:00",
            "temp": 285.0, "rain": 0.0, "snow": 0.0,
            "clouds": 20,  "weather": "Clear", "holiday": "None",
        },
        {
            "label"       : "Saturday Midnight",
            "datetime_str": "2024-03-16 00:00:00",
            "temp": 275.0, "rain": 0.0, "snow": 0.0,
            "clouds": 10,  "weather": "Clear", "holiday": "None",
        },
        {
            "label"       : "Friday Evening Rush — Heavy Rain",
            "datetime_str": "2024-03-22 17:00:00",
            "temp": 282.0, "rain": 2.5, "snow": 0.0,
            "clouds": 90,  "weather": "Rain", "holiday": "None",
        },
        {
            "label"       : "Christmas Day Midday — Snow",
            "datetime_str": "2024-12-25 12:00:00",
            "temp": 270.0, "rain": 0.0, "snow": 1.0,
            "clouds": 60,  "weather": "Snow", "holiday": "Christmas Day",
        },
    ]

    print("=" * 55)
    print("TRAFFIC VOLUME PREDICTIONS")
    print("=" * 55)

    for case in test_cases:
        result = predict(
            model, scaler, feature_names,
            datetime_str = case["datetime_str"],
            temp         = case["temp"],
            rain         = case["rain"],
            snow         = case["snow"],
            clouds       = case["clouds"],
            weather      = case["weather"],
            holiday      = case["holiday"]
        )
        hour_str = case["datetime_str"][11:13]
        is_rush  = hour_str in ["07", "08", "09", "16", "17", "18"]
        print(f"\n  Scenario : {case['label']}")
        print(f"  DateTime : {case['datetime_str']}")
        print(f"  Weather  : {case['weather']}, Temp={case['temp']}K, "
              f"Rain={case['rain']}mm, Snow={case['snow']}mm")
        print(f"  Rush Hour: {'Yes' if is_rush else 'No'}")
        print(f"  Predicted Traffic: {result:,} vehicles/hour")

    print("\n" + "=" * 55)
    print("Usage in your application:")
    print("  from predict import build_model, predict")
    print("  model, scaler, feature_names = build_model()  # once")
    print("  volume = predict(model, scaler, feature_names,")
    print('                   datetime_str="2024-06-10 08:00:00",')
    print("                   temp=285, rain=0, snow=0, clouds=10,")
    print('                   weather="Clear", holiday="None")')
    print("=" * 55)
