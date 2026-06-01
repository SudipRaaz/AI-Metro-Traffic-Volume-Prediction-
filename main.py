import sys
import os
import pandas as pd
import numpy as np

# Import your existing custom modules
from src.data.preprocessing import preprocess, NUMERICAL_COLS
from src.models.run_model import train_models

# =============================================================================
# 1. ETHICAL GUARDRAIL ENGINE (Built-in Ethical Complexity)
# =============================================================================
def check_ethical_guardrails(datetime_str):
    """
    Acts as the automated ethical layer required by the assignment guidelines.
    Cross-references inputs with known performance drops established during your audit.
    """
    try:
        dt = pd.to_datetime(datetime_str)
    except Exception:
        return ["INVALID TIMESTAMP: Unable to parse temporal data for audit analysis."]
        
    hour = dt.hour
    is_weekend = dt.dayofweek >= 5
    warnings = []
    
    # Check Weekend Subgroup Performance Drop
    if is_weekend:
        warnings.append("WEEKEND DATA FLAGGED: Model accuracy decreases on weekends (R² drops compared to weekdays).")
        
    # Check Critical Daytime Subgroup Bias
    if 10 <= hour <= 15:
        warnings.append("CRITICAL DAYTIME BIAS: Extremely low model explanation variance between 10am and 3pm (R²: 0.1067).")
        
    # Check Rush Hour Variances
    elif hour in [7, 8, 9]:
        warnings.append("MORNING RUSH DRIFT: High absolute errors detected during peak morning shifts (MAE: ~587.8 vehicles/hour).")
    elif hour in [16, 17, 18]:
        warnings.append("EVENING RUSH DRIFT: Model underpredicts during peak outward transit adjustments.")
        
    # Check Overnight Subgroup Bias
    elif hour >= 23 or hour <= 5:
        warnings.append("OVERNIGHT DRIFT: Baseline sampling limitations decrease operational model validation (R²: 0.5562).")
        
    return warnings

# =============================================================================
# 2. CUSTOM SCENARIO DATA TRANSFORMER
# =============================================================================
def prepare_input_vector(datetime_str, temp_k, rain_mm, snow_mm, clouds_pct, 
                         weather_main, holiday_name, scaler, feature_names):
    """
    Transforms human input parameters into a normalized vector matching 
    the exact columns and one-hot encoding structure expected by your models.
    """
    dt = pd.to_datetime(datetime_str)
    
    # Recreate the time-based features from your add_time_features pipeline
    row = {
        "temp": temp_k,
        "rain_1h": rain_mm,
        "snow_1h": snow_mm,
        "clouds_all": clouds_pct,
        "hour": dt.hour,
        "day_of_week": dt.dayofweek,
        "month": dt.month,
        "is_weekend": int(dt.dayofweek >= 5),
        "rush_hour": int(dt.hour in [7, 8, 9, 16, 17, 18])
    }
    
    df_custom = pd.DataFrame([row])
    
    # Initialize all possible one-hot categorical columns to 0
    for col in feature_names:
        if col not in df_custom.columns:
            df_custom[col] = 0
            
    # Activate the selected weather categorical column
    weather_col = f"weather_main_{weather_main}"
    if weather_col in df_custom.columns:
        df_custom[weather_col] = 1
        
    # Activate the selected holiday categorical column
    if holiday_name != "None":
        holiday_col = f"holiday_{holiday_name}"
        if holiday_col in df_custom.columns:
            df_custom[holiday_col] = 1
            
    # Force the dataframe to have the identical column sequence as the training set
    df_custom = df_custom[feature_names]
    
    # Scale numerical variables using the fitted scaler from preprocess()
    df_custom[NUMERICAL_COLS] = scaler.transform(df_custom[NUMERICAL_COLS])
    
    return df_custom

# =============================================================================
# 3. INTERACTIVE SYSTEM DRIVER
# =============================================================================
def main():
    print("=" * 65)
    print(" ETHICAL AI PROTOTYPE: TRAFFIC VOLUME PREDICTION ADVISORY TOOL ")
    print("=" * 65)
    print("Initializing system pipelines and training models...")
    
    # Use your existing preprocessing module (extracting datasets and the scaler object)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess(verbose=False)
    feature_names = list(X_train.columns)
    
    # Train all 4 estimators using your custom train_models function
    models = train_models(X_train, y_train)
    
    # Preset scenarios to allow the examiner to test easily
    presets = {
        "1": {
            "label": "Monday Morning Rush Hour (High Volatility Window)",
            "datetime": "2026-06-01 08:00:00", "temp": 285.0, "rain": 0.0, "snow": 0.0, "clouds": 20, "weather": "Clear", "holiday": "None"
        },
        "2": {
            "label": "Wednesday Midday (Critical Daytime Subgroup Bias Window)",
            "datetime": "2026-06-03 13:00:00", "temp": 291.5, "rain": 0.8, "snow": 0.0, "clouds": 80, "weather": "Rain", "holiday": "None"
        },
        "3": {
            "label": "Optimal Prediction Window (Clear Autumn Night)",
            "datetime": "2026-10-14 20:00:00", "temp": 277.0, "rain": 0.0, "snow": 0.0, "clouds": 5, "weather": "Clear", "holiday": "None"
        }
    }

    while True:
        print("\n" + "-" * 55)
        print("PROTOTYPE CONTROL PANEL")
        print("-" * 55)
        for key, details in presets.items():
            print(f"[{key}] Evaluate Preset: {details['label']}")
        print("[4] Enter Custom Real-Time Parameters")
        print("[5] Terminate Application Environment")
        
        choice = input("\nSelect an operational path (1-5): ").strip()
        
        if choice in ["1", "2", "3"]:
            scen = presets[choice]
            execute_prediction_cycle(scen["datetime"], scen["temp"], scen["rain"], 
                                     scen["snow"], scen["clouds"], scen["weather"], 
                                     scen["holiday"], models, scaler, feature_names)
            
        elif choice == "4":
            print("\n--- Custom Parameter Entry ---")
            dt_in = input("Timestamp (YYYY-MM-DD HH:MM:SS) [Default: 2026-06-01 12:00:00]: ") or "2026-06-01 12:00:00"
            temp_in = float(input("Temperature in Kelvin (e.g., 285.0): ") or 285.0)
            rain_in = float(input("1-Hour Rain in mm (e.g., 0.0): ") or 0.0)
            snow_in = float(input("1-Hour Snow in mm (e.g., 0.0): ") or 0.0)
            clouds_in = int(input("Cloud Cover % (0-100): ") or 20)
            weather_in = input("Weather Main Category (Clear, Clouds, Rain, Snow, Mist): ") or "Clear"
            holiday_in = input("Holiday string or 'None': ") or "None"
            
            execute_prediction_cycle(dt_in, temp_in, rain_in, snow_in, 
                                     clouds_in, weather_in, holiday_in, 
                                     models, scaler, feature_names)
            
        elif choice == "5":
            print("\nShutting down software prototype safely.")
            break
        else:
            print("Selection out of bounds. Please input an integer from 1 to 5.")

# =============================================================================
# 4. PREDICTION RUNTIME ENGINE
# =============================================================================
def execute_prediction_cycle(dt_str, temp, rain, snow, clouds, weather, holiday, models, scaler, features):
    print("\n" + "="*60)
    print(" INCOMING REQUEST DATA STREAM")
    print("="*60)
    print(f" Timestamp : {dt_str}")
    print(f" Conditions: {weather} | Temp: {temp}K | Rain: {rain}mm | Clouds: {clouds}%")
    print(f" Holiday   : {holiday}")
    
    # 1. Run the real-time Ethical Analysis
    bias_alerts = check_ethical_guardrails(dt_str)
    
    if bias_alerts:
        print("\n🛑 COGNITIVE ETHICAL GUARDRAILS TRIGGERED:")
        for alert in bias_alerts:
            print(f"  ⚠ {alert}")
        print("\n[OPERATIONAL PROTOCOL]: This scenario operates within an automated-embargo zone.")
        print("Human oversight mandatory. Do NOT pipe predictions directly to physical infrastructure.")
    else:
        print("\n✅ ETHICAL STATUS CONFORMANCE: Active input parameters fall within nominal bounds.")
        print("Model predictions verified safe for baseline automated advisory deployment.")
        
    print("\nPROTOTYPE INFERENCE STREAM (Vehicles/Hour):")
    
    # 2. Standardize data shape and scales 
    input_vector = prepare_input_vector(dt_str, temp, rain, snow, clouds, weather, holiday, scaler, features)
    
    # 3. Predict across your entire trained ensemble
    for name, model in models.items():
        prediction = model.predict(input_vector)[0]
        # Restrict negative traffic variables mathematically
        final_count = int(max(0, round(prediction)))
        print(f"  ↳ {name.ljust(20)} : {final_count:,} vehicles/hour")
    print("="*60)


if __name__ == "__main__":
    main()
# ======================================================================

# # Import required libraries
# from ucimlrepo import fetch_ucirepo
# import pandas as pd

# # Import custom modules
# from src.data.eda import run_eda, plot_actual_vs_predicted
# from src.data.preprocessing import preprocess
# from src.models.run_model import train_models
# from src.evaluation.evaluate import evaluate_models, error_analysis


# def main():
#     # Load dataset from UCI repository
#     print("Loading dataset...")
#     dataset = fetch_ucirepo(id=492)

#     # Extract features and target
#     X = dataset.data.features
#     y = dataset.data.targets

#     # Convert target to Series if needed
#     if isinstance(y, pd.DataFrame):
#         y = y.iloc[:, 0]

#     # Perform exploratory data analysis
#     run_eda(X, y)

#     # Preprocess data and split into train and test sets
#     X_train, y_train, X_test, y_test = preprocess()

#     # Train models using training data
#     models = train_models(X_train, y_train)

#     # Evaluate models on test data
#     results = evaluate_models(models, X_test, y_test)

#     # Select model with highest R2 score
#     best_model_name = max(results.items(), key=lambda x: x[1]["R2"])[0]

#     # Print best model name
#     print(f"\nBest Model: {best_model_name}")


# if __name__ == "__main__":
#     main()