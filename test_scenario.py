import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Import your actual preprocessing pipeline
from src.data.preprocessing import preprocess, NUMERICAL_COLS, CATEGORICAL_COLS

# =============================================================================
# 1. TRAIN ALL MODELS (Based on your results)
# =============================================================================
def train_all_models():
    print("Loading dataset and preprocessing...")
    # Fetch data and retrieve the fitted scaler from your pipeline
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess(verbose=False)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=15, # Matches your current training setup
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name} on {len(X_train)} samples...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    # We must return the scaler and feature names to replicate processing on custom inputs
    feature_names = list(X_train.columns)
    return trained_models, scaler, feature_names

# =============================================================================
# 2. GENERATE A SCENARIO PREDICTION ROW
# =============================================================================
def prepare_custom_scenario(datetime_str, temp_k, rain_mm, snow_mm, clouds_pct, 
                            weather_main, holiday_name, scaler, feature_names):
    """
    Transforms human input into a single-row DataFrame matching the exact mathematical
    shapes, scaling, and one-hot encodings expected by the trained estimators.
    """
    dt = pd.to_datetime(datetime_str)
    hour = dt.hour
    day_of_week = dt.dayofweek          # 0=Monday, 6=Sunday
    month = dt.month
    is_weekend = int(day_of_week >= 5)
    rush_hour = int(hour in [7, 8, 9, 16, 17, 18]) # Base rush feature alignment
    
    # Construct base dictionary
    row = {
        "temp": temp_k,
        "rain_1h": rain_mm,
        "snow_1h": snow_mm,
        "clouds_all": clouds_pct,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "rush_hour": rush_hour,
    }
    
    df = pd.DataFrame([row])
    
    # Initialize missing one-hot variables to 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
            
    # Active weather categorization matching one-hot rules
    weather_col = f"weather_main_{weather_main}"
    if weather_col in df.columns:
        df[weather_col] = 1
        
    # Active holiday categorization matching one-hot rules
    if holiday_name != "None":
        holiday_col = f"holiday_{holiday_name}"
        if holiday_col in df.columns:
            df[holiday_col] = 1
            
    # Force exact column sequencing
    df = df[feature_names]
    
    # Apply identical feature distribution normalization (StandardScaler)
    df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])
    
    return df

# =============================================================================
# 3. RUN INTERACTIVE TEST SCENARIOS
# =============================================================================
if __name__ == "__main__":
    # Train and lock models
    trained_models, scaler, feature_names = train_all_models()
    
    # Define custom testing inputs
    scenarios = [
        {
            "label": "Monday Morning Rush Hour (Clear Weather)",
            "datetime_str": "2026-06-01 08:00:00", # Monday at 8 AM
            "temp_k": 285.0,
            "rain_mm": 60.0,
            "snow_mm": 100.0,
            "clouds_pct": 10,
            "weather_main": "Clear",
            "holiday_name": "None"
        },
        {
            "label": "Sunday Night (Low Traffic / Clear)",
            "datetime_str": "2026-05-31 23:00:00", # Sunday at 11 PM
            "temp_k": 280.0,
            "rain_mm": 0.0,
            "snow_mm": 0.0,
            "clouds_pct": 5,
            "weather_main": "Clear",
            "holiday_name": "None"
        }
    ]
    
    print("\n" + "="*60)
    print("          SCENARIO EVALUATION MATRIX")
    print("="*60)
    
    for case in scenarios:
        print(f"\n▶ Scenario: {case['label']}")
        print(f"  Time & Date: {case['datetime_str']} | Weather: {case['weather_main']}")
        
        # Format the entry row
        input_row = prepare_custom_scenario(
            case["datetime_str"], case["temp_k"], case["rain_mm"], 
            case["snow_mm"], case["clouds_pct"], case["weather_main"], 
            case["holiday_name"], scaler, feature_names
        )
        
        # Query every model using the formatted row
        for name, model in trained_models.items():
            pred = model.predict(input_row)[0]
            # Ensure predictions do not output lower than zero vehicles
            final_volume = int(max(0, round(pred))) 
            print(f"  ↳ {name.ljust(20)} : {final_volume:,} vehicles/hour")
            
    print("\n" + "="*60)