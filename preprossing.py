# =============================================================================
# PRO502 - Assignment 2
# File: preprocessing.py
# Step 1: Data Loading, Cleaning, Feature Engineering & Splitting
# Dataset: Metro Interstate Traffic Volume (UCI ML Repository, ID=492)
# =============================================================================

import pandas as pd
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


# =============================================================================
# STEP 1A: LOAD DATA
# =============================================================================

def load_data():
    """
    Fetch the Metro Interstate Traffic Volume dataset directly from UCI.
    Returns a single combined DataFrame with features + target.
    """
    print("Loading dataset from UCI ML Repository...")

    metro = fetch_ucirepo(id=492)

    X = metro.data.features   # All input features
    y = metro.data.targets    # Target: traffic_volume

    # Combine into one DataFrame for easier preprocessing
    df = pd.concat([X, y], axis=1)

    print(f"Dataset loaded. Shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")

    return df


# =============================================================================
# STEP 1B: PARSE DATE_TIME → EXTRACT TIME FEATURES
# =============================================================================

def parse_datetime_features(df):
    """
    The raw 'date_time' column (e.g. "2012-10-02 09:00:00") is a string.
    Models cannot use strings directly, so we extract meaningful time signals:

    - hour       : captures rush hour patterns (8am, 5pm = high traffic)
    - day_of_week: weekdays vs weekends behave differently
    - month      : seasonal effects (winter = less traffic)
    - is_weekend : simple binary flag (1 = Sat/Sun, 0 = weekday)

    This process is called Feature Engineering — creating new useful
    features from raw ones.
    """
    print("\n--- Parsing datetime features ---")

    # Convert string to Python datetime so we can extract parts
    df['date_time'] = pd.to_datetime(df['date_time'])

    df['hour']        = df['date_time'].dt.hour           # 0 to 23
    df['day_of_week'] = df['date_time'].dt.dayofweek      # 0=Monday, 6=Sunday
    df['month']       = df['date_time'].dt.month          # 1 to 12
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)  # 1 if Sat or Sun

    # Drop the original column — we no longer need it
    df = df.drop(columns=['date_time'])

    print("Extracted: hour, day_of_week, month, is_weekend")
    print(f"Shape after datetime parsing: {df.shape}")

    return df


# =============================================================================
# STEP 1C: ENCODE CATEGORICAL VARIABLES
# =============================================================================

def encode_categoricals(df):
    """
    Machine learning models only understand numbers.
    Text columns like 'weather_main' = "Rain" or 'holiday' = "Columbus Day"
    must be converted into numerical form.

    We use ONE-HOT ENCODING:
    Instead of Rain=1, Clear=2 (which implies an order that doesn't exist),
    we create a separate binary column for each category:
        weather_main_Rain  = 1 or 0
        weather_main_Clear = 1 or 0
        etc.

    drop_first=True drops one column per category to avoid the
    "dummy variable trap" (perfect multicollinearity).

    weather_description is dropped — it has too many unique values and
    is already captured by weather_main.
    """
    print("\n--- Encoding categorical variables ---")

    # Drop weather_description — too granular, redundant
    df = df.drop(columns=['weather_description'])
    print("Dropped: weather_description")

    # One-hot encode weather_main and holiday
    df = pd.get_dummies(df, columns=['weather_main', 'holiday'], drop_first=True)
    print(f"One-hot encoded: weather_main, holiday")
    print(f"Shape after encoding: {df.shape}")

    return df


# =============================================================================
# STEP 1D: CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# =============================================================================

def split_data(df):
    """
    For time-series data, we NEVER shuffle before splitting.
    Why? Because the model should learn from the past and be tested on the future.
    Shuffling would let the model train on future data — that's cheating.

    Split ratio:
        70% → Training   (model learns from this)
        15% → Validation (tune hyperparameters, check for overfitting)
        15% → Test       (final unbiased evaluation — touch this last!)
    """
    print("\n--- Splitting data chronologically ---")

    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    print(f"Train size : {len(train)} rows ({70}%)")
    print(f"Val size   : {len(val)} rows ({15}%)")
    print(f"Test size  : {len(test)} rows ({15}%)")

    return train, val, test


# =============================================================================
# STEP 1E: SCALE NUMERICAL FEATURES
# =============================================================================

def scale_features(X_train, X_val, X_test):
    """
    Different features have very different ranges:
        temp       → 200 to 310 (Kelvin)
        rain_1h    → 0.0 to 2.0
        clouds_all → 0 to 100

    Without scaling, models may wrongly treat 'temp' as more important
    just because its numbers are larger.

    StandardScaler transforms each feature to have:
        mean = 0
        standard deviation = 1

    CRITICAL RULE:
    We fit the scaler ONLY on training data, then apply it to val and test.
    If we fit on val/test too, the scaler would use future information
    to scale past data — this is called DATA LEAKAGE and must be avoided.
    """
    print("\n--- Scaling numerical features ---")

    numerical_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']

    scaler = StandardScaler()

    # Fit on training data ONLY, then transform all three sets
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_val[numerical_cols]   = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols]  = scaler.transform(X_test[numerical_cols])

    print(f"Scaled columns: {numerical_cols}")
    print("Scaler fitted on training data only (no data leakage).")

    return X_train, X_val, X_test, scaler


# =============================================================================
# MASTER FUNCTION: RUN ALL PREPROCESSING STEPS
# =============================================================================

def preprocess():
    """
    Runs all preprocessing steps in order and returns clean,
    split, and scaled data ready for model training.

    Returns:
        X_train, y_train  → model learns from these
        X_val,   y_val    → used to tune hyperparameters
        X_test,  y_test   → used only for final evaluation
        scaler            → saved so we can inverse-transform predictions later
    """

    # 1. Load raw data
    df = load_data()

    # 2. Extract time features from date_time
    df = parse_datetime_features(df)

    # 3. Encode text/categorical columns
    df = encode_categoricals(df)

    # 4. Split into train / val / test (chronologically)
    train, val, test = split_data(df)

    # 5. Separate features (X) from target (y) in each split
    target = 'traffic_volume'

    X_train = train.drop(columns=[target]).copy()
    y_train = train[target].copy()

    X_val   = val.drop(columns=[target]).copy()
    y_val   = val[target].copy()

    X_test  = test.drop(columns=[target]).copy()
    y_test  = test[target].copy()

    # 6. Scale numerical features (fit only on train)
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    # Final summary
    print("\n========== PREPROCESSING COMPLETE ==========")
    print(f"X_train shape : {X_train.shape}")
    print(f"X_val shape   : {X_val.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"Features used : {X_train.columns.tolist()}")
    print("=============================================\n")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


