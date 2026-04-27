import pandas as pd
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET = "traffic_volume"

NUMERICAL_COLS = ["temp", "rain_1h", "snow_1h", "clouds_all"]

CATEGORICAL_COLS = ["weather_main", "holiday"]

DROP_COLS = ["weather_description", "date_time"]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(verbose: bool = True) -> pd.DataFrame:
    """Load dataset from UCI repository."""
    
    metro = fetch_ucirepo(id=492)
    df = pd.concat([metro.data.features, metro.data.targets], axis=1)

    if verbose:
        print(f"Dataset shape: {df.shape}")

    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from datetime."""
    
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])

    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # HD feature
    df["rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

    return df.drop(columns=["date_time"])


# =============================================================================
# ENCODING
# =============================================================================

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features."""

    df = df.copy()

    if "weather_description" in df.columns:
        df = df.drop(columns=["weather_description"])

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    return df


# =============================================================================
# SPLITTING (TIME-AWARE)
# =============================================================================

def split_time_series(df: pd.DataFrame):
    """Chronological split (no shuffling)."""

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test


# =============================================================================
# FEATURE / TARGET SPLIT
# =============================================================================

def split_X_y(df: pd.DataFrame):
    """Separate features and target."""

    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].copy()

    return X, y


# =============================================================================
# SCALING
# =============================================================================

def scale_data(X_train, X_val, X_test):
    """Scale numerical features using StandardScaler."""

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
    X_val[NUMERICAL_COLS] = scaler.transform(X_val[NUMERICAL_COLS])
    X_test[NUMERICAL_COLS] = scaler.transform(X_test[NUMERICAL_COLS])

    return X_train, X_val, X_test, scaler


# =============================================================================
# MASTER PIPELINE
# =============================================================================

def preprocess(verbose: bool = True):
    """Full preprocessing pipeline."""

    # Load
    df = load_data(verbose)

    # Feature engineering
    df = add_time_features(df)

    # Encoding
    df = encode_features(df)

    # Split (time-aware)
    train, val, test = split_time_series(df)

    # Split X/y
    X_train, y_train = split_X_y(train)
    X_val, y_val = split_X_y(val)
    X_test, y_test = split_X_y(test)

    # Scale
    X_train, X_val, X_test, scaler = scale_data(X_train, X_val, X_test)

    if verbose:
        print("\n=== FINAL DATA SHAPES ===")
        print(f"Train: {X_train.shape}")
        print(f"Val  : {X_val.shape}")
        print(f"Test : {X_test.shape}")

    return X_train, y_train, X_test, y_test