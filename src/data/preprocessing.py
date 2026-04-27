# Import required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


# Configuration
TARGET = "traffic_volume"
NUMERICAL_COLS = ["temp", "rain_1h", "snow_1h", "clouds_all"]
CATEGORICAL_COLS = ["weather_main", "holiday"]
DROP_COLS = ["weather_description", "date_time"]


def load_data(verbose: bool = True) -> pd.DataFrame:
    # Load dataset from UCI repository
    metro = fetch_ucirepo(id=492)
    df = pd.concat([metro.data.features, metro.data.targets], axis=1)

    if verbose:
        print(f"Dataset shape: {df.shape}")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Create time-based features from datetime column
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])

    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Create rush hour feature
    df["rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

    return df.drop(columns=["date_time"])


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    # Convert categorical variables into numeric using one-hot encoding
    df = df.copy()

    if "weather_description" in df.columns:
        df = df.drop(columns=["weather_description"])

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    return df


def split_time_series(df: pd.DataFrame):
    # Split dataset chronologically into train, validation, and test sets
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test


def split_X_y(df: pd.DataFrame):
    # Separate features and target variable
    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].copy()

    return X, y


def scale_data(X_train, X_val, X_test):
    # Scale numerical columns using StandardScaler
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
    X_val[NUMERICAL_COLS] = scaler.transform(X_val[NUMERICAL_COLS])
    X_test[NUMERICAL_COLS] = scaler.transform(X_test[NUMERICAL_COLS])

    return X_train, X_val, X_test, scaler


def preprocess(verbose: bool = True):
    # Run full preprocessing pipeline

    # Load data
    df = load_data(verbose)

    # Add time-based features
    df = add_time_features(df)

    # Encode categorical variables
    df = encode_features(df)

    # Split data chronologically
    train, val, test = split_time_series(df)

    # Separate features and target
    X_train, y_train = split_X_y(train)
    X_val, y_val = split_X_y(val)
    X_test, y_test = split_X_y(test)

    # Scale numerical features
    X_train, X_val, X_test, scaler = scale_data(X_train, X_val, X_test)

    if verbose:
        print("\n=== FINAL DATA SHAPES ===")
        print(f"Train: {X_train.shape}")
        print(f"Val  : {X_val.shape}")
        print(f"Test : {X_test.shape}")

    # Return only train and test sets for modeling
    return X_train, y_train, X_test, y_test