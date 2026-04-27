# =============================================================================
# PRO502 - Assignment 2
# File: model_linear_regression.py
# Step 2: Model 1 - Linear Regression (Baseline Model)
# =============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Import our preprocessing pipeline from Step 1
from preprocessing import preprocess


# =============================================================================
# TRAIN LINEAR REGRESSION
# =============================================================================

def train_linear_regression(X_train, y_train):
    """
    Linear Regression finds the best-fit line through training data.
    It learns one weight (coefficient) per feature.

    Formula:
        traffic_volume = w1*hour + w2*temp + w3*is_weekend + ... + bias

    No hyperparameters to tune here — this is why it's a good baseline.
    Simple, fast, interpretable.
    """
    print("\n--- Training Linear Regression ---")

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Training complete.")
    return model


# =============================================================================
# EVALUATE MODEL
# =============================================================================

def evaluate_model(model, X, y, split_name="Validation"):
    """
    Evaluate model performance using 3 key regression metrics:

    MAE  (Mean Absolute Error):
        Average absolute difference between predicted and actual values.
        Easy to interpret — same units as traffic_volume (vehicles/hour).
        e.g. MAE=500 means predictions are off by ~500 vehicles on average.

    RMSE (Root Mean Squared Error):
        Penalises large errors more than MAE.
        More sensitive to outliers (e.g. extreme traffic spikes).

    R²   (R-squared / Coefficient of Determination):
        How much variance in traffic_volume the model explains.
        R²=1.0 → perfect predictions
        R²=0.0 → model is no better than predicting the mean
        R²<0.0 → model is worse than just predicting the mean (bad!)
    """
    predictions = model.predict(X)

    mae  = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2   = r2_score(y, predictions)

    print(f"\n--- {split_name} Results ---")
    print(f"MAE  : {mae:.2f}  (avg error in vehicles/hour)")
    print(f"RMSE : {rmse:.2f} (penalises large errors more)")
    print(f"R²   : {r2:.4f}  (1.0 = perfect, 0.0 = no better than mean)")

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "predictions": predictions}


# =============================================================================
# UNDERSTAND THE MODEL — FEATURE WEIGHTS
# =============================================================================

def show_feature_importance(model, feature_names):
    """
    One advantage of Linear Regression is interpretability.
    We can see exactly how much each feature influences the prediction.

    A large positive coefficient → feature increases traffic
    A large negative coefficient → feature decreases traffic
    A coefficient near 0 → feature has little effect
    """
    print("\n--- Feature Coefficients (Weights) ---")
    print("These show how much each feature pushes traffic up or down:\n")

    # Pair each feature name with its learned weight
    coefficients = list(zip(feature_names, model.coef_))

    # Sort by absolute value — biggest impact first
    coefficients.sort(key=lambda x: abs(x[1]), reverse=True)

    for feature, coef in coefficients[:10]:  # Show top 10
        direction = "↑ increases" if coef > 0 else "↓ decreases"
        print(f"  {feature:<35} {coef:+.2f}  ({direction} traffic)")


# =============================================================================
# PLOT PREDICTIONS VS ACTUAL
# =============================================================================

def plot_predictions(y_actual, y_predicted, title="Linear Regression: Predicted vs Actual"):
    """
    Visual check — if the model is good, points should cluster
    along the diagonal line (predicted = actual).
    Points far from the line = large errors.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_predicted, alpha=0.3, color='steelblue', s=5)
    plt.plot([y_actual.min(), y_actual.max()],
             [y_actual.min(), y_actual.max()],
             'r--', linewidth=2, label='Perfect prediction line')

    plt.xlabel("Actual Traffic Volume")
    plt.ylabel("Predicted Traffic Volume")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("linear_regression_predictions.png", dpi=150)
    plt.show()
    print("Plot saved as: linear_regression_predictions.png")


# =============================================================================
# RUN THIS FILE DIRECTLY
# =============================================================================

if __name__ == "__main__":

    # Step 1: Load and preprocess data
    print("=" * 50)
    print("Loading preprocessed data...")
    print("=" * 50)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess()

    # Step 2: Train model on training data
    model = train_linear_regression(X_train, y_train)

    # Step 3: Evaluate on validation set (use this to compare models)
    val_results = evaluate_model(model, X_val, y_val, split_name="Validation")

    # Step 4: Understand what the model learned
    show_feature_importance(model, X_train.columns.tolist())

    # Step 5: Visual check
    plot_predictions(y_val, val_results["predictions"])

    # Step 6: Quick note on what these results mean
    print("\n--- What These Results Tell Us ---")
    print(f"The model's average prediction error is {val_results['MAE']:.0f} vehicles/hour.")
    print(f"It explains {val_results['R2']*100:.1f}% of variance in traffic volume.")
    print("We will compare this R² against Decision Tree, Random Forest, SVR, and LSTM.")
    print("A good model should have R² > 0.80 and low MAE.")
