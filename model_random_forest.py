# =============================================================================
# PRO502 - Assignment 2
# File: model_random_forest.py
# Step 2: Model 3 - Random Forest Regressor
# =============================================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import preprocess


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Random Forest has 3 key hyperparameters we tune:

    1. n_estimators (number of trees):
       - More trees = more stable predictions
       - But returns diminish after ~100-200 trees
       - And training gets slower with more trees

    2. max_depth (depth of each individual tree):
       - Same concept as Decision Tree
       - Random Forest is already more resistant to overfitting
         because of averaging, so we can allow deeper trees

    3. min_samples_split (minimum samples needed to split a node):
       - Higher value = simpler trees = less overfitting
       - Lower value = more complex trees

    We use a GRID SEARCH approach — test combinations and
    pick the one with the best validation R².

    Note: Full grid search across all combinations would take very
    long (~48k rows). We test a focused set of values instead.
    This is called a 'manual grid search' or 'coarse grid search'.
    """
    print("\n--- Tuning Random Forest Hyperparameters ---")
    print("Testing combinations of n_estimators and max_depth...\n")

    # Define values to test
    n_estimators_options = [50, 100, 200]
    max_depth_options    = [8, 12, 16, None]

    best_val_r2  = -999
    best_params  = {}
    all_results  = []

    print(f"{'n_est':<8} {'depth':<8} {'Train R²':<12} {'Val R²':<12} {'Note'}")
    print("-" * 58)

    for n_est in n_estimators_options:
        for depth in max_depth_options:

            model = RandomForestRegressor(
                n_estimators     = n_est,
                max_depth        = depth,
                min_samples_split= 10,    # Reasonable default
                random_state     = 42,
                n_jobs           = -1     # Use all CPU cores (faster training)
            )
            model.fit(X_train, y_train)

            train_r2 = r2_score(y_train, model.predict(X_train))
            val_r2   = r2_score(y_val,   model.predict(X_val))

            gap  = train_r2 - val_r2
            note = "⚠ Overfitting" if gap > 0.10 else "✓ Good"

            depth_label = str(depth) if depth else "None"
            print(f"{n_est:<8} {depth_label:<8} {train_r2:<12.4f} {val_r2:<12.4f} {note}")

            all_results.append({
                "n_estimators": n_est,
                "max_depth"   : depth,
                "train_r2"    : train_r2,
                "val_r2"      : val_r2
            })

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_params = {"n_estimators": n_est, "max_depth": depth}

    print(f"\n✓ Best Parameters:")
    print(f"  n_estimators = {best_params['n_estimators']}")
    print(f"  max_depth    = {best_params['max_depth']}")
    print(f"  Val R²       = {best_val_r2:.4f}")

    return best_params, all_results


# =============================================================================
# TRAIN FINAL RANDOM FOREST
# =============================================================================

def train_random_forest(X_train, y_train, best_params):
    """
    Train the final Random Forest using the best hyperparameters
    found during tuning.

    n_jobs=-1 tells sklearn to use ALL available CPU cores in parallel.
    Training 100 trees in parallel is much faster than one at a time.

    random_state=42 ensures reproducibility.
    """
    print(f"\n--- Training Final Random Forest ---")
    print(f"Parameters: {best_params}")

    model = RandomForestRegressor(
        n_estimators     = best_params["n_estimators"],
        max_depth        = best_params["max_depth"],
        min_samples_split= 10,
        random_state     = 42,
        n_jobs           = -1
    )
    model.fit(X_train, y_train)

    print("Training complete.")
    return model


# =============================================================================
# EVALUATE MODEL
# =============================================================================

def evaluate_model(model, X, y, split_name="Validation"):
    """
    Same metrics as previous models for fair comparison:
    MAE, RMSE, R²
    """
    predictions = model.predict(X)

    mae  = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2   = r2_score(y, predictions)

    print(f"\n--- {split_name} Results ---")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "predictions": predictions}


# =============================================================================
# PLOT: FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(model, feature_names):
    """
    Random Forest feature importance works differently to Decision Tree:

    Instead of measuring importance from ONE tree,
    it AVERAGES importance across ALL trees.

    This gives a much more reliable picture of which features
    truly matter — not just which ones one particular tree happened to use.

    Compare this plot to the Decision Tree's feature importance.
    They should be similar but Random Forest's will be smoother/more stable.
    """
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:12]  # Top 12

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(indices)),
            importances[indices],
            color='forestgreen', alpha=0.8)
    plt.xticks(range(len(indices)),
               [feature_names[i] for i in indices],
               rotation=45, ha='right')
    plt.ylabel("Feature Importance Score")
    plt.title("Random Forest: Top 12 Most Important Features\n(averaged across all trees)")
    plt.tight_layout()
    plt.savefig("random_forest_feature_importance.png", dpi=150)
    plt.show()
    print("Plot saved as: random_forest_feature_importance.png")


# =============================================================================
# PLOT: N_ESTIMATORS VS PERFORMANCE
# =============================================================================

def plot_n_estimators_effect(X_train, y_train, X_val, y_val, best_depth):
    """
    This plot shows how val R² improves as we add more trees,
    then levels off (diminishing returns).

    It answers the question:
    'How many trees is enough? When does adding more stop helping?'

    This is great evidence for your report to justify
    your chosen n_estimators value.
    """
    print("\n--- Plotting n_estimators effect ---")

    n_range  = [10, 25, 50, 75, 100, 150, 200]
    val_scores = []

    for n in n_range:
        m = RandomForestRegressor(
            n_estimators=n,
            max_depth=best_depth,
            random_state=42,
            n_jobs=-1
        )
        m.fit(X_train, y_train)
        val_scores.append(r2_score(y_val, m.predict(X_val)))
        print(f"  n_estimators={n:<5} → Val R²={val_scores[-1]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(n_range, val_scores, 'o-', color='forestgreen', linewidth=2)
    plt.xlabel("Number of Trees (n_estimators)")
    plt.ylabel("Validation R²")
    plt.title("Random Forest: More Trees → Better (then diminishing returns)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("random_forest_n_estimators.png", dpi=150)
    plt.show()
    print("Plot saved as: random_forest_n_estimators.png")


# =============================================================================
# PLOT: PREDICTED VS ACTUAL
# =============================================================================

def plot_predictions(y_actual, y_predicted):
    """
    Compare this scatter plot to Linear Regression's version.
    Random Forest should cluster MUCH more tightly along the diagonal.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_predicted, alpha=0.2, color='forestgreen', s=5)
    plt.plot([y_actual.min(), y_actual.max()],
             [y_actual.min(), y_actual.max()],
             'r--', linewidth=2, label='Perfect prediction line')
    plt.xlabel("Actual Traffic Volume")
    plt.ylabel("Predicted Traffic Volume")
    plt.title("Random Forest: Predicted vs Actual Traffic Volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig("random_forest_predictions.png", dpi=150)
    plt.show()
    print("Plot saved as: random_forest_predictions.png")


# =============================================================================
# RUN THIS FILE DIRECTLY
# =============================================================================

if __name__ == "__main__":

    # Load preprocessed data
    print("=" * 55)
    print("Loading preprocessed data...")
    print("=" * 55)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess()

    # Tune hyperparameters
    best_params, all_results = tune_hyperparameters(
        X_train, y_train, X_val, y_val
    )

    # Train final model
    model = train_random_forest(X_train, y_train, best_params)

    # Evaluate on validation set
    val_results = evaluate_model(model, X_val, y_val, split_name="Validation")

    # Feature importance plot
    plot_feature_importance(model, list(X_train.columns))

    # Show how n_estimators affects performance
    plot_n_estimators_effect(
        X_train, y_train, X_val, y_val,
        best_depth=best_params["max_depth"]
    )

    # Predicted vs actual plot
    plot_predictions(y_val, val_results["predictions"])

    # Comparison summary so far
    print("\n--- Model Comparison So Far ---")
    print(f"Linear Regression  Val R²: ~0.40-0.55  (baseline)")
    print(f"Decision Tree      Val R²: ~0.80-0.85  (better, but overfits)")
    print(f"Random Forest      Val R²: {val_results['R2']:.4f}         (best so far)")
    print("\nRandom Forest should be the best traditional ML model.")
    print("Next: SVR and LSTM will challenge it.")
