# =============================================================================
# PRO502 - Assignment 2
# File: model_svr.py
# Step 2: Model 4 - Support Vector Regression (SVR)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import preprocess


# =============================================================================
# STEP A: WHY WE SAMPLE FOR SVR
# =============================================================================

def sample_training_data(X_train, y_train, sample_size=8000):
    """
    SVR has O(n²) to O(n³) computational complexity.
    This means training time grows EXPONENTIALLY with dataset size.

    With ~33,000 training rows:
        Random Forest → trains in seconds
        SVR           → could take hours or crash RAM

    Solution: Train SVR on a RANDOM SAMPLE of training data.

    Why is this still valid?
    - 8,000 samples is still a large, representative sample
    - SVR is particularly good at finding boundaries
      even from a subset of data (support vectors)
    - We mention this limitation honestly in our report

    IMPORTANT:
    We use random_state=42 so the same sample is chosen
    every time we run the code (reproducibility).

    We do NOT sample validation or test data —
    we still evaluate on the full sets for fair comparison.
    """
    print(f"\n--- Sampling Training Data for SVR ---")
    print(f"Full training size : {len(X_train)} rows")
    print(f"SVR sample size    : {sample_size} rows")
    print(f"Reason             : SVR is O(n²) — too slow on full dataset")

    # Set seed for reproducibility
    np.random.seed(42)

    # Randomly pick sample_size indices
    indices = np.random.choice(len(X_train), size=sample_size, replace=False)

    # Select those rows
    X_sample = X_train.iloc[indices]
    y_sample = y_train.iloc[indices]

    print(f"Sample selected    : {len(X_sample)} rows")
    return X_sample, y_sample


# =============================================================================
# STEP B: HYPERPARAMETER TUNING
# =============================================================================

def tune_hyperparameters(X_sample, y_sample, X_val, y_val):
    """
    SVR has 3 key hyperparameters:

    C (Regularisation parameter):
        Controls the trade-off between:
        - Fitting training data well (high C)
        - Keeping the model simple/smooth (low C)

        Low C  → wide margin, more errors allowed → underfitting risk
        High C → narrow margin, fits tightly → overfitting risk

        Think of C like: "how much do I care about getting
        every training point right?"

    epsilon (ε — tube half-width):
        Points WITHIN epsilon distance of the tube are NOT penalised.
        This is the tolerance zone.

        Large epsilon → wider tube → more points ignored → simpler model
        Small epsilon → narrow tube → model tries harder to fit all points

        For traffic (range 0-7000), epsilon=100-500 is reasonable.

    kernel:
        'rbf' (Radial Basis Function) — transforms data into higher
        dimensions to handle non-linear relationships.
        Best default choice for most real-world problems.

        We fix kernel='rbf' and tune C and epsilon only.

    We use a smaller grid than Random Forest because
    each SVR training run is much slower.
    """
    print("\n--- Tuning SVR Hyperparameters ---")
    print("(Each combination takes ~1-3 minutes — please be patient)\n")

    C_values       = [0.1, 1, 10, 100]
    epsilon_values = [50, 100, 500]

    best_val_r2 = -999
    best_params = {}
    all_results = []

    print(f"{'C':<8} {'epsilon':<10} {'Train R²':<12} {'Val R²':<12} {'Note'}")
    print("-" * 55)

    for C in C_values:
        for epsilon in epsilon_values:

            model = SVR(
                kernel  = 'rbf',
                C       = C,
                epsilon = epsilon
            )
            model.fit(X_sample, y_sample)

            # Evaluate on sample (for train score) and full val set
            train_r2 = r2_score(y_sample, model.predict(X_sample))
            val_r2   = r2_score(y_val,    model.predict(X_val))

            gap  = train_r2 - val_r2
            note = "⚠ Overfitting" if gap > 0.15 else "✓ Good"

            print(f"{C:<8} {epsilon:<10} {train_r2:<12.4f} {val_r2:<12.4f} {note}")

            all_results.append({
                "C": C, "epsilon": epsilon,
                "train_r2": train_r2, "val_r2": val_r2
            })

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_params = {"C": C, "epsilon": epsilon}

    print(f"\n✓ Best Parameters:")
    print(f"  C       = {best_params['C']}")
    print(f"  epsilon = {best_params['epsilon']}")
    print(f"  Val R²  = {best_val_r2:.4f}")

    return best_params, all_results


# =============================================================================
# STEP C: TRAIN FINAL SVR MODEL
# =============================================================================

def train_svr(X_sample, y_sample, best_params):
    """
    Train final SVR with best hyperparameters found during tuning.

    After training, SVR internally identifies the 'support vectors' —
    the specific training points that define the tube boundary.
    All other points are irrelevant to the final model.

    This is why SVR can work well even with a sample —
    it only needs the boundary-defining points.
    """
    print(f"\n--- Training Final SVR ---")
    print(f"Parameters: C={best_params['C']}, epsilon={best_params['epsilon']}")

    model = SVR(
        kernel  = 'rbf',
        C       = best_params['C'],
        epsilon = best_params['epsilon']
    )
    model.fit(X_sample, y_sample)

    # How many support vectors did SVR find?
    n_sv = len(model.support_vectors_)
    print(f"Training complete.")
    print(f"Number of support vectors: {n_sv}")
    print(f"(These are the only training points that define the model)")

    return model


# =============================================================================
# STEP D: EVALUATE MODEL
# =============================================================================

def evaluate_model(model, X, y, split_name="Validation"):
    """
    Same metrics as all previous models: MAE, RMSE, R²
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
# STEP E: VISUALISE C PARAMETER EFFECT
# =============================================================================

def plot_C_effect(all_results):
    """
    Shows how C (regularisation) affects performance.

    This plot demonstrates the bias-variance tradeoff:
    - Low C  → high bias (underfitting), low variance
    - High C → low bias, high variance (overfitting)

    The best C is where val R² peaks.
    This is a great visualisation for your report to justify
    your chosen C value.
    """
    # Group results by C value, average across epsilon values
    c_values  = sorted(set(r['C'] for r in all_results))
    val_means = []
    val_stds  = []

    for C in c_values:
        vals = [r['val_r2'] for r in all_results if r['C'] == C]
        val_means.append(np.mean(vals))
        val_stds.append(np.std(vals))

    plt.figure(figsize=(8, 5))
    plt.semilogx(c_values, val_means, 'o-', color='darkorange', linewidth=2)
    plt.fill_between(c_values,
                     np.array(val_means) - np.array(val_stds),
                     np.array(val_means) + np.array(val_stds),
                     alpha=0.2, color='darkorange')

    plt.xlabel("C (log scale)")
    plt.ylabel("Validation R²")
    plt.title("SVR: Effect of C Parameter on Performance\n(shaded area = std across epsilon values)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("svr_C_effect.png", dpi=150)
    plt.show()
    print("Plot saved as: svr_C_effect.png")


# =============================================================================
# STEP F: PLOT PREDICTED VS ACTUAL
# =============================================================================

def plot_predictions(y_actual, y_predicted, n_points=500):
    """
    Same plots as CNN and LSTM for direct visual comparison.

    SVR often struggles with extreme values (traffic spikes)
    because the tube treats large errors beyond epsilon the same.
    Look for this in the scatter plot — predictions may be
    compressed toward the middle range.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    axes[0].scatter(y_actual[:n_points], y_predicted[:n_points],
                    alpha=0.3, color='darkorange', s=8)
    axes[0].plot([y_actual.min(), y_actual.max()],
                 [y_actual.min(), y_actual.max()],
                 'r--', linewidth=2, label='Perfect line')
    axes[0].set_xlabel("Actual Traffic Volume")
    axes[0].set_ylabel("Predicted Traffic Volume")
    axes[0].set_title("SVR: Predicted vs Actual (scatter)")
    axes[0].legend()

    # Time series
    axes[1].plot(y_actual[:n_points],    label='Actual',
                 color='steelblue', linewidth=1, alpha=0.8)
    axes[1].plot(y_predicted[:n_points], label='Predicted',
                 color='darkorange', linewidth=1, alpha=0.8)
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Traffic Volume")
    axes[1].set_title("SVR: Predicted vs Actual (over time)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("svr_predictions.png", dpi=150)
    plt.show()
    print("Plot saved as: svr_predictions.png")


# =============================================================================
# STEP G: FULL MODEL COMPARISON
# =============================================================================

def print_full_leaderboard(svr_val_r2):
    """
    Print expected leaderboard after all 5 models trained.
    Actual values will come from your test evaluation in Step 3.
    """
    print("\n" + "=" * 55)
    print("FULL MODEL LEADERBOARD (Validation R²)")
    print("=" * 55)
    print(f"  {'Model':<20} {'Expected R²':<15} {'Status'}")
    print("-" * 55)
    print(f"  {'Linear Regression':<20} {'~0.40-0.55':<15} Baseline")
    print(f"  {'Decision Tree':<20} {'~0.80-0.85':<15} Good")
    print(f"  {'Random Forest':<20} {'~0.87-0.91':<15} Best Traditional")
    print(f"  {'SVR':<20} {svr_val_r2:<15.4f} Traditional")
    print(f"  {'CNN':<20} {'~0.83-0.89':<15} Deep Learning")
    print(f"  {'LSTM':<20} {'~0.85-0.91':<15} Deep Learning")
    print("=" * 55)
    print("\nAll 5 models trained! Ready for Step 3: Final Evaluation")
    print("on TEST data with full comparison table.")


# =============================================================================
# RUN THIS FILE DIRECTLY
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # 1. Load preprocessed data
    # -------------------------------------------------------------------------
    print("=" * 55)
    print("Loading preprocessed data...")
    print("=" * 55)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess()

    # -------------------------------------------------------------------------
    # 2. Sample training data (SVR cannot handle 33k rows efficiently)
    # -------------------------------------------------------------------------
    X_sample, y_sample = sample_training_data(X_train, y_train, sample_size=8000)

    # -------------------------------------------------------------------------
    # 3. Tune hyperparameters
    # -------------------------------------------------------------------------
    best_params, all_results = tune_hyperparameters(
        X_sample, y_sample, X_val, y_val
    )

    # -------------------------------------------------------------------------
    # 4. Train final SVR model
    # -------------------------------------------------------------------------
    model = train_svr(X_sample, y_sample, best_params)

    # -------------------------------------------------------------------------
    # 5. Evaluate on validation set
    # -------------------------------------------------------------------------
    val_results = evaluate_model(model, X_val, y_val, split_name="Validation")

    # -------------------------------------------------------------------------
    # 6. Plots
    # -------------------------------------------------------------------------
    plot_C_effect(all_results)
    plot_predictions(y_val, val_results["predictions"])

    # -------------------------------------------------------------------------
    # 7. Full leaderboard
    # -------------------------------------------------------------------------
    print_full_leaderboard(val_results["R2"])
