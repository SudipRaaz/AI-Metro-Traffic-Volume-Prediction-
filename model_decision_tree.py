# =============================================================================
# PRO502 - Assignment 2
# File: model_decision_tree.py
# Step 2: Model 2 - Decision Tree Regressor
# =============================================================================

from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import preprocess


# =============================================================================
# HYPERPARAMETER TUNING — FIND BEST max_depth
# =============================================================================

def tune_max_depth(X_train, y_train, X_val, y_val):
    """
    max_depth controls how many levels deep the tree can grow.

    Too shallow (max_depth=2):
        - Tree can only learn very basic rules
        - Underfits — misses important patterns
        - Both train and val scores are low

    Too deep (max_depth=None, unlimited):
        - Tree keeps splitting until every leaf has 1 sample
        - Memorises training data perfectly (train R²≈1.0)
        - But performs badly on new data (overfitting)

    Just right:
        - Val score is highest
        - Train and val scores are close together
        - This is what we are searching for here

    We use the VALIDATION SET to find this sweet spot.
    We NEVER use the test set for tuning.
    """
    print("\n--- Tuning max_depth using Validation Set ---")
    print(f"{'Depth':<10} {'Train R²':<15} {'Val R²':<15} {'Note'}")
    print("-" * 55)

    depths = [2, 4, 6, 8, 10, 15, 20, None]
    best_depth = None
    best_val_r2 = -999
    results = []

    for depth in depths:
        # Train a tree with this depth
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)

        # Score on both train and val
        train_r2 = r2_score(y_train, model.predict(X_train))
        val_r2   = r2_score(y_val,   model.predict(X_val))

        # Detect overfitting: big gap between train and val score
        gap = train_r2 - val_r2
        if gap > 0.15:
            note = "⚠ Overfitting (train >> val)"
        elif val_r2 < 0.5:
            note = "⚠ Underfitting"
        else:
            note = "✓ Good"

        depth_label = str(depth) if depth is not None else "None (unlimited)"
        print(f"{depth_label:<10} {train_r2:<15.4f} {val_r2:<15.4f} {note}")

        results.append((depth, train_r2, val_r2))

        # Track best validation score
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_depth = depth

    print(f"\n✓ Best max_depth = {best_depth} (Val R² = {best_val_r2:.4f})")
    return best_depth, results


# =============================================================================
# TRAIN FINAL DECISION TREE WITH BEST DEPTH
# =============================================================================

def train_decision_tree(X_train, y_train, best_depth):
    """
    Train the final Decision Tree using the best max_depth
    found during hyperparameter tuning.

    random_state=42 ensures reproducibility —
    running the code twice gives the same result.
    """
    print(f"\n--- Training Decision Tree (max_depth={best_depth}) ---")

    model = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
    model.fit(X_train, y_train)

    print("Training complete.")
    return model


# =============================================================================
# EVALUATE MODEL
# =============================================================================

def evaluate_model(model, X, y, split_name="Validation"):
    """
    Same 3 metrics as Linear Regression so we can compare fairly:
    MAE, RMSE, R²

    Comparing the same metrics across all models is how we
    objectively decide which model performs best.
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
# SHOW WHAT RULES THE TREE LEARNED
# =============================================================================

def show_tree_rules(model, feature_names, max_depth_to_show=3):
    """
    One big advantage of Decision Trees over other models:
    you can READ what rules it learned.

    This is called MODEL INTERPRETABILITY — important in real-world
    applications like traffic management where humans need to
    understand and trust the model's decisions.
    """
    print(f"\n--- Top Decision Rules (first {max_depth_to_show} levels) ---")
    print("Read this like a flowchart — each branch is an if/else rule:\n")

    rules = export_text(model,
                        feature_names=list(feature_names),
                        max_depth=max_depth_to_show)
    print(rules)


# =============================================================================
# PLOT: DEPTH TUNING CURVE (SHOWS OVERFITTING VISUALLY)
# =============================================================================

def plot_depth_tuning(results):
    """
    This plot is very useful for your report.
    It visually shows:
    - Where the model underfits (low depth, both scores low)
    - Where overfitting starts (train score rises but val score drops)
    - The sweet spot (highest val score)
    """
    depths      = [str(d) if d is not None else "None" for d, _, _ in results]
    train_scores = [tr for _, tr, _ in results]
    val_scores   = [vr for _, _, vr in results]

    plt.figure(figsize=(9, 5))
    plt.plot(depths, train_scores, 'o-', color='steelblue', label='Train R²')
    plt.plot(depths, val_scores,   's-', color='tomato',    label='Val R²')

    plt.xlabel("max_depth")
    plt.ylabel("R² Score")
    plt.title("Decision Tree: Overfitting vs Underfitting by Depth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("decision_tree_depth_tuning.png", dpi=150)
    plt.show()
    print("Plot saved as: decision_tree_depth_tuning.png")


# =============================================================================
# PLOT: FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(model, feature_names):
    """
    Decision Trees calculate feature importance by measuring how much
    each feature reduces prediction error across all splits.

    Higher importance = the tree uses this feature more for decisions.
    This tells us WHICH features matter most for predicting traffic.

    Compare this to Linear Regression's coefficients — similar idea,
    different calculation method.
    """
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:12]  # Top 12 features

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(indices)),
            importances[indices],
            color='steelblue', alpha=0.8)
    plt.xticks(range(len(indices)),
               [feature_names[i] for i in indices],
               rotation=45, ha='right')
    plt.ylabel("Feature Importance Score")
    plt.title("Decision Tree: Top 12 Most Important Features")
    plt.tight_layout()
    plt.savefig("decision_tree_feature_importance.png", dpi=150)
    plt.show()
    print("Plot saved as: decision_tree_feature_importance.png")


# =============================================================================
# RUN THIS FILE DIRECTLY
# =============================================================================

if __name__ == "__main__":

    # Load preprocessed data
    print("=" * 55)
    print("Loading preprocessed data...")
    print("=" * 55)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess()

    # Tune max_depth using validation set
    best_depth, tuning_results = tune_max_depth(X_train, y_train, X_val, y_val)

    # Train final model with best depth
    model = train_decision_tree(X_train, y_train, best_depth)

    # Evaluate on validation set
    val_results = evaluate_model(model, X_val, y_val, split_name="Validation")

    # Show what rules the tree actually learned
    show_tree_rules(model, X_train.columns, max_depth_to_show=3)

    # Plot overfitting curve
    plot_depth_tuning(tuning_results)

    # Plot which features the tree used most
    plot_feature_importance(model, list(X_train.columns))

    # Compare with Linear Regression
    print("\n--- Comparison Note ---")
    print(f"Decision Tree Val R²  : {val_results['R2']:.4f}")
    print(f"Linear Regression R²  : ~0.35-0.55 (from previous model)")
    print("Decision Tree should significantly outperform Linear Regression")
    print("because it captures non-linear rush hour patterns.")
