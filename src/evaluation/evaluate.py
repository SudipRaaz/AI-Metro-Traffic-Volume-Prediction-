# Import required libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from src.data.eda import plot_actual_vs_predicted


def evaluate_models(models, X_test, y_test):
    # Store evaluation results for each model
    results = {}

    print("\n================ MODEL COMPARISON ================\n")

    # Loop through all trained models
    for name, model in models.items():
        # Generate predictions
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Save results
        results[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

        # Print results
        print(f"--- {name} ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE : {mae:.4f}")
        print(f"R2  : {r2:.4f}\n")

        # Plot actual vs predicted values
        plot_actual_vs_predicted(y_test, y_pred, model_name=name)

        # Analyze error distribution
        error_analysis(y_test, y_pred)

    return results


def error_analysis(y_test, y_pred):
    # Calculate prediction errors
    errors = y_test - y_pred

    # Plot distribution of errors
    plt.figure()
    plt.hist(errors)
    plt.title("Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.show()