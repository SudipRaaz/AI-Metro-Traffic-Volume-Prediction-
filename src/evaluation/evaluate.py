from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def evaluate_models(models, X_test, y_test):
    """
    Compare multiple models and print metrics
    """

    results = {}

    print("\n================ MODEL COMPARISON ================\n")

    for name, model in models.items():
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

        print(f"--- {name} ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE : {mae:.4f}")
        print(f"R2  : {r2:.4f}\n")

    return results