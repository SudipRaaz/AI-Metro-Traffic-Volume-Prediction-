# Import required libraries
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Import custom modules
from src.data.eda import run_eda, plot_actual_vs_predicted
from src.data.preprocessing import preprocess
from src.models.run_model import train_models
from src.evaluation.evaluate import evaluate_models, error_analysis


def main():
    # Load dataset from UCI repository
    print("Loading dataset...")
    dataset = fetch_ucirepo(id=492)

    # Extract features and target
    X = dataset.data.features
    y = dataset.data.targets

    # Convert target to Series if needed
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # Perform exploratory data analysis
    run_eda(X, y)

    # Preprocess data and split into train and test sets
    X_train, y_train, X_test, y_test = preprocess()

    # Train models using training data
    models = train_models(X_train, y_train)

    # Evaluate models on test data
    results = evaluate_models(models, X_test, y_test)

    # Select model with highest R2 score
    best_model_name = max(results.items(), key=lambda x: x[1]["R2"])[0]

    # Print best model name
    print(f"\nBest Model: {best_model_name}")


if __name__ == "__main__":
    main()