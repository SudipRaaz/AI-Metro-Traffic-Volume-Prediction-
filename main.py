from ucimlrepo import fetch_ucirepo
import pandas as pd

from src.data.eda import run_eda,plot_actual_vs_predicted
from src.data.preprocessing import preprocess
from src.models.run_model import train_models
from src.evaluation.evaluate import evaluate_models,  error_analysis


def main():

    print("Loading dataset...")
    dataset = fetch_ucirepo(id=492)

    X = dataset.data.features
    y = dataset.data.targets

    # Ensure y is a Series
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # ---------------- EDA ----------------
    run_eda(X,y)

    # Preprocess first to generate features for insights
    X_train, y_train,X_test, y_test = preprocess()

   

    # ---------------- TRAIN ----------------
    models = train_models(X_train, y_train)

    # ---------------- EVALUATE ----------------
    results = evaluate_models(models, X_test, y_test)


    # Best model selection
    best_model_name = max(results.items(), key=lambda x: x[1]["R2"])[0]

    print(f"\nBest Model: {best_model_name}")




if __name__ == "__main__":
    main()