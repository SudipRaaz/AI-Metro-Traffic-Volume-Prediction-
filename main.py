from ucimlrepo import fetch_ucirepo

from src.data.eda import run_eda
from src.data.preprossing import preprocess
from src.models.run_model import train_models
from src.evaluation.evaluate import evaluate_models


def main():

    print("Loading dataset...")
    dataset = fetch_ucirepo(id=492)

    X = dataset.data.features
    y = dataset.data.targets

    # 1. EDA
    run_eda(X, y)

    # 2. Preprocessing
    X_train, y_train, X_test, y_test = preprocess()

    # 3. Train multiple models
    models = train_models(X_train, y_train)

    # 4. Evaluate all models
    results = evaluate_models(models, X_test, y_test)
    best_model = max(results.items(), key=lambda x: x[1]["R2"])
    print("Best Model:", best_model[0])


if __name__ == "__main__":
    main()