from src.config import get_pipeline
from src.config import random_search_cv
import joblib

def train(X_train, y_train):
    # Get pipelines from config
    pipelines = get_pipeline()

    # Create empty dic to store params and results
    best_models = {}
    cv_results = {}

    for name in pipelines:
        grid_search = random_search_cv(X_train, y_train, pipelines[name], name)
        joblib.dump(grid_search, f"models/{name}.pkl")

        best_models[name] = grid_search.best_estimator_
        cv_results[name] = {
            'best_params': grid_search.best_params_,
            'best_score_cv': grid_search.best_score_,
            'all_cv_results': grid_search.cv_results_
        }

        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best cross-validation recall: {grid_search.best_score_:.4f}")
        print("-" * 50)

    return best_models, cv_results
