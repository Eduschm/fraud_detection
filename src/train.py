from config.config import get_pipeline
from config.config import random_search_cv
import joblib
from utils.logger import Logger

def train(X_train, y_train, quick=False, all_models=True):
    """Train machine learning models using RandomizedSearchCV and save the best models.
    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
        quick (bool, optional): If True, uses a smaller set of hyperparameters for quick training. Defaults to False.
        all_models (bool, optional): If True, trains all models defined in the pipeline. Defaults to True.
    Returns:
        dict: A dictionary containing the best models and their cross-validation results.
    """
    # Get pipelines from config
    log = Logger("ModelTraining", level="INFO").get()
    log.info("Retrieving model pipelines...")
    pipelines = get_pipeline(all_models=all_models)

    # Create empty dic to store params and results
    best_models = {}
    cv_results = {}

    log.info("Starting model training with RandomizedSearchCV...")

    try:
        for name in pipelines:
            log.info(f"Training model: {name}")
            grid_search = random_search_cv(X_train, y_train, pipelines[name], name)
            log.info(f"Best parameters for {name}: {grid_search.best_params_}")
            log.info(f"Best cross-validation recall: {grid_search.best_score_:.4f}")

            # Save the best model
            log.info(f"Saving the best model for {name}...")
            joblib.dump(grid_search, f"models/{name}.pkl")

            best_models[name] = grid_search.best_estimator_
            cv_results[name] = {
                'best_params': grid_search.best_params_,
                'best_score_cv': grid_search.best_score_,
                'all_cv_results': grid_search.cv_results_
            }

    except Exception as e:
        log.error(f"An error occurred during model training: {str(e)}")

    log.info("Model training completed.")

    return best_models, cv_results
