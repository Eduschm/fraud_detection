from utils.pipeline import get_pipeline
from utils.pipeline import random_search_cv
import joblib
from utils.logger import Logger
from utils.load_data import get_feature_type
from sklearn.ensemble import VotingClassifier

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
    cat_features, num_features = get_feature_type(X_train)
    pipelines = get_pipeline(cat_features, num_features, all_models=all_models)

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
            joblib.dump(grid_search.best_estimator_, f"models/{name}.pkl")

            # Save best params for model 
            log.info(f"Storing best model and CV results for {name}...")
            joblib.dump(grid_search.best_params_, f"config/{name}_params.pkl")

            best_models[name] = grid_search.best_estimator_
            cv_results[name] = {
                'best_params': grid_search.best_params_,
                'best_score_cv': grid_search.best_score_,
                'all_cv_results': grid_search.cv_results_
            }
        voting_clf = VotingClassifier(
            estimators=[(name, model) for name, model in best_models.items()],
            voting='soft'
        )
        voting_clf.fit(X_train, y_train)
        joblib.dump(voting_clf, "models/VotingClassifier.pkl")
        best_models['VotingClassifier'] = voting_clf


    except Exception as e:
        log.error(f"An error occurred during model training: {str(e)}")
        raise e

    log.info("Model training completed.")

    return best_models, cv_results
