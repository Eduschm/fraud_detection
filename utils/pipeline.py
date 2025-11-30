
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils.logger import Logger  
import yaml
# import column transformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def random_search_cv(X_train, y_train, model, model_name, quick=False):
    #Define Hyperparameters for each model
    log = Logger("RandomSearchCV", level="INFO").get()
    log.info(f"Setting up RandomizedSearchCV for {model_name}...")
    param_grids = yaml.safe_load(open('config/param_grids.yaml'))


    # Create KFold object
    try:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        #Randomized search with specified parameters
        model = RandomizedSearchCV(
            model, 
            cv=kf if not quick else 2, 
            param_distributions=param_grids[model_name], 
            verbose=1, 
            n_jobs=-1, 
            scoring='recall', 
            n_iter=20 if not quick else 5, 
            random_state=42)
        # Model fit
        model.fit(X_train, y_train)

    except Exception as e:
        log.error(f"An error occurred during RandomizedSearchCV for {model_name}: {str(e)}")
        raise e
    
    return model

def get_pipeline(cat_features, num_features, all_models=True,):

    """ Get machine learning pipelines for different models.
    Args:
        all_models (bool, optional): If True, includes all models. If False, includes only XGBClassifier. Defaults to True.
        cat_features (list): List of categorical feature names.
        num_features (list): List of numerical feature names.
    Returns:
        dict: A dictionary containing machine learning pipelines.
    """
    # Creates a pipeline for each model, Scaler, feature selection and classification)

    preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', StandardScaler(with_mean=False), num_features)
    ],
    remainder='passthrough'
)

    pipelines =  {
    'LogisticRegression': Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_classif)),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ]),

    'RandomForest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),

    'XGBClassifier': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])
}
    if not all_models:
        pipelines = {
            'XGBClassifier': pipelines['XGBClassifier']
        }
    return pipelines