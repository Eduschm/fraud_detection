
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils.logger import setup_logger  

def random_search_cv(X_train, y_train, model, model_name):
    #Define Hyperparameters for each model
    log = setup_logger("RandomSearchCV", level="INFO")
    log.info(f"Setting up RandomizedSearchCV for {model_name}...")
    param_grids = {
        'LogisticRegression': {
            'classifier__C': [100, 1000, 10000],
            'classifier__solver': ['lbfgs', 'liblinear'],
            'feature_selection__k': [8, 'all']
        },
        'RandomForest': {
            'classifier__n_estimators': [50, 75],
            'classifier__max_depth': [20, 25],
            'classifier__min_samples_split': [4, 5],
        },
        'XGBClassifier': {
            'classifier__n_estimators': [500, 700],
            'classifier__max_depth': [3, 4],
            'classifier__learning_rate': [0.05, 0.07],
            'classifier__subsample': [0.7, 0.9],
            'classifier__colsample_bytree': [0.6, 0.7],
            'classifier__min_child_weight': [0.5, 1],
            'classifier__gamma': [0, 1],
            'classifier__scale_pos_weight': [80, 100, 120]
}
    }
    # Create KFold object
    try:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        #Randomized search with specified parameters
        model = RandomizedSearchCV(model, cv=kf, param_distributions=param_grids[model_name], verbose=3, n_jobs=-1, scoring='recall', n_iter=5)
        # Model fit
        model.fit(X_train, y_train)
    except Exception as e:
        log.error(f"An error occurred during RandomizedSearchCV for {model_name}: {str(e)}")

    return model

def get_pipeline():
    # Creates a pipeline for each model, Scaler, feature selection and classification
    pipelines = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif)),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        'XGBClassifier': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
        ])

    }
    return pipelines