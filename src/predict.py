import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score, recall_score
from utils.logger import Logger
import json

class ModelPredictor():
    """
    ModelPredictor is a utility class for loading trained machine learning models and making predictions or evaluating their performance.
    Attributes:
        None
    Methods:
        evaluate(X):
            Loads a specific model ('models/XGBClassifier.pkl') and returns predictions for the input features X.
        evaluate_models(X_test, y_test):
            Iterates over all model files in the 'models' directory, loads each model, makes predictions on X_test, computes evaluation metrics (recall, f1-score, precision, classification report, confusion matrix), prints results, and stores them in the results attribute.
    Example:
        predictions = predictor.evaluate(X)
        evaluation_results = predictor.evaluate_models(X_test, y_test)
    """

    def __init__(self):
        
        self.models = []
        self.results = {}

    def evaluate(self, X):
        model = joblib.load('models/XGBClassifier.pkl')
        y_pred = model.predict(X)
        return y_pred
    
    def evaluate_models(self, X_test, y_test, type='evaluate'):
        """Evaluate all models in the 'models' directory on the provided test data.
        Args:
            X_test (DataFrame): Test features.
            y_test (Series): True labels for the test data.
        Returns:
            dict: A dictionary containing evaluation metrics for each model.
        """
        log = Logger("ModelEvaluation", level="INFO").get()
        log.info("Evaluating models...")
        for file in os.listdir('models'):
            # Load model
            model = joblib.load(f"models/{file}")
            log.info(f"Calculating metrics for model: {file}")
            self.predict_model(model, X_test, y_test, file)
            return self.results

            

    def predict_model(self, model, X_test, y_test, file):
        """Predict using a specific model.
        Args:
            model_name (str): Name of the model file (e.g., 'XGBClassifier.pkl').
            X (DataFrame): Input features for prediction.
            y_test (Series): True labels for the test data.
            file: str
                Model file name.
        Returns:
            array: Predicted labels.
        """
        # Predicts with trained model, adjusting the threshold to 0.3
        y_pred = model.predict(X_test)
        # Calculate metrics
        
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        # Store results
        self.results[file.replace('.pkl', '')] = {
            'Recall': recall,
            'f1': f1,
            'Precision': precision,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }

        # Print accuracy
        print(f"{file.replace('.pkl', '')} Recall: {recall:.4f}, f1: {f1}")
        print(conf_matrix)
        print(report)
        return self.results
        
