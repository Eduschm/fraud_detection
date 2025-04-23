import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score, recall_score

def predict(X_test, y_test):

    results = {}

    for file in os.listdir('models'):
        # Load model
        model = joblib.load(f"models/{file}")
        # Predicts with trained model
        y_pred = model.predict(X_test)

        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        # Store results
        results[file.replace('.pkl', '')] = {
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
    return results