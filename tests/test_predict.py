import pytest
from src.predict import ModelPredictor
from src.load_data import load_data

def test_evaluate_models():
    X_test, y_test = load_data()

    model_predictor = ModelPredictor()
    results = model_predictor.evaluate_models(X_test, y_test)

    assert isinstance(results, dict)
    for model_name, metrics in results.items():
        assert 'Recall' in metrics
        assert 'f1' in metrics
        assert 'Precision' in metrics
        assert 'classification_report' in metrics
        assert 'confusion_matrix' in metrics