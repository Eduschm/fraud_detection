import pytest
from utils.load_data import load_data



@pytest.fixture
def df():
    return load_data("data/sample_data.csv", n_rows=1000)

def test_load_data_returns_dataframe(df):
    assert df is not None
    assert len(df) > 0

def test_load_data_has_required_columns(df):
    required_columns = [
        'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
        'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest'
    ]
    for col in required_columns:
        assert col in df.columns

def test_load_data_target_variable(df):
    assert 'isFraud' in df.columns
    assert df['isFraud'].nunique() == 2  # Binary target variable

def test_load_data_no_missing_values(df):
    assert df.isnull().sum().sum() == 0

def test_column_dtypes(df):
    expected_types = {
        'step': 'int64',
        'type': 'object',
        'amount': 'float64',
        'nameOrig': 'object',
        'oldbalanceOrg': 'float64',
        'newbalanceOrig': 'float64',
        'nameDest': 'object',
        'oldbalanceDest': 'float64',
        'newbalanceDest': 'float64',
        'isFraud': 'int64'  # only if predict=False / target present
    }
    for col, dtype in expected_types.items():
        if col in df.columns:
            assert str(df[col].dtype) == dtype, f"{col} dtype {df[col].dtype} != {dtype}"

def test_numeric_non_negative(df):
    numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    for col in numeric_cols:
        if col in df.columns:
            assert (df[col] >= 0).all(), f"Negative values in {col}"
@pytest.fixture
def test_train_test_validation_split(df):
    from utils.load_data import train_test_validation_split
    train_df, val_df = train_test_validation_split(df, test_size=0.2, validation_size=0.1)
    
    total_len = len(df)
    assert len(train_df) + len(val_df) == total_len
    
    # Check approximate sizes
    assert abs(len(val_df) - 0.1 * total_len) < 0.05 * total_len




def test_evaluate_models():
    from src.predict import ModelPredictor

    model_predictor = ModelPredictor()
    results = model_predictor.evaluate_models(X_test, y_test)

    assert isinstance(results, dict)
    for model_name, metrics in results.items():
        assert 'Recall' in metrics
        assert 'f1' in metrics
        assert 'Precision' in metrics
        assert 'classification_report' in metrics
        assert 'confusion_matrix' in metrics