import pytest
from src.data import load_data

@pytest.fixture
def df():
    return load_data("data/sample_data.csv", predict=True)

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
