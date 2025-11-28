import pytest
from src.load_data import load_data 

def test_load_data_sample():

    def test_load_data_returns_dataframe():
        df = load_data("data/sample_data.csv", nrows=500)
        assert df is not None
        assert len(df) > 0

    def test_load_data_has_required_columns():
        df = load_data("data/sample_data.csv", nrows=500)
        required_columns = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                            'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
        for col in required_columns:
            assert col in df.columns

    def test_load_data_nrows_parameter():
        df = load_data("data/sample_data.csv", nrows=100)
        assert len(df) <= 100
    


