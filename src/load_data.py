import pandas as pd
from utils.logger import Logger

# Load fraud data from CSV and drop isFraud,isFlaggedFraud
def load_data(X=None, predict=False, data_path='data/fraud_data.csv'):
    '''
    Load data from CSV file and prepare features and target variable.
    args:
        X: DataFrame, optional, default=None
            If provided and predict=True, will be used as input features.
        data_path: str, optional, default='data/fraud_data.csv'
            Path to the CSV data file.
    returns:
        X: DataFrame
            Features for model training.
        y: Series
            Target variable indicating fraud.
    raises: Exception
        If there is an error loading the data.
    '''

    log = Logger(name="DataLoader", level="INFO").get()

    log.info("Loading data from CSV...")

    log.info("Preparing features and target variable...")
    try:
        df = pd.read_csv(data_path)
        X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
        X = X.drop(['nameDest', 'nameOrig', 'step'], axis=1)
        X = pd.get_dummies(X)
        y = df['isFraud']
    except Exception as e:
        log.error(f"An error occurred while loading data: {str(e)}")
        raise e
    return X, y


