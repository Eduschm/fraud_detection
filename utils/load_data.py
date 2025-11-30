import pandas as pd
from utils.logger import Logger

# Load fraud data from CSV and drop isFraud,isFlaggedFraud
def load_data(X=None, data_path='data/fraud_data.csv', n_rows=None):
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
        df = pd.read_csv(data_path) if not n_rows else pd.read_csv(data_path, nrows=n_rows)
        return df
    except Exception as e:
        log.error(f"An error occurred while loading data: {str(e)}")
        raise e


def test_validation_split(df, validation_size=0.1, random_state=42):
    '''
    Split the DataFrame into training, validation, and test sets.
    args:
        df: DataFrame
            The complete dataset to be split.
        validation_size: float, optional, default=0.1
            Proportion of the dataset to include in the validation split.
        random_state: int, optional, default=42
            Random seed for reproducibility.    
    returns:
        train_val_df: DataFrame
            Training set.
        val_df: DataFrame
            Validation set.
    raises: Exception
        If there is an error during the splitting process.
    '''
    from sklearn.model_selection import train_test_split

    log = Logger(name="DataSplitter", level="INFO").get()
    log.info("Splitting data into train, validation, and test sets...")
    try:
        train_val_df, test_df = train_test_split(df, random_state=random_state)
        log.info(f"Data split completed: {len(train_val_df)} training samples, {len(test_df)} validation samples.")
        #Save validation set
        test_df.to_csv('data/validation_set.csv', index=False)
        return  train_val_df, test_df
    except Exception as e:
        log.error(f"An error occurred during data splitting: {str(e)}")
        raise e

def get_feature_type(df):
    '''
    Identify categorical and numerical features in the DataFrame.
    args:
        df: DataFrame
            The dataset to analyze.
    returns:
        categorical_features: list
            List of categorical feature names.
        numerical_features: list
            List of numerical feature names.
    raises: Exception
        If there is an error during feature type identification.
    '''
    log = Logger(name="FeatureTypeIdentifier", level="INFO").get()
    log.info("Identifying categorical and numerical features...")
    try:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = df.select_dtypes(include=['number']).columns.tolist()
        log.info(f"Identified {len(categorical_features)} categorical features and {len(numerical_features)} numerical features.")
        return categorical_features, numerical_features
    except Exception as e:
        log.error(f"An error occurred while identifying feature types: {str(e)}")
        raise e
