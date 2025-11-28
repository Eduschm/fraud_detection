import pandas as pd

# Load fraud data from CSV and drop isFraud,isFlaggedFraud
def load_data(X=None, predict=False, data_path='data/fraud_data.csv'):
    if predict:
        X = X.drop(['nameDest', 'nameOrig', 'step'], axis=1)
        X = pd.get_dummies(X)
        return X
    
    df = pd.read_csv(data_path)
    X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
    X = X.drop(['nameDest', 'nameOrig', 'step'], axis=1)
    X = pd.get_dummies(X)
    y = df['isFraud']

    return X, y
