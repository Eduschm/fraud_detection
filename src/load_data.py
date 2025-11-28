import pandas as pd

# Load fraud data from CSV and drop isFraud,isFlaggedFraud
def load_data(file_path):
    df = pd.read_csv('data/fraud_data.csv')
    X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
    y = df['isFraud']
    return X, y
