from src.load_data import load_data
from sklearn.model_selection import train_test_split
from src.train import train
from src.predict import ModelPredictor
import argparse
import uvicorn

def main():
    # Create parser to choose between train or test model
    parser = argparse.ArgumentParser(description="Train or predict using ML model")
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help='Select operation mode')
    args = parser.parse_args()

    # Create X and y from the dataset
    X, y = load_data('data/fraud_data.csv')
    # Train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)

    if args.mode == 'train':
        models, cv_results = train(X_train, y_train)
    elif args.mode == 'predict':
        model_predictor = ModelPredictor()
        model_predictor.evaluate_models(X_test, y_test)

if __name__ == "__main__":
    #uvicorn.run("app.main:FraudDetectionApp", host="0.0.0.0", port=8000, reload=True)
    main()
