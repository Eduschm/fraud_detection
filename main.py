from src.load_data import load_data
from sklearn.model_selection import train_test_split
from src.train import train
from src.predict import ModelPredictor
import argparse
import uvicorn
from utils.logger import setup_logger

def main():

    log = setup_logger("FraudDetectionApp", level="INFO")
    log.info("Starting Fraud Detection Application")
    # Create parser to choose between train or test model
    parser = argparse.ArgumentParser(description="Train or predict using ML model")
    parser.add_argument('--mode', choices=['train', 'predict', 'quick_train'], required=True, help='Select operation mode')
    parser.add_argument('--data-path', type=str, default='data/fraud_data.csv', help='Path to the dataset CSV file')

    args = parser.parse_args()

    # Create X and y from the dataset
    log.info("Loading data...")
    X, y = load_data('data/fraud_data.csv')
    # Train_test_split
    log.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)

    try:
        log.info(f"Operation mode selected: {args.mode}")
        if args.mode == 'train':
            log.info("Training models...")
            models, cv_results = train(X_train, y_train)
        elif args.mode == 'predict':
            log.info("Evaluating models...")
            model_predictor = ModelPredictor()
            model_predictor.evaluate_models(X_test, y_test)
    except Exception as e:
        log.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    #uvicorn.run("app.main:FraudDetectionApp", host="0.0.0.0", port=8000, reload=True)
    main()
