# Banking Fraud Detection System Documentation

## Project Overview

This fraud detection system employs machine learning techniques to identify fraudulent banking transactions. The project uses a structured pipeline for data preprocessing, model training, and evaluation with a focus on optimizing recall scores to effectively identify fraudulent activities.

## System Architecture

The project is organized into several modules:

1. **Data Loading and Preparation** (`load_data.py`)
2. **Model Configuration** (`config.py`)
3. **Model Training** (`train.py`)
4. **Model Evaluation** (`predict.py`)
5. **Main Application** (`main.py`)

## Data

The system processes banking transaction data from `fraud_data.csv`, which contains information about bank transactions. During preprocessing:
To download the data, access https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset/data

- Identifiers (`nameOrig`, `nameDest`) and temporal data (`step`) are removed
- Categorical variables are encoded using one-hot encoding
- Data is split into features (X) and target variable (y), where `isFraud` is the target variable

## Models

Three different machine learning models are implemented and compared:

1. **Logistic Regression**
   - Uses feature selection with SelectKBest
   - Hyperparameters tuned: regularization strength (C), solver algorithm, number of features

2. **Random Forest**
   - Hyperparameters tuned: number of estimators, maximum depth, minimum samples split

3. **XGBoost**
   - Hyperparameters tuned: number of estimators, max depth, learning rate, subsample ratio, column subsample ratio, minimum child weight, gamma, class weight

## Pipeline Architecture

Each model follows a consistent ML pipeline:
- Data standardization using StandardScaler
- Optional feature selection (for Logistic Regression)
- Model training with hyperparameter optimization

## Hyperparameter Optimization

The system uses `RandomizedSearchCV` with 5-fold cross-validation to find optimal hyperparameters for each model. Optimization is specifically targeted to maximize recall, prioritizing the detection of fraudulent transactions (reducing false negatives).

## Evaluation Metrics

Models are evaluated using several metrics with emphasis on:
- **Recall**: The ability to detect all fraudulent transactions
- **Precision**: The accuracy of fraud predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of true/false positives and negatives
- **Classification Report**: Comprehensive performance metrics

## Usage Instructions

### Requirements

The project requires Python with the following packages:
```
scikit-learn==1.6.1
pandas==2.2.3
xgboost==3.0.0
joblib==1.4.2
```
(See requirements.txt for complete list)

### Training Models

To train the models:
```bash
python main.py --mode train
```

This will:
1. Load and preprocess the transaction data
2. Train all three models with hyperparameter optimization
3. Save the best models in the `models` directory

### Making Predictions

To evaluate models on the test set:
```bash
python main.py --mode predict
```

This will:
1. Load the trained models from the `models` directory
2. Generate predictions on the test data
3. Calculate and display performance metrics

## Implementation Details

### Data Split Strategy

The dataset is split with 70% for training and 30% for testing with stratified sampling to maintain class distribution. This strategy ensures:
- Sufficient data for testing model performance in a production-like environment
- Preservation of the fraud/non-fraud ratio across splits

### Performance Optimization

- All available CPU cores are utilized during training (`n_jobs=-1`)
- Models are serialized using `joblib` for persistence

## Extending the System

To add new models to the pipeline:
1. Add the model configuration to the `get_pipeline()` function in `config.py`
2. Define hyperparameter grid in the `param_grids` dictionary in `random_search_cv()`

## Notes on Banking Fraud Detection

- The system prioritizes recall to minimize missed fraud cases
- XGBoost includes `scale_pos_weight` to address class imbalance, which is common in fraud detection
- Standardization is applied to all features to ensure optimal model performance
