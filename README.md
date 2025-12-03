# Banking Fraud Detection System

A production-ready machine learning system for detecting fraudulent banking transactions. Built with FastAPI, XGBoost, and modern MLOps tools.

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.122.0-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## Why This Project?

Fraud detection is one of those problems where getting it wrong has real consequences. Miss a fraudulent transaction and the bank loses money. Flag too many legitimate transactions and customers get frustrated. This project tackles that balance by prioritizing recall (catching fraud) while keeping false positives manageable.

The dataset is highly imbalanced (typical for fraud), which made it a good learning ground for handling class imbalance, hyperparameter tuning, and thinking through real-world tradeoffs.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/eduschm/fraud-detection.git
cd fraud-detection

# Install dependencies
pip install -r requirements.txt

# Download the dataset
# Get it from: https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset
# Place fraud_data.csv in the data/ directory

# Train models (this takes a while)
python main.py --mode train --rows 10000  # Start with subset

# Evaluate
python main.py --mode predict

# Run API
uvicorn app.main:app --reload

# Run Streamlit app
streamlit run app.py
```

## Project Structure

```
fraud-detection/
├── app/
│   ├── main.py           # FastAPI application
│   └── __init__.py
├── config/
│   └── config.py         # Model configs and hyperparameters
├── data/
│   ├── fraud_data.csv    # Main dataset (not in repo)
│   └── sample_data.csv   # Sample for testing
├── models/               # Saved models (gitignored)
├── src/
│   ├── train.py          # Training pipeline
│   ├── predict.py        # Inference and evaluation
│   └── __init__.py
├── tests/
│   ├── test_pipeline.py
│   └── test_predict.py
├── utils/
│   ├── load_data.py      # Data loading and preprocessing
│   ├── logger.py         # Logging utility
│   └── __init__.py
├── app.py                # Streamlit interface
├── main.py               # CLI entry point
├── Dockerfile
├── requirements.txt
└── README.md
```

## The Data

The dataset contains ~6.3M banking transactions with these features:

- **step**: Time unit (1 step = 1 hour)
- **type**: Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
- **amount**: Transaction amount
- **nameOrig**: Customer ID (removed during preprocessing)
- **oldbalanceOrg**: Initial balance before transaction
- **newbalanceOrig**: Balance after transaction
- **nameDest**: Recipient ID (removed during preprocessing)
- **oldbalanceDest**: Initial recipient balance
- **newbalanceDest**: New recipient balance
- **isFraud**: Target variable (1 = fraud, 0 = legitimate)

### Preprocessing Steps

1. Drop identifier columns (nameOrig, nameDest, step)
2. One-hot encode transaction types
3. Standardize numerical features
4. Split 80/20 train/test with stratification

The class imbalance is significant (frauds are rare), which is why I focused on recall as the primary metric.

## Models

I tested three algorithms to see what works best:

### 1. Logistic Regression
Fast baseline with feature selection (SelectKBest). Good for interpretability but limited by linear decision boundaries.

### 2. Random Forest
Handles non-linear patterns and provides feature importance. Better than logistic regression but slower to train.

### 3. XGBoost (Final Choice)
Best performance overall. Handles class imbalance well with `scale_pos_weight` parameter and gives good feature importance for investigation teams.

### Hyperparameter Tuning

Used RandomizedSearchCV with 5-fold cross-validation, optimizing for recall. The search space was extensive:

- XGBoost: learning rate, max depth, subsample ratios, min_child_weight
- Random Forest: n_estimators, max_depth, min_samples_split
- Logistic Regression: regularization (C), solver, feature count

## Why Recall Over Precision?

In fraud detection, the cost of missing fraud (false negative) is typically higher than investigating a legitimate transaction (false positive). A false negative means actual money lost. A false positive just means a transaction gets flagged for review.

That said, precision still matters - too many false alarms and the review team gets overwhelmed. The goal is high recall while keeping precision reasonable.

## API Reference

### Endpoints

**GET /** - Health check
```bash
curl http://localhost:8000/
# Returns: {"status": "ok"}
```

**GET /predict** - Make prediction
```bash
curl -X GET "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": 1000, "type": "TRANSFER", ...}'
```

**GET /get_samples** - Get sample transactions
```bash
curl http://localhost:8000/get_samples
```

## Results

Results from the test set (will vary based on data split):

Coming soon!

(Run `python main.py --mode predict` to see actual numbers)

## Deployment Roadmap

Planning to deploy this with AWS:

1. **Model Storage**: S3 bucket for versioned models
2. **API**: Lambda for serverless inference (or EC2 if Lambda times out)
3. **Frontend**: Streamlit on App Runner
4. **Database**: RDS for logging predictions and feedback
5. **Orchestration**: Airflow for retraining pipeline
6. **Monitoring**: MLflow for experiment tracking and model registry

## MLOps Stack

- **MLflow**: Experiment tracking and model versioning
- **Airflow**: Scheduled retraining and data pipeline
- **Docker**: Containerization for consistent environments
- **Joblib**: Model serialization

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# With coverage
pytest --cov=src tests/
```

## Docker

```bash
# Build image
docker build -t fraud-detection .

# Run container
docker run -p 8080:8080 fraud-detection

# Access API at http://localhost:8080
```

## Known Issues & TODO

### Critical (Fix Before Production)
- [ ] Fix broken test functions in `test_pipeline.py` (nested function definitions)
- [ ] Fix `/predict` endpoint - currently doesn't parse query_params correctly
- [ ] Add input validation with Pydantic models
- [ ] Add proper error handling in API endpoints
- [ ] Model files aren't included in Docker image (need to download or mount)

### High Priority
- [ ] Add data validation (check for nulls, outliers, schema changes)
- [ ] Implement threshold tuning (0.5 might not be optimal)
- [ ] Add CORS configuration for web clients
- [ ] Create proper train/val/test split (currently just train/test)
- [ ] Add model explainability (SHAP values for investigations)
- [ ] Set up CI/CD pipeline (GitHub Actions)

### Nice to Have
- [ ] Add model monitoring and drift detection
- [ ] Implement A/B testing framework
- [ ] Add authentication/rate limiting to API
- [ ] Create interactive dashboard for model metrics
- [ ] Add feature engineering (transaction velocity, account age, etc.)
- [ ] Implement ensemble methods or model stacking
- [ ] Add automated retraining based on performance degradation
- [ ] Create comprehensive API documentation with examples

### Documentation
- [ ] Add architecture diagram
- [ ] Write blog post about interesting challenges
- [ ] Add troubleshooting section
- [ ] Document MLflow setup and usage
- [ ] Add contribution guidelines

## What I Learned

1. **Class imbalance is hard**: Simple accuracy is useless when 99.5% of transactions are legitimate. Had to dig into precision-recall tradeoffs.

2. **Hyperparameter tuning takes forever**: RandomizedSearchCV on the full dataset took hours. Learned to prototype on subsets first.

3. **Production is different from notebooks**: Making something that actually works as an API requires way more error handling and validation than I expected.

4. **MLOps matters**: Without MLflow, I was losing track of which hyperparameters produced which results. Experiment tracking is essential.

## Contributing

Found a bug? Have a suggestion? Open an issue or submit a PR. This is a learning project so feedback is welcome.

## License

MIT License - feel free to use this for learning or as a starting point for your own projects.


---

*