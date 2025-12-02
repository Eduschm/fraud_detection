from fastapi import FastAPI
from src.predict import ModelPredictor
import pandas as pd

app = FastAPI()


sample_data = pd.read_csv('data/validation_set.csv')

@app.get("/")
def status():
    return {"status": "ok"}

@app.get("/predict")
def predict(query_params):
    return ModelPredictor().evaluate(query_params)

@app.get("/get_samples")
def get_samples():
    samples = sample_data.sample(n=1000).drop('isFraud', axis=1)
    print(samples)
    y_pred = predict(samples)
    samples['isFraud_Prediction'] = y_pred
    # Create samples csv if not exists
    
    return samples