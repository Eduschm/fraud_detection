from fastapi import FastAPI
from src.predict import ModelPredictor
import pandas as pd
from utils.load_data import load_data
class FraudDetectionApp:
    def __init__(self, query_params: dict = None):
        a = 1 # Placeholder to avoid empty constructor

        
    def predict(self, query_params):
        query_params = pd.DataFrame([query_params])
        X = load_data(query_params, predict=True)
        return ModelPredictor().evaluate(X)

app = FastAPI()


sample_data = pd.read_csv('data/sample_data.csv')

@app.get("/")
def status():
    return {"status": "ok"}

@app.get("/predict")
def predict(query_params):
    return ModelPredictor().evaluate(query_params)

@app.get("/get_samples")
def get_samples():
    samples = sample_data.sample(n=5).to_dict(orient='records')
    return {"samples": samples}