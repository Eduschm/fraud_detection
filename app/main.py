from fastapi import FastAPI
from src.predict import ModelPredictor
import pandas as pd

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