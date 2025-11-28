from fastapi import FastAPI
from src.predict import ModelPredictor
import pandas as pd
from src.load_data import load_data
class FraudDetectionApp:
    def __init__(self, query_params: dict = None):
        self.app = FastAPI()
        self.sample_data = pd.read_csv('data/sample_data.csv')

        @self.app.get("/")
        def status():
            return {"status": "ok"}

        @self.app.get("/predict")
        def predict():
            return ModelPredictor().evaluate(query_params['X'])
        
        @self.app.get("/get_samples")
        def get_samples():
            samples = self.sample_data.sample(n=5).to_dict(orient='records')
            return {"samples": samples}
        
    def predict(self, query_params):
        query_params = pd.DataFrame([query_params])
        X = load_data(query_params, predict=True)
        return ModelPredictor().evaluate(X)

