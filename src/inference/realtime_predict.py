from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("models/xgboost_model.pkl")

# Define input schema
class LoanFeatures(BaseModel):
    features: list

# Create FastAPI instance
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Loan prediction API is live!"}

@app.post("/predict")
def predict(data: LoanFeatures):
    try:
        input_data = np.array(data.features).reshape(1, -1)
        prediction = model.predict(input_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
