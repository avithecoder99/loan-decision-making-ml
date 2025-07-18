from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
import io

# Load model
model = joblib.load("models/xgboost_model.pkl")

# Azure Blob setup
account_url = "https://<your-storage-account>.blob.core.windows.net"
sas_token = "<your-sas-token>"
blob_service = BlobServiceClient(account_url=account_url, credential=sas_token)

# Blob details
container_name = "loan-data"
base_file_blob = "synthetic_loan_data.csv"
review_blob = "needs_review_output.csv"

# FastAPI init
app = FastAPI()

# Request schema
class LoanRequest(BaseModel):
    name: str
    address: str
    email: str
    features: list  # [Age, Income, Loan_Amount, Credit_Score, Debt_To_Income, Loan_Term, Self_Employed, Unemployed]

@app.get("/")
def root():
    return {"message": "Loan prediction API is live!"}

@app.post("/predict")
def predict(data: LoanRequest):
    try:
        # Prepare input
        input_data = np.array(data.features).reshape(1, -1)
        prediction = int(model.predict(input_data)[0])

        # Define full column structure
        columns = ["Name", "Address", "Email",
                   "Age", "Income", "Loan_Amount", "Credit_Score", "Debt_To_Income", "Loan_Term",
                   "Employment_Status_Self-Employed", "Employment_Status_Unemployed",
                   "Eligibility"]

        full_row = [data.name, data.address, data.email] + data.features + [prediction]
        new_df = pd.DataFrame([full_row], columns=columns)

        # === 1. Update synthetic_loan_data.csv ===
        base_blob = blob_service.get_blob_client(container=container_name, blob=base_file_blob)
        base_existing = pd.read_csv(io.BytesIO(base_blob.download_blob().readall()))
        updated_df = pd.concat([base_existing, new_df], ignore_index=True)
        base_blob.upload_blob(updated_df.to_csv(index=False), overwrite=True)

        # === 2. Update needs_review_output.csv if prediction == 2 ===
        if prediction == 2:
            review_blob_client = blob_service.get_blob_client(container=container_name, blob=review_blob)
            if review_blob_client.exists():
                review_existing = pd.read_csv(io.BytesIO(review_blob_client.download_blob().readall()))
                review_updated = pd.concat([review_existing, new_df], ignore_index=True)
            else:
                review_updated = new_df

            review_blob_client.upload_blob(review_updated.to_csv(index=False), overwrite=True)

        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
