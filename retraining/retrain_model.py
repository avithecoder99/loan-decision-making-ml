import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from azure.storage.blob import BlobServiceClient
import io

# Blob access
account_url = "https://<your-storage>.blob.core.windows.net"
sas_token = "<your-sas-token>"
container_name = "loan-data"
data_blob = "synthetic_loan_data.csv"

# Download dataset
blob_service = BlobServiceClient(account_url=account_url, credential=sas_token)
blob_client = blob_service.get_blob_client(container=container_name, blob=data_blob)
data = pd.read_csv(io.BytesIO(blob_client.download_blob().readall()))

# Preprocessing
X = data.drop(columns=["Name", "Address", "Email", "Eligibility"])
y = data["Eligibility"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train XGBoost
xgb_model = XGBClassifier(objective="multi:softmax", num_class=3)
xgb_model.fit(X_train, y_train)

# Train Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Save models
joblib.dump(xgb_model, "retraining/xgboost_model.pkl")
joblib.dump(rf_model, "retraining/random_forest_model.pkl")

# Upload new models to blob
for model_name in ["xgboost_model.pkl", "random_forest_model.pkl"]:
    model_path = f"retraining/{model_name}"
    model_blob = blob_service.get_blob_client(container=container_name, blob=f"models/{model_name}")
    with open(model_path, "rb") as model_file:
        model_blob.upload_blob(model_file, overwrite=True)

print("✅ Models retrained and uploaded.")
