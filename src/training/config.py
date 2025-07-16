# src/training/config.py
# config.py or top of your training script
import os
from azure.storage.blob import BlobServiceClient

from azure.storage.blob import BlobServiceClient

sas_token = os.getenv("AZURE_SAS_TOKEN")
account_url = "https://loanprojectstorage.blob.core.windows.net"

blob_service = BlobServiceClient(
    account_url=f"{account_url}?{sas_token}"
)

# Blob details
CONTAINER_NAME = "preprocessed-loan-data"
BLOB_FILE_NAME = "preprocessed_loan_data.csv"
LOCAL_DATA_PATH = "preprocessed_loan_data.csv"

# Model output paths
XGBOOST_MODEL_PATH = "models/xgboost_model.pkl"
RF_MODEL_PATH = "models/random_forest_model.pkl"

