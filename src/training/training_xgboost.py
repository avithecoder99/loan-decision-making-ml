import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from azure.storage.blob import BlobServiceClient
import os

# Constants
ACCOUNT_NAME = "loanprojectstorage"
CONTAINER_NAME = "preprocessed-loan-data"
BLOB_FILE_NAME = "preprocessed_loan_data.csv"
LOCAL_DATA_PATH = "preprocessed_loan_data.csv"
MODEL_OUTPUT_PATH = "models/xgboost_model.pkl"

# Use SAS token from environment
SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN")
if not SAS_TOKEN:
    raise ValueError("AZURE_SAS_TOKEN environment variable is not set!")

# Blob client with SAS
account_url = f"https://{ACCOUNT_NAME}.blob.core.windows.net"
blob_service = BlobServiceClient(account_url=f"{account_url}?{SAS_TOKEN}")
blob_client = blob_service.get_blob_client(container=CONTAINER_NAME, blob=BLOB_FILE_NAME)

# Download the CSV
with open(LOCAL_DATA_PATH, "wb") as f:
    f.write(blob_client.download_blob().readall())

# Load and split data
df = pd.read_csv(LOCAL_DATA_PATH)
X = df.drop("Eligibility", axis=1)
y = df["Eligibility"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate XGBoost
model = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_OUTPUT_PATH)
print(f"XGBoost model saved to: {MODEL_OUTPUT_PATH}")
