import pandas as pd
import joblib
from azure.storage.blob import BlobServiceClient
from io import StringIO
from datetime import datetime

# === Azure Config ===
AZURE_ACCOUNT_URL = "https://loanprojectstorage.blob.core.windows.net"
AZURE_SAS_TOKEN = "sp=r&st=2025-07-17T00:00:45Z&se=2025-07-17T07:57:45Z&sv=2024-11-04&sr=c&sig=RbrBwOMz8nstO4bS9F17emvO5YeLTvu4R%2F3rR79yGso%3D"  # 🔐 Replace with actual SAS token (omit '?')
AZURE_CONTAINER_NAME = "loan-data"
INPUT_BLOB = "needs_review_output.csv"
OUTPUT_BLOB = "batch_predictions.csv"

# === Model Path ===
MODEL_PATH = "models/random_forest_model.pkl"

# === Connect to Azure Blob Storage ===
blob_service_client = BlobServiceClient(
    account_url=AZURE_ACCOUNT_URL,
    credential=AZURE_SAS_TOKEN
)

# === Load Random Forest model ===
model = joblib.load(MODEL_PATH)

# === Load review data from blob ===
blob_client_in = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=INPUT_BLOB)

try:
    blob_data = blob_client_in.download_blob().readall()
    df = pd.read_csv(StringIO(blob_data.decode()))
except Exception as e:
    print(f"[ERROR] Failed to download review data: {e}")
    exit(1)

# === Preprocess features ===
try:
    df["features"] = df["features"].apply(eval) if isinstance(df.iloc[0]["features"], str) else df["features"]
    X = pd.DataFrame(df["features"].tolist())
except Exception as e:
    print(f"[ERROR] Feature extraction failed: {e}")
    exit(1)

# === Predict with Random Forest ===
df["batch_prediction"] = model.predict(X)

# === Save predictions back to blob ===
blob_client_out = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=OUTPUT_BLOB)

csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False)

try:
    blob_client_out.upload_blob(csv_buffer.getvalue(), overwrite=True)
    print(f"[INFO] Batch predictions written to blob: {OUTPUT_BLOB}")
except Exception as e:
    print(f"[ERROR] Failed to upload predictions: {e}")
