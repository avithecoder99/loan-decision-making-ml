from azure.storage.blob import BlobServiceClient
import os

ACCOUNT_NAME = "loanprojectstorage"
ACCOUNT_KEY = os.getenv("AZURE_STORAGE_KEY")

blob_service = BlobServiceClient(
    account_url=f"https://loanprojectstorage.blob.core.windows.net",
    credential=ACCOUNT_KEY
)

# List containers to test
for container in blob_service.list_containers():
    print(" Found container:", container["name"])
