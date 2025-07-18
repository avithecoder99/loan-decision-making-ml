import logging
import azure.functions as func
from azure.storage.blob import BlobServiceClient
import pandas as pd
import io
import os
import datetime
from sklearn.metrics import mutual_info_score
from opencensus.ext.azure.log_exporter import AzureLogHandler

logger = logging.getLogger("DriftLogger")
logger.setLevel(logging.INFO)

# Replace with your App Insights connection string
app_insights_conn_str = os.getenv("APPINSIGHTS_CONNECTION_STRING")
if app_insights_conn_str:
    logger.addHandler(AzureLogHandler(connection_string=app_insights_conn_str))


def compute_psi(expected, actual, buckets=10):
    expected = pd.Series(expected)
    actual = pd.Series(actual)
    breakpoints = [expected.min()] + \
                  list(expected.quantile([i / buckets for i in range(1, buckets)])) + \
                  [expected.max()]
    expected_percents = pd.cut(expected, bins=breakpoints).value_counts(normalize=True, sort=False)
    actual_percents = pd.cut(actual, bins=breakpoints).value_counts(normalize=True, sort=False)
    psi = ((expected_percents - actual_percents) * np.log(expected_percents / actual_percents)).sum()
    return psi

def main(blob: func.InputStream):
    logging.info(f"Blob trigger: {blob.name}, Size: {blob.length} bytes")

    # Load current file
    current_df = pd.read_csv(blob)
    current_df = current_df.drop(columns=['Name', 'Address', 'Email', 'Eligibility'], errors='ignore')

    # Load previous training data (from versioned blob)
    account_url = "https://<your-storage-account>.blob.core.windows.net"
    sas_token = "<your-sas-token>"
    container = "loan-data"
    old_blob = "previous_training_data.csv"

    blob_service = BlobServiceClient(account_url=account_url, credential=sas_token)
    old_data = pd.read_csv(io.BytesIO(
        blob_service.get_blob_client(container=container, blob=old_blob).download_blob().readall()
    ))

    old_df = old_data.drop(columns=['Name', 'Address', 'Email', 'Eligibility'], errors='ignore')

    # Check for PSI drift
    drift_detected = False
    for col in current_df.columns:
        psi = compute_psi(old_df[col], current_df[col])
        logging.info(f"PSI for {col}: {psi:.4f}")
        if psi > 0.1:
            drift_detected = True

    # Collect drift metrics
    drift_results = []
    for col in current_df.columns:
        psi = compute_psi(old_df[col], current_df[col])
        drift_results.append({
            "Feature": col,
            "PSI": round(psi, 4),
            "Drift_Threshold": 0.1,
            "Drift_Detected": psi > 0.1
        })

    # Add metadata
    report_df = pd.DataFrame(drift_results)
    report_df["Timestamp"] = datetime.datetime.utcnow().isoformat()
    report_df["Retraining_Triggered"] = drift_detected

    # Save to Blob Storage
    report_filename = f"drift_report_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
    drift_report_blob = blob_service.get_blob_client(
        container=container,
        blob=f"drift_reports/{report_filename}"
    )

    # Upload CSV to blob
    drift_csv_bytes = io.BytesIO()
    report_df.to_csv(drift_csv_bytes, index=False)
    drift_csv_bytes.seek(0)
    drift_report_blob.upload_blob(drift_csv_bytes, overwrite=True)

    logging.info(f" Drift report saved: drift_reports/{report_filename}")
    if drift_detected:
        logger.warning("Drift detected – triggering retrain!")
        os.system("python trigger_aml_pipeline.py")
    else:
        logging.info(" No significant drift detected.")
