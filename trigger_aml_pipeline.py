import subprocess

print("Triggering model retraining...")
subprocess.run(["python", "retraining/retrain_model.py"], check=True)
print("Retraining completed and models uploaded.")