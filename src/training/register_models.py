from azureml.core import Workspace, Model
import os

# Directly define the workspace details
ws = Workspace(
    subscription_id="your sbscription id",
    resource_group="loan-project-rg",                        
    workspace_name="loan-decision-ml-studio"                 
)

model_dir = "models"
model_files = {
    "xgboost_model": "xgboost_model.pkl",
    "random_forest_model": "random_forest_model.pkl"
}

for name, file in model_files.items():
    model_path = os.path.join(model_dir, file)
    registered_model = Model.register(
        workspace=ws,
        model_path=model_path,
        model_name=name,
        tags={"project": "LoanApproval", "type": "real-time" if "xgboost" in name else "batch"},
        description=f"{name} for loan approval system"
    )
    print(f" Registered: {registered_model.name}, Version: {registered_model.version}")
