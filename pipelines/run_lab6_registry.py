import mlflow
import mlflow.sklearn

from mlflow import MlflowClient
import joblib


print("=========================================")
print("Starting Lab 6: Model Registry")
print("=========================================")


# --------------------------------------------------
# 1. Load the best model
# --------------------------------------------------

model_path = "models/logistic_regression.pkl"

model = joblib.load(model_path)

print("[INFO] Best model loaded: Logistic Regression")


# --------------------------------------------------
# 2. Create / select MLflow experiment
# --------------------------------------------------

mlflow.set_experiment("Netflix_Model_Registry")


# --------------------------------------------------
# 3. Start MLflow run
# --------------------------------------------------

with mlflow.start_run(run_name="Netflix_Best_Model") as run:

    # Log model information
    mlflow.log_param(
        "model_name",
        "Logistic Regression"
    )

    mlflow.log_param(
        "dataset",
        "Netflix Titles"
    )

    mlflow.log_param(
        "purpose",
        "Netflix rating classification"
    )

    # Log the model
    model_info = mlflow.sklearn.log_model(
        model,
        name="netflix_logistic_regression",
    )

    model_uri = model_info.model_uri

    print("\n[INFO] Model logged successfully!")
    print("[INFO] Model URI:", model_uri)


# --------------------------------------------------
# 4. Register model
# --------------------------------------------------

model_name = "Netflix_Rating_Classifier"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print("\n[INFO] Model registered successfully!")
print("[INFO] Registered Model:", model_name)
print("[INFO] Version:", registered_model.version)


# --------------------------------------------------
# 5. Display model information
# --------------------------------------------------

client = MlflowClient()

model_versions = client.search_model_versions(
    f"name='{model_name}'"
)

print("\n=========================================")
print("MODEL REGISTRY")
print("=========================================")

for version in model_versions:

    print("Model Name :", version.name)
    print("Version    :", version.version)
    print("Stage      :", version.current_stage)
    print("Status     :", version.status)


print("\n=========================================")
print("Lab 6 Model Registry Completed!")
print("=========================================")