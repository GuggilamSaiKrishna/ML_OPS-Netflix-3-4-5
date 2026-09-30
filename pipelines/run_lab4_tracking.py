import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


print("=========================================")
print("Starting Lab 4: MLflow Experiment Tracking")
print("=========================================")


# --------------------------------------------------
# 1. Load test data
# --------------------------------------------------

X_test = np.load("data/processed/X_test.npy")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()


# --------------------------------------------------
# 2. Set MLflow experiment
# --------------------------------------------------

mlflow.set_experiment("Netflix_Classification")


# --------------------------------------------------
# 3. Models
# --------------------------------------------------

models = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Random Forest": "models/random_forest.pkl"
}


# --------------------------------------------------
# 4. Track each model
# --------------------------------------------------

for model_name, model_path in models.items():

    print(f"\n[INFO] Tracking {model_name}...")

    model = joblib.load(model_path)

    with mlflow.start_run(run_name=model_name):

        # Prediction
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        # Log parameters
        mlflow.log_param(
            "model",
            model_name
        )

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
        
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")


print("\n=========================================")
print("Lab 4 MLflow Tracking Completed!")
print("=========================================")