import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


print("Starting Model Evaluation...")


# --------------------------------------------------
# 1. Load test data
# --------------------------------------------------

X_test = np.load("data/processed/X_test.npy")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()


# --------------------------------------------------
# 2. Models to evaluate
# --------------------------------------------------

models = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Random Forest": "models/random_forest.pkl"
}


results = []


# --------------------------------------------------
# 3. Evaluate each model
# --------------------------------------------------

for name, path in models.items():

    print(f"\nEvaluating {name}...")

    model = joblib.load(path)

    y_pred = model.predict(X_test)

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

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


# --------------------------------------------------
# 4. Create results DataFrame
# --------------------------------------------------

results_df = pd.DataFrame(results)


# --------------------------------------------------
# 5. Save evaluation results
# --------------------------------------------------

results_df.to_csv(
    "outputs/evaluation_results.csv",
    index=False
)


# --------------------------------------------------
# 6. Find best model
# --------------------------------------------------

best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\n========================================")
print("BEST MODEL")
print("========================================")

print("Model:", best_model["Model"])
print("F1 Score:", round(best_model["F1 Score"], 4))

print("\nEvaluation completed successfully!")