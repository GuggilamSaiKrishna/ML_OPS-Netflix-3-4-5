import numpy as np
import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


print("Starting Model Training...")


# --------------------------------------------------
# 1. Load processed data
# --------------------------------------------------

X_train = np.load("data/processed/X_train.npy")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()


# --------------------------------------------------
# 2. Create models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# --------------------------------------------------
# 3. Create model directory
# --------------------------------------------------

os.makedirs("models", exist_ok=True)


# --------------------------------------------------
# 4. Train and save models
# --------------------------------------------------

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    file_name = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        f"models/{file_name}"
    )

    print(f"{name} saved successfully!")


print("Model training completed successfully!")