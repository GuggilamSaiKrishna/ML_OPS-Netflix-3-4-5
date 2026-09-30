import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

print("Starting Preprocessing Pipeline...")

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

data_path = "data/raw/netflix_titles.xlsx"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Remove rows where rating is missing
# --------------------------------------------------

df = df.dropna(subset=["rating"])


# --------------------------------------------------
# 3. Remove invalid rating values
# --------------------------------------------------

invalid_ratings = ["74 min", "84 min", "66 min"]

df = df[~df["rating"].isin(invalid_ratings)]


# --------------------------------------------------
# 4. Select required features
# --------------------------------------------------

features = [
    "type",
    "country",
    "release_year",
    "duration",
    "listed_in"
]

target = "rating"

df = df[features + [target]].copy()


# --------------------------------------------------
# 5. Fill missing values
# --------------------------------------------------

for column in features:
    df[column] = df[column].fillna("Unknown")


# --------------------------------------------------
# 6. Separate input and target
# --------------------------------------------------

X = df[features]
y = df[target]


# --------------------------------------------------
# 7. One-hot encoding
# --------------------------------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_encoded = encoder.fit_transform(X)


# --------------------------------------------------
# 8. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 9. Create processed folder
# --------------------------------------------------

os.makedirs("data/processed", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)


# --------------------------------------------------
# 10. Save processed data
# --------------------------------------------------

import numpy as np

np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)

y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)


# --------------------------------------------------
# 11. Save encoder
# --------------------------------------------------

joblib.dump(
    encoder,
    "artifacts/encoder.pkl"
)

print("Preprocessing completed successfully!")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)