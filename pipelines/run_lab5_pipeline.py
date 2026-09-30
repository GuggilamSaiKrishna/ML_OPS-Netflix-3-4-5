import subprocess
import sys

print("=========================================")
print("Starting Lab 5: Complete ML Pipeline")
print("=========================================")


# --------------------------------------------------
# Step 1: Preprocessing
# --------------------------------------------------

print("\n[INFO] ---> Step 1: Preprocessing")

subprocess.run(
    [sys.executable, "src/preprocess.py"],
    check=True
)


# --------------------------------------------------
# Step 2: Training
# --------------------------------------------------

print("\n[INFO] ---> Step 2: Model Training")

subprocess.run(
    [sys.executable, "src/train.py"],
    check=True
)


# --------------------------------------------------
# Step 3: Evaluation
# --------------------------------------------------

print("\n[INFO] ---> Step 3: Model Evaluation")

subprocess.run(
    [sys.executable, "src/evaluate.py"],
    check=True
)


# --------------------------------------------------
# Step 4: MLflow Tracking
# --------------------------------------------------

print("\n[INFO] ---> Step 4: MLflow Tracking")

subprocess.run(
    [sys.executable, "pipelines/run_lab4_tracking.py"],
    check=True
)


print("\n=========================================")
print("Lab 5 Complete ML Pipeline Finished!")
print("=========================================")