import subprocess
import sys

print("=========================================")
print("Starting Lab 3: Baseline ML Pipeline")
print("=========================================")


# --------------------------------------------------
# Step 1: Preprocessing
# --------------------------------------------------

print("\n[INFO] ---> Executing src/preprocess.py...")

subprocess.run(
    [sys.executable, "src/preprocess.py"],
    check=True
)


# --------------------------------------------------
# Step 2: Model Training
# --------------------------------------------------

print("\n[INFO] ---> Executing src/train.py...")

subprocess.run(
    [sys.executable, "src/train.py"],
    check=True
)


# --------------------------------------------------
# Step 3: Model Evaluation
# --------------------------------------------------

print("\n[INFO] ---> Executing src/evaluate.py...")

subprocess.run(
    [sys.executable, "src/evaluate.py"],
    check=True
)


print("\n=========================================")
print("Lab 3 Baseline Pipeline Completed!")
print("=========================================")