import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import predict_traffic, feature_names


# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "MachineLearningCSV",
    "MachineLearningCVE"
)


# ------------------------------------------------------------
# DATASETS TO TEST
# ------------------------------------------------------------

datasets = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
]


# ------------------------------------------------------------
# MULTI-ATTACK TEST
# ------------------------------------------------------------

results = []

for filename in datasets:

    path = os.path.join(DATASET_DIR, filename)

    print("\nLoading:", filename)

    if not os.path.exists(path):
        print("File not found - skipped")
        continue

    df = pd.read_csv(path)

    # Remove spaces from column names


    # Find attack rows
    attack_rows = df[
        df[" Label"].astype(str).str.strip().str.upper() != "BENIGN"
    ]

    # If no attack exists, use first row
    if len(attack_rows) == 0:
        sample = df.iloc[0]
    else:
        sample = attack_rows.iloc[0]

    # Get 78 model features
    X_sample = sample[feature_names].values

    # Predict
    result = predict_traffic(X_sample)

    results.append({
        "Dataset": filename,
        "Actual": sample[" Label"],
        "Predicted": result["attack_type"],
        "Risk": result["risk_level"]
    })


# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("MULTI-ATTACK VALIDATION RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))

print("\nValidation completed successfully!")
