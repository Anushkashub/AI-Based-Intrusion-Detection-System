# ============================================================
# AI-BASED INTRUSION DETECTION SYSTEM
# Real Traffic Validation
# Actual Label vs Predicted Label
# ============================================================

import os
import numpy as np
import pandas as pd

from predict import predict_traffic, feature_names


# ============================================================
# DATASET PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "MachineLearningCSV",
    "MachineLearningCVE",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("REAL TRAFFIC VALIDATION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# FIND LABEL COLUMN
# ============================================================

label_column = None

for column in df.columns:
    if column.lower() == "label":
        label_column = column
        break

if label_column is None:
    raise ValueError("Label column not found.")

print("Label column:", label_column)


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df.drop(columns=[label_column]).copy()

X.columns = X.columns.str.strip()

column_map = {
    column.strip(): column
    for column in X.columns
}

clean_feature_names = [
    feature.strip()
    for feature in feature_names
]


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in clean_feature_names
    if feature not in column_map
]

if missing_features:

    print("\nMissing features:")
    print(missing_features)

    raise ValueError(
        "Some model features are missing."
    )


# ============================================================
# SELECT FEATURES IN MODEL ORDER
# ============================================================

X = X[
    [
        column_map[feature]
        for feature in clean_feature_names
    ]
]


# ============================================================
# CONVERT TO NUMERIC
# ============================================================

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# HANDLE INVALID VALUES
# ============================================================

X = X.mask(np.isinf(X))

valid_mask = X.notna().all(axis=1)

X = X[valid_mask].copy()

y = df.loc[
    valid_mask,
    label_column
].copy()

print("Valid traffic rows:", len(X))


# ============================================================
# CLEAN ACTUAL LABELS
# ============================================================

y = y.astype(str).str.strip()

# Normalize common label variations
y = y.replace({
    "Web Attack – Brute Force":
        "Web Attack - Brute Force",

    "Web Attack – XSS":
        "Web Attack - XSS",

    "Web Attack – Sql Injection":
        "Web Attack - Sql Injection"
})


# ============================================================
# SELECT TEST RECORDS
# ============================================================

TEST_SIZE = 20

test_indices = np.linspace(
    0,
    len(X) - 1,
    TEST_SIZE,
    dtype=int
)

X_test = X.iloc[test_indices]
y_test = y.iloc[test_indices]


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\n")
print("=" * 60)
print("ACTUAL vs PREDICTED")
print("=" * 60)

correct = 0
total = len(X_test)

results = []


for i in range(total):

    values = X_test.iloc[i].to_numpy(
        dtype=float
    )

    actual_label = y_test.iloc[i]

    try:

        result = predict_traffic(values)

        predicted_label = result["attack_type"]
        risk_level = result["risk_level"]

        is_correct = (
            actual_label == predicted_label
        )

        if is_correct:
            correct += 1
            status = "CORRECT"
        else:
            status = "INCORRECT"

        print("\nRecord:", i + 1)
        print("-" * 40)

        print(
            "Actual Attack   :",
            actual_label
        )

        print(
            "Predicted Attack:",
            predicted_label
        )

        print(
            "Risk Level      :",
            risk_level
        )

        print(
            "Class ID        :",
            result["class_id"]
        )

        print(
            "Status           :",
            status
        )

        results.append({
            "Record": i + 1,
            "Actual": actual_label,
            "Predicted": predicted_label,
            "Risk": risk_level,
            "Status": status
        })

    except Exception as e:

        print(
            "\nRecord:",
            i + 1
        )

        print(
            "Prediction Error:",
            e
        )


# ============================================================
# CALCULATE TEST ACCURACY
# ============================================================

if total > 0:

    accuracy = (
        correct / total
    ) * 100

    print("\n")
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(
        "Total records tested:",
        total
    )

    print(
        "Correct predictions:",
        correct
    )

    print(
        "Incorrect predictions:",
        total - correct
    )

    print(
        "Test Accuracy:",
        f"{accuracy:.2f}%"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_dir = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    results_dir,
    exist_ok=True
)

results_path = os.path.join(
    results_dir,
    "real_traffic_validation.csv"
)

results_df = pd.DataFrame(results)

results_df.to_csv(
    results_path,
    index=False
)

print("\nResults saved to:")
print(results_path)


# ============================================================
# COMPLETION
# ============================================================

print("\n")
print("=" * 60)
print("REAL TRAFFIC VALIDATION COMPLETED")
print("=" * 60)