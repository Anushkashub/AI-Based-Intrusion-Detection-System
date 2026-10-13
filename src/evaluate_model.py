# ============================================================
# AI-BASED INTRUSION DETECTION SYSTEM
# MODEL EVALUATION MODULE
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from predict import predict_traffic, feature_names


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "MachineLearningCSV",
    "MachineLearningCVE"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

# Create results folder if it does not exist
os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# DATASET FILES
# ============================================================

DATASET_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
]


# ============================================================
# SETTINGS
# ============================================================

SAMPLES_PER_CLASS = 100


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(label):

    label = str(label).strip()

    # Handle different dash/encoding variations
    label = label.replace("Web Attack �", "Web Attack -")
    label = label.replace("Web Attack –", "Web Attack -")
    label = label.replace("Web Attack —", "Web Attack -")

    label = " ".join(label.split())

    return label


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("=" * 70)
print("LOADING REAL NETWORK TRAFFIC")
print("=" * 70)

X_validation = []
y_validation = []


for filename in DATASET_FILES:

    file_path = os.path.join(
        DATA_DIR,
        filename
    )

    print(f"\nLoading: {filename}")

    if not os.path.exists(file_path):

        print("File not found. Skipping...")

        continue

    try:

        df = pd.read_csv(
            file_path,
            low_memory=False
        )

        print(f"Rows: {len(df)}")

        # ----------------------------------------------------
        # FIND LABEL COLUMN SAFELY
        # ----------------------------------------------------

        label_column = None

        for col in df.columns:

            if str(col).strip() == "Label":

                label_column = col
                break

        if label_column is None:

            print("Label column not found. Skipping...")

            continue

        # ----------------------------------------------------
        # GET UNIQUE LABELS
        # ----------------------------------------------------

        labels = df[label_column].apply(
            normalize_label
        )

        unique_labels = labels.unique()

        # ----------------------------------------------------
        # PROCESS EACH CLASS
        # ----------------------------------------------------

        for attack_class in unique_labels:

            # Select rows belonging to this class
            class_mask = (
                labels == attack_class
            )

            class_df = df.loc[
                class_mask
            ]

            # ------------------------------------------------
            # CHECK REQUIRED FEATURES
            # ------------------------------------------------

            missing_features = [
                feature
                for feature in feature_names
                if feature not in class_df.columns
            ]

            if missing_features:

                print(
                    f"Missing features for {attack_class}. "
                    "Skipping class."
                )

                continue

            # ------------------------------------------------
            # SELECT FEATURES
            # ------------------------------------------------

            sample_features = class_df[
                feature_names
            ].copy()

            # ------------------------------------------------
            # CONVERT TO NUMERIC
            # ------------------------------------------------

            sample_features = sample_features.apply(
                pd.to_numeric,
                errors="coerce"
            )

            # ------------------------------------------------
            # REPLACE INFINITY
            # ------------------------------------------------

            sample_features = sample_features.replace(
                [np.inf, -np.inf],
                np.nan
            )

            # ------------------------------------------------
            # REMOVE INVALID ROWS
            # ------------------------------------------------

            valid_mask = (
                sample_features.notna().all(axis=1)
            )

            sample_features = sample_features.loc[
                valid_mask
            ]

            # ------------------------------------------------
            # TAKE SAMPLE
            # ------------------------------------------------

            if len(sample_features) > SAMPLES_PER_CLASS:

                sample_features = sample_features.sample(
                    n=SAMPLES_PER_CLASS,
                    random_state=42
                )

            number_of_samples = len(
                sample_features
            )

            if number_of_samples == 0:

                continue

            # ------------------------------------------------
            # STORE DATA
            # ------------------------------------------------

            X_validation.append(
                sample_features
            )

            y_validation.extend(
                [attack_class] * number_of_samples
            )

            print(
                f"  {attack_class}: "
                f"{number_of_samples} valid samples"
            )

    except Exception as e:

        print(
            f"Error loading {filename}: {e}"
        )


# ============================================================
# COMBINE VALIDATION DATA
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION DATA PREPARED")
print("=" * 70)

if len(X_validation) == 0:

    raise ValueError(
        "No valid validation samples were found."
    )


X_validation = pd.concat(
    X_validation,
    ignore_index=True
)

y_validation = np.array(
    y_validation
)

print(
    f"Total valid samples: {len(y_validation)}"
)


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\nRunning predictions...")

predictions = []

total_samples = len(
    X_validation
)

for i in range(
    total_samples
):

    values = X_validation.iloc[i].values

    result = predict_traffic(
        values
    )

    predictions.append(
        result["attack_type"]
    )

    if (
        (i + 1) % 500 == 0
        or
        (i + 1) == total_samples
    ):

        print(
            f"Processed "
            f"{i + 1}/{total_samples} samples"
        )


y_pred = np.array(
    predictions
)

# Normalize actual labels
y_actual = np.array(
    [
        normalize_label(label)
        for label in y_validation
    ]
)


# ============================================================
# MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_actual,
    y_pred
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_actual,
    y_pred,
    zero_division=0
)


print("\n")
print("=" * 70)
print("MODEL EVALUATION RESULTS")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report")
print("-" * 70)

print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AI-BASED INTRUSION DETECTION SYSTEM\n"
    )

    file.write(
        "MODEL EVALUATION - CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        "Model: Random Forest\n"
    )

    file.write(
        f"Number of features: {len(feature_names)}\n"
    )

    file.write(
        "Number of classes: 15\n"
    )

    file.write(
        f"Total validation samples: "
        f"{len(y_actual)}\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    file.write(report)


print(
    "\nClassification report saved to:"
)

print(report_path)


# ============================================================
# CONFUSION MATRIX
# ============================================================

class_labels = sorted(
    set(y_actual) | set(y_pred)
)

cm = confusion_matrix(
    y_actual,
    y_pred,
    labels=class_labels
)


print("\nConfusion Matrix")
print("-" * 70)

print(cm)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 14)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_labels
)

disp.plot(
    ax=ax,
    xticks_rotation=90
)

plt.title(
    "AI-Based Intrusion Detection System\n"
    "Random Forest Confusion Matrix"
)

plt.tight_layout()


# ============================================================
# SAVE CONFUSION MATRIX GRAPH
# ============================================================

cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

print(
    "\nConfusion matrix saved to:"
)

print(cm_path)

plt.show()


# ============================================================
# SAVE COMPLETE EVALUATION SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "evaluation_results.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AI-BASED INTRUSION DETECTION SYSTEM\n"
    )

    file.write(
        "MODEL EVALUATION SUMMARY\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        "Model: Random Forest\n"
    )

    file.write(
        f"Number of features: "
        f"{len(feature_names)}\n"
    )

    file.write(
        "Number of classes: 15\n"
    )

    file.write(
        f"Validation samples: "
        f"{len(y_actual)}\n\n"
    )

    file.write(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(report)

    file.write(
        "\n\nConfusion Matrix\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        str(cm)
    )


print(
    "\nEvaluation summary saved to:"
)

print(summary_path)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nFinal Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Validation Samples: "
    f"{len(y_actual)}"
)

print(
    "\nSaved Results:"
)

print(
    "1. classification_report.txt"
)

print(
    "2. evaluation_results.txt"
)

print(
    "3. confusion_matrix.png"
)

print(
    "\nResults folder:"
)

print(RESULTS_DIR)