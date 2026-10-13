
# ============================================================
# AI-BASED INTRUSION DETECTION SYSTEM
# Prediction Module
# Optimized Single + Batch Prediction
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PACKAGE_PATH = os.path.join(
    BASE_DIR,
    "ids_model_package.pkl"
)


# ============================================================
# LOAD MODEL PACKAGE
# ============================================================

print("Loading model package...")

package = joblib.load(PACKAGE_PATH)

model = package["model"]
scaler = package["scaler"]
feature_names = package["feature_names"]
class_mapping = package["class_mapping"]

print("Model package loaded successfully.")


# ============================================================
# MODEL CPU CONTROL
# ============================================================

try:

    if hasattr(model, "n_jobs"):

        model.set_params(
            n_jobs=1
        )

        print(
            "Prediction workers: 1"
        )

except Exception:

    pass


# ============================================================
# PREDICTION SETTINGS
# ============================================================

BATCH_SIZE = 10000


# ============================================================
# RISK LEVEL FUNCTION
# ============================================================

def get_risk_level(attack_name):

    if attack_name == "BENIGN":

        return "Low"


    critical_attacks = [

        "Bot",

        "Web Attack - Sql Injection"

    ]


    high_attacks = [

        "DDoS",

        "DoS Hulk",

        "DoS GoldenEye",

        "DoS slowloris",

        "DoS Slowhttptest",

        "Infiltration",

        "Heartbleed"

    ]


    if attack_name in critical_attacks:

        return "Critical"


    if attack_name in high_attacks:

        return "High"


    return "Medium"


# ============================================================
# GET ATTACK NAME
# ============================================================

def get_attack_name(prediction):

    prediction = int(
        prediction
    )


    attack_name = class_mapping.get(
        str(prediction)
    )


    if attack_name is None:

        attack_name = class_mapping.get(
            prediction,
            "Unknown"
        )


    return attack_name


# ============================================================
# SINGLE TRAFFIC PREDICTION
# ============================================================

def predict_traffic(values):

    data = np.asarray(
        values,
        dtype=float
    )


    if data.size != len(feature_names):

        raise ValueError(

            f"Expected "
            f"{len(feature_names)} features, "
            f"but received "
            f"{data.size}."

        )


    data = data.reshape(
        1,
        -1
    )


    data_df = pd.DataFrame(

        data,

        columns=feature_names

    )


    data_df = data_df.replace(

        [np.inf, -np.inf],

        np.nan

    )


    if data_df.isna().any().any():

        raise ValueError(

            "Input contains missing or invalid values."

        )


    # IMPORTANT:
    # Keep DataFrame feature names when scaling.
    # This prevents the sklearn feature-name warning.

    data_scaled = scaler.transform(
        data_df
    )


    prediction = model.predict(
        data_scaled
    )[0]


    prediction = int(
        prediction
    )


    attack_name = get_attack_name(
        prediction
    )


    risk_level = get_risk_level(
        attack_name
    )


    return {

        "class_id":
            prediction,

        "attack_type":
            attack_name,

        "risk_level":
            risk_level

    }


# ============================================================
# INTERNAL CHUNK PREDICTION
# ============================================================

def _predict_chunk(data_df):

    # --------------------------------------------------------
    # SELECT FEATURES IN EXACT MODEL ORDER
    # --------------------------------------------------------

    data_df = data_df[
        feature_names
    ].copy()


    # --------------------------------------------------------
    # CONVERT TO NUMERIC
    # --------------------------------------------------------

    data_df = data_df.apply(

        pd.to_numeric,

        errors="coerce"

    )


    # --------------------------------------------------------
    # HANDLE INFINITE VALUES
    # --------------------------------------------------------

    data_df = data_df.replace(

        [np.inf, -np.inf],

        np.nan

    )


    # --------------------------------------------------------
    # CHECK INVALID VALUES
    # --------------------------------------------------------

    if data_df.isna().any().any():

        raise ValueError(

            "Input contains missing or invalid values."

        )


    # --------------------------------------------------------
    # SCALE DATA
    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT convert the DataFrame to NumPy here.
    #
    # The scaler was fitted using feature names.
    # Passing the DataFrame preserves those names and
    # removes the sklearn warning.

    data_scaled = scaler.transform(
        data_df
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(
        data_scaled
    )


    return predictions


# ============================================================
# BATCH TRAFFIC PREDICTION
# ============================================================

def predict_batch(

    data,

    batch_size=BATCH_SIZE,

    show_progress=True

):

    """
    Predict multiple network traffic records
    using memory-safe chunks.

    Parameters
    ----------
    data : pandas DataFrame or numpy array

    batch_size : int
        Number of rows processed at one time.

    show_progress : bool
        Display prediction progress.

    Returns
    -------
    list of dictionaries
    """


    # --------------------------------------------------------
    # PREPARE INPUT
    # --------------------------------------------------------

    if isinstance(
        data,
        pd.DataFrame
    ):

        data_df = data.copy()


    else:

        data_array = np.asarray(

            data,

            dtype=float

        )


        data_df = pd.DataFrame(

            data_array,

            columns=feature_names

        )


    # --------------------------------------------------------
    # CHECK FEATURE COUNT
    # --------------------------------------------------------

    if data_df.shape[1] != len(feature_names):

        raise ValueError(

            f"Expected "
            f"{len(feature_names)} features, "
            f"but received "
            f"{data_df.shape[1]}."

        )


    # --------------------------------------------------------
    # RESTORE EXACT FEATURE NAMES
    # --------------------------------------------------------

    data_df.columns = feature_names


    # --------------------------------------------------------
    # RESET INDEX
    # --------------------------------------------------------

    data_df = data_df.reset_index(
        drop=True
    )


    total_rows = len(
        data_df
    )


    if total_rows == 0:

        return []


    # --------------------------------------------------------
    # CALCULATE NUMBER OF BATCHES
    # --------------------------------------------------------

    total_batches = (

        total_rows
        + batch_size
        - 1

    ) // batch_size


    if show_progress:

        print()

        print(
            "=" * 60
        )

        print(
            "STARTING BATCH PREDICTION"
        )

        print(
            "=" * 60
        )

        print(

            f"Total records : "
            f"{total_rows:,}"

        )

        print(

            f"Batch size    : "
            f"{batch_size:,}"

        )

        print(

            f"Total batches : "
            f"{total_batches:,}"

        )

        print(
            "=" * 60
        )


    # --------------------------------------------------------
    # STORE PREDICTIONS
    # --------------------------------------------------------

    all_predictions = []


    # --------------------------------------------------------
    # PROCESS DATA IN CHUNKS
    # --------------------------------------------------------

    for batch_number, start in enumerate(

        range(

            0,

            total_rows,

            batch_size

        ),

        start=1

    ):


        end = min(

            start + batch_size,

            total_rows

        )


        # ----------------------------------------------------
        # GET CURRENT CHUNK
        # ----------------------------------------------------

        chunk = data_df.iloc[

            start:end

        ].copy()


        # ----------------------------------------------------
        # PREDICT CURRENT CHUNK
        # ----------------------------------------------------

        predictions = _predict_chunk(

            chunk

        )


        # ----------------------------------------------------
        # STORE PREDICTIONS
        # ----------------------------------------------------

        all_predictions.extend(

            predictions.tolist()

        )


        # ----------------------------------------------------
        # SHOW PROGRESS
        # ----------------------------------------------------

        if show_progress:

            print(

                f"Processed "
                f"{end:,}/{total_rows:,} "
                f"records "
                f"({batch_number}/{total_batches})"

            )


    # --------------------------------------------------------
    # CREATE RESULT LIST
    # --------------------------------------------------------

    results = []


    for prediction in all_predictions:

        prediction = int(
            prediction
        )


        attack_name = get_attack_name(

            prediction

        )


        risk_level = get_risk_level(

            attack_name

        )


        results.append({

            "class_id":
                prediction,

            "attack_type":
                attack_name,

            "risk_level":
                risk_level

        })


    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    if show_progress:

        print(
            "=" * 60
        )

        print(
            "BATCH PREDICTION COMPLETED"
        )

        print(

            f"Total predictions: "
            f"{len(results):,}"

        )

        print(
            "=" * 60
        )

        print()


    return results


# ============================================================
# MODULE INFORMATION
# ============================================================

print()

print(
    "=" * 60
)

print(
    "PREDICTION MODULE LOADED"
)

print(
    "=" * 60
)


print(

    "Model:",

    type(model).__name__

)


print(

    "Features:",

    len(feature_names)

)


print(

    "Classes:",

    len(class_mapping)

)


print(

    "Batch size:",

    BATCH_SIZE

)


print(
    "=" * 60
)


# ============================================================
# TEST PREDICTION
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=" * 60
    )

    print(
        "PREDICTION MODULE TEST"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # SINGLE PREDICTION TEST
    # ========================================================

    test_values = np.zeros(

        len(feature_names)

    )


    result = predict_traffic(

        test_values

    )


    print()

    print(
        "Single Prediction"
    )

    print(
        "-" * 30
    )


    print(

        "Class ID:",

        result["class_id"]

    )


    print(

        "Attack Type:",

        result["attack_type"]

    )


    print(

        "Risk Level:",

        result["risk_level"]

    )


    # ========================================================
    # BATCH PREDICTION TEST
    # ========================================================

    test_batch = np.zeros(

        (

            5,

            len(feature_names)

        )

    )


    batch_results = predict_batch(

        test_batch,

        batch_size=5,

        show_progress=True

    )


    print()

    print(
        "Batch Prediction"
    )

    print(
        "-" * 30
    )


    print(

        "Records:",

        len(batch_results)

    )


    for i, result in enumerate(

        batch_results,

        start=1

    ):

        print(

            f"Record {i}: "

            f"{result['attack_type']} "

            f"({result['risk_level']})"

        )


    print()

    print(
        "=" * 60
    )

    print(
        "PREDICTION MODULE TEST COMPLETED"
    )

    print(
        "=" * 60
    )
