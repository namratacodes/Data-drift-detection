"""
run_experiment.py
------------------
Phase 3: THE MAIN EXPERIMENT (now with 6 detection methods, including the
autoencoder-based deep learning detector).

For each window (1 through 5), tests 4 scenarios (no_drift control,
covariate_age, label_income, concept_education) using all 6 detectors:
KS, PSI, KL, Classifier_RF, Classifier_LR, and Autoencoder.

Results saved to results/experiment_results.csv.
"""

import pandas as pd

from drift_injection import (
    inject_covariate_drift,
    inject_label_drift,
    inject_concept_drift,
)
from drift_detectors import (
    ks_test_drift,
    calculate_psi,
    calculate_kl_divergence,
    classifier_drift_detector,
    autoencoder_drift_detector,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
N_WINDOWS = 6
REFERENCE_WINDOW_PATH = "data/window_0.csv"

NUMERIC_COL_FOR_COVARIATE_DRIFT = "age"
TARGET_COL = "income"
POSITIVE_CLASS = ">50K"
CONCEPT_DRIFT_CONDITION_COL = "education"
CONCEPT_DRIFT_CONDITION_VALUE = "Bachelors"

CLASSIFIER_FEATURE_COLS = ["age", "hours.per.week", "capital.gain", "capital.loss"]
# The autoencoder uses the categorical column too, since it can natively
# handle the one-hot encoded version via its own preprocessing step.
AUTOENCODER_FEATURE_COLS = ["age", "hours.per.week", "capital.gain", "capital.loss", "education"]

# IMPORTANT: this default was calibrated on synthetic validation data.
# Re-check it against YOUR dataset's no_drift control scores once this
# script has run once (see the printed summary at the end), and adjust
# if needed -- see the detailed note inside drift_detectors.py.
AUTOENCODER_EXCEEDANCE_THRESHOLD = 0.07
AUTOENCODER_EPOCHS = 50


def run_numeric_detectors(reference_col, comparison_col):
    return [
        ks_test_drift(reference_col, comparison_col),
        calculate_psi(reference_col, comparison_col),
        calculate_kl_divergence(reference_col, comparison_col),
    ]


def run_categorical_detectors(reference_col, comparison_col):
    return [
        calculate_psi(reference_col, comparison_col),
        calculate_kl_divergence(reference_col, comparison_col),
    ]


def evaluate_window(reference_df, comparison_df, window_id, scenario,
                     ground_truth_drift, drift_metadata=None):
    rows = []

    for result in run_numeric_detectors(reference_df[NUMERIC_COL_FOR_COVARIATE_DRIFT],
                                         comparison_df[NUMERIC_COL_FOR_COVARIATE_DRIFT]):
        rows.append({
            "window_id": window_id, "scenario": scenario,
            "column_checked": NUMERIC_COL_FOR_COVARIATE_DRIFT,
            "method": result["method"], "score": result["score"],
            "detected_drift": result["drifted"], "ground_truth_drift": ground_truth_drift,
            "drift_metadata": drift_metadata,
        })

    for result in run_categorical_detectors(reference_df[CONCEPT_DRIFT_CONDITION_COL],
                                             comparison_df[CONCEPT_DRIFT_CONDITION_COL]):
        rows.append({
            "window_id": window_id, "scenario": scenario,
            "column_checked": CONCEPT_DRIFT_CONDITION_COL,
            "method": result["method"], "score": result["score"],
            "detected_drift": result["drifted"], "ground_truth_drift": ground_truth_drift,
            "drift_metadata": drift_metadata,
        })

    for model_type in ["random_forest", "logistic_regression"]:
        clf_result = classifier_drift_detector(reference_df, comparison_df, CLASSIFIER_FEATURE_COLS,
                                                 model_type=model_type)
        rows.append({
            "window_id": window_id, "scenario": scenario, "column_checked": "multi_feature",
            "method": clf_result["method"], "score": clf_result["score"],
            "detected_drift": clf_result["drifted"], "ground_truth_drift": ground_truth_drift,
            "drift_metadata": drift_metadata,
        })

    # --- Autoencoder-based (deep learning) check, multi-feature ---
    ae_result = autoencoder_drift_detector(
        reference_df, comparison_df, AUTOENCODER_FEATURE_COLS,
        epochs=AUTOENCODER_EPOCHS, exceedance_threshold=AUTOENCODER_EXCEEDANCE_THRESHOLD,
    )
    rows.append({
        "window_id": window_id, "scenario": scenario, "column_checked": "multi_feature",
        "method": ae_result["method"], "score": ae_result["score"],
        "detected_drift": ae_result["drifted"], "ground_truth_drift": ground_truth_drift,
        "drift_metadata": drift_metadata,
    })

    return rows


def main():
    reference_df = pd.read_csv(REFERENCE_WINDOW_PATH)
    all_results = []

    for i in range(1, N_WINDOWS):
        window_df = pd.read_csv(f"data/window_{i}.csv")
        print(f"\n=== Processing window_{i} ===")

        print("  -> scenario: no_drift (control)")
        all_results.extend(
            evaluate_window(reference_df, window_df, window_id=i,
                             scenario="no_drift", ground_truth_drift=False)
        )

        print("  -> scenario: covariate drift (age)")
        drifted_cov, meta_cov = inject_covariate_drift(
            window_df, column=NUMERIC_COL_FOR_COVARIATE_DRIFT, shift_std=1.5
        )
        all_results.extend(
            evaluate_window(reference_df, drifted_cov, window_id=i,
                             scenario="covariate_age", ground_truth_drift=True,
                             drift_metadata=meta_cov)
        )

        print("  -> scenario: label drift (income)")
        drifted_label, meta_label = inject_label_drift(
            window_df, target_col=TARGET_COL, target_value=POSITIVE_CLASS,
            new_positive_rate=0.6
        )
        all_results.extend(
            evaluate_window(reference_df, drifted_label, window_id=i,
                             scenario="label_income", ground_truth_drift=True,
                             drift_metadata=meta_label)
        )

        print("  -> scenario: concept drift (education)")
        drifted_concept, meta_concept = inject_concept_drift(
            window_df, target_col=TARGET_COL,
            condition_col=CONCEPT_DRIFT_CONDITION_COL,
            condition_value=CONCEPT_DRIFT_CONDITION_VALUE,
            flip_fraction=0.5
        )
        all_results.extend(
            evaluate_window(reference_df, drifted_concept, window_id=i,
                             scenario="concept_education", ground_truth_drift=True,
                             drift_metadata=meta_concept)
        )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/experiment_results.csv", index=False)
    print(f"\nSaved {len(results_df)} result rows to results/experiment_results.csv")

    summary = results_df.groupby(["method", "scenario"])["detected_drift"].mean()
    print("\n=== Quick summary: detection rate by method x scenario ===")
    print(summary)

    # Flag the autoencoder's no_drift scores specifically, since this is
    # exactly the data you should use to re-calibrate AUTOENCODER_EXCEEDANCE_THRESHOLD
    # for your real dataset (see the note in drift_detectors.py).
    ae_nodrift_scores = results_df[
        (results_df["method"] == "Autoencoder") & (results_df["scenario"] == "no_drift")
    ]["score"]
    print(f"\nAutoencoder no_drift scores (use these to sanity-check the {AUTOENCODER_EXCEEDANCE_THRESHOLD} threshold):")
    print(ae_nodrift_scores.describe())


if __name__ == "__main__":
    main()