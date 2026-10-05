"""
analyze_results.py
--------------------
Phase 4: ANALYSIS. Now updated for 6 detection methods (KS, PSI, KL,
Classifier_RF, Classifier_LR, Autoencoder).

Produces:
    1. Detection performance table (per method): precision, recall, FPR,
       overall accuracy vs. ground truth -- works automatically for any
       number of methods present in experiment_results.csv.
    2. A detection-rate heatmap (method x scenario) -- now 6 rows instead
       of 5.
    3. A model-accuracy bar chart (model x scenario).
    4. Score-vs-accuracy-drop analysis for BOTH a classical method
       (Classifier_RF) AND the deep learning method (Autoencoder),
       side by side -- so you can directly compare whether the DL
       detector's scores track real damage any better or worse than the
       classical classifier-based detector's scores do.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_style("whitegrid")

# Both detectors we compare against real accuracy. Classifier_RF is the
# strongest classical method from Phase 3; Autoencoder is the new deep
# learning method -- comparing both tells us whether going deep-learning
# actually improves on the classical result, or not.
DETECTORS_TO_COMPARE = ["Classifier_RF", "Autoencoder"]


# ---------------------------------------------------------------------------
# 1. DETECTION PERFORMANCE (works for any number of methods automatically)
# ---------------------------------------------------------------------------
def compute_detection_performance(detection_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in detection_df.groupby("method"):
        tp = ((group["detected_drift"] == True) & (group["ground_truth_drift"] == True)).sum()
        fp = ((group["detected_drift"] == True) & (group["ground_truth_drift"] == False)).sum()
        fn = ((group["detected_drift"] == False) & (group["ground_truth_drift"] == True)).sum()
        tn = ((group["detected_drift"] == False) & (group["ground_truth_drift"] == False)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        overall_accuracy = (tp + tn) / len(group) if len(group) > 0 else float("nan")

        rows.append({
            "method": method,
            "precision": round(precision, 3),
            "recall_TPR": round(recall, 3),
            "false_positive_rate": round(fpr, 3),
            "overall_accuracy": round(overall_accuracy, 3),
            "n_observations": len(group),
        })

    return pd.DataFrame(rows).sort_values("overall_accuracy", ascending=False)


def plot_detection_heatmap(detection_df: pd.DataFrame):
    pivot = detection_df.groupby(["method", "scenario"])["detected_drift"].mean().unstack()

    plt.figure(figsize=(9, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0, vmax=1,
                cbar_kws={"label": "Detection Rate"})
    plt.title("Drift Detection Rate by Method x Scenario (6 methods)")
    plt.ylabel("Detection Method")
    plt.xlabel("Drift Scenario")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/detection_rate_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR}/detection_rate_heatmap.png")


# ---------------------------------------------------------------------------
# 2. MODEL PERFORMANCE UNDER DRIFT (unchanged logic)
# ---------------------------------------------------------------------------
def plot_model_accuracy_by_scenario(model_df: pd.DataFrame):
    plt.figure(figsize=(9, 5))
    sns.barplot(data=model_df, x="scenario", y="accuracy", hue="model")
    plt.title("Model Accuracy by Drift Scenario")
    plt.ylabel("Accuracy")
    plt.xlabel("Scenario")
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_accuracy_by_scenario.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR}/model_accuracy_by_scenario.png")


# ---------------------------------------------------------------------------
# 3. SCORE VS ACCURACY DROP -- now runs once per detector in DETECTORS_TO_COMPARE
# ---------------------------------------------------------------------------
def build_merged_score_vs_accuracy_table(detection_df: pd.DataFrame, model_df: pd.DataFrame,
                                          detector_method: str) -> pd.DataFrame:
    scores = (
        detection_df[detection_df["method"] == detector_method]
        .groupby(["window_id", "scenario"])["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "drift_score", "window_id": "window"})
    )

    accuracy = (
        model_df
        .groupby(["window", "scenario"])["accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"accuracy": "avg_accuracy"})
    )

    merged = pd.merge(scores, accuracy, on=["window", "scenario"], how="inner")
    baseline_acc = model_df[model_df["scenario"] == "no_drift"]["accuracy"].mean()
    merged["accuracy_drop"] = baseline_acc - merged["avg_accuracy"]
    merged["detector"] = detector_method
    return merged


def plot_score_vs_accuracy_drop(merged_df: pd.DataFrame, detector_method: str):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=merged_df, x="drift_score", y="accuracy_drop", hue="scenario", s=100)
    plt.title(f"{detector_method} Drift Score vs. Model Accuracy Drop")
    plt.xlabel(f"{detector_method} Drift Score")
    plt.ylabel("Accuracy Drop from Baseline")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.legend(title="Scenario", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    safe_name = detector_method.lower()
    plt.savefig(f"{PLOTS_DIR}/drift_score_vs_accuracy_drop_{safe_name}.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR}/drift_score_vs_accuracy_drop_{safe_name}.png")


def plot_combined_detector_comparison(all_merged: pd.DataFrame):
    """
    Side-by-side comparison: does the deep learning detector's score
    track real accuracy damage any better than the classical
    classifier-based detector's score does? One scatter panel per
    detector, sharing the same y-axis so they're directly comparable.
    """
    detectors = all_merged["detector"].unique()
    fig, axes = plt.subplots(1, len(detectors), figsize=(7 * len(detectors), 6), sharey=True)
    if len(detectors) == 1:
        axes = [axes]

    for ax, detector in zip(axes, detectors):
        subset = all_merged[all_merged["detector"] == detector]
        sns.scatterplot(data=subset, x="drift_score", y="accuracy_drop", hue="scenario", s=100, ax=ax, legend=(ax is axes[-1]))
        ax.set_title(f"{detector}")
        ax.set_xlabel(f"{detector} Drift Score")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Accuracy Drop from Baseline")
    if len(detectors) > 1:
        axes[-1].legend(title="Scenario", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.suptitle("Classical vs. Deep Learning Detector: Score vs. Real Accuracy Drop")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/classical_vs_dl_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR}/classical_vs_dl_comparison.png")


def main():
    detection_df = pd.read_csv(f"{RESULTS_DIR}/experiment_results.csv")
    model_df = pd.read_csv(f"{RESULTS_DIR}/model_metrics.csv")

    # --- 1. Detection performance table (now covers all 6 methods automatically) ---
    perf_table = compute_detection_performance(detection_df)
    perf_table.to_csv(f"{RESULTS_DIR}/detection_performance_summary.csv", index=False)
    print("=== Detection Performance by Method (6 methods) ===")
    print(perf_table.to_string(index=False))
    print(f"\nSaved {RESULTS_DIR}/detection_performance_summary.csv\n")

    plot_detection_heatmap(detection_df)

    # --- 2. Model accuracy under drift ---
    plot_model_accuracy_by_scenario(model_df)
    print("\n=== Average Model Accuracy by Scenario ===")
    print(model_df.groupby(["model", "scenario"])["accuracy"].mean().round(4).to_string())

    # --- 3. Score vs accuracy drop, for EACH detector being compared ---
    all_merged = []
    all_scenario_summaries = []

    for detector_method in DETECTORS_TO_COMPARE:
        if detector_method not in detection_df["method"].unique():
            print(f"\nSkipping {detector_method} -- not found in experiment_results.csv")
            continue

        merged = build_merged_score_vs_accuracy_table(detection_df, model_df, detector_method)
        all_merged.append(merged)
        plot_score_vs_accuracy_drop(merged, detector_method)

        scenario_summary = (
            merged.groupby("scenario")[["drift_score", "accuracy_drop"]]
            .mean().round(4).reset_index()
        )
        scenario_summary["detector"] = detector_method
        all_scenario_summaries.append(scenario_summary)

        correlation = merged["drift_score"].corr(merged["accuracy_drop"])
        print(f"\n[{detector_method}] Pooled correlation (drift score vs accuracy drop): {correlation:.3f}")
        print(f"[{detector_method}] Scenario-level breakdown:")
        print(scenario_summary.to_string(index=False))

    if all_merged:
        combined_df = pd.concat(all_merged, ignore_index=True)
        combined_df.to_csv(f"{RESULTS_DIR}/drift_score_vs_accuracy.csv", index=False)
        plot_combined_detector_comparison(combined_df)

    if all_scenario_summaries:
        combined_summary = pd.concat(all_scenario_summaries, ignore_index=True)
        combined_summary.to_csv(f"{RESULTS_DIR}/scenario_score_vs_accuracy_summary.csv", index=False)
        print(f"\nSaved {RESULTS_DIR}/scenario_score_vs_accuracy_summary.csv")

    # --- Key finding: concept drift blind spot, now checked across ALL methods ---
    print("\n--- Key finding: concept drift detection rate, by method ---")
    concept_by_method = (
        detection_df[detection_df["scenario"] == "concept_education"]
        .groupby("method")["detected_drift"].mean()
    )
    print(concept_by_method.to_string())
    concept_accuracy_drop = model_df[model_df["scenario"] == "no_drift"]["accuracy"].mean() - \
        model_df[model_df["scenario"] == "concept_education"]["accuracy"].mean()
    print(f"\nConcept drift (education) average real accuracy drop: {concept_accuracy_drop:.3f}")
    print("If every method above shows ~0% detection despite a real accuracy drop,")
    print("this confirms the blind spot is structural -- it persists even for the")
    print("deep learning detector, not just the classical statistical methods.")

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()