"""
drift_detectors.py
Six drift detection methods: KS-test, PSI, KL Divergence, Classifier-based
(Random Forest and Logistic Regression variants), and an Autoencoder-based
deep learning detector.
Each function compares a reference window to a comparison window and returns:
    {"score": float, "drifted": bool, "method": str}
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TensorFlow's startup noise
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import logging
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# 1. KS-TEST (numeric features only)
# ---------------------------------------------------------------------------
def ks_test_drift(reference: pd.Series, comparison: pd.Series, alpha: float = 0.05):
    reference = reference.dropna()
    comparison = comparison.dropna()
    stat, p_value = ks_2samp(reference, comparison)
    return {
        "method": "KS",
        "score": float(stat),
        "p_value": float(p_value),
        "drifted": bool(p_value < alpha),
    }


# ---------------------------------------------------------------------------
# 2. PSI (Population Stability Index)
# ---------------------------------------------------------------------------
def calculate_psi(reference: pd.Series, comparison: pd.Series, buckets: int = 10,
                   threshold: float = 0.2):
    reference = reference.dropna()
    comparison = comparison.dropna()

    if not pd.api.types.is_numeric_dtype(reference):
        categories = set(reference.unique()) | set(comparison.unique())
        ref_counts = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
        comp_counts = comparison.value_counts(normalize=True).reindex(categories, fill_value=0)
    else:
        breakpoints = np.quantile(reference, np.linspace(0, 1, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        breakpoints = np.unique(breakpoints)

        ref_binned = pd.cut(reference, bins=breakpoints)
        comp_binned = pd.cut(comparison, bins=breakpoints)

        ref_counts = ref_binned.value_counts(normalize=True).sort_index()
        comp_counts = comp_binned.value_counts(normalize=True).sort_index()

    eps = 1e-4
    ref_counts = ref_counts.replace(0, eps)
    comp_counts = comp_counts.replace(0, eps)

    psi_value = np.sum((comp_counts - ref_counts) * np.log(comp_counts / ref_counts))

    return {
        "method": "PSI",
        "score": float(psi_value),
        "drifted": bool(psi_value > threshold),
    }


# ---------------------------------------------------------------------------
# 3. KL DIVERGENCE
# ---------------------------------------------------------------------------
def calculate_kl_divergence(reference: pd.Series, comparison: pd.Series, buckets: int = 10,
                             threshold: float = 0.1):
    reference = reference.dropna()
    comparison = comparison.dropna()

    if not pd.api.types.is_numeric_dtype(reference):
        categories = set(reference.unique()) | set(comparison.unique())
        p = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
        q = comparison.value_counts(normalize=True).reindex(categories, fill_value=0)
    else:
        breakpoints = np.quantile(reference, np.linspace(0, 1, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        breakpoints = np.unique(breakpoints)

        p = pd.cut(reference, bins=breakpoints).value_counts(normalize=True).sort_index()
        q = pd.cut(comparison, bins=breakpoints).value_counts(normalize=True).sort_index()

    eps = 1e-4
    p = p.replace(0, eps)
    q = q.replace(0, eps)

    kl_value = np.sum(p * np.log(p / q))

    return {
        "method": "KL",
        "score": float(kl_value),
        "drifted": bool(kl_value > threshold),
    }


# ---------------------------------------------------------------------------
# 4. CLASSIFIER-BASED DRIFT DETECTOR (Random Forest / Logistic Regression)
# ---------------------------------------------------------------------------
def classifier_drift_detector(reference_df: pd.DataFrame, comparison_df: pd.DataFrame,
                               feature_cols: list, auc_threshold: float = 0.6,
                               random_state: int = 42, model_type: str = "random_forest"):
    ref = reference_df[feature_cols].copy()
    comp = comparison_df[feature_cols].copy()

    ref["__label__"] = 0
    comp["__label__"] = 1

    combined = pd.concat([ref, comp], axis=0, ignore_index=True)
    categorical_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(combined[c])]
    combined = pd.get_dummies(combined, columns=categorical_cols)

    X = combined.drop(columns="__label__")
    y = combined["__label__"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=6)
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        method_name = "Classifier_RF"

    elif model_type == "logistic_regression":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X_train_scaled, y_train)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        method_name = "Classifier_LR"

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'random_forest' or 'logistic_regression'.")

    auc = roc_auc_score(y_test, y_proba)

    return {
        "method": method_name,
        "score": float(auc),
        "drifted": bool(auc > auc_threshold),
    }


# ---------------------------------------------------------------------------
# 5. AUTOENCODER-BASED DRIFT DETECTOR (deep learning)
# ---------------------------------------------------------------------------
def _preprocess_for_autoencoder(reference_df: pd.DataFrame, comparison_df: pd.DataFrame,
                                 feature_cols: list):
    """
    Shared preprocessing for the autoencoder: one-hot encode categoricals
    (columns fixed by the REFERENCE window's categories only -- a category
    that appears in comparison but not reference is intentionally left as
    all-zeros, since the autoencoder has never learned that category and
    SHOULD find it unfamiliar), then scale numeric columns using a scaler
    fit on the reference window only.
    """
    ref = reference_df[feature_cols].copy()
    comp = comparison_df[feature_cols].copy()

    categorical_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(ref[c])]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    if categorical_cols:
        ref_dummies = pd.get_dummies(ref[categorical_cols])
        comp_dummies = pd.get_dummies(comp[categorical_cols])
        # Align comparison's dummy columns to exactly match reference's
        # (new/unseen categories in comparison become all-zero rows)
        comp_dummies = comp_dummies.reindex(columns=ref_dummies.columns, fill_value=0)
    else:
        ref_dummies = pd.DataFrame(index=ref.index)
        comp_dummies = pd.DataFrame(index=comp.index)

    scaler = StandardScaler()
    ref_numeric_scaled = scaler.fit_transform(ref[numeric_cols]) if numeric_cols else np.empty((len(ref), 0))
    comp_numeric_scaled = scaler.transform(comp[numeric_cols]) if numeric_cols else np.empty((len(comp), 0))

    X_ref = np.hstack([ref_numeric_scaled, ref_dummies.values.astype(float)])
    X_comp = np.hstack([comp_numeric_scaled, comp_dummies.values.astype(float)])

    return X_ref, X_comp


def _build_autoencoder(input_dim: int, random_state: int = 42):
    """
    A small, standard dense autoencoder: compresses input_dim -> 8 -> 4
    (the 'bottleneck') -> 8 -> input_dim. Forcing data through this narrow
    bottleneck means the network can only reconstruct data well if it has
    genuinely learned the PATTERNS in the reference data -- not just
    memorized individual rows.
    """
    tf.random.set_seed(random_state)

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(4, activation="relu", name="bottleneck"),
        layers.Dense(8, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(input_dim, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def _single_autoencoder_run(reference_df, comparison_df, feature_cols, epochs, batch_size,
                             error_percentile, calibration_fraction, random_state, verbose):
    """One trained autoencoder -> one exceedance_fraction. Used internally by
    autoencoder_drift_detector(), which averages several of these runs."""
    ref_train_df, ref_calib_df = train_test_split(
        reference_df, test_size=calibration_fraction, random_state=random_state
    )

    X_ref_train, X_comp = _preprocess_for_autoencoder(ref_train_df, comparison_df, feature_cols)
    X_ref_train2, X_ref_calib = _preprocess_for_autoencoder(ref_train_df, ref_calib_df, feature_cols)

    autoencoder = _build_autoencoder(input_dim=X_ref_train.shape[1], random_state=random_state)
    autoencoder.fit(
        X_ref_train, X_ref_train,
        epochs=epochs, batch_size=batch_size, shuffle=True,
        validation_split=0.1, verbose=verbose,
    )

    calib_reconstructed = autoencoder.predict(X_ref_calib, verbose=0)
    comp_reconstructed = autoencoder.predict(X_comp, verbose=0)

    calib_errors = np.mean(np.square(X_ref_calib - calib_reconstructed), axis=1)
    comp_errors = np.mean(np.square(X_comp - comp_reconstructed), axis=1)

    threshold = np.percentile(calib_errors, error_percentile)
    return float(np.mean(comp_errors > threshold))


def autoencoder_drift_detector(reference_df: pd.DataFrame, comparison_df: pd.DataFrame,
                                feature_cols: list, epochs: int = 50, batch_size: int = 32,
                                error_percentile: float = 95, exceedance_threshold: float = 0.07,
                                calibration_fraction: float = 0.3, n_ensemble: int = 3,
                                random_state: int = 42, verbose: int = 0):
    """
    Deep-learning drift detector based on autoencoder reconstruction error.

    HOW IT WORKS:
    1. The reference window is split into a TRAIN portion and a held-out
       CALIBRATION portion. The autoencoder is trained only on the train
       portion -- it never sees the calibration portion OR the comparison
       window during training.
    2. We measure reconstruction error on the held-out CALIBRATION portion
       (data the network has never seen, but which is still genuinely
       "normal" reference data) and set our threshold at its
       `error_percentile`-th percentile (default: 95th). This is a more
       honest baseline than measuring error on the training data itself,
       which a network tends to reconstruct artificially well since it
       has already memorized it -- using held-out calibration data avoids
       this optimistic bias.
    3. By construction, ~5% of the calibration data naturally exceeds this
       threshold (that's what "95th percentile" means).
    4. We then check what fraction of the COMPARISON window exceeds that
       same threshold. If the data hasn't drifted, this exceedance
       fraction should sit close to the ~5% implied by the percentile
       choice (empirically it lands around 0.05-0.06 on validation
       testing). If it's meaingfully higher than that baseline -- above
       `exceedance_threshold` -- that signals the comparison data no
       longer matches the patterns the autoencoder learned from the
       reference window, i.e. drift.

       IMPORTANT -- calibrating exceedance_threshold for YOUR dataset:
       the default (0.07) was set empirically during validation testing,
       where no-drift comparisons scored ~0.05-0.06 and genuinely
       drifted comparisons scored ~0.07-0.11. This gap is dataset- and
       drift-magnitude-specific. Your existing experiment pipeline
       already runs a "no_drift" control scenario for every window
       (see run_experiment.py) -- use the autoencoder scores from THOSE
       control runs to empirically re-calibrate this threshold for your
       actual dataset, the same way the 0.07 default was derived here,
       rather than assuming this default transfers unchanged. This is
       worth describing explicitly as your calibration methodology.

    Returns the same {"method", "score", "drifted"} shape as every other
    detector in this file, so it drops directly into the existing
    experiment pipeline (run_experiment.py) as a sixth method.

    Note on reproducibility: a single neural network carries noticeable
    run-to-run variance from random weight initialization and training
    stochasticity -- especially on small / moderate-sized tabular data
    like this. To address this, this detector trains an ENSEMBLE of
    `n_ensemble` independently-initialized autoencoders (default 3) and
    averages their exceedance fractions, which substantially stabilizes
    the final score. This ensembling choice is itself worth describing
    explicitly in your paper's methodology section.
    """
    exceedance_fractions = []
    for i in range(n_ensemble):
        seed_i = random_state + i
        np.random.seed(seed_i)
        tf.random.set_seed(seed_i)
        frac = _single_autoencoder_run(
            reference_df, comparison_df, feature_cols, epochs, batch_size,
            error_percentile, calibration_fraction, seed_i, verbose,
        )
        exceedance_fractions.append(frac)

    avg_exceedance_fraction = float(np.mean(exceedance_fractions))

    return {
        "method": "Autoencoder",
        "score": avg_exceedance_fraction,   # e.g. 0.15 = 15% of new data looked "unfamiliar", averaged over the ensemble
        "drifted": bool(avg_exceedance_fraction > exceedance_threshold),
        "ensemble_scores": exceedance_fractions,  # individual runs, useful for reporting variance
    }


# ---------------------------------------------------------------------------
# Convenience wrapper: run KS/PSI/KL on one numeric/categorical column
# ---------------------------------------------------------------------------
def run_all_detectors_on_column(reference_df, comparison_df, column):
    ref_col = reference_df[column]
    comp_col = comparison_df[column]

    results = [
        ks_test_drift(ref_col, comp_col) if pd.api.types.is_numeric_dtype(ref_col) else None,
        calculate_psi(ref_col, comp_col),
        calculate_kl_divergence(ref_col, comp_col),
    ]
    return [r for r in results if r is not None]