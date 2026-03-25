# -*- coding: utf-8 -*-

"""Generate SHAP explainability plots for the classical SVM proxy model.

This script loads the multi-disease VOC dataset and the trained classical SVM
model, projects samples into PCA space, and computes a SHAP waterfall plot for
one representative non-Healthy sample. The resulting figure is saved as
``shap_explanation.png`` for downstream use in reports or dashboards.
"""

import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def main() -> None:
    """Entry point for SHAP-based explainability on the classical SVM."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    logging.info("--- Running SHAP Explainability ---")

    # Load dataset and artifacts
    try:
        df = pd.read_csv("data/VOC_MultiDisease_Dataset.csv")
    except FileNotFoundError as exc:
        logging.error(
            "Could not find data/VOC_MultiDisease_Dataset.csv. "
            "Ensure the dataset is available and classical_svm.py has been run.",
        )
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.error("Failed to load dataset: %s", exc)
        raise SystemExit(1) from exc

    target_col = "Disease Label"

    try:
        feature_cols = joblib.load("feature_cols.pkl")
        scaler = joblib.load("scaler.pkl")
        pca = joblib.load("pca.pkl")
        svm = joblib.load("classical_svm_model.pkl")
    except FileNotFoundError as exc:
        logging.error(
            "Required artifacts not found (feature_cols.pkl, scaler.pkl, "
            "pca.pkl, classical_svm_model.pkl). Please run classical_svm.py first.",
        )
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.error("Failed to load explainability artifacts: %s", exc)
        raise SystemExit(1) from exc

    X = df[feature_cols]

    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    # Choose a representative non-Healthy sample for the waterfall plot
    non_healthy_mask = df[target_col] != "Healthy"
    if non_healthy_mask.any():
        sample_index = df[non_healthy_mask].index[0]
        logging.info(
            "Using non-Healthy sample at index %d for SHAP waterfall.",
            sample_index,
        )
    else:
        sample_index = df.index[0]
        logging.warning(
            "No non-Healthy samples found; using the first sample as fallback.",
        )

    sample = X_pca[sample_index : sample_index + 1]

    # Downsample background to speed up SHAP
    background = shap.kmeans(X_pca, min(10, len(X_pca)))
    explainer = shap.KernelExplainer(svm.predict, background)

    # Compute SHAP values
    shap_values = explainer.shap_values(sample)

    # Robustly extract scalar base value
    base_val = np.atleast_1d(explainer.expected_value)[0]

    exp = shap.Explanation(
        values=shap_values[0],
        base_values=base_val,
        data=sample[0],
        feature_names=[
            "PCA-VOC 1",
            "PCA-VOC 2",
            "PCA-VOC 3",
            "PCA-VOC 4",
            "PCA-VOC 5",
        ],
    )

    plt.figure(figsize=(8, 5))
    shap.waterfall_plot(exp, show=False)
    plt.title("Why did QNose flag this breath sample?")
    plt.tight_layout()
    plt.savefig("shap_explanation.png")
    logging.info("Saved shap_explanation.png")


if __name__ == "__main__":
    main()