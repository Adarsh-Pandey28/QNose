# -*- coding: utf-8 -*-

"""Train classical multi-disease models and persist artifacts.

This script loads the VOC_MultiDisease dataset, performs basic preprocessing
(feature selection, standardization, PCA), trains several classical
classifiers (SVM, Random Forest, XGBoost), and saves all artifacts needed by
other components of the QNose project (including label encoders, PCA models,
and evaluation predictions).
"""

import logging
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def main() -> None:
    """Entry point for training classical models and saving artifacts."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    logging.info("--- Running Classical Multi-Disease SVM pipeline ---")

    # Use the multi-disease local dataset
    try:
        df = pd.read_csv("data/VOC_MultiDisease_Dataset.csv")
    except FileNotFoundError as exc:
        logging.error(
            "Could not find data/VOC_MultiDisease_Dataset.csv. "
            "Ensure the VOC dataset is available before running classical_svm.py.",
        )
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.error("Failed to load dataset: %s", exc)
        raise SystemExit(1) from exc

    # Ensure there are no id/text label columns leaking into features
    target_col = "Disease Label"

    # Filter out obvious non-numeric identifiers or label-like columns
    exclude_keywords = ["id", "label"]
    feature_cols = [
        c
        for c in df.columns
        if not any(x in c.lower() for x in exclude_keywords)
        and df[c].dtype in (np.float64, np.int64)
    ]
    X = df[feature_cols]

    # Map the disease strings to integers
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    joblib.dump(le, "label_encoder.pkl")

    # Calculate healthy mean for the dashboard
    if "Healthy" in le.classes_:
        healthy_mean = df[df[target_col] == "Healthy"][feature_cols].mean().values
    else:
        # Fallback to global feature means if "Healthy" is absent
        logging.warning(
            "Label 'Healthy' not found in dataset; using global feature means as baseline.",
        )
        healthy_mean = X.mean().values

    joblib.dump(healthy_mean, "healthy_mean.pkl")
    joblib.dump(feature_cols, "feature_cols.pkl")
    joblib.dump(X.mean().values, "x_mean.pkl")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")

    # PCA to 5 dimensions for the 5-qubit architecture
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, "pca.pkl")

    # Stratified split: 70% train, 30% test (test_size=0.3)
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca,
        y,
        test_size=0.3,  # hold out 30% for evaluation
        random_state=42,  # fixed seed for reproducibility
        stratify=y,
    )

    # Classical SVM baseline
    svm = SVC(kernel="rbf", probability=True, break_ties=True)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    # Random Forest baseline
    rf = RandomForestClassifier(
        n_estimators=100,  # standard ensemble size balancing bias/variance
        random_state=42,  # fixed seed for reproducibility
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # XGBoost baseline (use_label_encoder=False is deprecated and now default)
    xgb = XGBClassifier(
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,  # silence training logs for cleaner CLI output
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    logging.info("Classical SVM Accuracy: %.4f", accuracy_score(y_test, y_pred))
    logging.info(
        "Random Forest Accuracy: %.4f", accuracy_score(y_test, y_pred_rf)
    )
    logging.info(
        "XGBoost Accuracy:       %.4f", accuracy_score(y_test, y_pred_xgb)
    )
    logging.info("Total evaluated classes: %d", len(le.classes_))

    joblib.dump(svm, "classical_svm_model.pkl")
    joblib.dump(rf, "classical_rf_model.pkl")
    joblib.dump(xgb, "classical_xgb_model.pkl")

    np.save("y_test_classical.npy", y_test)
    np.save("y_pred_classical.npy", y_pred)
    np.save("y_pred_rf.npy", y_pred_rf)
    np.save("y_pred_xgb.npy", y_pred_xgb)


if __name__ == "__main__":
    main()