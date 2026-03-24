import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def main():
    print("--- Running SHAP Explainability ---")
    df = pd.read_csv('data/qnose_synthetic_dataset.csv')
    df = df[df['disease_label'].isin(['Parkinsons', 'Healthy'])]
    
    feature_cols = joblib.load('feature_cols.pkl')
    X = df[feature_cols]
    
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    svm = joblib.load('classical_svm_model.pkl')
    
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    
    # Take a sample Parkinson's patient for explainability
    parkinsons_idx = np.where(df['disease_label'] == 'Parkinsons')[0]
    if len(parkinsons_idx) > 0:
        sample = X_pca[parkinsons_idx[0]:parkinsons_idx[0]+1]
    else:
        sample = X_pca[0:1] # fallback
    
    # Downsample background to speed up SHAP
    background = shap.kmeans(X_pca, min(10, len(X_pca)))
    explainer = shap.KernelExplainer(svm.predict, background)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(sample)
    
    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        base_val = base_val[0]
        
    exp = shap.Explanation(values=shap_values[0], 
                           base_values=base_val, 
                           data=sample[0], 
                           feature_names=["PCA-VOC 1", "PCA-VOC 2", "PCA-VOC 3", "PCA-VOC 4", "PCA-VOC 5"])
    
    plt.figure(figsize=(8, 5))
    shap.waterfall_plot(exp, show=False)
    plt.title("Why did QNose flag this breath sample?")
    plt.tight_layout()
    plt.savefig('shap_explanation.png')
    print("Saved shap_explanation.png")

if __name__ == "__main__":
    main()