import pandas as pd
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("--- Running Classical SVM on General Disease Dataset ---")
    
    # Use the local synthetic dataset
    df = pd.read_csv('data/qnose_synthetic_dataset.csv')
    
    # Target: is_diseased (0 = Healthy, 1 = Diseased)
    y = df['is_diseased'].values
    
    # Features: Pick only the ppb and ppm sensory data
    feature_cols = [c for c in df.columns if c.endswith('_ppb') or c.endswith('_ppm')]
    X = df[feature_cols]
    
    # Compute the average healthy profile to use in the UI later
    healthy_mean = df[df['is_diseased'] == 0][feature_cols].mean().values
    joblib.dump(healthy_mean, 'healthy_mean.pkl')
    
    # Save feature names
    joblib.dump(feature_cols, 'feature_cols.pkl')
    
    # Better to use StandardScaler for general biological VOCs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save the global mean for padding
    joblib.dump(X.mean().values, 'x_mean.pkl')
    
    # PCA down to 5 components for the 5-qubit quantum embedding
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, 'pca.pkl')
    
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)
    
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(svm, 'classical_svm_model.pkl')
    np.save('y_test_classical.npy', y_test)
    np.save('y_pred_classical.npy', y_pred)

if __name__ == '__main__':
    main()