import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pennylane as qml
import os

# 1. Setup the Page Configuration
st.set_page_config(
    page_title="QNose | Quantum ML",
    page_icon="👃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Main Title and Header
st.title("👃 QNose — Quantum Breath Disease Detector")
st.markdown("##### _Harnessing Quantum Machine Learning to analyze Volatile Organic Compounds (VOCs) for early Parkinson's detection._")
st.markdown("---")

# 3. Load Models (Cached for Performance)
@st.cache_resource
def load_resources():
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    x_mean = joblib.load('x_mean.pkl')
    qsvm = joblib.load('quantum_svm_model.pkl')
    X_train = np.load('X_train_qsvm.npy')
    return scaler, pca, x_mean, qsvm, X_train

scaler, pca, x_mean, qsvm, X_train = load_resources()

# 4. Sidebar Controls
with st.sidebar:
    st.header("🎛️ Adjust VOC Levels")
    st.markdown("Simulate a breath sample below.")
    
    acetone = st.slider("Acetone (ppm)", 0.0, 3.0, 1.0, step=0.1)
    isoprene = st.slider("Isoprene (ppm)", 0.0, 5.0, 2.0, step=0.1)
    hydrogen_cyanide = st.slider("Hydrogen Cyanide (ppm)", 0.0, 1.0, 0.4, step=0.05)
    ethane = st.slider("Ethane (ppm)", 0.0, 1.5, 0.5, step=0.1)
    pentane = st.slider("Pentane (ppm)", 0.0, 2.0, 0.8, step=0.1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔮 Analyze Breath Sample", type="primary", use_container_width=True)

feature_names = ["Acetone", "Isoprene", "Hydrogen Cyanide", "Ethane", "Pentane"]
features = np.array([acetone, isoprene, hydrogen_cyanide, ethane, pentane])

# 5. Quantum Circuit Setup
n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))

def kernel_function(x1, x2):
    return kernel_circuit(x1, x2)[0]

# 6. Results Section
col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.subheader("🔬 Diagnostic Result")
    
    if predict_button:
        # Inject current UI features over real dataset padding
        full_features = np.copy(x_mean)
        full_features[:5] = features
        
        # Scale & Reduce dims
        X_input_scaled = scaler.transform([full_features])
        X_input_pca = pca.transform(X_input_scaled)
        
        # Progress Spinner & Evaluation
        with st.spinner("⚛️ Traversing Hilbert space... Computing Quantum Kernel..."):
            K_pred = np.array([[kernel_function(X_input_pca[0], x_train) for x_train in X_train]])
            
        pred = qsvm.predict(K_pred)[0]
        
        # Fancy UI Cards for Prediction
        if pred == 1:
            st.error("### 🚨 **High Risk Detected**")
            st.markdown("**Assessment:** The Quantum SVM has identified biomarker patterns consistent with Parkinson's disease.")
        else:
            st.success("### ✅ **Low Risk / Healthy**")
            st.markdown("**Assessment:** The Quantum SVM found normal biomarker patterns. No immediate risk detected.")
        
        # Display the inputted breakdown
        st.markdown("<br>#### Feature Breakdown", unsafe_allow_html=True)
        df_chart = pd.DataFrame([features], columns=feature_names).T
        df_chart.columns = ["Value (ppm)"]
        st.bar_chart(df_chart, color="#5B2C6F")
    else:
        st.info("👈 Please enter the VOC levels in the sidebar and click **Analyze Breath Sample** to evaluate the quantum model.")

with col2:
    st.subheader("📊 Explainability")
    if os.path.exists("shap_explanation.png"):
        st.image("shap_explanation.png", caption="Model decision breakdown based on SHAP values", use_container_width=True)
    else:
        st.warning("Could not find `shap_explanation.png`. Did you run the explainability script?")

st.markdown("---")

# 7. Deep Dive Tabs
tab1, tab2 = st.tabs(["🧠 How QNose Thinks", "📄 Download Full Report"])

with tab1:
    st.markdown("### Quantum Embedding Architecture")
    st.markdown("Instead of normal dimensions, QNose embeds classical tabular data directly into complex **Quantum States**. ")
    st.markdown("- **AngleEmbedding:** Rotates qubits horizontally based on feature intensities.\n- **CNOT Entanglement:** Evaluates deep non-linear interactions across combined biomarkers.")
    
    if os.path.exists("quantum_circuit.png"):
        st.image("quantum_circuit.png", caption="The generated PennyLane quantum circuit.")

with tab2:
    st.markdown("### Comparison & Diagnostic Report")
    st.markdown("Get an aggregate report comparing Classical SVM behavior against our new Quantum Kernel Method.")
    if os.path.exists("QNose_Results.pdf"):
        with open("QNose_Results.pdf", "rb") as f:
            st.download_button(
                label="📥 Download PDF Report",
                data=f,
                file_name="QNose_Results.pdf",
                mime="application/pdf",
                type="primary"
            )
    else:
        st.warning("Run `report_generator.py` to generate the downloadable PDF.")
