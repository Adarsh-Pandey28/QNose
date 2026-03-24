import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pennylane as qml
import os
import plotly.express as px
import plotly.graph_objects as go

# 1. Setup the Page Configuration
st.set_page_config(
    page_title="QNose | Quantum ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4B0082;
    }
    .metric-label {
        color: #555;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# 2. Main Title and Header
st.title("🤖 QNose — Quantum Biomarker Analysis System")
st.markdown("##### _Multi-Disease Early Detection AI using Quantum Support Vector Machines._")
st.markdown("---")

# 3. Load Models (Cached for Performance)
@st.cache_resource
def load_resources():
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    x_mean = joblib.load('x_mean.pkl')
    healthy_mean = joblib.load('healthy_mean.pkl')
    qsvm = joblib.load('quantum_svm_model.pkl')
    X_train = np.load('X_train_qsvm.npy')
    feature_cols = joblib.load('feature_cols.pkl')
    return scaler, pca, x_mean, healthy_mean, qsvm, X_train, feature_cols

scaler, pca, x_mean, healthy_mean, qsvm, X_train, feature_cols = load_resources()

idx_acetone = feature_cols.index('acetone_ppb')
idx_isoprene = feature_cols.index('isoprene_ppb')
idx_hc = feature_cols.index('hydrogen_cyanide_ppb')
idx_ethanol = feature_cols.index('ethanol_ppb')
idx_pentane = feature_cols.index('pentane_ppb')

# 4. Sidebar Controls
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/P_hybrid_circuit.svg/1024px-P_hybrid_circuit.svg.png", width=150)
    st.header("🎛️ Clinical Breath Inputs")
    st.markdown("Adjust key VOC parts-per-billion (ppb) detected in the sample.")
    
    acetone = st.slider("Acetone (ppb) [Diabetes/Metabolic]", 0.0, 3000.0, float(healthy_mean[idx_acetone]), step=10.0)
    isoprene = st.slider("Isoprene (ppb) [Cholesterol/Liver]", 0.0, 500.0, float(healthy_mean[idx_isoprene]), step=5.0)
    hydrogen_cyanide = st.slider("Hydrogen Cyanide (ppb) [Resp. Infection]", 0.0, 50.0, float(healthy_mean[idx_hc]), step=0.5)
    ethanol = st.slider("Ethanol (ppb) [Gut Microbiome]", 0.0, 1000.0, float(healthy_mean[idx_ethanol]), step=10.0)
    pentane = st.slider("Pentane (ppb) [Oxidative Stress]", 0.0, 200.0, float(healthy_mean[idx_pentane]), step=1.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🧬 Run Quantum Inference", type="primary", use_container_width=True)

ui_feature_names = ["Acetone", "Isoprene", "H. Cyanide", "Ethanol", "Pentane"]
ui_features = [acetone, isoprene, hydrogen_cyanide, ethanol, pentane]
healthy_base = [healthy_mean[idx_acetone], healthy_mean[idx_isoprene], healthy_mean[idx_hc], healthy_mean[idx_ethanol], healthy_mean[idx_pentane]]

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

# 6. Dashboard Layout
col1, col2 = st.columns([1, 1], gap="medium")

# Pre-computation to hold state
if 'prediction_run' not in st.session_state:
    st.session_state.prediction_run = False
    st.session_state.prob = 0.0
    st.session_state.pred = 0

if predict_button:
    full_features = np.copy(x_mean)
    full_features[idx_acetone] = acetone
    full_features[idx_isoprene] = isoprene
    full_features[idx_hc] = hydrogen_cyanide
    full_features[idx_ethanol] = ethanol
    full_features[idx_pentane] = pentane
    
    X_input_scaled = scaler.transform([full_features])
    X_input_pca = pca.transform(X_input_scaled)
    
    with st.spinner("⚛️ Calculating entangled feature states in Quantum Hilbert Space..."):
        K_pred = np.array([[kernel_function(X_input_pca[0], x_train) for x_train in X_train]])
        
    st.session_state.pred = qsvm.predict(K_pred)[0]
    
    if hasattr(qsvm, "predict_proba"):
        st.session_state.prob = qsvm.predict_proba(K_pred)[0][1] # Probability of disease class
    else:
        # fallback if probability=True wasn't set, though it should be
        st.session_state.prob = 0.95 if st.session_state.pred == 1 else 0.05
        
    st.session_state.prediction_run = True

with col1:
    st.subheader("🔬 Biomarker Radar Profile")
    # Radar Chart matching input values against healthy baseline
    df_radar = pd.DataFrame({
        'Feature': ui_feature_names * 2,
        'Value': ui_features + healthy_base,
        'Group': ['Patient Sample'] * 5 + ['Healthy Baseline'] * 5
    })
    
    fig_radar = px.line_polar(df_radar, r='Value', theta='Feature', color='Group', line_close=True,
                              color_discrete_sequence=['#ff4b4b', '#1f77b4'],
                              template="plotly_white")
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)))
    st.plotly_chart(fig_radar, use_container_width=True)

with col2:
    st.subheader("⚠️ Quantum Risk Assessment")
    
    if not st.session_state.prediction_run:
        st.info("Awaiting input. Click **Run Quantum Inference** in the sidebar.")
        
        # Blank placeholder gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 0,
            title = {'text': "Disease Probability"},
            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "lightgray"}}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        # Populate live gauge
        risk_pct = st.session_state.prob * 100
        gauge_color = "red" if risk_pct > 50 else "green"
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_pct,
            number={'suffix': "%"},
            title = {'text': "Anomalous Biomarker Probability"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': gauge_color},
                'steps' : [
                    {'range': [0, 30], 'color': "#e6ffe6"},
                    {'range': [30, 60], 'color': "#fff3e6"},
                    {'range': [60, 100], 'color': "#ffe6e6"}],
                'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if st.session_state.pred == 1:
            st.error("### 🚨 **Anomalous Signature Detected**")
            st.markdown("The quantum kernel has flagged systemic anomalies correlating strongly with clinical disease profiles.")
        else:
            st.success("### ✅ **Baseline Healthy Signature**")
            st.markdown("The biomarker distribution maps smoothly within the generalized healthy quantum hyperplane.")

st.markdown("---")

tab1, tab2 = st.tabs(["🧠 AI Explainability (SHAP)", "🗺️ Quantum Processing View"])

with tab1:
    st.markdown("### Decision Interpretation")
    st.markdown("Understanding *why* the SVM made its decision using Shapley additive explanations.")
    if os.path.exists("shap_explanation.png"):
        st.image("shap_explanation.png", use_container_width=False)
    else:
        st.warning("Generate explainability metrics by running `python explainability.py`.")

with tab2:
    st.markdown("### Topological Data Encoding")
    st.markdown("Visualizing the parameterised PennyLane operations bridging the classical-to-quantum step.")
    if os.path.exists("quantum_circuit.png"):
        st.image("quantum_circuit.png")
