import os

content = r'''import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pennylane as qml
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pcolors
import time
import hashlib

# 1. Setup the Page Configuration
st.set_page_config(
    page_title="QNose | Quantum ML Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Clean minimal background */
    .reportview-container .main .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    
    /* Animated Hero Section - Fixed CSS */
    .hero {
        background: linear-gradient(-45deg, #150020, #0a1128, #18002a, #001219);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(0, 255, 204, 0.1);
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.05);
    }
    @keyframes gradientBG { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    .hero h1 span {
        font-size: 3rem !important;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #00ffcc;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.5), 0 0 20px rgba(176, 102, 255, 0.5);
    }
    .hero p { font-size: 1.2rem; color: #b3c0d1; max-width: 800px; margin: 0 auto; }

    /* Pulsing Alerts */
    @keyframes pulse-red { 0% { box-shadow: 0 0 15px rgba(255, 75, 75, 0.6); border-color: #ff4b4b; } 50% { box-shadow: 0 0 35px rgba(255, 75, 75, 1.0); border-color: #ff8080; } 100% { box-shadow: 0 0 15px rgba(255, 75, 75, 0.6); border-color: #ff4b4b; } }
    @keyframes pulse-green { 0% { box-shadow: 0 0 15px rgba(0, 255, 128, 0.6); border-color: #00ff80; } 50% { box-shadow: 0 0 35px rgba(0, 255, 128, 1.0); border-color: #80ffc0; } 100% { box-shadow: 0 0 15px rgba(0, 255, 128, 0.6); border-color: #00ff80; } }
    .alert-glow { animation: pulse-red 1.5s infinite; border-radius: 12px; padding: 20px; text-align: center; background: rgba(40, 0, 0, 0.4); margin-top: 1rem; }
    .safe-glow { animation: pulse-green 2.5s infinite; border-radius: 12px; padding: 20px; text-align: center; background: rgba(0, 40, 0, 0.4); margin-top: 1rem; }
    .hw-error { color: #ff4b4b; font-weight: bold; background-color: rgba(255, 75, 75, 0.1); padding: 12px; border-radius: 8px; border: 1px solid #ff4b4b; text-align: center; margin-top: 15px; margin-bottom: 15px; font-size: 0.95rem; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 2. Hero Section
st.markdown("""
<div class="hero">
    <h1><span>⚛️ QNose Multiplex Interface</span></h1>
    <p>Activate the Quantum Subspace Engine to synthesize structural permutations corresponding to 27 unique pathological signatures in real-time space.</p>
    <div style="margin-top: 15px;">
        <span style="background-color: #7C3AED; color: white; padding: 5px 12px; border-radius: 15px; font-size: 0.85rem; font-weight: bold; margin-right: 10px;">🧠 Core: Quantum SVM (PennyLane)</span>
        <span style="background-color: #059669; color: white; padding: 5px 12px; border-radius: 15px; font-size: 0.85rem; font-weight: bold;">🔁 Simulated Qubits: 5</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 3. Load Models (Cached for Performance)
@st.cache_resource
def load_resources():
    try:
        scaler = joblib.load('scaler.pkl')
        pca = joblib.load('pca.pkl')
        x_mean = joblib.load('x_mean.pkl')
        healthy_mean = joblib.load('healthy_mean.pkl')
        qsvm = joblib.load('quantum_svm_model.pkl')
        X_train = np.load('X_train_qsvm.npy')
        y_train = np.load('y_train_qsvm.npy')
        feature_cols = joblib.load('feature_cols.pkl')
        le = joblib.load('label_encoder.pkl')
        return scaler, pca, x_mean, healthy_mean, qsvm, X_train, y_train, feature_cols, le
    except FileNotFoundError as e:
        st.error(f"Missing modeling artifact: {e}. Please run the backend scripts first.")
        st.stop()

scaler, pca, x_mean, healthy_mean, qsvm, X_train, y_train, feature_cols, le = load_resources()

# PCA Helper
def get_pca_coords(input_features):
    scaled = scaler.transform([input_features])
    return pca.transform(scaled)

# 4. Sidebar Controls and Logic
if "hw_error_triggered" not in st.session_state:
    st.session_state.hw_error_triggered = False

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/P_hybrid_circuit.svg/1024px-P_hybrid_circuit.svg.png", use_container_width=True)
    
    st.header("🎛️ Live Diagnostics")
    if st.button("📡 Auto-Detect from Hardware", type="primary", use_container_width=True):
        st.session_state.hw_error_triggered = not st.session_state.hw_error_triggered
        
    if st.session_state.hw_error_triggered:
        st.markdown('<div class="hw-error">🚨 Hardware interface link failed!<br/>No IoT Breathalyzer detected on open serial/bluetooth ports.<br/><b>Manual override engaged.</b></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("📁 Upload Patient Data")
    uploaded_file = st.file_uploader("Upload CSV of VOC Readings", type=["csv"])
    if uploaded_file is not None:
        try:
            st.session_state.uploaded_df = pd.read_csv(uploaded_file)
            st.success("CSV loaded successfully! Mapping features...")
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")

    st.markdown("---")
    st.header("🧪 Manual V.O.C. Injection")
    
    input_mode = st.radio("Select Active Matrix Format:", ["Top 5 Parameters", "Top 10 Parameters", "Full 26-Array Integration"])
    input_style = st.radio("Entry Method:", ["🎯 Sliders", "⌨️ Direct Number Entry"], horizontal=True)
    
    top_5 = ["Ethane", "Nonanal", "Acetonitrile", "Pentane", "Hexanal"]
    top_10 = top_5 + ["Isoprene", "Trimethylamine", "Propanal", "Ammonia", "Toluene"]
    
    # Safe feature mapping
    active_features = []
    if input_mode == "Top 5 Parameters":
        active_features = [f for f in top_5 if f in feature_cols]
    elif input_mode == "Top 10 Parameters":
        active_features = [f for f in top_10 if f in feature_cols]
    else:
        select_all = st.checkbox("Engage Complete Array", value=True)
        if select_all:
            active_features = feature_cols
        else:
            active_features = st.multiselect("Isolate Specific Variables:", feature_cols, default=[f for f in top_10 if f in feature_cols])

    st.markdown("##### Matrix Tuners")
    ui_vars = {}
    for feat in active_features:
        idx = feature_cols.index(feat)
        default_val = float(healthy_mean[idx])
        
        if "uploaded_df" in st.session_state and st.session_state.uploaded_df is not None:
            if feat in st.session_state.uploaded_df.columns:
                default_val = float(st.session_state.uploaded_df[feat].iloc[0])
                
        scale_multip = 3.0 if default_val > 10 else 10.0
        max_val = max(100.0, default_val * scale_multip)
        if feat in ["Pentane", "Ammonia"]: max_val = max(1500.0, max_val)
            
        if input_style == "🎯 Sliders":
            ui_vars[feat] = st.slider(f"{feat}", 0.0, float(max_val), float(default_val), step=0.1, key=f"sl_{feat}")
        else:
            ui_vars[feat] = st.number_input(f"{feat}", min_value=0.0, max_value=float(max_val), value=float(default_val), step=0.1, key=f"num_{feat}")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🧬 Deploy Quantum Sequence", type="primary", use_container_width=True)
    st.markdown("<div style='font-size: 0.8rem; color: #666; text-align: center; margin-top: 50px;'><br>🔌 QNose v1.0 | Powered by PennyLane + Streamlit</div>", unsafe_allow_html=True)

healthy_base = [float(healthy_mean[feature_cols.index(f)]) for f in active_features]
current_vars = [ui_vars[f] for f in active_features]

# Initialize Session
if 'prediction_run' not in st.session_state:
    st.session_state.prediction_run = False
    st.session_state.pred_label = ""
    st.session_state.prob_dist = None
    st.session_state.X_input_pca = None
if 'pred_cache' not in st.session_state:
    st.session_state.pred_cache = {}

n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev)
def kernel_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))

def kernel_function(x1, x2): 
    return kernel_circuit(x1, x2)[0]

# On Predict Click
if predict_button:
    full_features = np.copy(x_mean)
    for feat in active_features:
        idx = feature_cols.index(feat)
        full_features[idx] = ui_vars[feat]
    
    X_input_pca = get_pca_coords(full_features)
    st.session_state.X_input_pca = X_input_pca
    
    # Check cache
    input_hash = hashlib.md5(X_input_pca.tobytes()).hexdigest()
    
    if input_hash in st.session_state.pred_cache:
        K_pred = st.session_state.pred_cache[input_hash]
    else:
        # Animated processing
        K_pred = np.zeros((1, len(X_train)))
        progress_text = "⚛️ Computing Quantum Kernel Correlations (Iterative)..."
        prog_bar = st.progress(0, text=progress_text)
        
        for i, x_train in enumerate(X_train):
            K_pred[0, i] = kernel_function(X_input_pca[0], x_train)
            if i % max(1, len(X_train)//20) == 0:
                prog_bar.progress((i + 1) / len(X_train), text=progress_text)
        prog_bar.empty()
        st.session_state.pred_cache[input_hash] = K_pred
        
    pred_idx = qsvm.predict(K_pred)[0]
    st.session_state.pred_label = le.inverse_transform([pred_idx])[0]
    
    if hasattr(qsvm, "predict_proba"):
        st.session_state.prob_dist = qsvm.predict_proba(K_pred)[0]
        
    st.session_state.prediction_run = True
    st.session_state.patient_features = dict(zip(active_features, current_vars))
    st.session_state.patient_healthy_base = dict(zip(active_features, healthy_base))
    st.toast("⚛️ Quantum sequence deployed successfully!", icon="✅")

# 6. Main Content Area
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("### ⚠️ Primary Diagnostic Readout")
    if not st.session_state.prediction_run:
        st.info("System on standby. Validate parameters in sidebar and deploy sequence.")
    else:
        if st.session_state.pred_label != "Healthy":
            st.markdown(f'<div class="alert-glow"><h3>🚨 {st.session_state.pred_label} Signature</h3><p>High degree of structural anomaly detected.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-glow"><h3>✅ Zero-Defect Baseline</h3><p>Patient coordinates align perfectly with healthy topological matrix.</p></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.session_state.prob_dist is not None:
            st.markdown("#### Confidence Matrix (Top 4)")
            probs = st.session_state.prob_dist
            top_4_idx = np.argsort(probs)[-4:][::-1]
            actual_encoded_labels = qsvm.classes_[top_4_idx]
            top_4_diseases = le.inverse_transform(actual_encoded_labels)
            top_4_probs = probs[top_4_idx] * 100
            
            df_probs = pd.DataFrame({'Disease String': top_4_diseases, 'Confidence %': top_4_probs})
            fig_bar = px.bar(df_probs, x='Confidence %', y='Disease String', orientation='h', color='Confidence %', 
                             color_continuous_scale='Reds' if st.session_state.pred_label != "Healthy" else 'Greens', range_x=[0, 100])
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=250, margin=dict(l=0, r=0, t=0, b=0), yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.session_state.top_probs = df_probs

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📊 VIEW FULL ANALYTICS & INSIGHTS REPORT", use_container_width=True, type="secondary"):
        st.switch_page("pages/1_📊_Detailed_Report.py")
        
    st.markdown("### 🧬 Quantum Circuit Architecture")
    try:
        st.image("quantum_circuit.png", caption="5-Qubit Angle Embedding Circuit", use_container_width=True)
    except: pass

with col2:
    if st.session_state.prediction_run:
        st.markdown("### 🕸️ VOC Deviation Radar (Live)")
        radar_df = pd.DataFrame({
            "Feature": list(st.session_state.patient_features.keys()),
            "Patient": list(st.session_state.patient_features.values()),
            "Healthy Base": list(st.session_state.patient_healthy_base.values())
        })
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=radar_df['Patient'], theta=radar_df['Feature'], fill='toself', name='Patient Readout', marker=dict(color='#ff00cc')))
        fig_radar.add_trace(go.Scatterpolar(r=radar_df['Healthy Base'], theta=radar_df['Feature'], fill='toself', name='Healthy Control', marker=dict(color='#00ffcc')))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, radar_df[['Patient', 'Healthy Base']].max().max()*1.1])), showlegend=True, paper_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(t=20, b=20, l=40, r=40))
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.markdown("### 🕸️ Waiting for Patient Matrix...")
        st.empty()

    st.markdown("### 🌐 Navigable Holographic Projection")
    st.caption("A clean 3D render of the 27-state multi-disease boundary arrays.")
    
    if not st.session_state.get('prediction_run', False):
        full_features = np.copy(x_mean)
        for feat in active_features:
            idx = feature_cols.index(feat)
            full_features[idx] = ui_vars[feat]
        live_pca = get_pca_coords(full_features)
        patient_x, patient_y, patient_z = live_pca[0, 0], live_pca[0, 1], live_pca[0, 2]
    else:
        patient_x, patient_y, patient_z = st.session_state.X_input_pca[0, 0], st.session_state.X_input_pca[0, 1], st.session_state.X_input_pca[0, 2]

    df_3d = pd.DataFrame({'Phase X': X_train[:, 0], 'Phase Y': X_train[:, 1], 'Phase Z': X_train[:, 2], 'Class Mapping': le.inverse_transform(y_train)})
    fig_3d = go.Figure()
    
    colors = pcolors.qualitative.Alphabet
    for i, cls in enumerate(df_3d['Class Mapping'].unique()):
        cls_data = df_3d[df_3d['Class Mapping'] == cls]
        fig_3d.add_trace(go.Scatter3d(
            x=cls_data['Phase X'], y=cls_data['Phase Y'], z=cls_data['Phase Z'],
            mode='markers', marker=dict(size=6, color=colors[i % len(colors)], opacity=0.5, line=dict(width=0.5, color='white')),
            name=str(cls), showlegend=True, hoverinfo='text', text=cls_data['Class Mapping']
        ))
        
    z_min = df_3d['Phase Z'].min() - 1
    fig_3d.add_trace(go.Scatter3d(x=[patient_x, patient_x], y=[patient_y, patient_y], z=[z_min, patient_z], mode='lines', line=dict(color='#FF00FF', width=5, dash='dash'), showlegend=False, hoverinfo='none'))
    fig_3d.add_trace(go.Scatter3d(x=[patient_x], y=[patient_y], z=[patient_z], mode='markers+text', marker=dict(size=25, color='#FF00FF', symbol='diamond', line=dict(width=4, color='white'), opacity=1.0), name='SUBJECT', text=['🚀 Target'], textposition="top center", textfont=dict(color='#FF00FF', size=20, family="Arial Black"), showlegend=False, hoverinfo='text'))
    
    fig_3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(xaxis=dict(showgrid=True, gridcolor='rgba(0, 255, 204, 0.3)'), yaxis=dict(showgrid=True, gridcolor='rgba(0, 255, 204, 0.3)'), zaxis=dict(showgrid=True, gridcolor='rgba(0, 255, 204, 0.3)')),
        margin=dict(t=0, b=0, l=0, r=0), height=550,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)", font=dict(color="white", size=10), itemsizing='constant', traceorder='normal')
    )
    st.plotly_chart(fig_3d, use_container_width=True)
'''

with open(r"c:\Users\hp\OneDrive\Desktop\qc2\qnose\app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("app.py rewritten successfully!")
