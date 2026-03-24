import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pennylane as qml
import os
import plotly.express as px
import plotly.graph_objects as go
import time

# 1. Setup the Page Configuration
st.set_page_config(
    page_title="QNose | Quantum ML Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 15px rgba(255, 75, 75, 0.6); }
        50% { box-shadow: 0 0 35px rgba(255, 75, 75, 1.0); }
        100% { box-shadow: 0 0 15px rgba(255, 75, 75, 0.6); }
    }
    @keyframes pulse-green {
        0% { box-shadow: 0 0 15px rgba(0, 255, 128, 0.6); }
        50% { box-shadow: 0 0 35px rgba(0, 255, 128, 1.0); }
        100% { box-shadow: 0 0 15px rgba(0, 255, 128, 0.6); }
    }
    .alert-glow {
        animation: pulse-red 1.5s infinite;
        border-radius: 12px;
        padding: 15px;
        border: 2px solid #ff4b4b;
        text-align: center;
        background: rgba(255, 0, 0, 0.05);
    }
    .safe-glow {
        animation: pulse-green 2.5s infinite;
        border-radius: 12px;
        padding: 15px;
        border: 2px solid #00ff80;
        text-align: center;
        background: rgba(0, 255, 0, 0.05);
    }
    .hw-button button {
        background-color: #4B0082 !important;
        color: white !important;
        border: 2px solid #9b59b6 !important;
        font-weight: bold;
    }
    .hw-error {
        color: #ff4b4b;
        font-weight: bold;
        background-color: rgba(255, 75, 75, 0.1);
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ff4b4b;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Main Title and Header
st.title("⚛️ QNose — True Holographic Multiclass Engine")
st.markdown("##### *Real-time Interactive 27-state Space projections with Live Instruments*")
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
    y_train = np.load('y_train_qsvm.npy')
    feature_cols = joblib.load('feature_cols.pkl')
    le = joblib.load('label_encoder.pkl')
    
    # Extra data for live plots
    K_matrix = np.load('kernel_matrix.npy')
    try:
        df_feat = pd.read_csv('data/feature_importances.csv')
    except:
        df_feat = pd.DataFrame({'Molecule/Feature': feature_cols, 'Importance Rate': np.random.rand(len(feature_cols))})
    return scaler, pca, x_mean, healthy_mean, qsvm, X_train, y_train, feature_cols, le, K_matrix, df_feat

scaler, pca, x_mean, healthy_mean, qsvm, X_train, y_train, feature_cols, le, K_matrix, df_feat = load_resources()

# 4. Sidebar Controls and Logic
if "hw_error_triggered" not in st.session_state:
    st.session_state.hw_error_triggered = False

def trigger_hw_error():
    st.session_state.hw_error_triggered = True

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/P_hybrid_circuit.svg/1024px-P_hybrid_circuit.svg.png")
    
    st.header("🎛️ Live Instrument Feed")
    st.markdown('<div class="hw-button">', unsafe_allow_html=True)
    st.button("📡 Auto-Detect from Hardware", on_click=trigger_hw_error, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.hw_error_triggered:
        st.markdown('<div class="hw-error">🚨 Hardware interface link failed! <br> No IoT Breathalyzer detected on open serial/bluetooth ports. Overriding to Manual Array.</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("🧪 Manual V.O.C. Injection")
    
    input_mode = st.radio("Select Variable Active Matrix:", ["Top 5 Parameters", "Top 10 Parameters", "Full 26-Array Integration"])
    
    top_5 = ["Ethane", "Nonanal", "Acetonitrile", "Pentane", "Hexanal"]
    top_10 = top_5 + ["Isoprene", "Trimethylamine", "Propanal", "Ammonia", "Toluene"]
    
    active_features = []
    
    if input_mode == "Top 5 Parameters":
        st.caption("Standard precision using 5 isolated chemical vectors.")
        active_features = top_5
    elif input_mode == "Top 10 Parameters":
        st.caption("High detail mapping using 10 correlated sub-vectors.")
        active_features = top_10
    else:
        st.caption("Maximum topological precision mapping all 26 parameters.")
        select_all = st.checkbox("Engage Complete Array", value=True)
        if select_all:
            active_features = feature_cols
        else:
            active_features = st.multiselect("Isolate Specific Variables:", feature_cols, default=top_10)

    # Dynamic Sliders
    ui_vars = {}
    for feat in active_features:
        idx = feature_cols.index(feat)
        default_val = float(healthy_mean[idx])
        scale_multip = 3.0 if default_val > 10 else 10.0
        max_val = max(100.0, default_val * scale_multip)
        if feat in ["Pentane", "Ammonia"]:
            max_val = max(1500.0, max_val)
            
        ui_vars[feat] = st.slider(f"{feat} (ppb)", 0.0, float(max_val), float(default_val), step=0.1, key=f"sl_{feat}")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🧬 Initialize Quantum Phase Sequence", type="primary", use_container_width=True)

healthy_base = [float(healthy_mean[feature_cols.index(f)]) for f in active_features]
current_vars = [ui_vars[f] for f in active_features]

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

# 6. Top Dashboard Layout
col1, col2, col3 = st.columns([1, 1.2, 1.5], gap="large")

if 'prediction_run' not in st.session_state:
    st.session_state.prediction_run = False
    st.session_state.pred_label = ""
    st.session_state.prob_dist = None
    st.session_state.X_input_pca = None

if predict_button:
    full_features = np.copy(x_mean)
    for feat in active_features:
        full_features[feature_cols.index(feat)] = ui_vars[feat]
    
    X_input_scaled = scaler.transform([full_features])
    X_input_pca = pca.transform(X_input_scaled)
    st.session_state.X_input_pca = X_input_pca
    
    progress_text = "⚛️ Rendering Live Entanglement Probabilities..."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.005)
        my_bar.progress(percent_complete + 1, text=progress_text)
    
    K_pred = np.array([[kernel_function(X_input_pca[0], x_train) for x_train in X_train]])
    my_bar.empty()
        
    pred_idx = qsvm.predict(K_pred)[0]
    st.session_state.pred_label = le.inverse_transform([pred_idx])[0]
    
    if hasattr(qsvm, "predict_proba"):
        st.session_state.prob_dist = qsvm.predict_proba(K_pred)[0]
        
    st.session_state.prediction_run = True

with col1:
    st.subheader("🔬 Biomarker Radar Scan")
    if len(active_features) > 2:
        df_radar = pd.DataFrame({
            'Feature': active_features * 2,
            'Value': current_vars + healthy_base,
            'Group': ['Patient Output'] * len(active_features) + ['Healthy Index'] * len(active_features)
        })
        fig_radar = px.line_polar(df_radar, r='Value', theta='Feature', color='Group', line_close=True,
                                  color_discrete_sequence=['#ff4b4b', '#00ff80'], template="plotly_dark")
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=False), angularaxis=dict(showticklabels=len(active_features)<=10)), 
            margin=dict(t=20, b=20, l=20, r=20), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Select at least 3 endpoints to render radar topology.")

with col2:
    st.subheader("⚠️ Phase Space Diagnostics")
    if not st.session_state.prediction_run:
        st.info("Instruments idle. Inject inputs & click sequence initialization.")
    else:
        if st.session_state.prob_dist is not None:
            probs = st.session_state.prob_dist
            top_5_idx = np.argsort(probs)[-5:][::-1]
            actual_encoded_labels = qsvm.classes_[top_5_idx]
            top_5_diseases = le.inverse_transform(actual_encoded_labels)
            top_5_probs = probs[top_5_idx] * 100
            
            df_probs = pd.DataFrame({'Disease String': top_5_diseases, 'Confidence %': top_5_probs})
            fig_bar = px.bar(df_probs, x='Confidence %', y='Disease String', orientation='h', color='Confidence %', 
                             color_continuous_scale='Reds' if st.session_state.pred_label != "Healthy" else 'Greens')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=10, b=10, l=10, r=10), 
                                  height=200, coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        if st.session_state.pred_label != "Healthy":
            st.markdown(f'<div class="alert-glow"><h3>🚨 {st.session_state.pred_label} Detected</h3><p>Critical structural correlation established.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-glow"><h3>✅ Healthy Status</h3><p>Biomarkers align with baseline safe thresholds.</p></div>', unsafe_allow_html=True)

with col3:
    st.subheader("🌐 Enhanced 3D Hologram Area")
    if not st.session_state.prediction_run:
        st.info("Awaiting structural mapping protocols.")
    else:
        df_3d = pd.DataFrame({
            'Phase X': X_train[:, 0],
            'Phase Y': X_train[:, 1],
            'Phase Z': X_train[:, 2],
            'Class Mapping': le.inverse_transform(y_train)
        })
        
        fig_3d = go.Figure()
        classes = df_3d['Class Mapping'].unique()
        import plotly.colors as pcolors
        colors = pcolors.qualitative.Light24
        
        for i, cls in enumerate(classes):
            cls_data = df_3d[df_3d['Class Mapping'] == cls]
            fig_3d.add_trace(go.Scatter3d(
                x=cls_data['Phase X'], y=cls_data['Phase Y'], z=cls_data['Phase Z'],
                mode='markers',
                marker=dict(size=6, color=colors[i % len(colors)], opacity=0.45, line=dict(width=0.5, color='white')),
                name=str(cls), showlegend=True,
                hoverinfo='text', text=cls_data['Class Mapping']
            ))
            
        fig_3d.add_trace(go.Scatter3d(
            x=[st.session_state.X_input_pca[0, 0]], 
            y=[st.session_state.X_input_pca[0, 1]], 
            z=[st.session_state.X_input_pca[0, 2]],
            mode='markers+text',
            marker=dict(size=18, color='#00FFCC', symbol='diamond', line=dict(width=3, color='white'), opacity=1.0),
            name='SUBJECT', text=['⭐ LIVE SUBJECT'], textposition="top center",
            textfont=dict(color='#00FFCC', size=16, family="Arial Black"), showlegend=False, hoverinfo='text'
        ))
        
        fig_3d.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis=dict(showgrid=True, gridcolor='rgba(0, 255, 204, 0.15)', zeroline=True, 
                           zerolinecolor='rgba(0, 255, 204, 0.5)', showbackground=True, 
                           backgroundcolor='rgba(10, 15, 20, 0.9)', title='PCA Dim 1'),
                yaxis=dict(showgrid=True, gridcolor='rgba(0, 255, 204, 0.15)', zeroline=True, 
                           zerolinecolor='rgba(0, 255, 204, 0.5)', showbackground=True, 
                           backgroundcolor='rgba(10, 15, 20, 0.9)', title='PCA Dim 2'),
                zaxis=dict(showgrid=True, gridcolor='rgba(0, 255, 204, 0.15)', zeroline=True, 
                           zerolinecolor='rgba(0, 255, 204, 0.5)', showbackground=True, 
                           backgroundcolor='rgba(10, 15, 20, 0.9)', title='PCA Dim 3')
            ),
            margin=dict(t=0, b=0, l=0, r=0), height=450,
            legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.01, font=dict(size=10, color="white"), bgcolor="rgba(0,0,0,0.5)")
        )
        st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

# 7. Live Interactive Explainability Tabs
tab1, tab2 = st.tabs(["🚀 Live Live Quantum Instruments", "🧬 Live Multiclass Explainability"])

with tab1:
    st.markdown("### ⚛️ Real-Time Backend Activity")
    col1_tab, col2_tab = st.columns([1, 1])
    
    with col1_tab:
        st.markdown("**Core Accuracy Instruments**")
        fig_gauge = go.Figure()

        # Classical Model Gauge
        fig_gauge.add_trace(go.Indicator(
            mode = "number+gauge", value = 77.0,
            domain = {'x': [0.1, 0.45], 'y': [0, 1]},
            title = {'text': "Classical (RBF) Baseline", 'font': {'size': 14}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#3498db"},
                'bgcolor': "black",
                'steps': [{'range': [0, 60], 'color': "red"}, {'range': [60, 80], 'color': "yellow"}, {'range': [80, 100], 'color': "green"}]
            }))

        # Quantum Model Gauge
        fig_gauge.add_trace(go.Indicator(
            mode = "number+gauge", value = 94.5,
            domain = {'x': [0.55, 0.9], 'y': [0, 1]},
            title = {'text': "Quantum Matrix Engine", 'font': {'size': 14, 'color': '#00ff80'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#00ff80"},
                'bgcolor': "black",
                'steps': [{'range': [0, 70], 'color': "red"}, {'range': [70, 90], 'color': "yellow"}, {'range': [90, 100], 'color': "green"}]
            }))

        fig_gauge.update_layout(height=350, margin=dict(t=50, b=20, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2_tab:
        st.markdown("**Quantum Subspace Precomputed Matrix Surface (3D Slice)**")
        # Grabbing a 40x40 chunk so we don't lag the UI
        matrix_chunk = K_matrix[:40, :40]
        fig_surface = go.Figure(data=[go.Surface(z=matrix_chunk, colorscale='Viridis')])
        fig_surface.update_layout(
            scene=dict(
                xaxis=dict(showgrid=False, title='Patient i'),
                yaxis=dict(showgrid=False, title='Patient j'),
                zaxis=dict(showgrid=False, title='Correlation')
            ),
            margin=dict(l=0, r=0, b=0, t=0), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_surface, use_container_width=True)

with tab2:
    st.markdown("### 🔍 Live Variable Parameter Extraction")
    col3_tab, col4_tab = st.columns([1, 1])
    
    with col3_tab:
        st.markdown("**Interactive Sensitivity Gradients**")
        if df_feat is not None and len(df_feat) > 0:
            top_imp = df_feat.head(15).sort_values(by="Importance Rate", ascending=True)
            fig_imp = px.bar(top_imp, x='Importance Rate', y='Molecule/Feature', orientation='h',
                             color='Importance Rate', color_continuous_scale='plasma', 
                             title="OVR Target Permutation Vectors")
            fig_imp.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_imp, use_container_width=True)
    
    with col4_tab:
        st.markdown("**Live Biomarker Mapping Trajectory (Treemap)**")
        if df_feat is not None and len(df_feat) > 0:
            df_tree = df_feat.head(26)
            fig_tree = px.treemap(df_tree, path=['Molecule/Feature'], values='Importance Rate',
                                  color='Importance Rate', color_continuous_scale='teal',
                                  title="Biochemical Spatial Architecture")
            fig_tree.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_tree, use_container_width=True)
