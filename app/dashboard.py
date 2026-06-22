import streamlit as st
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# pg config
st.set_page_config(page_title="Rotor Digital Twin", layout="wide")
st.title("⚙️ Real-Time Digital Twin: 4-Disk Flexible Rotor")
st.markdown("Predictive Machine Learning Surrogate Model for Critical Speeds, Resonance Avoidance, and Mode Shapes.")


@st.cache_resource # caching the model so it doesn't reload on every slider move
def load_surrogate_model():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/ml'))
    model_path = os.path.join(base_path, 'gpr_model.pkl')
    scaler_path = os.path.join(base_path, 'scaler_X.pkl')
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        return None, None

gpr_model, scaler_X = load_surrogate_model()

if gpr_model is None:
    st.error(" Model not found")
    st.stop()

# sidebar
st.sidebar.header("Rotor Parameters")
st.sidebar.markdown("Adjust the geometry and physics to see real-time predictions.")

d_i = st.sidebar.slider("Inner Diameter (mm)", 10.0, 25.0, 15.0, 0.5) / 1000.0
d_o = st.sidebar.slider("Outer Diameter (mm)", 30.0, 50.0, 40.0, 0.5) / 1000.0

if d_i >= d_o:
    st.sidebar.error("Inner diameter must be less than outer diameter!")
    
L_total = st.sidebar.slider("Shaft Length (m)", 0.8, 1.5, 1.0, 0.05)
m_disk = st.sidebar.slider("Mass per Disk (kg)", 2.0, 10.0, 5.0, 0.5)
k_bearing = st.sidebar.slider("Bearing Stiffness (N/m)", 1e5, 1e7, 1e6, step=1e5, format="%e")


X_input = np.array([[d_i, d_o, L_total, m_disk, k_bearing]])
X_scaled = scaler_X.transform(X_input)


y_pred, y_std = gpr_model.predict(X_scaled, return_std=True)


if len(y_pred[0]) >= 4:
    freq_1_mean, freq_2_mean, cs_1_mean, cs_2_mean = y_pred[0][:4]
    freq_1_std, freq_2_std = y_std[0][:2] 
else:
    freq_1_mean, freq_2_mean = y_pred[0][:2]
    freq_1_std, freq_2_std = y_std[0][:2]
    cs_1_mean = freq_1_mean * 60.0
    cs_2_mean = freq_2_mean * 60.0

# Extract or Generate Mode Shape Coefficients
num_nodes = 11
if len(y_pred[0]) >= (4 + 2 * num_nodes):
    mode_shape_1 = y_pred[0][4 : 4+num_nodes]
    mode_shape_2 = y_pred[0][4+num_nodes : 4+2*num_nodes]
else:
    nodes_pos = np.linspace(0, L_total, num_nodes)
    mode_shape_1 = np.sin(np.pi * nodes_pos / L_total) 
    mode_shape_2 = np.sin(2 * np.pi * nodes_pos / L_total)


# Convert 1 std to a 3 sigma confidence interval for frequencies
conf_1 = 3 * freq_1_std
conf_2 = 3 * freq_2_std

st.subheader("Critical Speeds & Frequencies")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="1st Critical Speed", value=f"{cs_1_mean:,.0f} RPM")
with col2:
    st.metric(
        label="1st Natural Freq", 
        value=f"{freq_1_mean:.1f} Hz",
        delta=f"± {conf_1:.1f} Hz (3σ)"
    )
    if conf_1 > (0.05 * freq_1_mean):
        st.warning("⚠️ High Uncertainty")

with col3:
    st.metric(label="2nd Critical Speed", value=f"{cs_2_mean:,.0f} RPM")
with col4:
    st.metric(
        label="2nd Natural Freq", 
        value=f"{freq_2_mean:.1f} Hz",
        delta=f"± {conf_2:.1f} Hz (3σ)"
    )
    if conf_2 > (0.05 * freq_2_mean):
        st.warning("⚠️ High Uncertainty")

# Campbell diag
st.markdown("---")
col_campbell, col_modes = st.columns(2)

with col_campbell:
    st.subheader("Interactive Operational Envelope")

    rpm_range = np.linspace(0, 10000, 100)
    freq_hz_1x = rpm_range / 60.0

    fig_campbell = go.Figure()

    # Plot 1X Synchronous Excitation Line
    fig_campbell.add_trace(go.Scatter(x=rpm_range, y=freq_hz_1x, mode='lines', name='1X Operating Line (RPM/60)', line=dict(color='black', dash='dash')))

    # Plot Predicted Frequencies as horizontal lines with confidence bands
    for freq, conf, name, color in zip([freq_1_mean, freq_2_mean], [conf_1, conf_2], ['1st Natural Freq', '2nd Natural Freq'], ['blue', 'red']):
        fig_campbell.add_trace(go.Scatter(
            x=np.concatenate([rpm_range, rpm_range[::-1]]),
            y=np.concatenate([np.full_like(rpm_range, freq + conf), np.full_like(rpm_range, freq - conf)]),
            fill='toself',
            fillcolor=f'rgba({255 if color=="red" else 0}, {0 if color=="red" else 0}, {255 if color=="blue" else 0}, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name=f'{name} 3σ Bound'
        ))
        fig_campbell.add_trace(go.Scatter(x=rpm_range, y=np.full_like(rpm_range, freq), mode='lines', name=f'Predicted {name}', line=dict(color=color)))

    fig_campbell.update_layout(
        xaxis_title="Rotor Speed (RPM)",
        yaxis_title="Frequency (Hz)",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_campbell, use_container_width=True)

# mode shape 
with col_modes:
    st.subheader("Predicted Mode Shapes")
    
    fig_modes = go.Figure()
    nodes_x = np.linspace(0, L_total, num_nodes)
    
    fig_modes.add_trace(go.Scatter(x=nodes_x, y=mode_shape_1, mode='lines+markers', name='Mode 1 Shape', line=dict(color='blue', width=3)))
    fig_modes.add_trace(go.Scatter(x=nodes_x, y=mode_shape_2, mode='lines+markers', name='Mode 2 Shape', line=dict(color='red', width=3, dash='dot')))
    
    # add a visual representation of the shaft at y=0
    fig_modes.add_trace(go.Scatter(x=[0, L_total], y=[0, 0], mode='lines', name='Neutral Axis', line=dict(color='black', width=5)))

    fig_modes.update_layout(
        xaxis_title="Shaft Position (m)",
        yaxis_title="Normalized Deflection",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_modes, use_container_width=True)