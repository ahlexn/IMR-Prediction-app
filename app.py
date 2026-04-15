import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

# ----------------------------------------------------------------
# 1. MODEL ARCHITECTURE & LOADING
# ----------------------------------------------------------------
class PresentationANN(nn.Module):
    def __init__(self, input_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_model():
    # Reconciled with your GitHub 'deployment_startup' folder
    path = 'deployment_startup/deployment_model.pth'
    try:
        # Loading full object per previous serialization fix
        model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"⚠️ Model Load Error: {e}")
        return None

# ----------------------------------------------------------------
# 2. PAGE CONFIG & STYLING
# ----------------------------------------------------------------
st.set_page_config(page_title="Executive IMR Dashboard", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 40px; color: #007bff; font-weight: bold; }
    .stButton>button { border-radius: 8px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ District Health Decision Support System")
st.markdown("##### Infant Mortality Rate (IMR) Analytics for India's EAG States")

# ----------------------------------------------------------------
# 3. TOP ROW: INFERENCE ACTION (High Visibility)
# ----------------------------------------------------------------
st.divider()
model = load_model()

pc_names = [
    'Death_Rate', 'Population_And_Marriage', 'Vaccination', 'Population_Urban', 
    'Delivery', 'Foods', 'Death_Rate_Urban', 'Neo_Natal_Mortality', 'Birth_rate', 'Check_Up',
    'Government_Assist', 'BCG_No_Vaccination', 'Illiteracy', 'State'
]

# Create a clean row for the button and the results at the top
inf_col1, inf_col2, inf_col3 = st.columns([1, 1, 2])

with inf_col1:
    st.write("### 1. Execute")
    run_btn = st.button("🚀 RUN FORECAST", use_container_width=True)

with inf_col2:
    st.write("### 2. Prediction")
    prediction_placeholder = st.empty()

with inf_col3:
    st.write("### 3. Strategic Status")
    status_placeholder = st.empty()

st.divider()

# ----------------------------------------------------------------
# 4. DATA ROW: COMPACT INPUTS & CHART
# ----------------------------------------------------------------
col_sliders, col_chart = st.columns([1.2, 1], gap="medium")

inputs = {} 

with col_sliders:
    st.subheader("📍 Input Profile")
    # Grid view: 2 columns of 7 sliders each
    sub_col1, sub_col2 = st.columns(2)
    
    for i, name in enumerate(pc_names):
        target_col = sub_col1 if i < 7 else sub_col2
        
        with target_col:
            label = name.replace('_', ' ')
            # Defaulting to 0.0 (The mathematical mean)
            inputs[name] = st.slider(label, -5.0, 5.0, 0.0, key=f"inp_{name}")

with col_chart:
    st.subheader("📊 Visual Variance")
    # Create the data for the chart based on slider values
    chart_data = pd.DataFrame({"Factor": pc_names, "Strength": [inputs[n] for n in pc_names]})
    # Height adjusted to match the 7-row slider grid
    st.bar_chart(chart_data.set_index("Factor"), height=400)

# ----------------------------------------------------------------
# 5. TRIGGER PREDICTION
# ----------------------------------------------------------------
if run_btn and model:
    # Prepare tensor
    input_list = [inputs[n] for n in pc_names]
    input_tensor = torch.tensor([input_list], dtype=torch.float32)
    
    with st.spinner('Analyzing...'):
        with torch.no_grad():
            prediction = model(input_tensor).item()
        time.sleep(0.3) 
    
    # Push results to the Top Row placeholders
    prediction_placeholder.metric(label="Predicted IMR", value=f"{prediction:.2f}")
    
    if prediction > 50:
        status_placeholder.error("**CRITICAL RISK**\n\nImmediate neonatal intervention required.")
    elif prediction > 35:
        status_placeholder.warning("**ELEVATED RISK**\n\nHigh-intensity support recommended.")
    else:
        status_placeholder.success("**STABLE BASELINE**\n\nStandard maintenance levels.")

# ----------------------------------------------------------------
# 6. FOOTER & LIMITATIONS
# ----------------------------------------------------------------
st.divider()
with st.expander("🛠️ Methodology & Strategic Limitations"):
    st.write("**Architecture:** 4-Layer MLP | **Feature Engineering:** 14-Cluster PCA Transformation")
    st.warning("This model is calibrated for India's 9 EAG states. Predictions for developed urban centers outside of these states may be inaccurate.")
