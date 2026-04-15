import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

# ----------------------------------------------------------------
# 1. MODEL ARCHITECTURE
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
    path = 'deployment_startup/deployment_model.pth'
    try:
        model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"⚠️ Deployment Error: {e}")
        return None

# ----------------------------------------------------------------
# 2. UI CONFIG & STYLING
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
# 3. TOP ROW: INFERENCE ACTION
# ----------------------------------------------------------------
st.divider()
model = load_model()

# PCA names for the first 13 HEALTH components
health_pcs = [
    'Death_Rate', 'Population_And_Marriage', 'Vaccination', 'Population_Urban', 
    'Delivery', 'Foods', 'Death_Rate_Urban', 'Neo_Natal_Mortality', 'Birth_rate', 'Check_Up',
    'Government_Assist', 'BCG_No_Vaccination', 'Illiteracy'
]

# State Mapping - REPLACE 0.0 WITH YOUR ACTUAL PCA VALUES
state_map = {
    "Bihar": 0.0, 
    "Chhattisgarh": 0.0, 
    "Jharkhand": 0.0, 
    "Madhya Pradesh": 0.0, 
    "Odisha": 0.0, 
    "Rajasthan": 0.0, 
    "Uttar Pradesh": 0.0, 
    "Uttarakhand": 0.0, 
    "Assam": 0.0
}

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
# 4. DATA ROW: INPUTS & CHART
# ----------------------------------------------------------------
col_sliders, col_chart = st.columns([1.2, 1], gap="medium")

inputs = {} 

with col_sliders:
    st.subheader("📍 Input Profile")
    sub_col1, sub_col2 = st.columns(2)
    
    # Generate sliders for the 13 health metrics
    for i, name in enumerate(health_pcs):
        target_col = sub_col1 if i < 7 else sub_col2
        with target_col:
            label = name.replace('_', ' ')
            inputs[name] = st.slider(label, -5.0, 5.0, 0.0, key=f"inp_{name}")
    
    # The 14th component: Categorical State Selection
    with sub_col2:
        st.write("---")
        selected_state = st.selectbox("Select Target State", options=list(state_map.keys()), help="Sets the baseline geographic variance.")
        inputs['State_Val'] = state_map[selected_state]

with col_chart:
    st.subheader("📊 Visual Variance")
    # Prepare chart labels: 13 health PCs + the selected State name
    chart_labels = health_pcs + [selected_state]
    chart_values = [inputs[n] for n in health_pcs] + [inputs['State_Val']]
    
    chart_df = pd.DataFrame({"Factor": chart_labels, "Strength": chart_values})
    st.bar_chart(chart_df.set_index("Factor"), height=415)

# ----------------------------------------------------------------
# 5. TRIGGER PREDICTION
# ----------------------------------------------------------------
if run_btn and model:
    # Build tensor in the exact order the model expects
    input_list = [inputs[n] for n in health_pcs] + [inputs['State_Val']]
    input_tensor = torch.tensor([input_list], dtype=torch.float32)
    
    with st.spinner('Analyzing...'):
        with torch.no_grad():
            prediction = model(input_tensor).item()
        time.sleep(0.3) 
    
    prediction_placeholder.metric(label="Predicted IMR", value=f"{prediction:.2f}")
    
    if prediction > 50:
        status_placeholder.error("**CRITICAL RISK**\n\nImmediate neonatal intervention required.")
    elif prediction > 35:
        status_placeholder.warning("**ELEVATED RISK**\n\nHigh-intensity support recommended.")
    else:
        status_placeholder.success("**STABLE BASELINE**\n\nStandard maintenance levels.")

# ----------------------------------------------------------------
# 6. FOOTER
# ----------------------------------------------------------------
st.divider()
with st.expander("🛠️ Methodology & Strategic Limitations"):
    st.write("**Architecture:** 4-Layer MLP | **Feature Engineering:** 14-Cluster PCA Transformation")
    st.warning("This model is calibrated for India's 9 EAG states[cite: 49]. Predictions for urban centers outside of these states may be inaccurate[cite: 51].")
