import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

# ----------------------------------------------------------------
# 1. MODEL ARCHITECTURE (Required for Object Reconstruction)
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
    # Path reconciled with your GitHub folder structure
    path = 'deployment_startup/deployment_model.pth'
    try:
        # Loading as a full object per your previous serialization fix
        model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"⚠️ Deployment Error: Could not locate model at {path}. Error: {e}")
        return None

# ----------------------------------------------------------------
# 2. UI CONFIG & STYLING
# ----------------------------------------------------------------
st.set_page_config(page_title="Executive IMR Dashboard", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    div[data-testid="stMetricValue"] { font-size: 48px; color: #007bff; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #007bff; color: white; font-weight: bold; }
    .readout { font-family: 'monospace'; font-weight: bold; color: #007bff; font-size: 1.2em; text-align: center; padding-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ District Health Decision Support System")
st.markdown("##### Infant Mortality Rate (IMR) Strategic Forecasting | India's EAG States")

# ----------------------------------------------------------------
# 3. TOP ACTION BAR (The "Hero" Section)
# ----------------------------------------------------------------
st.divider()
model = load_model()

# Header Row for immediate feedback
top_col1, top_col2, top_col3 = st.columns([1, 1, 2])

with top_col1:
    st.write("### 🚀 Step 1")
    run_btn = st.button("RUN NEURAL FORECAST")

with top_col2:
    st.write("### 📊 Step 2: Prediction")
    res_placeholder = st.empty() # Filled after inference

with top_col3:
    st.write("### 🚨 Step 3: Status")
    status_placeholder = st.empty() # Filled after inference

st.divider()

# ----------------------------------------------------------------
# 4. THE ROW-BASED INPUT GRID
# ----------------------------------------------------------------
# PCA names for the model tensor
pc_names = [
    'Death_Rate', 'Population_And_Marriage', 'Vaccination', 'Population_Urban', 
    'Delivery', 'Foods', 'Death_Rate_Urban', 'Neo_Natal_Mortality', 'Birth_rate', 'Check_Up',
    'Government_Assist', 'BCG_No_Vaccination', 'Illiteracy', 'State'
]

col_sliders, col_chart = st.columns([1.5, 1], gap="large")

final_values = []

with col_sliders:
    st.subheader("📍 Input Profile (Z-Scores)")
    st.info("Defaults set to 0.0 (Regional Mean). Adjust rows to simulate district variance.")
    
    # Building the Row-Based architecture
    for name in pc_names:
        label = name.replace('_', ' ')
        row_col1, row_col2 = st.columns([4, 1])
        
        with row_col1:
            # All defaults now set to 0.0
            val = st.slider(label, -5.0, 5.0, 0.0, key=f"sl_{name}")
            final_values.append(val)
        
        with row_col2:
            # Digital readout for professional aesthetic
            st.markdown(f"<p class='readout'>{val:.2f}</p>", unsafe_allow_html=True)

with col_chart:
    st.subheader("📊 Feature Variance Chart")
    st.markdown("Visual mapping of the selected profile across the 14 clusters.")
    
    # Sync chart with the live slider values
    chart_df = pd.DataFrame({"Factor": pc_names, "Strength": final_values})
    st.bar_chart(chart_df.set_index("Factor"), height=750)

# ----------------------------------------------------------------
# 5. INFERENCE EXECUTION
# ----------------------------------------------------------------
if run_btn and model:
    input_tensor = torch.tensor([final_values], dtype=torch.float32)
    
    with st.spinner('Neural Network Inference in progress...'):
        with torch.no_grad():
            prediction = model(input_tensor).item()
        time.sleep(0.4) # UX polish to show "work" being done
    
    # Update the Top Hero Row
    res_placeholder.metric(label="Predicted IMR", value=f"{prediction:.2f}")
    
    if prediction > 50:
        status_placeholder.error("**CRITICAL RISK**\n\nImmediate Priority: Neonatal intervention & specialized care.")
    elif prediction > 35:
        status_placeholder.warning("**ELEVATED RISK**\n\nPriority: Enhance immunization & institutional delivery access.")
    else:
        status_placeholder.success("**STABLE BASELINE**\n\nDistrict is performing within expected average parameters.")

# ----------------------------------------------------------------
# 6. DOCUMENTATION & LIMITATIONS
# ----------------------------------------------------------------
st.divider()
with st.expander("🛠️ Methodology & Limitations"):
    st.write("**Architecture:** 4-Layer Multi-Layer Perceptron ($ANN$)")
    st.write("**Data Pipeline:** Robust Scaler → 14-Cluster Principal Component Analysis ($PCA$)")
    st.warning("This tool is calibrated for the 9 EAG states of India. Applying this model to highly developed urban centers outside of these states may result in out-of-distribution inaccuracies.")
