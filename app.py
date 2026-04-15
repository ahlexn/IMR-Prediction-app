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

# ----------------------------------------------------------------
# 2. MODEL LOADING (Object-loading logic)
# ----------------------------------------------------------------
@st.cache_resource
def load_model():
    path = 'deployment_startup/deployment_model.pth'
    try:
        model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model file: {e}")
        return None

# ----------------------------------------------------------------
# 3. PAGE CONFIG & STYLING
# ----------------------------------------------------------------
st.set_page_config(page_title="Executive IMR Dashboard", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 50px; color: #007bff; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #007bff; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ District Health Decision Support System")
st.markdown("##### Infant Mortality Rate (IMR) Strategic Forecasting for EAG States")
st.divider()

# ----------------------------------------------------------------
# 4. DATA SYNCHRONIZATION: COLUMNS VS SIDEBAR
# ----------------------------------------------------------------
pc_names = [
    'Death_Rate', 'Population_And_Marriage', 'Vaccination', 'Population_Urban', 
    'Delivery', 'Foods', 'Death_Rate_Urban', 'Neo_Natal_Mortality', 'Birth_rate', 'Check_Up',
    'Government_Assist', 'BCG_No_Vaccination', 'Illiteracy', 'State'
]

# We create two main columns to replace the sidebar misalignment
col_input, col_chart = st.columns([1, 1], gap="large")

inputs = []

with col_input:
    st.subheader("📍 Input Profile")
    st.info("Adjust the factors below to simulate district conditions.")
    
    # We display all 14 sliders in the main area for better alignment with the chart
    for name in pc_names:
        label = name.replace('_', ' ')
        # Special help for BCG based on your question
        help_text = "Bacille Calmette-Guérin (TB Vaccine) Coverage Component" if "BCG" in name else None
        
        val = st.slider(label, -5.0, 5.0, 0.0, key=f"main_{name}", help=help_text)
        inputs.append(val)

with col_chart:
    st.subheader("📊 Visual Variance")
    st.markdown("Real-time distribution of influence across the 14 PCA clusters.")
    
    # We force the chart to have a larger height to match the long list of sliders
    chart_data = pd.DataFrame({"Factor": pc_names, "Strength": inputs})
    st.bar_chart(chart_data.set_index("Factor"), height=650) 

# ----------------------------------------------------------------
# 5. INFERENCE & STRATEGIC ADVISORY
# ----------------------------------------------------------------
st.divider()
model = load_model()

if model:
    # Action area for the big button and results
    action_col, result_col = st.columns([1, 2])

    with action_col:
        st.write("### Ready for Inference?")
        run_btn = st.button("🚀 RUN NEURAL NETWORK FORECAST")

    with result_col:
        if run_btn:
            with st.spinner('Calculating complex regional interactions...'):
                input_tensor = torch.tensor([inputs], dtype=torch.float32)
                with torch.no_grad():
                    prediction = model(input_tensor).item()
                time.sleep(0.5) # UX polish
            
            # Prediction Results
            res_1, res_2 = st.columns(2)
            with res_1:
                st.metric(label="Predicted IMR", value=f"{prediction:.2f}")
                st.caption("Units: Deaths per 1,000 live births.")
            
            with res_2:
                if prediction > 50:
                    st.error("**STATUS: CRITICAL**\n\nPriority: Urgent Neonatal Care Intervention.")
                elif prediction > 35:
                    st.warning("**STATUS: ELEVATED**\n\nPriority: High-Intensity Vaccination & Delivery Support.")
                else:
                    st.success("**STATUS: STABLE**\n\nPriority: Standard District Maintenance.")

# ----------------------------------------------------------------
# 6. DOCUMENTATION & LIMITATIONS
# ----------------------------------------------------------------
st.divider()
with st.expander("🛠️ Methodology & Strategic Limitations"):
    st.write("**Architecture:** 4-Layer MLP | **Feature Engineering:** 14-Cluster PCA Transformation")
    st.warning("This model is calibrated for India's 9 EAG states. Predictions for developed urban centers outside of these states may be inaccurate due to socioeconomic variance.")
