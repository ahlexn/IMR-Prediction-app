import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

# ----------------------------------------------------------------
# 1. MODEL DEFINITION & LOADING
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
    # Points to the folder structure established in your GitHub repo
    path = 'deployment_startup/deployment_model.pth'
    try:
        model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {path}. Please check your GitHub structure.")
        return None

# ----------------------------------------------------------------
# 2. PAGE CONFIGURATION & UI STYLING
# ----------------------------------------------------------------
st.set_page_config(page_title="Executive IMR Dashboard", page_icon="📈", layout="wide")

# Custom CSS for a clean, modern aesthetic
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 45px; color: #007bff; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .stExpander { background-color: #ffffff; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ District Health Decision Support System")
st.markdown("##### Infant Mortality Rate (IMR) Predictive Analytics for EAG States")
st.divider()

# ----------------------------------------------------------------
# 3. SIDEBAR: INPUT PARAMETERS (The PCA Components)
# ----------------------------------------------------------------
st.sidebar.header("📍 District Health Profile")
st.sidebar.markdown("Adjust the **Principal Components** below to simulate district conditions.")

pc_names = [
    'Death_Rate', 'Population_And_Marriage', 'Vaccination', 'Population_Urban', 
    'Delivery', 'Foods', 'Death_Rate_Urban', 'Neo_Natal_Mortality', 'Birth_rate', 'Check_Up',
    'Government_Assist', 'BCG_No_Vaccination', 'Illiteracy', 'State'
]

inputs = []
# Grouping sliders to avoid a "wall of sliders"
with st.sidebar:
    with st.expander("Vital & Demographics", expanded=True):
        for name in pc_names[:4]:
            inputs.append(st.slider(name.replace('_', ' '), -5.0, 5.0, 0.0))
    
    with st.expander("Clinical & Nutrition", expanded=False):
        for name in pc_names[4:9]:
            inputs.append(st.slider(name.replace('_', ' '), -5.0, 5.0, 0.0))
            
    with st.expander("Socio-Economic & State Factors", expanded=False):
        for name in pc_names[9:]:
            inputs.append(st.slider(name.replace('_', ' '), -5.0, 5.0, 0.0))

# ----------------------------------------------------------------
# 4. MAIN DASHBOARD: PREDICTION & VISUALIZATION
# ----------------------------------------------------------------
model = load_model()

if model:
    # Split layout: Results on the left, Data Visualization on the right
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Predictive Forecast")
        if st.button("🚀 Run Neural Network Inference"):
            with st.spinner('Calculating complex regional interactions...'):
                input_tensor = torch.tensor([inputs], dtype=torch.float32)
                with torch.no_grad():
                    prediction = model(input_tensor).item()
                time.sleep(0.4) # UX pause for "work" effect
            
            # Outcome Metric
            st.metric(label="Predicted Infant Mortality Rate", value=f"{prediction:.2f}")
            st.caption("Units: Deaths per 1,000 live births.")
            
            # Dynamic Advisory Logic
            if prediction > 50:
                st.error("**Action Required:** High mortality risk. Prioritize neonatal critical care resources.")
            elif prediction > 35:
                st.warning("**Alert:** Elevated risk. Review vaccination coverage and institutional delivery access.")
            else:
                st.success("**Stable:** Projected IMR is within the baseline range for identified EAG districts.")

    with col2:
        st.subheader("Input Variance")
        # Visualizing the input PCA components for context
        chart_data = pd.DataFrame({"Factor": pc_names, "Strength": inputs})
        st.bar_chart(chart_data.set_index("Factor"))

# ----------------------------------------------------------------
# 5. TECHNICAL DOCUMENTATION (For Executive Defense Q&A)
# ----------------------------------------------------------------
st.divider()
footer_1, footer_2 = st.columns(2)

with footer_1:
    with st.expander("🛠️ Deployment Methodology"):
        st.write("""
        - **Pipeline:** Raw Data → Robust Scaler → 14-Cluster PCA → ANN.
        - **Architecture:** 4-Layer Multi-Layer Perceptron ($MLP$).
        - **Operational readiness:** Fully containerized via Streamlit Cloud.
        """)

with footer_2:
    with st.expander("⚠️ Strategic Limitations"):
        st.info("""
        **User Warning:** This model is calibrated for India's 9 EAG states. 
        Predictions for highly developed urban centers outside of these states 
        may be inaccurate due to demographic variance.
        """)
