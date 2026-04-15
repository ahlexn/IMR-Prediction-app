import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

# 1. Model Architecture
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

# 2. Load Model
@st.cache_resource
def load_model():
    path = 'deployment_startup/deployment_model.pth'
    try:
        model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

# 3. Page Config
st.set_page_config(page_title="IMR Prediction Dashboard", page_icon="📈", layout="wide")

st.title("🛡️ District Health Decision Support System")
st.markdown("---")

# 4. Sidebar: Direct Sliders (No Expanders for better visibility)
st.sidebar.header("📍 District Health Profile")
st.sidebar.markdown("Adjust the values below to simulate district conditions.")

pc_names = [
    'Death_Rate', 'Population_And_Marriage', 'Vaccination', 'Population_Urban', 
    'Delivery', 'Foods', 'Death_Rate_Urban', 'Neo_Natal_Mortality', 'Birth_rate', 'Check_Up',
    'Government_Assist', 'BCG_No_Vaccination', 'Illiteracy', 'State'
]

# We use a dictionary to store the slider values
inputs = []
with st.sidebar:
    for name in pc_names:
        # We add a unique key and format the label for clarity
        label = name.replace('_', ' ')
        val = st.slider(label, min_value=-5.0, max_value=5.0, value=0.0, key=f"slider_{name}")
        inputs.append(val)

# 5. Main Dashboard
model = load_model()

if model:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Predictive Forecast")
        if st.button("🚀 Run Neural Network Inference", use_container_width=True):
            with st.spinner('Calculating regional interactions...'):
                input_tensor = torch.tensor([inputs], dtype=torch.float32)
                with torch.no_grad():
                    prediction = model(input_tensor).item()
                time.sleep(0.4)
            
            st.metric(label="Predicted Infant Mortality Rate", value=f"{prediction:.2f}")
            
            if prediction > 50:
                st.error("**Critical:** High mortality risk detected.")
            elif prediction > 35:
                st.warning("**Elevated:** Increased risk levels.")
            else:
                st.success("**Stable:** Within baseline expectations.")

    with col2:
        st.subheader("Factor Variance")
        chart_data = pd.DataFrame({"Factor": pc_names, "Strength": inputs})
        st.bar_chart(chart_data.set_index("Factor"))

# 6. Documentation
st.divider()
with st.expander("⚠️ Strategic Limitations & Methodology"):
    st.info("This model is calibrated for India's 9 EAG states. Predictions for developed urban centers outside of these states may be inaccurate.")
    st.write("Architecture: 4-Layer MLP | Input: 14 Principal Components")
