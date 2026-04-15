import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

# 1. Define the Architecture (Required for PyTorch to reconstruct the object)
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

# 2. Load the Model (Using your specific directory and object-loading fix)
@st.cache_resource
def load_model():
    path = 'deployment_startup/deployment_model.pth'
    model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    return model

# 3. Page Configuration
st.set_page_config(page_title="Executive IMR Dashboard", page_icon="📈", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_code=True)

st.title("🛡️ District Health Decision Support System")
st.markdown("---")

# 4. Sidebar Inputs (Organized for Operational Readiness)
st.sidebar.header("📍 District Parameters")
st.sidebar.info("Adjust the PCA components below to simulate regional health profiles.")

pc_names = [
    'Death_Rate', 'Population_And_Marriage', 'Vaccination', 'Population_Urban', 
    'Delivery', 'Foods', 'Death_Rate_Urban', 'Neo_Natal_Mortality', 'Birth_rate', 'Check_Up',
    'Government_Assist', 'BCG_No_Vaccination', 'Illiteracy', 'State'
]

inputs = []
with st.sidebar:
    for name in pc_names:
        val = st.slider(f"{name.replace('_', ' ')}", -5.0, 5.0, 0.0, help=f"Adjusted factor for {name}")
        inputs.append(val)

# 5. Main Area - Results & Metrics
model = load_model()

# Use columns to separate the prediction from the interpretation
main_col, side_col = st.columns([2, 1])

with main_col:
    st.subheader("Predictive Analytics")
    if st.button("Generate Strategic Forecast", use_container_width=True):
        with st.spinner('Running PCA Transform and ANN Inference...'):
            input_tensor = torch.tensor([inputs], dtype=torch.float32)
            with torch.no_grad():
                prediction = model(input_tensor).item()
            time.sleep(0.5) # UX Pause
        
        # Display the "Hero" Number
        st.metric(label="Predicted Infant Mortality Rate", value=f"{prediction:.2f}")
        st.caption("Units: Deaths per 1,000 live births.")
        
        # Visual Status
        if prediction > 50:
            st.error(f"**Critical Level Identified:** This district requires immediate neonatal intervention.")
        elif prediction > 35:
            st.warning(f"**Elevated Risk:** Targeted vaccination and delivery assistance recommended.")
        else:
            st.success(f"**Stable Baseline:** District is performing within standard EAG state expectations.")

with side_col:
    st.subheader("Model Insights")
    st.write("Current analysis profile:")
    # Show a small bar chart of the inputs for visual feedback
    input_df = pd.DataFrame({"Component": pc_names, "Value": inputs})
    st.bar_chart(input_df.set_index("Component"))

# 6. Technical Transparency (Expanders)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True) # Change made here
