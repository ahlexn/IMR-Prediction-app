import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle

# 1. Define the Architecture (Must match your Slide 13 exactly)
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

# 2. Load the Model
@st.cache_resource
def load_model():
    model = PresentationANN(input_dim=14)
    # Ensure you have saved your model weights as 'final_ann_model.pth'
    model.load_state_dict(torch.load('deployment_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# 3. App UI
st.set_page_config(page_title="IMR Predictor", page_icon="👶")
st.title("District-Level Infant Mortality Predictor")
st.markdown("""
This tool uses a **Custom Artificial Neural Network** to predict Infant Mortality Rates (IMR) 
in India's EAG states based on 14 Principal Component inputs.
""")

st.sidebar.header("Input Health Metrics (PCA Values)")

# Generate 14 sliders for the user to input the PCA values
# We use sliders because PCA components are usually centered around 0
inputs = []
pc_names = [
    'Death_Rate','Population_Marriage','Vaccination','Population_Urban',
    'Delivery','Foods','Death_Rate_Urban','Neo_Natal_Mortality','Birth_rate',
    'Check_Up','Govt_Assist','BCG_Vaccination','Illiteracy', 'State_Encoded'
]

for name in pc_names:
    val = st.sidebar.slider(f"Component: {name}", -5.0, 5.0, 0.0)
    inputs.append(val)

# 4. Prediction Logic
model = load_model()

if st.button("Predict Infant Mortality Rate"):
    input_tensor = torch.tensor([inputs], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    st.subheader(f"Predicted IMR: {prediction:.2f}")
    st.write("Deaths per 1,000 live births.")
    
    # Interpretation Guidance
    if prediction > 50:
        st.error("⚠️ Warning: This district is predicted to have a Critical IMR level.")
    elif prediction > 35:
        st.warning("ℹ️ Notice: This district is predicted to have an Elevated IMR level.")
    else:
        st.success("✅ Stable: This district is predicted to have a standard IMR for EAG states.")

st.divider()
st.info("**Limitations & Warnings:** This model is designed for districts in EAG states. Predictions for developed urban centers outside of these states may be inaccurate. PCA inputs must be pre-calculated based on original scaling.")
