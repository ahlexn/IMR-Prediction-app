import streamlit as st
import os 
import pickle
from PIL import Image

import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import TargetEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import shap
import streamlit_shap as st_shap

# Function to load the target encoder from training
def load_encoder(filepath = 'target_encoder.pkl'):
    """Loads an encoder from a pickle file."""
    with open(filepath, 'rb') as f:
        encoder = pickle.load(f)
    print(f"Encoder loaded from {filepath}")
    return encoder

# Function to load the robust scaler from training
def load_scaler(filepath = 'robust_scaler.pkl'):
    """Loads an encoder from a pickle file."""
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {filepath}")
    return scaler

# Function to load the pickle files of pca_models for conversion of new input
def load_pca_models(folder="pca_models"):
    """Reloads all pickle files from a folder into a dictionary."""
    models_dict = {}
    if not os.path.exists(folder):
        print("Folder not found.")
        return models_dict

    for filename in os.listdir(folder):
        if filename.endswith(".pkl"):
            name = filename.replace(".pkl", "")
            filepath = os.path.join(folder, filename)
            with open(filepath, 'rb') as f:
                models_dict[name] = pickle.load(f)
            print(f"Loaded: {name}")
    
    return models_dict

# Reload the cluster features to order new inputs
with open("feat_clusters.pkl", "rb") as f:
    feat_clusters = pickle.load(f)

# Reload all necessary features
with open("all_feats.pkl", "rb") as f:
    all_feats = pickle.load(f)

with open("features_mini.pkl", "rb") as f:
    features_mini = pickle.load(f)
    
# Add the state name to the clusters
feat_clusters.append(['State_Name'])

def create_processable_df(base):
    # Create a blank matrix of zeros
    frame = pd.DataFrame(0, index=base.index, columns=all_feats)
    
    # Add information from the input
    frame.update(base)

    return frame

# Function to process a csv input
def process_input_data(df):
    # Load the encoder and scaler
    encoder = load_encoder()
    scaler = load_scaler()
    
    # Check that all necessary features are present
    if model_type == full_model and set(all_feats).issubset(list(df.columns)):
        df = df[all_feats]

        # 1. Load the target encoder from training
        # 2. Target encode the state names
        # 3. Load the robust scaler from training
        # 4. Scale the features
        
        # Encoding
        df[['State_Name']] = encoder.transform(df[['State_Name']])
        
        # Scaling
        scaled = scaler.transform(df)
        df = pd.DataFrame(scaled, columns=df.columns)
        
        # Load the pca_models
        pca_models = load_pca_models()
        
        #For each cluster in the feat_clusters
        # 1. Take a cluster of features
        # 2. Take the corresponding pca model
        # 3. Compress the cluster into a single component
        # 4. Add the component to the compressed_data
        # 5. Rename the components for interpretability
        # 6. Append the components into a single DataFrame
        
        compressed_data = []
        for clus in range(len(feat_clusters)):
            pca = pca_models[f'pca_clus_{clus+1}']
            compressed_data.append(pd.DataFrame(pca.transform(df[feat_clusters[clus]])))
        
        # Component names from training
        pc_names = ['Death_Rate', 'Population_And_Marriage', 'Vaccination', 'Population_Urban', 
                'Delivery', 'Foods', 'Death_Rate_Urban', 'Neo_Natal_Mortality', 'Birth_rate', 'Check_Up',
                'Government_Assist', 'BCG_No_Vaccination', 'Illiteracy', 'State']
        
        ## Rename PC groups for interpretability
        for x in range(len(compressed_data)):
            compressed_data[x] = compressed_data[x].rename(columns={0: pc_names[x]})
        
        # Append all of the compenents into a single data frame
        input_features = pd.DataFrame()
        
        for x in range(len(compressed_data)):
            input_features = pd.concat([input_features, compressed_data[x]], axis=1)

    elif model_type == mini_model and set(features_mini).issubset(list(df.columns)):
        # Create a base dataframe that fits the encoder and scaler
        df = create_processable_df(df[features_mini])
        # Encoding
        df[['State_Name']] = encoder.transform(df[['State_Name']])
        
        # Scaling
        scaled = scaler.transform(df)
        df = pd.DataFrame(scaled, columns=df.columns)

        # Extract the featureset
        input_features = df[features_mini]
        
    else:
        st.write('Missing Features!')
        return None
        
    return input_features

# Function to convert a dataframe to a tensor for the model
def df_to_tensor(df):
    """ converts an input df to a PyTorch tensor """
    values = df.values
    tensor = torch.tensor(values, dtype=torch.float)
    return tensor

# Function to make a prediction
def make_prediction(model, input_tensor, districts):
    preds = model(input_tensor).squeeze(dim=1).tolist()
    preds = pd.DataFrame(preds, columns=['Predicted IMR'])
    ret = pd.concat([districts, preds], axis=1)
    return ret
    
def runner(df, districts):

    # Define necessary elements
    processed_data = process_input_data(df)
    input_tensor = df_to_tensor(processed_data)

    # Load the requested model
    if model_type == full_model:
        model = torch.load('deployment_model.pth', weights_only = False)
    elif model_type == mini_model:
        model = torch.load('ann_mini.pth', weights_only = False)
        
    # Set the model to evaluation mode    
    model.eval()

    # Make Prediction
    with torch.no_grad():
        results = make_prediction(model, input_tensor, districts)
        
    # Display the predictions
    st.write(""" #### IMR Predictions """)
    st.dataframe(results, hide_index=True)

def file_upload(uploaded_file=None, template=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # preview the uploaded file
        st.write("### Uploaded file preview")
        st.dataframe(df)
        
    elif template is not None:
        df = template

    districts = df[["State_District_Name"]]
    districts = districts.rename(columns={'State_District_Name': 'District Name'})

    runner(df, districts)
## Main
full_model = 'IMR Predictor'
model_type = full_model
df = st.file_uploader('Upload Your Data')

file_upload(df)