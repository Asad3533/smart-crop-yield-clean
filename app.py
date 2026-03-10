import os
import pickle
import streamlit as st
import pandas as pd

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Smart Crop Yield Prediction", layout="wide")
MODEL_PATH = "crop_yield_model.pkl"
DATA_PATH = "final_crop_dataset.csv"

# -----------------------------
# Model Check
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.warning(
        "⚠️ Model file not found!\n"
        "Please download 'crop_yield_model.pkl' from Google Drive and place it in this folder.\n\n"
        "[Download here](https://drive.google.com/file/d/1Yx_Ew5qDdLjmk7mGb3tcGnCKdILBS_pS/view?usp=sharing)"
    )
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Load Dataset for Reference
# -----------------------------
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = pd.DataFrame()  # empty dataframe if CSV missing

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Input Parameters")

# Example inputs: adjust based on your dataset columns
year = st.sidebar.number_input("Year", min_value=int(df['Year'].min()) if not df.empty else 2000,
                               max_value=int(df['Year'].max()) if not df.empty else 2030,
                               value=2025)
area = st.sidebar.selectbox("Area", sorted(df['Area'].unique()) if 'Area' in df.columns else ["Pakistan"])
crop = st.sidebar.selectbox("Crop", sorted(df['Item'].unique()) if 'Item' in df.columns else ["Wheat"])
rainfall = st.sidebar.number_input("Average Rainfall (mm)", min_value=0.0, value=500.0)
temperature = st.sidebar.number_input("Average Temperature (°C)", min_value=-10.0, value=25.0)
pesticide = st.sidebar.number_input("Pesticide Usage (units)", min_value=0.0, value=10.0)

# -----------------------------
# Prediction Button
# -----------------------------
if st.sidebar.button("Predict Crop Yield"):
    # Prepare input for model: match your feature engineering
    input_df = pd.DataFrame({
        'Year': [year],
        'Rainfall': [rainfall],
        'Temperature': [temperature],
        'Pesticide': [pesticide],
        'Area_' + area: [1],  # one-hot encoding simulation
        'Item_' + crop: [1]
    })

    # Ensure all model features exist in input
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0  # add missing columns as 0

    # Predict
    pred = model.predict(input_df)[0]
    st.success(f"🌾 Predicted Crop Yield: {pred:.2f} units")

# -----------------------------
# Data Exploration (Optional)
# -----------------------------
st.header("Dataset Overview")
st.dataframe(df.head(10))

st.header("Yield Distribution")
if not df.empty and 'Yield' in df.columns:
    st.bar_chart(df.groupby('Item')['Yield'].mean())