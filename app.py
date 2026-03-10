import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ------------------------------
# Load pre-trained ML model
# ------------------------------
model = pickle.load(open("crop_yield_model.pkl", "rb"))

# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv("final_crop_dataset.csv")

# Extract one-hot encoded columns
area_cols = [col for col in df.columns if col.startswith("Area_")]
item_cols = [col for col in df.columns if col.startswith("Item_")]

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🌱 Smart Crop Yield Prediction System")

# Select Country
country = st.selectbox("Select Country", [c.replace("Area_","") for c in area_cols])

# Select Crop
crop = st.selectbox("Select Crop", [c.replace("Item_","") for c in item_cols])

# Input numeric features
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, value=25.0)
pesticide = st.number_input("Pesticide Usage (kg/ha)", min_value=0.0, value=50.0)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Yield"):
    # Create input vector
    input_data = pd.DataFrame(columns=df.drop("Yield", axis=1).columns)
    input_data.loc[0] = 0  # initialize all zeros

    # Fill numeric values
    input_data.at[0, "Rainfall"] = rainfall
    input_data.at[0, "Temperature"] = temperature
    input_data.at[0, "Pesticide"] = pesticide

    # Set one-hot encoded features
    input_data.at[0, "Area_" + country] = 1
    input_data.at[0, "Item_" + crop] = 1

    # Predict
    pred = model.predict(input_data)[0]
    st.success(f"Predicted Yield: {pred:.2f} tons/hectare")

# ------------------------------
# Global Yield Map
# ------------------------------
st.subheader("🌍 Global Crop Yield Map")

# Compute average yield per country
yield_map = []
for col in area_cols:
    mean_yield = df.loc[df[col] == 1, "Yield"].mean()
    country_name = col.replace("Area_", "")
    yield_map.append({"Country": country_name, "Yield": mean_yield})

yield_map_df = pd.DataFrame(yield_map)

# Plot interactive map
fig = px.choropleth(
    yield_map_df,
    locations="Country",
    locationmode="country names",
    color="Yield",
    hover_name="Country",
    color_continuous_scale="Viridis",
    title="Global Crop Yield Distribution"
)

st.plotly_chart(fig, use_container_width=True)