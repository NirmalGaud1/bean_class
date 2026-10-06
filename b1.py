#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="centered"
)

@st.cache_resource
def load_model():
    package = joblib.load("knn_dry_bean_model.pkl")
    return package

package = load_model()

model = package["model"]
scaler = package["scaler"]
le = package["label_encoder"]
features = package["features"]
best_k = package["best_k"]

st.title("🌱 Dry Bean Classification")
st.write("K-Nearest Neighbors (KNN) Classification")

st.info(f"Selected Features: {len(features)} | Best K: {best_k}")

st.subheader("Enter Bean Measurements")

perimeter = st.number_input(
    "Perimeter",
    min_value=0.0,
    value=650.0
)

aspect_ratio = st.number_input(
    "Aspect Ratio",
    min_value=0.0,
    value=1.85
)

roundness = st.number_input(
    "Roundness",
    min_value=0.0,
    value=0.72
)

shape_factor4 = st.number_input(
    "Shape Factor 4",
    min_value=0.0,
    value=0.99
)

solidity = st.number_input(
    "Solidity",
    min_value=0.0,
    value=0.98
)

extent = st.number_input(
    "Extent",
    min_value=0.0,
    value=0.75
)

if st.button("🔍 Predict Bean Class", use_container_width=True):

    input_data = pd.DataFrame([{
        "Perimeter": perimeter,
        "AspectRation": aspect_ratio,
        "roundness": roundness,
        "ShapeFactor4": shape_factor4,
        "Solidity": solidity,
        "Extent": extent
    }])

    input_data = input_data[features]

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    predicted_class = le.inverse_transform(prediction)[0]

    st.success(f"Predicted Bean Class: **{predicted_class}**")

    st.subheader("Input Data")

    st.dataframe(input_data, use_container_width=True)

