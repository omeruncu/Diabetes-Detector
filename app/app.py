import streamlit as st
import joblib
import numpy as np

model = joblib.load("models/random_forest_diabetes_model.pkl")
features = joblib.load("models/selected_features.pkl")

st.title("Diabetes Risk Predictor")

user_input = []
for feature in features:
    val = st.number_input(f"{feature}", value=0.0)
    user_input.append(val)

if st.button("Tahmin Et"):
    prediction = model.predict([user_input])[0]
    prob = model.predict_proba([user_input])[0][1]
    st.write(f"**Sonuç:** {'Diyabetli' if prediction == 1 else 'Diyabetli Değil'}")
    st.write(f"**Olasılık:** {prob:.2%}")