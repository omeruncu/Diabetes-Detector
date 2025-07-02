import pandas as pd
import numpy as np
import streamlit as st
import joblib
import sys
import os

# Yol ayarı
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Özel modüller
import src.features.build_features as fb
import src.features.encode_scale as fes

# Model, özellik listesi ve scaler'ı yükle
model = joblib.load("models/best_model.pkl")
features = joblib.load("models/selected_features.pkl")
scaler = joblib.load("models/scaler.pkl")  # Eğitimde kaydedilen scaler

# Başlık
st.title("🩺 Diabetes Risk Predictor")
st.markdown("### Lütfen aşağıdaki bilgileri girin:")

# Kullanıcıdan ham verileri al
PREGNANCIES = st.slider("Pregnancies", 0, 20, 1)
GLUCOSE = st.slider("Glucose", 50, 200, 100)
BLOODPRESSURE = st.slider("Blood Pressure", 30, 130, 70)
SKINTHICKNESS = st.slider("Skin Thickness", 0, 100, 20)
INSULIN = st.slider("Insulin", 0, 900, 80)
BMI = st.slider("BMI", 10.0, 70.0, 25.0)
DIABETESPEDIGREEFUNCTION = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
AGE = st.slider("Age", 10, 100, 33)

# DataFrame oluştur
input_df = pd.DataFrame([{
    "PREGNANCIES": PREGNANCIES,
    "GLUCOSE": GLUCOSE,
    "BLOODPRESSURE": BLOODPRESSURE,
    "SKINTHICKNESS": SKINTHICKNESS,
    "INSULIN": INSULIN,
    "BMI": BMI,
    "DIABETESPEDIGREEFUNCTION": DIABETESPEDIGREEFUNCTION,
    "AGE": AGE
}])

# Özellik mühendisliği
input_df = fb.apply_all_feature_engineering(input_df)
input_df["INSULIN_FLAG"] = fb.add_insulin_flag(input_df)["INSULIN_FLAG"]
input_df["IS_OUTLIER"] = 0
input_df["PREGNANCY_FLAG"] = np.where(input_df["PREGNANCIES"] > 0, 1, 0)

# Encoding
input_df = fes.encode_categorical_features(input_df)

# Ölçeklenecek sütunlar
scaled_cols = [
    "PREGNANCIES", "BLOODPRESSURE", "SKINTHICKNESS",
    "GLUCOSE", "BMI", "AGE", "DIABETESPEDIGREEFUNCTION",
    "INSULIN", "AGE_X_PREGNANCIES"
]

# ✅ Eğitimde kullanılan scaler ile transform
input_df[scaled_cols] = scaler.transform(input_df[scaled_cols])

# Modelin beklediği sıraya göre input vektörü oluştur
user_input = input_df[features].values

# Tahmin
if st.button("Tahmin Et"):
    prediction = model.predict(user_input)[0]
    prob = model.predict_proba(user_input)[0][1]

    st.markdown("---")
    st.markdown(f"### 🔍 Sonuç: {'🟥 **Diyabetli**' if prediction == 1 else '🟩 **Diyabetli Değil**'}")
    st.markdown(f"### 📊 Olasılık: **{prob:.2%}**")