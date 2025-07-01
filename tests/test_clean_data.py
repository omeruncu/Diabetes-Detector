import os
import pandas as pd
import numpy as np
from src.data.clean_data import clean_diabetes_data

# Örnek veri oluştur
def get_sample_df():
    return pd.DataFrame({
        "PREGNANCIES": [1, 2, 3],
        "GLUCOSE": [0, 120, 130],
        "BLOODPRESSURE": [70, 0, 80],
        "SKINTHICKNESS": [0, 35, 40],
        "INSULIN": [0, 100, 200],
        "BMI": [0, 30.0, 35.0],
        "DIABETESPEDIGREEFUNCTION": [0.5, 0.6, 0.7],
        "AGE": [25, 35, 45],
        "OUTCOME": [0, 1, 1]
    })


def test_replace_and_fill_missing_values():
    df = get_sample_df()
    cleaned_df = clean_diabetes_data(df.copy())

    # Eksik değer kalmamalı
    assert not cleaned_df.isnull().values.any(), "Eksik değerler kaldı!"

    # 0 değerleri medyan ile doldurulmuş olmalı
    assert (cleaned_df[["GLUCOSE", "BLOODPRESSURE", "BMI"]] != 0).all().all(), "0 değerleri medyanla doldurulmadı!"

    # KNN ile doldurulan sütunlarda 0 kalmamalı
    assert (cleaned_df[["SKINTHICKNESS", "INSULIN"]] != 0).all().all(), "KNN ile doldurulan sütunlarda 0 kaldı!"