from sklearn.impute import KNNImputer
import numpy as np


def clean_diabetes_data(df):
    """
    Diyabet veri setini temizler:
    - 0 değerlerini NaN yapar
    - Eksik değerleri KNN ile doldurur (bağlam sütunları dahil)
    """
    # Eksik değer olarak kabul edilen sütunlar
    zero_cols = ["GLUCOSE", "BLOODPRESSURE", "BMI", "SKINTHICKNESS", "INSULIN"]

    # 1. 0'ları NaN yap
    df_none = df.copy()
    df_none[zero_cols] = df_none[zero_cols].replace(0, np.nan)

    # 2. KNN için kullanılacak sütunlar (eksik + bağlam)
    knn_features = [
        "GLUCOSE", "BLOODPRESSURE", "BMI", "SKINTHICKNESS", "INSULIN",
        "AGE", "PREGNANCIES", "DIABETESPEDIGREEFUNCTION"
    ]

    # 3. KNN imputasyonu
    df_cleaned = df_none.copy()
    imputer = KNNImputer(n_neighbors=5)
    df_cleaned[knn_features] = imputer.fit_transform(df_cleaned[knn_features])

    return df_none, df_cleaned