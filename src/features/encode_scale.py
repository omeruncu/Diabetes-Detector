import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib


def encode_categorical_features(df):
    """
    Kategorik sütunları one-hot encoding ile dönüştürür.

    Args:
        df (pd.DataFrame): Girdi veri seti

    Returns:
        pd.DataFrame: Encode edilmiş veri seti
    """
    df = pd.get_dummies(df, columns=[
        "AGE_GROUP", "BMI_CATEGORY", "GLUCOSE_LEVEL"
    ], drop_first=True)
    return df


def scale_numerical_features(df, columns, method="standard"):
    """
    Belirtilen sayısal sütunları ölçekler.

    Args:
        df (pd.DataFrame): Girdi veri seti
        columns (list): Ölçeklenecek sütun isimleri
        method (str): 'standard' veya 'robust' ölçekleme yöntemi

    Returns:
        pd.DataFrame: Ölçeklenmiş veri seti
        scaler: Kullanılan scaler nesnesi
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("method parametresi 'standard' veya 'robust' olmalıdır.")

    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


