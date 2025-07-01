import pandas as pd

def add_age_group(df):
    """
    Yaş değerine göre yaş grubu etiketi ekler.

    Args:
        df (pd.DataFrame): Girdi veri seti

    Returns:
        pd.DataFrame: Yeni AGE_GROUP sütunu eklenmiş veri
    """
    df = df.copy()
    df["AGE_GROUP"] = pd.cut(
        df["AGE"],
        bins=[20, 30, 45, 60, 100],
        labels=["YOUNG", "MID", "MATURE", "OLD"]
    )
    return df


def add_bmi_category(df):
    """
    BMI değerine göre kategorik sınıflandırma ekler.

    Args:
        df (pd.DataFrame): Girdi veri seti

    Returns:
        pd.DataFrame: Yeni BMI_CATEGORY sütunu eklenmiş veri
    """
    df = df.copy()
    df["BMI_CATEGORY"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Skinny", "Normal", "Overweight", "Obese"]
    )
    return df


def add_glucose_level(df):
    """
    Glukoz değerine göre diyabet riski sınıflandırması ekler.

    Args:
        df (pd.DataFrame): Girdi veri seti

    Returns:
        pd.DataFrame: Yeni GLUCOSE_LEVEL sütunu eklenmiş veri
    """
    df = df.copy()
    df["GLUCOSE_LEVEL"] = pd.cut(
        df["GLUCOSE"],
        bins=[0, 99, 125, 200],
        labels=["Normal", "Prediabetes", "Diabetes"]
    )
    return df


def add_insulin_flag(df):
    """
    İnsülin değeri 0 olanları 'unmeasured' olarak işaretler.

    Args:
        df (pd.DataFrame): Girdi veri seti

    Returns:
        pd.DataFrame: Yeni INSULIN_FLAG sütunu eklenmiş veri
    """
    df = df.copy()
    df["INSULIN_FLAG"] = df["INSULIN"].apply(lambda x: "UNMEASURED" if x == 0 else "MEASURED")
    return df


def add_pregnancy_flag(df):
    """
    Gebelik sayısı > 0 olanları işaretler.

    Args:
        df (pd.DataFrame): Girdi veri seti

    Returns:
        pd.DataFrame: Yeni PREGNANCY_FLAG sütunu eklenmiş veri
    """
    df = df.copy()
    df["PREGNANCY_FLAG"] = df["PREGNANCIES"].apply(lambda x: "YES" if x > 0 else "NO")
    return df


def add_outlier_flag(df, outlier_index):
    """
    LOF ile tespit edilen aykırı gözlemleri işaretleyen bir sütun ekler.

    Args:
        df (pd.DataFrame): Girdi veri seti
        outlier_index (array-like): Aykırı gözlemlerin indeksleri

    Returns:
        pd.DataFrame: is_outlier sütunu eklenmiş veri
    """
    df = df.copy()
    df["IS_OUTLIER"] = 0
    df.loc[outlier_index, "IS_OUTLIER"] = 1
    return df

def add_age_pregnancy_interaction(df):
    """
    Yaş ve gebelik sayısı etkileşimini içeren yeni bir özellik ekler.

    Args:
        df (pd.DataFrame): Girdi veri seti

    Returns:
        pd.DataFrame: AGE_X_PREGNANCIES sütunu eklenmiş veri
    """
    df = df.copy()
    df["AGE_X_PREGNANCIES"] = df["AGE"] * df["PREGNANCIES"]
    return df


def apply_all_feature_engineering(df):
    """
    Tüm feature engineering adımlarını sırayla uygular.

    Args:
        df (pd.DataFrame): Girdi veri seti

    Returns:
        pd.DataFrame: Yeni özelliklerle zenginleştirilmiş veri
    """
    df = add_age_group(df)
    df = add_bmi_category(df)
    df = add_glucose_level(df)
    df = add_pregnancy_flag(df)
    df = add_age_pregnancy_interaction(df)

    return df

