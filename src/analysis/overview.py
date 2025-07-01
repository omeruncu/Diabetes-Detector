import pandas as pd

def data_overview(df: pd.DataFrame) -> None:
    """Veri setinin genel yapısını özetler (NaN odaklı)."""
    print("Veri Seti Genel Bilgisi\n")
    print(f"Gözlem sayısı: {df.shape[0]}")
    print(f"Özellik sayısı: {df.shape[1]}")
    print(f"Sütunlar: {list(df.columns)}\n")

    print("Veri Tipleri ve Bellek Kullanımı:\n")
    print(df.info())

    print("\nTemel İstatistikler:\n")
    print(df.describe().T)

    print("\nEksik (NaN) Değer:\n")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(missing)
    else:
        print("Eksik (NaN) değer bulunmuyor.")


def compare_descriptive_stats(df_before, df_after, columns):
    """
    Belirtilen sütunlar için temizlik öncesi ve sonrası describe() karşılaştırması yapar.

    Args:
        df_before (pd.DataFrame): Temizlik öncesi veri
        df_after (pd.DataFrame): Temizlik sonrası veri
        columns (list): Karşılaştırılacak sütunlar

    Returns:
        tuple: (df_before.describe(), df_after.describe())
    """
    return df_before[columns].describe(), df_after[columns].describe()


def summarize_filled_values(df_before, df_after, columns):
    """
    0 olan değerlerin KNN ile doldurulmuş versiyonlarının istatistiksel özetini verir.

    Args:
        df_before (pd.DataFrame): Temizlik öncesi veri
        df_after (pd.DataFrame): Temizlik sonrası veri
        columns (list): Doldurulan sütunlar

    Returns:
        dict: Her sütun için describe() çıktısı
    """
    summaries = {}
    for col in columns:
        filled = df_after.loc[df_before[col] == 0, col]
        summaries[col] = filled.describe()
    return summaries