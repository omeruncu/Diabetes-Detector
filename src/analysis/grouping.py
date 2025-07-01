def groupby_outcome_means(df, target="OUTCOME", sort_by=1, ascending=False):
    """
    OUTCOME'a göre gruplama yaparak ortalama değerleri karşılaştırır.

    Args:
        df (pd.DataFrame): Girdi veri seti
        target (str): Hedef değişken
        sort_by (int): Hangi sınıfa göre sıralanacağı (0 veya 1)
        ascending (bool): Sıralama yönü

    Returns:
        pd.DataFrame: Gruplanmış ve sıralanmış ortalama değerler
    """
    grouped = df.groupby(target).mean().T
    return grouped.sort_values(by=sort_by, ascending=ascending)