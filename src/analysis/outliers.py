import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA


def detect_lof_outliers(df, target="OUTCOME", contamination=0.05, n_neighbors=20, visualize=True):
    """
    Local Outlier Factor (LOF) ile aykırı değerleri tespit eder ve işaretler.

    Args:
        df (pd.DataFrame): Girdi veri seti
        target (str): Hedef değişken (analiz dışı bırakılır)
        contamination (float): Aykırı oranı varsayımı
        n_neighbors (int): LOF için komşu sayısı
        visualize (bool): PCA ile 2D görselleştirme yapılacak mı?

    Returns:
        df_out (pd.DataFrame): Aykırı gözlemler işaretlenmiş veri seti
        summary (dict): Aykırı gözlem sayısı ve oranı
    """
    df_out = df.copy()

    # 1. Sayısal sütunları seç
    features = df_out.select_dtypes(include=["int64", "float64"]).drop(columns=[target])

    # 2. Eksik değerleri medyanla doldur
    X = features.fillna(features.median())

    # 3. LOF modeli
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(X)

    # 4. Aykırıları işaretle
    df_out["LOF_Outlier"] = y_pred

    # 5. Özet
    n_outliers = (y_pred == -1).sum()
    outlier_ratio = n_outliers / len(df_out)
    summary = {
        "n_outliers": n_outliers,
        "outlier_ratio": outlier_ratio
    }

    # 6. Görselleştirme (PCA)
    if visualize:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[y_pred == 1, 0], X_pca[y_pred == 1, 1], label="Normal", alpha=0.6)
        plt.scatter(X_pca[y_pred == -1, 0], X_pca[y_pred == -1, 1], label="Aykırı", color="red", alpha=0.8)
        plt.title("LOF ile Aykırı Değer Tespiti (PCA 2D)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return df_out, summary
