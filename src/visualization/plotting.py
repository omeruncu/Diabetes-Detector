import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import missingno as msno


def plot_outcome_distribution(df):
    """
    OUTCOME değişkeninin sınıf dağılımını çubuk grafikle gösterir.
    """
    sns.countplot(x="OUTCOME", data=df, palette="Set2")
    plt.title("Diyabet Dağılımı (0: Yok, 1: Var)")
    plt.xlabel("OUTCOME")
    plt.ylabel("Kişi Sayısı")
    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(df, exclude_target="OUTCOME", bins=20):
    """
    Sayısal değişkenlerin histogramlarını çizer.

    Args:
        df (pd.DataFrame): Veri seti
        exclude_target (str): Hedef değişken (grafik dışı bırakılır)
        bins (int): Histogram aralık sayısı
    """
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if exclude_target in num_cols:
        num_cols.remove(exclude_target)

    df[num_cols].hist(figsize=(15, 10), bins=bins, color="#69b3a2", edgecolor="black")
    plt.suptitle("Sayısal Değişken Dağılımları", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_outlier_boxplots(df, exclude_target="OUTCOME", cols_per_row=3):
    """
    Sayısal değişkenler için boxplot çizerek aykırı değerleri görselleştirir.

    Args:
        df (pd.DataFrame): Veri seti
        exclude_target (str): Hedef değişken (grafik dışı bırakılır)
        cols_per_row (int): Her satırda kaç grafik gösterileceği
    """
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if exclude_target in num_cols:
        num_cols.remove(exclude_target)

    total = len(num_cols)
    rows = (total + cols_per_row - 1) // cols_per_row

    plt.figure(figsize=(5 * cols_per_row, 4 * rows))
    for i, col in enumerate(num_cols):
        plt.subplot(rows, cols_per_row, i + 1)
        sns.boxplot(x=df[col], color="#ffb347")
        plt.title(f"{col} - Boxplot")
        plt.xlabel("")
    plt.tight_layout()
    plt.show()


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


def plot_correlation_matrix(df, figsize=(10, 8), cmap="coolwarm", annot=True):
    """
    Korelasyon matrisi ısı haritasını çizer.

    Args:
        df (pd.DataFrame): Girdi veri seti
        figsize (tuple): Grafik boyutu
        cmap (str): Renk haritası
        annot (bool): Hücrelere değer yazılsın mı
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), annot=annot, cmap=cmap, fmt=".2f", square=True)
    plt.title("Korelasyon Matrisi", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_feature_distributions_by_target(df, target="OUTCOME", exclude_target=True, cols_per_row=3):
    """
    Sayısal değişkenlerin hedef değişkene göre KDE dağılım grafiklerini çizer.

    Args:
        df (pd.DataFrame): Girdi veri seti
        target (str): Hedef değişken
        exclude_target (bool): Hedef değişkeni grafiklerden hariç tut
        cols_per_row (int): Her satırda kaç grafik gösterileceği
    """
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if exclude_target and target in num_cols:
        num_cols.remove(target)

    total = len(num_cols)
    rows = (total + cols_per_row - 1) // cols_per_row

    plt.figure(figsize=(5 * cols_per_row, 4 * rows))
    for i, col in enumerate(num_cols):
        plt.subplot(rows, cols_per_row, i + 1)
        sns.kdeplot(data=df, x=col, hue=target, fill=True, common_norm=True, palette="Set2")
        plt.title(f"{col} - {target} Dağılımı")
        plt.xlabel("")
    plt.tight_layout()
    plt.show()


def visualize_missing_values(df, title_prefix=""):
    """
    Eksik değerleri bar, matrix ve heatmap olarak görselleştirir.

    Args:
        df (pd.DataFrame): Veri seti
        title_prefix (str): Grafik başlıklarına eklenecek ön ek
    """
    msno.bar(df)
    plt.title(f"{title_prefix}Eksik Değerler - Bar")
    plt.show()

    msno.matrix(df)
    plt.title(f"{title_prefix}Eksik Değerler - Matris")
    plt.show()

    msno.heatmap(df)
    plt.title(f"{title_prefix}Eksik Değer Korelasyonları")
    plt.show()


def compare_distributions_before_after(df_before, df_after, columns):
    """
    Belirtilen sütunlar için temizlik öncesi ve sonrası dağılım karşılaştırması yapar.

    Args:
        df_before (pd.DataFrame): Temizlik öncesi veri
        df_after (pd.DataFrame): Temizlik sonrası veri
        columns (list): Karşılaştırılacak sütunlar
    """
    for col in columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(df_before[col], kde=True, ax=axes[0], color="red")
        axes[0].set_title(f"{col} - Temizlikten Önce")
        axes[0].set_xlabel(col)

        sns.histplot(df_after[col], kde=True, ax=axes[1], color="green")
        axes[1].set_title(f"{col} - Temizlikten Sonra")
        axes[1].set_xlabel(col)

        plt.suptitle(f"{col} Dağılımı Karşılaştırması", fontsize=14)
        plt.tight_layout()
        plt.show()

