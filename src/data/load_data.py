import pandas as pd
import yaml
import os


def read_params(config_path: str = "config/params.yaml") -> dict:
    """YAML dosyasından parametreleri okur."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_data(config_path: str = "config/params.yaml") -> pd.DataFrame:
    """
    YAML dosyasındaki veri yolunu kullanarak CSV dosyasını yükler.
    Sütun adlarını büyütür ve boşlukları alt çizgiyle değiştirir.
    """
    config = read_params(config_path)
    data_path = config["data_source"]["raw_data"]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_path}")

    df = pd.read_csv(data_path)

    # Sütun adlarını temizle
    df.columns = [col.replace(" ", "_").upper() for col in df.columns]

    return df