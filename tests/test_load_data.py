import pandas as pd
from src.data.load_data import load_data

def test_load_data_columns_cleaned():
    df = load_data(config_path="config/params.yaml")
    assert all(col == col.upper() for col in df.columns)
    assert all(" " not in col for col in df.columns)

def test_load_data_file_exists():
    try:
        df = load_data(config_path="config/params.yaml")
        assert isinstance(df, pd.DataFrame)
    except FileNotFoundError:
        assert False, "Veri dosyası bulunamadı!"