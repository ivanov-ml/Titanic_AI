import pandas as pd

def test_data_has_no_nulls():
    df = pd.read_csv('data_clean_fe.csv')
    assert df.isnull().sum().sum() == 0, "Есть пропуски в данных!"

def test_target_distribution():
    df = pd.read_csv('data_clean_fe.csv')
    assert 0.3 < df['Survived'].mean() < 0.5, "Странное распределение таргета"