import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score


def test_model_quality():
    model = joblib.load('models/titanic_model.pkl')
    df = pd.read_csv('data_clean_fe.csv')
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    y_pred = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred)

    assert auc > 0.85, f"ROC-AUC слишком низкий: {auc:.3f}"  # твой порог