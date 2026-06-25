from fastapi.testclient import TestClient
from app.main import app
import httpx
import requests

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", data={
        "pclass": 3,
        "sex": "male",
        "age": 22,
        "sibsp": 1,
        "parch": 0,
        "fare": 7.25,
        "embarked": "S"
    })
    assert response.status_code == 200
    # Проверяем, что страница содержит слово "ВЫЖИВЕТ" или "НЕ ВЫЖИВЕТ"
    assert "ВЫЖИВЕТ" in response.text or "НЕ ВЫЖИВЕТ" in response.text