from fastapi.testclient import TestClient
from app.main import app
import httpx
import requests

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", data={  # ← data, а не json!
        "pclass": 3,
        "sex": "male",
        "age": 22,
        "sibsp": 1,
        "parch": 0,
        "fare": 7.25,
        "embarked": "S"
    })
    assert response.status_code == 200
    # Проверь, что возвращается, например, "prediction" или "probability"
    assert "prediction" in response.text  # или response.json() если возвращаешь JSON