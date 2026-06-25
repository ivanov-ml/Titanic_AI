from fastapi.testclient import TestClient
from app.main import app
import httpx
import requests

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    })
    assert response.status_code == 200
    assert "survival_probability" in response.json()