from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

app = FastAPI(title="Titanic Survival Predictor")

# Метрики
REQUESTS = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')


@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    REQUESTS.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    LATENCY.observe(duration)
    return response


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

templates = Jinja2Templates(directory="templates")

# Загрузка модели
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "titanic_pipeline.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    """
    Преобразует входные данные в DataFrame с 12 признаками (как при обучении)
    """
    # Создаём DataFrame с одним пассажиром
    data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": 1 if sex.lower() == "male" else 0,
        "Age": float(age),
        "SibSp": int(sibsp),
        "Parch": int(parch),
        "Fare": float(fare),
        "Embarked": {"C": 0, "Q": 1, "S": 2}.get(embarked.upper(), 2)
    }])

    # ====== ДОБАВЛЯЕМ ПРИЗНАКИ (как в сабмит-скрипте) ======
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    data['IsChild'] = (data['Age'] < 12).astype(int)
    data['WomanWithChild'] = ((data['Sex'] == 0) & (data['Parch'] > 0)).astype(int)

    # FareBin: нужно использовать те же границы, что при обучении
    # Если у тебя были границы из тренировочных данных, укажи их здесь
    # Например, для Titanic стандартные границы:
    fare_bins = [0, 8, 15, 32, 600]  # примерные границы
    data['FareBin'] = pd.cut(data['Fare'], bins=fare_bins, labels=False, include_lowest=True)
    data['FareBin'] = data['FareBin'].fillna(0).astype(int)

    # Приводим колонки к тому же порядку, что при обучении
    expected_columns = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
        'FamilySize', 'IsAlone', 'IsChild', 'WomanWithChild', 'FareBin'
    ]
    data = data[expected_columns]

    return data


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(
        request: Request,
        pclass: int = Form(...),
        sex: str = Form(...),
        age: float = Form(...),
        sibsp: int = Form(...),
        parch: int = Form(...),
        fare: float = Form(...),
        embarked: str = Form(...)
):
    # Подготовка данных (12 признаков!)
    input_data = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)

    # Предсказание вероятности
    proba = model.predict_proba(input_data)[0][1]
    survival = 1 if proba >= 0.5 else 0

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": survival,
        "probability": round(proba * 100, 2),
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked
    })