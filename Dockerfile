FROM python:3.10-slim

WORKDIR /app

# Установка системных библиотек для LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/
COPY templates/ ./templates/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
