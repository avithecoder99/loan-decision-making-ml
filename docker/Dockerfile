FROM python:3.10-slim
WORKDIR /app

COPY models/xgboost_model.pkl ./models/xgboost_model.pkl
COPY src/inference/realtime_predict.py ./app.py
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
