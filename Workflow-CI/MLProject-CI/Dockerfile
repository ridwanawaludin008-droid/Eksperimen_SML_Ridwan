FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install mlflow scikit-learn pandas numpy matplotlib joblib

CMD ["python", "modelling.py"]
