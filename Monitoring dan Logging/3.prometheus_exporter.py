from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Histogram, Gauge, generate_latest
import requests
import json
import time

app = Flask(__name__)

REQUEST_COUNT = Counter('ml_requests_total', 'Total inference requests')
SUCCESS_COUNT = Counter('ml_success_total', 'Total successful inference')
ERROR_COUNT = Counter('ml_error_total', 'Total failed inference')
LATENCY = Histogram('ml_latency_seconds', 'Inference latency in seconds')
CONFIDENCE = Gauge('ml_confidence_score', 'Last prediction confidence score')

MLFLOW_URL = "http://127.0.0.1:5001/invocations"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    REQUEST_COUNT.inc()

    start = time.time()
    try:
        response = requests.post(
            MLFLOW_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"inputs": data["inputs"]})
        )
        duration = time.time() - start
        LATENCY.observe(duration)

        result = response.json()
        SUCCESS_COUNT.inc()

        if "predictions" in result:
            pred = result["predictions"][0]
            CONFIDENCE.set(float(pred) if isinstance(pred, (int, float)) else 0)

        return jsonify({"prediction": result, "latency": duration})

    except Exception as e:
        ERROR_COUNT.inc()
        return jsonify({"error": str(e)}), 500

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}

if __name__ == '__main__':
    start_http_server(8000)
    app.run(host='0.0.0.0', port=8001)
