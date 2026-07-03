from prometheus_client import start_http_server, Counter, Gauge, Histogram
import random
import time

request_counter = Counter('ml_requests_total', 'Jumlah Request')
error_counter = Counter('ml_error_total', 'Jumlah Error')
latency = Histogram('ml_latency_seconds', 'Waktu Prediksi')
cpu_usage = Gauge('cpu_usage_percent', 'CPU Usage')
memory_usage = Gauge('memory_usage_percent', 'Memory Usage')

start_http_server(8000)

while True:
    request_counter.inc()

    latency_value = random.uniform(0.1, 1.0)
    latency.observe(latency_value)

    if latency_value > 0.8:
        error_counter.inc()

    cpu_usage.set(random.randint(20, 80))
    memory_usage.set(random.randint(30, 70))

    time.sleep(5)
