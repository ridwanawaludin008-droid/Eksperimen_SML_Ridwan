import requests
import json

url = "http://127.0.0.1:5000/invocations"

headers = {
    "Content-Type": "application/json"
}

data = {
    "inputs": [
        [5.1,3.5,1.4,0.2]
    ]
}

response = requests.post(
    url,
    headers=headers,
    data=json.dumps(data)
)

print(response.json())