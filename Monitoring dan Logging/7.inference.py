import requests

url = "http://127.0.0.1:8001/predict"

data = {
    "inputs": [
        {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
        {"sepal length (cm)": 6.2, "sepal width (cm)": 3.4, "petal length (cm)": 5.4, "petal width (cm)": 2.3},
        {"sepal length (cm)": 5.9, "sepal width (cm)": 3.0, "petal length (cm)": 4.2, "petal width (cm)": 1.5}
    ]
}

for i in range(10):
    response = requests.post(url, json=data)
    print(response.json())
