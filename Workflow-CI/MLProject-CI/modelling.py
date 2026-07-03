import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

train = pd.read_csv("iris_preprocessing/iris_train.csv")
test = pd.read_csv("iris_preprocessing/iris_test.csv")

X_train = train.drop("target", axis=1)
y_train = train["target"]
X_test = test.drop("target", axis=1)
y_test = test["target"]

mlflow.set_experiment("Workflow-CI")

with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    joblib.dump(model, "model.pkl")

    print("Accuracy:", acc)