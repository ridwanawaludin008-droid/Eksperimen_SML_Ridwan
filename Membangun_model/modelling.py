import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

TRAIN_PATH = 'iris_preprocessing/iris_train.csv'
TEST_PATH = 'iris_preprocessing/iris_test.csv'


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    X_train, y_train = train.drop('target', axis=1), train['target']
    X_test, y_test = test.drop('target', axis=1), test['target']
    return X_train, X_test, y_train, y_test


def main():
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.sklearn.autolog(log_models=True)

    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name='RandomForest_autolog'):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.sklearn.log_model(model, "model")
        print(f'Test Accuracy: {acc:.4f}')


if __name__ == '__main__':
    main()
