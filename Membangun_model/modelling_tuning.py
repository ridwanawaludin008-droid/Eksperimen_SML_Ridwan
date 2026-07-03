import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import GridSearchCV

TRAIN_PATH = 'titanic_preprocessing/titanic_train.csv'
TEST_PATH = 'titanic_preprocessing/titanic_test.csv'

DAGSHUB_USERNAME = 'ridwanawaludin008-droid'
DAGSHUB_REPO = 'Membangun_model'
DAGSHUB_TOKEN = 'ca819be2396cacdc7500f00ba5fa9d05c0b28a1c'


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    X_train = train.drop('Survived', axis=1)
    y_train = train['Survived']
    X_test = test.drop('Survived', axis=1)
    y_test = test['Survived']
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(cm, path='confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    classes = ['Not Survived', 'Survived']
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_feature_importance(model, feature_names, path='feature_importance.png'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=20, ha='right')
    ax.set_title('Feature Importances')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def main():
    os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN

    mlflow.set_tracking_uri(f'https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow')
    mlflow.set_experiment('titanic-classification')

    X_train, X_test, y_train, y_test = load_data()

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print(f'Best params: {best_params}')

    with mlflow.start_run(run_name='RandomForest_tuning'):
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param('cv_folds', 5)

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        cv_best_score = grid_search.best_score_

        mlflow.log_metric('test_accuracy', acc)
        mlflow.log_metric('test_precision_weighted', precision)
        mlflow.log_metric('test_recall_weighted', recall)
        mlflow.log_metric('test_f1_weighted', f1)
        mlflow.log_metric('test_roc_auc', roc_auc)
        mlflow.log_metric('cv_best_accuracy', cv_best_score)

        print(f'Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}')

        cm = confusion_matrix(y_test, y_pred)
        cm_path = plot_confusion_matrix(cm)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        fi_path = plot_feature_importance(best_model, list(X_train.columns))
        mlflow.log_artifact(fi_path)
        os.remove(fi_path)

        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = 'classification_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)
        os.remove(report_path)

        mlflow.sklearn.log_model(best_model, 'model')


if __name__ == '__main__':
    main()
