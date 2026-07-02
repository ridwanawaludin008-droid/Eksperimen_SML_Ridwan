import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os


def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    return df


def preprocess(df):
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df = df.drop_duplicates().dropna()

    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[feature_cols]
    y = df['Survived']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = X_train.copy()
    train_df['Survived'] = y_train.values

    test_df = X_test.copy()
    test_df['Survived'] = y_test.values

    return train_df, test_df


def save_results(train_df, test_df, output_dir='titanic_preprocessing'):
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(f'{output_dir}/titanic_train.csv', index=False)
    test_df.to_csv(f'{output_dir}/titanic_test.csv', index=False)
    print(f'Saved to {output_dir}/ — Train: {train_df.shape}, Test: {test_df.shape}')


def main():
    df = load_data()
    os.makedirs('../titanic_raw', exist_ok=True)
    df.to_csv('../titanic_raw/titanic_raw.csv', index=False)

    train_df, test_df = preprocess(df)
    save_results(train_df, test_df)

    return train_df, test_df


if __name__ == '__main__':
    main()
