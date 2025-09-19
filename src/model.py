from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    df_X = pd.DataFrame(data=X, columns=feature_names)
    df_y = pd.DataFrame(data=y, columns=['target'])
    # df = pd.concat([df_X, df_y], axis=1)
    return df_X, df_y

def data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    return model

def evaluate_model(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    logging.info(f"Model accuracy: {accuracy}")
    return accuracy

def save_model(model, model_path):
    import joblib
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = data_split(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, "E:/my_projects/build_it/model/random_forest_model.joblib")

if __name__ == "__main__":
    main()