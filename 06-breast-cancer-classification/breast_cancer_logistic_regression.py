"""
Breast Cancer Classification with Logistic Regression
------------------------------------------------------
Predicts whether a tumor is malignant (M) or benign (B) from 30 numeric
features, using scikit-learn's LogisticRegression.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- Configuration ---
DATA_PATH = "dataset/brca.csv"
TEST_SIZE = 0.15
RANDOM_STATE = 42
MAX_ITER = 10000


def load_data(path):
    """Load the dataset and split it into features (X) and labels (y)."""
    data = pd.read_csv(path)
    data = data.drop(columns=["Unnamed: 0"], errors="ignore")

    y = (data["y"] == "M").astype(int)  # M (malignant) = 1, B (benign) = 0
    X = data.drop(columns=["y"])
    return X, y


def main():
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Scale features to the 0-1 range (fit on training data only)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=MAX_ITER)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    errors = (y_test != predictions).sum()
    print(f"Wrong predictions: {errors} / {len(y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"Iterations used: {model.n_iter_[0]}")


if __name__ == "__main__":
    main()
