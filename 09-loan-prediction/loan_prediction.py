"""
Loan Approval Prediction
------------------------
Predicts whether a loan application is approved (Loan_Status = Y) from
applicant details such as income, education, credit history and property
area. Compares two classifiers: k-Nearest Neighbors and Logistic Regression.

Data: Loan Prediction Problem Dataset (train file) from Kaggle.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Configuration ---
DATA_PATH = "Data/train_u6lujuX_CVtuZ9i.csv"
TARGET = "Loan_Status"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_NEIGHBORS = 5


def load_data(path):
    """Load the CSV, drop rows with missing values and drop the ID column."""
    data = pd.read_csv(path)
    data = data.dropna()  # must happen BEFORE encoding, or NaN becomes a fake category
    return data.drop(columns=["Loan_ID"])  # unique per row, useless for prediction


def plot_categorical(data):
    """Save a count plot for every text (categorical) column."""
    columns = list(data.select_dtypes(include="object").columns)
    n_rows = math.ceil(len(columns) / 3)

    plt.figure(figsize=(15, 5 * n_rows))
    for i, column in enumerate(columns, start=1):
        plt.subplot(n_rows, 3, i)
        sns.countplot(x=data[column])
        plt.title(column)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("categorical_counts.png")


def encode_categorical(data):
    """Turn every text column into numbers (e.g. Female/Male -> 0/1)."""
    data = data.copy()
    for column in data.select_dtypes(include="object").columns:
        data[column] = LabelEncoder().fit_transform(data[column])
    return data


def plot_correlation(data):
    """Save a heatmap of the correlation between all columns."""
    plt.figure(figsize=(11, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")


def evaluate_models(X_train, X_test, y_train, y_test):
    """Train each model and print its test accuracy."""
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=N_NEIGHBORS),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        print(f"{name} accuracy: {accuracy:.4f}")


def main():
    data = load_data(DATA_PATH)
    print(f"Rows after dropping missing values: {len(data)}")

    plot_categorical(data)  # plotted before encoding, so labels stay readable
    data = encode_categorical(data)
    plot_correlation(data)

    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Scale features (fit on training data only to avoid data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    evaluate_models(X_train, X_test, y_train, y_test)
    plt.show()


if __name__ == "__main__":
    main()
