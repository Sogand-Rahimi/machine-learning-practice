"""
Fruit Classification
--------------------
Classifies fruits (apple, mandarin, orange, lemon) from their mass, width,
height and color score, and compares four scikit-learn classifiers:
Logistic Regression, Decision Tree, k-Nearest Neighbors and SVC.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# --- Configuration ---
DATA_PATH = "data/fruit_data_with_colors.txt"
FEATURE_NAMES = ["height", "width", "mass", "color_score"]
TARGET = "fruit_label"
TEST_SIZE = 0.2
RANDOM_STATE = 0


def load_data(path):
    """Load the dataset and split it into features (X) and labels (y)."""
    fruits = pd.read_table(path)
    return fruits[FEATURE_NAMES], fruits[TARGET]


def plot_data(X, y):
    """Save a scatter matrix and a pairplot to visualize the feature spread."""
    scatter_matrix(
        X, c=y, marker="o", s=40, hist_kwds={"bins": 15},
        figsize=(9, 9), cmap=plt.get_cmap("gnuplot"),
    )
    plt.suptitle("Scatter-matrix for each input variable")
    plt.savefig("fruits_scatter_matrix.png")

    sns.pairplot(X.assign(**{TARGET: y}), hue=TARGET, palette="viridis")
    plt.savefig("fruits_pairplot.png")
    plt.show()


def evaluate_models(X_train, X_test, y_train, y_test):
    """Train each model and print its accuracy on the training and test sets."""
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNeighbors": KNeighborsClassifier(),
        "SVC": SVC(),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name}")
        print(f"  Training accuracy: {model.score(X_train, y_train):.2f}")
        print(f"  Test accuracy:     {model.score(X_test, y_test):.2f}")


def main():
    X, y = load_data(DATA_PATH)
    plot_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Scale features to 0-1 (fit on training data only to avoid data leakage)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)


if __name__ == "__main__":
    main()
