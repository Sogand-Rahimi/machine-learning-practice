"""
Bitcoin Next-Day Price Direction
--------------------------------
Predicts whether the Bitcoin closing price will go up (1) or down (0) the
next day, using simple engineered features, and compares three classifiers:
Logistic Regression, Decision Tree and k-Nearest Neighbors.

Data: daily BTC-USD prices (Open, High, Low, Close, Volume) from Yahoo Finance.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --- Configuration ---
DATA_PATH = "bitcoin.csv"
TICKER = "BTC-USD"
START_DATE = "2014-09-17"
END_DATE = "2023-05-22"
PRICE_FEATURES = ["Open", "High", "Low", "Volume"]
MODEL_FEATURES = ["Open_Close", "Low_High", "is_quarter_end"]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def download_data(path):
    """Download daily BTC-USD prices from Yahoo Finance and save them as CSV."""
    import yfinance as yf

    data = yf.download(TICKER, start=START_DATE, end=END_DATE)
    data.to_csv(path)


def load_data(path):
    """Load the CSV (downloading it first if missing) with a real Date column."""
    if not os.path.exists(path):
        download_data(path)

    # yfinance writes 2 extra header rows (Ticker and Date), so skip them
    df = pd.read_csv(path, skiprows=[1, 2], index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df.reset_index()


def add_features(df):
    """Add date parts, engineered features and the Target column."""
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    # 1 for the last month of each quarter (March, June, September, December)
    df["is_quarter_end"] = np.where(df["Month"] % 3 == 0, 1, 0)
    df["Open_Close"] = df["Open"] - df["Close"]
    df["Low_High"] = df["Low"] - df["High"]

    # Target: 1 if tomorrow's close is higher than today's, otherwise 0
    next_close = df["Close"].shift(-1)
    df["Target"] = np.where(next_close > df["Close"], 1, 0)

    # The last row has no "tomorrow", so its target would be wrong. Drop it.
    return df.iloc[:-1]


def plot_data(df):
    """Save exploratory plots: price history, distributions, yearly means, target."""
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Close"])
    plt.title("Bitcoin Price Over Time", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.savefig("bitcoin_price.png")

    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(PRICE_FEATURES):
        plt.subplot(2, 2, i + 1)
        sns.histplot(df[feature], kde=True)
    plt.tight_layout()
    plt.savefig("bitcoin_distributions.png")

    yearly = df.groupby("Year")[PRICE_FEATURES].mean()
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(PRICE_FEATURES):
        plt.subplot(2, 2, i + 1)
        yearly[feature].plot(kind="bar", ax=plt.gca())
        plt.title(feature)
        plt.ylabel("Average Value")
    plt.tight_layout()
    plt.savefig("bitcoin_yearly_means.png")

    plt.figure()
    counts = df["Target"].value_counts().sort_index()  # 0 = decrease, 1 = increase
    plt.pie(counts, labels=["Decrease", "Increase"], autopct="%1.1f%%")
    plt.title("Distribution of Target Variable (Next Day Price Movement)")
    plt.savefig("bitcoin_target_distribution.png")

    plt.show()


def evaluate_models(X_train, X_test, y_train, y_test):
    """Train each model and print its test accuracy."""
    models = [
        LogisticRegression(),
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        KNeighborsClassifier(),
    ]

    for model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        print(f"{model.__class__.__name__}: accuracy = {accuracy:.4f}")


def main():
    df = add_features(load_data(DATA_PATH))
    plot_data(df)

    X = df[MODEL_FEATURES]
    y = df["Target"]

    # shuffle=False keeps time order: train on the past, test on the future
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )

    # Fit the scaler on the training data only to avoid data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    evaluate_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
