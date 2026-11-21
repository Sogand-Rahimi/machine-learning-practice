# Bitcoin Next-Day Price Direction

A binary classification project that predicts whether the Bitcoin closing price will go up or down the next day. It builds simple features from daily prices and compares three scikit-learn classifiers.

---

## Key Features

* **Data Download:** Gets daily BTC-USD prices (Open, High, Low, Close, Volume) from Yahoo Finance with `yfinance`, and saves them as `bitcoin.csv`.
* **Exploratory Plots:** Price history, feature distributions, yearly averages and the balance of up/down days. All are saved as `.png` files.
* **Feature Engineering:** Creates `Open_Close` (open minus close), `Low_High` (low minus high) and `is_quarter_end` (1 for March, June, September, December).
* **Target Label:** `Target` is 1 if tomorrow's close is higher than today's, otherwise 0.
* **Time-Ordered Split:** Trains on the older 80% of days and tests on the newest 20%, so the model never sees the future.
* **Model Comparison:** Evaluates Logistic Regression, Decision Tree and k-Nearest Neighbors by test accuracy.

---

## Prerequisites

```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn
```

## Dataset

If `bitcoin.csv` is missing, the script downloads it automatically from Yahoo Finance (BTC-USD, 2014-09-17 to 2023-05-22). No account is needed.

## Usage

```bash
python bitcoin_price_direction.py
```

The script saves the plots in the current folder and prints the test accuracy of each model.

## Note

Predicting price direction from these features is very hard. Accuracy close to 50% (a coin flip) is normal, so this project is for learning the workflow, not for trading.
