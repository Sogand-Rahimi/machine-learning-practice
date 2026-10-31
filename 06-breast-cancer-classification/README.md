# Breast Cancer Classification with Logistic Regression

A binary classification project that predicts whether a breast tumor is malignant or benign from 30 numeric features, using scikit-learn's Logistic Regression.

---

## Key Features

* **Label Encoding:** Converts the diagnosis labels into numbers (`M` = malignant = 1, `B` = benign = 0).
* **Feature Scaling:** Normalizes all features to the 0-1 range with `MinMaxScaler`, fitted on the training data only to avoid data leakage.
* **Train/Test Split:** Uses 85% of the data for training and 15% for testing, with a fixed random seed for reproducible results.
* **Evaluation:** Prints the number of wrong predictions, the accuracy, and the number of iterations the model needed to converge.

---

## Prerequisites

```bash
pip install pandas scikit-learn
```

## Dataset

The script expects the dataset at `dataset/brca.csv`. The last column `y` holds the diagnosis (`B` or `M`), and the other columns are numeric tumor measurements (radius, texture, perimeter, area, etc.).

## Usage

```bash
python breast_cancer_logistic_regression.py
```

## Example Output

```
Wrong predictions: 3 / 86
Accuracy: 0.965
Iterations used: 25
```

(Your exact numbers may differ.)
