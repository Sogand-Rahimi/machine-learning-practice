# Loan Approval Prediction

A binary classification project that predicts whether a loan application is approved, based on applicant details such as income, education, credit history and property area. It compares k-Nearest Neighbors and Logistic Regression.

---

## Key Features

* **Data Cleaning:** Drops rows with missing values before encoding, and removes the `Loan_ID` column because it is unique for every row and useless for prediction.
* **Categorical Plots:** Count plots show how the values of each text column (Gender, Married, Education, etc.) are distributed.
* **Label Encoding:** Converts text columns into numbers (for example Female/Male to 0/1), using a separate encoder for each column.
* **Correlation Heatmap:** Shows how strongly each column is related to the others and to the loan status.
* **Feature Scaling:** Standardizes all features with `StandardScaler`, fitted on the training data only to avoid data leakage.
* **Model Comparison:** Trains KNN and Logistic Regression on the same train/test split and prints the test accuracy of each.

---

## Prerequisites

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Dataset

The dataset is not included in this repo. Download the **Loan Prediction Problem Dataset** from Kaggle and save the training file (`train_u6lujuX_CVtuZ9i.csv`) as `Data/train_u6lujuX_CVtuZ9i.csv`.

The target column is `Loan_Status` (Y = approved, N = not approved). Ignore the `test_...` file, because it has no `Loan_Status` column.

## Usage

```bash
python loan_prediction.py
```

The script saves `categorical_counts.png` and `correlation_heatmap.png` in the current folder and prints the accuracy of each model.

(Run the script and replace this with your real numbers, or delete this section.)
