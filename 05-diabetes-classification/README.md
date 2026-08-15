# Diabetes Prediction using Logistic Regression

A binary classification pipeline predicting diabetes onset using health metrics from the Pima Indians Diabetes dataset and Scikit-Learn.

---

## Overview

This module predicts whether a patient has diabetes (`Outcome = 1`) or not (`Outcome = 0`) based on medical features such as glucose levels, blood pressure, insulin, and BMI.

---

## Key Improvements & Implementation

* **Stratified Splitting:** Used `stratify=y` to preserve the target class balance across train and test sets.
* **Scikit-Learn Evaluation Metrics:** Replaced manual loops with standard `confusion_matrix` and `accuracy_score` functions.
* **Reproducibility:** Applied explicit random seeding (`random_state=42`) across splitting and model initialization.

---

## Dataset & Usage

1. Download `diabetes.csv` from Kaggle and place it inside `dataset/diabetes.csv`.
2. Run the pipeline:

```bash
python diabetes_classification.py
