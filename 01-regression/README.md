# Linear Regression for CO₂ Emissions Prediction

An end-to-end Machine Learning module that predicts vehicle CO₂ emissions based on engine characteristics and fuel metrics using modern Data Science best practices and Scikit-Learn.

---

## Table of Contents
- [Overview](#overview)
- [Key Improvements & Modern Practices](#key-improvements--modern-practices)
- [Dataset](#dataset)
- [Installation & Setup](#installation--setup)
- [Project Workflow](#project-workflow)
- [Results & Evaluation](#results--evaluation)

---

## Overview

Vehicle emissions contribute significantly to greenhouse gas pollution. This project builds and evaluates a **Linear Regression** model to accurately estimate carbon dioxide ($CO_2$) emissions based on vehicle design and engine features. 

The goal of this project is to demonstrate clean, reproducible, and production-ready Python workflows for exploratory data analysis (EDA), data preprocessing, model training, and performance evaluation.

---

## Key Improvements & Modern Practices

This script incorporates production-level Machine Learning practices:

* **Fixed Target Distribution EDA:** Replaced inappropriate discrete count plots with kernel density estimation (`sns.histplot`) suitable for continuous target variables.
* **Pandas Compatibility:** Handled non-numeric feature correlations explicitly using modern `numeric_only` arguments to prevent breaking runtime changes.
* **Reproducible Splitting:** Implemented strict random seeding (`random_state=42`) across train/test splits for verifiable model evaluation.
* **Enhanced Model Evaluation:** Expanded beyond Mean Squared Error (MSE) to include the **$R^2$ Score** for better interpretability.
* **Visual Diagnostics:** Added a reference parity line ($y = x$) to actual vs. predicted scatter plots to visually assess variance and bias.

---

## Dataset

The dataset contains vehicle features and corresponding emission levels:
* **Features ($X$):** Engine size, cylinders, fuel consumption metrics, etc.
* **Target ($y$):** CO₂ emissions (continuous numerical output).
* **Excluded Features:** Unnecessary or redundant columns were dropped to reduce multicollinearity.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/machine-learning-practice.git](https://github.com/YOUR-USERNAME/machine-learning-practice.git)
