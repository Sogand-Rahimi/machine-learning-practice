# Handwritten Digit Classification

An end-to-end multi-class classification module predicting handwritten digits (0–9) using Scikit-Learn's `load_digits` dataset and Logistic Regression.

---

## Overview

This module demonstrates multi-class classification on $8 \times 8$ pixel grayscale digit images. It evaluates model accuracy, generates precision/recall metrics, and plots a confusion matrix to highlight misclassified digits.

---

## Workflow & Implementation

1. **Data Preprocessing**: Split dataset 80/20 with `random_state=42` for consistent reproducibility.
2. **Model Selection**: Multi-class Logistic Regression solver.
3. **Evaluation**: Evaluated using accuracy score, precision, recall, and F1-score.
4. **Visualization**: Plotted confusion matrix heatmap.

---

## How to Run

```bash
python digit_classification.py
