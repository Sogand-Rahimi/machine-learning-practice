# Fruit Classification

A multi-class classification project that predicts the type of a fruit (apple, mandarin, orange, lemon) from its mass, width, height and color score. It trains four scikit-learn classifiers and compares their accuracy.

---

## Key Features

* **Data Visualization:** A scatter matrix and a seaborn pairplot show how the four features spread and how well each fruit type separates.
* **Feature Scaling:** Scales all features to the 0-1 range with `MinMaxScaler`, fitted on the training data only to avoid data leakage.
* **Model Comparison:** Trains and evaluates Logistic Regression, Decision Tree, k-Nearest Neighbors and SVC on the same train/test split.
* **Evaluation:** Prints the training and test accuracy of each model, so you can spot overfitting.

---

## Prerequisites

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Dataset

The dataset is not included in this repo. Download the **Fruits with colors dataset** (`fruit_data_with_colors.txt`) and save it as `data/fruit_data_with_colors.txt`.

The file is tab-separated. The columns are `fruit_label`, `fruit_name`, `fruit_subtype`, `mass`, `width`, `height` and `color_score`.

## Usage

```bash
python fruit_classification.py
```

The script saves `fruits_scatter_matrix.png` and `fruits_pairplot.png` in the current folder and prints the accuracy of each model.

## Example Output

```
Logistic Regression
  Training accuracy: 0.xx
  Test accuracy:     0.xx
...
```
