import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("dataset/diabetes.csv")

# 2. Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. Train/Test Split (fixed seed for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train Logistic Regression Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 5. Model Evaluation
predictions = model.predict(X_test)

# Built-in evaluation instead of manual loops
acc = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

# Extracts correct vs incorrect counts directly from confusion matrix diagonal
correct = cm.diagonal().sum()
incorrect = cm.sum() - correct

print(f"Correct Predictions: {correct}")
print(f"Incorrect Predictions: {incorrect}")
print(f"Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, predictions))
