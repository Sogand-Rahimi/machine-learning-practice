import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Load dataset
digits = datasets.load_digits()
X, y = digits.data, digits.target

# 2. Split dataset with a fixed seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train Logistic Regression Model
model = linear_model.LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 4. Predict on Test Set
predictions = model.predict(X_test)

# 5. Evaluate Performance
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, predictions))

# 6. Generate Confusion Matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.title("Confusion Matrix - Digits Classification")
plt.savefig("02-classification/confusion_matrix.png")
plt.show()
