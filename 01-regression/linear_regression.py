import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load data
df = pd.read_csv("co2.csv")

# 2. EDA - Target Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["out1"], kde=True)
plt.title("CO2 Emission Distribution")
plt.xlabel("CO2 Emissions")
plt.show()

# 3. EDA - Correlation Heatmap (Fixed numeric_only)
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 4. Prepare data
X = df.drop(columns=["out1", "fuelcomb", "cylandr"])
y = df["out1"]

# 5. Train-test split (Fixed random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predict & Evaluate
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# 8. Visualization: Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual CO2")
plt.ylabel("Predicted CO2")
plt.title("Actual vs Predicted CO2 Emissions")
plt.show()
