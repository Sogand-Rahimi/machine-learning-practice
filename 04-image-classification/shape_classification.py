import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Configuration & Path Setup
# Using os.path to build a path that works on Windows, Mac, and Linux
DATASET_DIR = os.path.join("dataset", "dataset") 
NUM_IMAGES = 20
IMAGE_SIZE = (64, 64)

# 2. Data Preprocessing & Feature Extraction
X = []
for i in range(1, NUM_IMAGES + 1):
    img_path = os.path.join(DATASET_DIR, f"{i}.bmp")
    
    if os.path.exists(img_path):
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Extract single channel (Red) and flatten into a 1D vector (4096 pixels)
        channel_red = img_array[:, :, 0]
        flattened_img = channel_red.flatten()
        
        # Normalize pixel values to range [0, 1]
        normalized_img = flattened_img / 255.0
        X.append(normalized_img)
    else:
        print(f"Warning: File {img_path} not found.")

X = np.array(X)

# Binary targets: 0 for Squares (images 1-10), 1 for Circles (images 11-20)
y = np.array([0]*10 + [1]*10)

# 3. Visual Check of First Sample
if len(X) > 0:
    plt.imshow(X[0].reshape(IMAGE_SIZE), cmap="gray")
    plt.title("Sample Preprocessed Image (Square)")
    plt.axis("off")
    plt.show()

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train Model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 6. Evaluation
predictions = model.predict(X_test)

print("Predictions:", predictions)
print("Actual Labels:", y_test)
print(f"\nAccuracy Score: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, predictions))
