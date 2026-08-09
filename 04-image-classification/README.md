# Custom Image Preprocessing & Shape Classification

A computer vision classification pipeline that processes raw `.bmp` image files, extracts pixel intensity features, normalizes pixel data, and classifies geometric shapes (Squares vs. Circles) using Logistic Regression.

---

## Overview

This project demonstrates manual image processing for machine learning from scratch without relying on high-level automated computer vision loaders.

---

## Key Steps & Concepts Learned

* **Channel Extraction:** Converted 3D RGB image arrays ($64 \times 64 \times 3$) into 2D arrays by extracting a single color channel.
* **Dimensionality Reduction & Flattening:** Reshaped $64 \times 64$ spatial pixel matrices into a 1D feature vector of $4,096$ values per image.
* **Pixel Normalization:** Scaled pixel values from $[0, 255]$ down to $[0, 1]$ for faster gradient descent convergence and numerical stability.
* **Dataset Assembly:** Dynamically constructed training arrays using NumPy and PIL image streams.
* **Supervised Classification:** Binary shape classification (0 = Square, 1 = Circle) using Logistic Regression.

---

## How to Run

1. Place your dataset inside the `dataset/dataset/` directory (`1.bmp` to `20.bmp`).
2. Run the processing and classification pipeline:

```bash
python shape_classification.py
