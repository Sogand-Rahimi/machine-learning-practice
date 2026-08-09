# K-Means Unsupervised Clustering

An unsupervised machine learning module implementing **K-Means Clustering** on synthetic multi-cluster spatial data generated using Scikit-Learn's `make_blobs`.

---

## Overview

This module demonstrates unsupervised grouping of 2D data points into distinct clusters based on Euclidean distance to cluster centroids.

---

## Workflow & Implementation

1. **Synthetic Data Generation**: Generated 400 2D data points across 4 centers with standard deviation $0.60$.
2. **K-Means Modeling**: Trained a 4-cluster K-Means model using explicit random seeding (`random_state=0`) for consistent cluster assignment.
3. **Centroid Extraction & Visualization**: Computed cluster centers and overlaid centroids on the classified data space using Seaborn and Matplotlib.

---

## How to Run

```bash
python kmeans_clustering.py
