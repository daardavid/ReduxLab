import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the test data
data = pd.read_csv('test_data.csv')

# Assume all columns are features (no target column)
features = data.columns
X = data.values

# Standardize the data (common for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Get the results
explained_variance_ratios = pca.explained_variance_ratio_
components = pca.components_

print("Explained Variance Ratios:")
for i, ratio in enumerate(explained_variance_ratios):
    print(f"PC{i+1}: {ratio:.6f}")

print("\nPrincipal Components (rows are PCs, columns are original features):")
for i, component in enumerate(components):
    print(f"PC{i+1}: {component}")

print("\nTransformed Data (first 5 rows):")
print(X_pca[:5])