# %%
# Customer Segmentation using K-Means

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# 1. Generate mock customer data
np.random.seed(42)
n_customers = 200
data = {
    'Customer_ID': [f"C{i:04d}" for i in range(1, n_customers + 1)],
    'Age': np.random.randint(18, 70, size=n_customers),
    'Annual_Income (k$)': np.random.normal(60, 25, size=n_customers).astype(int),
    'Visit_Frequency': np.random.poisson(12, size=n_customers),
    'Spending_Score': np.random.uniform(0, 100, size=n_customers)
}
df = pd.DataFrame(data)

# Save original dataset
df.to_csv("customer_data.csv", index=False)

# 2. Preprocess and scale the data
features = ['Age', 'Annual_Income (k$)', 'Visit_Frequency', 'Spending_Score']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# 3. Elbow Method to determine optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.tight_layout()
plt.savefig("elbow_plot.png")
plt.show()

# 4. Apply KMeans with chosen k (e.g., 4)
k = 4
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(scaled_features)
df['Cluster'] = kmeans.labels_

# 5. Visualize 2D Clusters using PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
df['PCA1'] = principal_components[:, 0]
df['PCA2'] = principal_components[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=60)
plt.title('Customer Segments (PCA 2D View)')
plt.tight_layout()
plt.savefig("customer_clusters_2D.png")
plt.show()

# 6. Visualize 3D Clusters using PCA
pca_3d = PCA(n_components=3)
pca_components_3d = pca_3d.fit_transform(scaled_features)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_components_3d[:, 0], pca_components_3d[:, 1], pca_components_3d[:, 2],
                     c=df['Cluster'], cmap='Set2', s=60)
ax.set_title('Customer Segments (PCA 3D View)')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
plt.tight_layout()
plt.savefig("customer_clusters_3D.png")
plt.show()

print("\nCustomer Segmentation complete. Dataset saved as 'customer_data.csv' with cluster labels.")
print("Elbow plot saved as 'elbow_plot.png', 2D plot as 'customer_clusters_2D.png', and 3D plot as 'customer_clusters_3D.png'.")



