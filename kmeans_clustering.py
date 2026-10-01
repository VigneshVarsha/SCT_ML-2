import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Resolve paths relative to this script so it runs from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Locate and Load Dataset
file_candidates = [
    "Mall_Customers.csv",
    "data/Mall_Customers.csv",
    "Mall Customer Segmentation Data.csv"
]

csv_path = None
for path in file_candidates:
    candidate = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
    if os.path.exists(candidate):
        csv_path = candidate
        break

if csv_path:
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
else:
    print("Local CSV not found. Generating default sample dataframe.")
    np.random.seed(42)
    df = pd.DataFrame({
        'CustomerID': range(1, 201),
        'Gender': np.random.choice(['Male', 'Female'], size=200),
        'Age': np.random.randint(18, 70, size=200),
        'Annual Income (k$)': np.random.randint(15, 137, size=200),
        'Spending Score (1-100)': np.random.randint(1, 100, size=200)
    })

# Display basic information
print("\nDataset Preview:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe())

# 2. Select Features for Clustering
# Standard features: Annual Income (k$) and Spending Score (1-100)
income_matches = [col for col in df.columns if 'income' in col.lower()]
spending_matches = [col for col in df.columns if 'spending' in col.lower()]

if not income_matches or not spending_matches:
    raise SystemExit(
        "Could not find income/spending columns. Available columns: "
        + ", ".join(map(str, df.columns))
    )

income_col = income_matches[0]
spending_col = spending_matches[0]

df = df.dropna(subset=[income_col, spending_col])
if df.empty:
    raise SystemExit("No rows left after removing missing feature values.")

X = df[[income_col, spending_col]].values

# 3. Determine Optimal Clusters Using Elbow Method
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


def find_optimal_k(k_values, wcss_values):
    """Pick k using the maximum distance from the line joining the elbow curve ends."""
    k = np.asarray(list(k_values), dtype=float)
    w = np.asarray(wcss_values, dtype=float)

    # Normalize both axes so the distance is scale-independent
    k_norm = (k - k.min()) / (k.max() - k.min())
    w_norm = (w - w.min()) / (w.max() - w.min())

    m, b = np.polyfit([k_norm[0], k_norm[-1]], [w_norm[0], w_norm[-1]], 1)
    distances = np.abs(m * k_norm - w_norm + b) / np.hypot(1, m)
    return int(k[int(np.argmax(distances))])


optimal_k = find_optimal_k(k_range, wcss)
if optimal_k < 2:  # a 1-cluster "elbow" is never useful
    optimal_k = 5
print(f"\nElbow method suggests optimal k = {optimal_k}")

# Plot Elbow Curve
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, marker='o', linewidth=2, color='darkorange')
plt.axvline(optimal_k, linestyle='--', color='gray', alpha=0.7,
            label=f'Optimal k = {optimal_k}')
plt.title("Elbow Method to Determine Optimal K", fontsize=12)
plt.xlabel("Number of Clusters (k)", fontsize=10)
plt.ylabel("WCSS (Within-Cluster Sum of Squares)", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 4. Fit K-Means with Optimal Clusters
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)
df['Cluster'] = cluster_labels

# Cluster centroids
centroids = kmeans.cluster_centers_

# 5. Label clusters from their centroids (K-Means label order is arbitrary)
income_mean = centroids[:, 0].mean()
spending_mean = centroids[:, 1].mean()
TOLERANCE = 0.10  # +/- 10% around the mean counts as "moderate"


def classify(value, average):
    if value <= average * (1 - TOLERANCE):
        return 'Low'
    if value >= average * (1 + TOLERANCE):
        return 'High'
    return 'Moderate'


SEGMENT_SUFFIX = {
    ('High', 'High'): 'Target',
    ('High', 'Low'): 'Careful',
    ('Low', 'High'): 'Careless',
    ('Low', 'Low'): 'Sensible',
    ('Moderate', 'Moderate'): 'Standard',
}

cluster_names = []
for cx, cy in centroids:
    income_level = classify(cx, income_mean)
    spending_level = classify(cy, spending_mean)
    suffix = SEGMENT_SUFFIX.get((income_level, spending_level))
    name = f"{income_level} Income, {spending_level} Spenders"
    cluster_names.append(f"{name} ({suffix})" if suffix else name)

# Sort clusters by income (desc) then spending (desc) for a readable legend
order = sorted(range(optimal_k), key=lambda i: (-centroids[i, 0], -centroids[i, 1]))
cluster_names = [cluster_names[i] for i in order]
sorted_centroids = centroids[order]

# 6. Visualize Customer Segments
plt.subplot(1, 2, 2)
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange', 'purple']

for i in range(optimal_k):
    plt.scatter(
        X[cluster_labels == order[i], 0],
        X[cluster_labels == order[i], 1],
        s=60,
        c=colors[i % len(colors)],
        label=cluster_names[i],
        alpha=0.7,
        edgecolors='k'
    )

# Plot Centroids
plt.scatter(
    sorted_centroids[:, 0],
    sorted_centroids[:, 1],
    s=200,
    c='yellow',
    marker='X',
    edgecolors='black',
    label='Centroids'
)

plt.title("Customer Segments (K-Means Clustering)", fontsize=12)
plt.xlabel(income_col, fontsize=10)
plt.ylabel(spending_col, fontsize=10)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()

# Save visualization
plot_path = os.path.join(BASE_DIR, "customer_segments_plot.png")
plt.savefig(plot_path, dpi=300)
print(f"\nVisualization saved as '{plot_path}'")
plt.show()

# 7. Save Clustered Output to CSV
output_path = os.path.join(BASE_DIR, "segmented_customers.csv")
df.to_csv(output_path, index=False)
print(f"Clustered results saved to '{output_path}'")

# Cluster summary
print("\nCluster Summary:")
summary = df.groupby('Cluster')[[income_col, spending_col]].mean().round(2)
summary['Customers'] = df.groupby('Cluster').size()
summary['Segment'] = [cluster_names[order.index(i)] for i in summary.index]
print(summary.to_string())
