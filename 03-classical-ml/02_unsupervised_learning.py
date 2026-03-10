"""
=============================================================
FASE 3 — MODUL 2: UNSUPERVISED LEARNING
=============================================================
Unsupervised = tidak ada label. Model mencari STRUKTUR dalam data.

Tiga kategori utama:
1. Clustering (K-Means, DBSCAN, Hierarchical)
2. Dimensionality Reduction (PCA, t-SNE, UMAP)
3. Anomaly Detection (Isolation Forest, LOF)

Koneksi Teknik Elektro:
- PCA = Karhunen-Loève Transform (KLT) → optimal decorrelation
- Clustering = signal segmentation
- Anomaly detection = fault detection tanpa labeled data!

Durasi target: 3-4 jam
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score

np.random.seed(42)


# ===========================================================
# 📖 BAGIAN 1: K-Means Clustering
# ===========================================================

X_blobs, y_true = make_blobs(n_samples=300, centers=4,
                              cluster_std=0.8, random_state=42)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_blobs)

print("=== K-Means Clustering ===")
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X_blobs, labels):.4f}")
print(f"Adjusted Rand Index: {adjusted_rand_score(y_true, labels):.4f}")

# Elbow Method — cara menentukan K optimal
inertias = []
sil_scores = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_blobs)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_blobs, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('K')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method (cari "siku")')

axes[1].plot(K_range, sil_scores, 'ro-')
axes[1].set_xlabel('K')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score (higher = better)')

plt.tight_layout()
plt.savefig('01_elbow_silhouette.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 01_elbow_silhouette.png")


# ===========================================================
# 📖 BAGIAN 2: DBSCAN — Density-Based Clustering
# ===========================================================
# Keunggulan DBSCAN vs K-Means:
# - Tidak perlu tentukan K
# - Bisa mendeteksi cluster berbentuk aneh
# - Bisa mengidentifikasi outliers (noise points)

X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# K-Means on moons (GAGAL!)
km = KMeans(n_clusters=2, random_state=42)
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=km.fit_predict(X_moons),
                cmap='Set1', s=20)
axes[0].set_title('K-Means (GAGAL pada non-convex)')

# DBSCAN (BERHASIL!)
db = DBSCAN(eps=0.2, min_samples=5)
labels_db = db.fit_predict(X_moons)
axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_db, cmap='Set1', s=20)
axes[1].set_title(f'DBSCAN (clusters: {len(set(labels_db)) - (1 if -1 in labels_db else 0)})')

# Hierarchical
hc = AgglomerativeClustering(n_clusters=2, linkage='single')
axes[2].scatter(X_moons[:, 0], X_moons[:, 1], c=hc.fit_predict(X_moons),
                cmap='Set1', s=20)
axes[2].set_title('Hierarchical (single linkage)')

plt.tight_layout()
plt.savefig('02_clustering_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n📊 Saved: 02_clustering_comparison.png")


# ===========================================================
# 📖 BAGIAN 3: PCA — Principal Component Analysis
# ===========================================================
# PCA = cari arah variance terbesar dalam data
# Mathematically: eigendecomposition dari covariance matrix
#
# Sebagai engineer: ini KLT (Karhunen-Loève Transform)!
# Optimal linear transform untuk decorrelation & compression.

# Generate high-dimensional data
from sklearn.datasets import load_digits
digits = load_digits()
X_digits = digits.data   # 64 features (8x8 pixel)
y_digits = digits.target  # 10 kelas (digit 0-9)

print(f"\n=== PCA pada Digits Dataset ===")
print(f"Original shape: {X_digits.shape}")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_digits)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Variance explained
cumulative_var = np.cumsum(pca.explained_variance_ratio_)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Variance explained
axes[0].plot(pca.explained_variance_ratio_[:30], 'bo-', markersize=4)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained')
axes[0].set_title('Variance per PC')
axes[0].grid(True)

# Cumulative variance
axes[1].plot(cumulative_var, 'r-', linewidth=2)
axes[1].axhline(y=0.95, color='k', linestyle='--', alpha=0.5, label='95% threshold')
n_95 = np.argmax(cumulative_var >= 0.95) + 1
axes[1].axvline(x=n_95, color='k', linestyle='--', alpha=0.5)
axes[1].annotate(f'{n_95} components\nfor 95% variance',
                  xy=(n_95, 0.95), fontsize=10)
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance Explained')
axes[1].set_title('Cumulative Variance')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('03_pca_variance.png', dpi=100, bbox_inches='tight')
plt.close()
print(f"  Components for 95% variance: {n_95} (dari 64)")
print("📊 Saved: 03_pca_variance.png")

# 2D visualization
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='tab10',
                     s=10, alpha=0.7)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Digits Dataset — PCA 2D Projection')
plt.colorbar(scatter)
plt.savefig('04_pca_2d.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 04_pca_2d.png")


# ===========================================================
# 📖 BAGIAN 4: t-SNE — Non-linear Dimensionality Reduction
# ===========================================================
# PCA = linear → preserves global structure
# t-SNE = non-linear → preserves LOCAL structure (neighborhood)
#
# Sangat bagus untuk visualisasi cluster di high-dimensional data
# TAPI: mahal komputasi, dan perplexity harus di-tune

print("\n=== t-SNE ===")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='tab10', s=5, alpha=0.5)
axes[0].set_title('PCA (linear)')

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10', s=5, alpha=0.5)
axes[1].set_title('t-SNE (non-linear)')

plt.tight_layout()
plt.savefig('05_pca_vs_tsne.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 05_pca_vs_tsne.png")
print("→ t-SNE biasanya memisahkan cluster lebih jelas untuk visualisasi")


# ===========================================================
# 📖 BAGIAN 5: Anomaly Detection
# ===========================================================
# Sangat relevan untuk Teknik Elektro:
# - Predictive maintenance (deteksi anomali mesin)
# - Power quality monitoring
# - Network intrusion detection

from sklearn.ensemble import IsolationForest

# Simulate normal sensor data + anomalies
n_normal = 200
n_anomaly = 20
X_normal = np.random.randn(n_normal, 2) * 0.5
X_anomaly = np.random.randn(n_anomaly, 2) * 2 + 3
X_mixed = np.vstack([X_normal, X_anomaly])
y_truth = np.array([1] * n_normal + [-1] * n_anomaly)

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
y_pred_if = iso_forest.fit_predict(X_mixed)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X_mixed[:, 0], X_mixed[:, 1], c=y_truth, cmap='RdYlGn', s=20)
axes[0].set_title('Ground Truth (hijau=normal, merah=anomaly)')

axes[1].scatter(X_mixed[:, 0], X_mixed[:, 1], c=y_pred_if, cmap='RdYlGn', s=20)
axes[1].set_title('Isolation Forest Prediction')

plt.tight_layout()
plt.savefig('06_anomaly_detection.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n📊 Saved: 06_anomaly_detection.png")


# ===========================================================
# 🏋️ EXERCISE 9: Unsupervised Analysis
# ===========================================================
"""
1. Implementasi K-Means FROM SCRATCH (NumPy only):
   - Random initialization
   - E-step: assign each point to nearest centroid
   - M-step: recompute centroids
   - Implement K-means++ initialization (smarter seeding)

2. Implementasi PCA FROM SCRATCH:
   - Compute covariance matrix
   - Eigendecomposition
   - Project data

3. Gunakan PCA + K-Means pada real dataset:
   - Apply PCA untuk reduce dimensionality
   - Tentukan K optimal (elbow + silhouette)
   - Cluster dan analisis: apa yang dikelompokkan?
"""


# ===========================================================
# 🔥 CHALLENGE: Anomaly Detection untuk Power Quality
# ===========================================================
"""
Simulasi monitoring kualitas daya listrik:

1. Generate data normal: sinyal 50Hz, THD < 5%, voltage ±5%
2. Inject anomalies:
   - Voltage sag (tegangan turun > 10%)
   - Voltage swell (tegangan naik > 10%)
   - Harmonic distortion (THD > 8%)
   - Transient spikes
   - Frequency deviation

3. Extract features dari setiap window (misal 1 detik):
   - RMS voltage
   - THD
   - Crest factor
   - Frequency
   - Spectral features

4. Gunakan unsupervised methods untuk:
   a. Clustering: apakah anomali tercluster terpisah?
   b. PCA: komponen mana yang membedakan normal vs anomali?
   c. Isolation Forest: deteksi anomali tanpa label

5. Bandingkan performance (pakai ground truth labels untuk evaluasi)

Ini SANGAT relevan untuk power systems engineer!
"""

print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 03-classical-ml/03_feature_engineering.py")
print("="*50)
