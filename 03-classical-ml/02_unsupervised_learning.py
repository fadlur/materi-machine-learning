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
# K-Means = algoritma clustering paling populer.
# Goal: partisi data ke k clusters dengan within-cluster variance minimum.
#
# Algoritma (Lloyd's algorithm):
# 1. Inisialisasi k centroid (random)
# 2. Assign setiap point ke centroid terdekat (E-step)
# 3. Update centroid = mean dari points di cluster (M-step)
# 4. Ulangi sampai convergen

X_blobs, y_true = make_blobs(n_samples=300, centers=4,
                              cluster_std=0.8, random_state=42)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_blobs)

print("=== K-Means Clustering ===")
print(f"Inertia: {kmeans.inertia_:.2f}")
# Inertia = within-cluster sum of squares (WCSS)
# Semakin kecil → cluster semakin compact
print(f"Silhouette Score: {silhouette_score(X_blobs, labels):.4f}")
# Silhouette = (-1, 1). 1 = perfect clustering, 0 = overlapping, -1 = wrong
print(f"Adjusted Rand Index: {adjusted_rand_score(y_true, labels):.4f}")
# ARI = (-1, 1). 1 = perfect match dengan ground truth

# Elbow Method — cara menentukan K optimal
# Plot inertia vs K, cari "siku" (elbow)
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
# eps = maximum distance antar 2 samples untuk dianggap neighbors
# min_samples = minimum points untuk membentuk core point
db = DBSCAN(eps=0.2, min_samples=5)
labels_db = db.fit_predict(X_moons)
axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_db, cmap='Set1', s=20)
axes[1].set_title(f'DBSCAN (clusters: {len(set(labels_db)) - (1 if -1 in labels_db else 0)})')
# -1 = noise points (outliers)

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
# PCA sensitive ke scale → harus standardize terlebih dahulu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_digits)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Variance explained
# Setiap PC menjelaskan sebagian variance dari data
cumulative_var = np.cumsum(pca.explained_variance_ratio_)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Variance explained per PC
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
# Prinsip: outliers lebih "isolated" dan lebih mudah dipisahkan
# Mengisolasi outlier membutuhkan fewer splits dari normal points
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
🎯 Learning Objectives:
   - Mengimplementasikan K-Means dari nol
   - Mengimplementasikan PCA dari nol
   - Mengaplikasikan unsupervised methods ke real dataset

📋 LANGKAH-LANGKAH:

STEP 1: Implementasi K-Means FROM SCRATCH (NumPy only)
───────────────────────────────────────────────────────
Buat class KMeansScratch dengan algoritma Lloyd:

   a) __init__(self, k, max_iters=100, tol=1e-4)
   b) fit(X):
      - Random initialization centroid (pilih k random points dari X)
      - For iter in range(max_iters):
        * Assign: labels = argmin ||X - centroids||²
        * Update: centroids = mean(X[labels==i]) untuk setiap i
        * Check convergence: if change < tol, break
        
   c) predict(X): return argmin ||X - centroids||²
   
   💡 KENAPA from scratch?
     - Memahami algoritma secara mendalam
     - Memahami sensitivity ke inisialisasi
     - Memahami convergence criteria

   🧪 Verification:
     - Compare dengan sklearn KMeans pada dataset yang sama
     - Inertia harus mendekati
     - Labels bisa berbeda (permutation) tapi clustering sama


STEP 2: Implementasi K-Means++ Initialization
─────────────────────────────────────────────
K-Means++ = smart initialization untuk centroid.

   Algoritma:
   a) Pilih centroid pertama secara random
   b) Untuk setiap point, hitung D(x)² = jarak ke centroid terdekat
   c) Pilih centroid baru dengan probabilitas ∝ D(x)²
   d) Ulangi sampai k centroid
   
   💡 KENAPA K-Means++?
     - Menghindari poor initialization
     - Convergence lebih cepat
     - Hasil lebih konsisten
     - Default di sklearn


STEP 3: Implementasi PCA FROM SCRATCH
──────────────────────────────────────
Buat class PCAScratch:

   a) fit(X):
      - Center data: X_centered = X - mean
      - Covariance matrix: C = X_centered.T @ X_centered / (n-1)
      - Eigendecomposition: eigenvalues, eigenvectors = np.linalg.eigh(C)
      - Sort by eigenvalues descending
      
   b) transform(X, n_components):
      - Project X ke top-k eigenvectors
      - return X @ eigenvectors[:, :k]
      
   💡 KENAPA from scratch?
     - Memahami bahwa PCA = eigendecomposition of covariance
     - Memahami bahwa PC = eigenvectors
     - Memahami bahwa variance explained = eigenvalues

   🧪 Verification:
     - Compare dengan sklearn PCA
     - Eigenvalues harus sama (ordering bisa berbeda untuk degenerate)
     - Transformasi harus sama (sign bisa berbeda)


STEP 4: Gunakan PCA + K-Means pada Real Dataset
────────────────────────────────────────────────
   Dataset: Digits (sudah di-load di atas)
   
   a) Apply PCA untuk reduce dimensionality ke 10, 20, 30
   b) Untuk setiap jumlah PC:
      - Tentukan K optimal (elbow + silhouette)
      - Cluster dengan K-Means
      - Compare clusters dengan true labels (ARI)
      
   c) Analisis:
      - Berapa PC yang optimal untuk clustering?
      - Apakah clustering menangkap digit classes?
      - Mana digit yang paling sulit dipisahkan?


💡 HINTS:
   - np.linalg.eigh untuk symmetric matrix (covariance)
   - np.argsort untuk sorting eigenvalues
   - np.linalg.norm(X[:, None] - centroids, axis=2) untuk distance matrix
   - np.argmin(distance_matrix, axis=1) untuk labels

⚠️ COMMON MISTAKES:
   - Tidak center data sebelum PCA
   - Lupa sort eigenvalues descending
   - K-Means tanpa multiple init → stuck di local minimum
   - Menghitung covariance dengan n bukan n-1 (bias correction)

🎯 EXPECTED OUTPUT:
   - KMeansScratch yang matching dengan sklearn
   - PCAScratch yang matching dengan sklearn
   - Analysis: optimal PC untuk digit clustering
   - Insight: digit mana yang sering "tercluster bersama"
"""


# ===========================================================
# 🔥 CHALLENGE: Anomaly Detection untuk Power Quality
# ===========================================================
"""
🎯 Learning Objectives:
   - Membangun anomaly detection system untuk domain power systems
   - Menggabungkan time domain dan frequency domain features
   - Membandingkan multiple unsupervised methods

📋 LANGKAH-LANGKAH:

STEP 1: Generate Data Normal dan Anomali
─────────────────────────────────────────
Simulasi monitoring kualitas daya listrik:

   a) Data normal (800 samples):
      - Sinyal 50Hz sinusoidal, THD < 5%
      - Voltage: 220V ± 5%
      - Duration: 1 detik per sample
      - Sampling rate: 1000 Hz
      
   b) Inject anomalies (200 samples):
      - Voltage sag (tegangan turun > 10%)
      - Voltage swell (tegangan naik > 10%)
      - Harmonic distortion (THD > 8%)
      - Transient spikes (impulse noise)
      - Frequency deviation (49-51 Hz → 48 atau 52 Hz)

   💡 KENAPA anomalies ini?
     - Realistic untuk power systems
     - Setiap anomaly punya signature yang berbeda
     - Penting untuk protective relaying


STEP 2: Extract Features
────────────────────────
Dari setiap window (1 detik = 1000 samples):

   Time domain:
   - RMS voltage
   - Peak voltage
   - Crest factor (peak/RMS)
   - THD (Total Harmonic Distortion)
   
   Frequency domain:
   - Dominant frequency
   - Harmonic content (3rd, 5th, 7th)
   - Spectral energy
   
   💡 KENAPA features ini?
     - Voltage sag/swell terdeteksi di time domain
     - Harmonic distortion terdeteksi di frequency domain
     - Crest factor sensitif terhadap transients


STEP 3: Apply Unsupervised Methods
───────────────────────────────────
   a) Isolation Forest:
      - contamination = estimated anomaly ratio (0.2)
      - Evaluate: precision, recall, F1
      
   b) One-Class SVM:
      - Train hanya pada data normal
      - Test: apakah anomaly terdeteksi?
      
   c) Gaussian Mixture Model (GMM):
      - Model data normal sebagai Gaussian
      - Points dengan low likelihood = anomaly
      
   d) DBSCAN:
      - Anomaly = noise points (-1)
      - Tune eps dan min_samples


STEP 4: Compare Performance
───────────────────────────
   Metrics:
   - Precision: berapa detected anomaly yang benar?
   - Recall: berapa anomaly yang tertangkap?
   - F1-score
   - False alarm rate
   
   Visualisasi:
   - ROC curve (untuk methods yang output score)
   - Confusion matrix per method
   - Feature space dengan anomaly highlighted


STEP 5: Analyze Results
───────────────────────
   a) Method mana yang terbaik untuk setiap jenis anomaly?
   b) Fitur mana yang paling diskriminatif?
   c) Apakah ada anomaly yang tidak tertangkap? Kenapa?
   d) Bagaimana handle false alarms di production?


💡 HINTS:
   - np.fft.rfft untuk frequency domain analysis
   - THD = sqrt(sum(harmonics²)) / fundamental
   - IsolationForest(contamination=0.2) untuk 20% anomaly
   - One-Class SVM nu parameter ≈ expected anomaly ratio

⚠️ COMMON MISTAKES:
   - Training anomaly detector pada data dengan anomaly → overfit
   - Tidak scale features → distance-based methods bias
   - Mengabaikan temporal patterns → anomaly mungkin sequential
   - Threshold terlalu strict → banyak false negatives

🎯 EXPECTED OUTPUT:
   - Anomaly detection system dengan F1 > 0.85
   - Per-method comparison table
   - Feature importance analysis
   - Recommendation untuk deployment

Ini SANGAT relevan untuk power systems engineer!
"""

print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 03-classical-ml/03_feature_engineering.py")
print("="*50)
