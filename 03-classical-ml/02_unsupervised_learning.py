"""
=============================================================
FASE 3 - MODUL 2: UNSUPERVISED LEARNING
=============================================================
Unsupervised = tidak ada label. Model mencari STRUKTUR dalam data.

Tiga kategori utama:
1. Clustering (K-Means, DBSCAN, Hierarchical)
2. Dimensionality Reduction (PCA, t-SNE, UMAP)
3. Anomaly Detection (Isolation Forest, LOF)

Koneksi Teknik Elektro:
- PCA = Karhunen-Loeve Transform (KLT) -> optimal decorrelation
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
# BAGIAN 1: K-Means Clustering
# ===========================================================
# K-Means = algoritma clustering paling populer.
# Goal: partisi data ke k clusters dengan within-cluster variance minimum.
#
# Matematika K-Means:
# Objective: minimize Sum_{i=1}^{k} Sum_{x in C_i} ||x - mu_i||**2
#   dimana C_i = cluster i, mu_i = centroid cluster i.
# Ini adalah NP-hard problem, tapi Lloyd's algorithm memberikan
# local optimum yang biasanya cukup baik.
#
# Algoritma (Lloyd's algorithm):
# 1. Inisialisasi k centroid (random)
# 2. Assign setiap point ke centroid terdekat (E-step / Expectation)
# 3. Update centroid = mean dari points di cluster (M-step / Maximization)
# 4. Ulangi sampai convergen
#
# Kenapa K-Means penting?
# - Simplicity: algoritma paling sederhana dan cepat untuk clustering.
# - Scalability: O(n * k * i * d) dimana n=samples, k=clusters,
#   i=iterations, d=dimensions. Bisa handle dataset besar.
# - Foundation: banyak algoritma clustering lebih canggih
#   mengembangkan ide K-Means.
#
# Hyperparameter kunci:
# - n_clusters (k): jumlah cluster. Biasanya tidak diketahui!
#   Gunakan Elbow Method atau Silhouette Score untuk estimasi.
# - init: metode inisialisasi. 'k-means++' (default) lebih baik
#   dari random karena menghindari poor initialization.
# - n_init: berapa kali K-Means dijalankan dengan init berbeda.
#   Default di sklearn >= 1.4 adalah "auto" (= 10 untuk k-means++).
#   Pilih best run berdasarkan inertia.
#
# Keterbatasan K-Means:
# - Asumsikan cluster berbentuk convex (bulat).
# - Tidak bisa handle non-convex shapes (contoh: moons).
# - Sensitif terhadap skala fitur -> WAJIB standardize!
# - Sensitif terhadap outliers (karena menggunakan mean).
# - Random initialization bisa menyebabkan local minimum yang buruk.
#
# Koneksi Teknik Elektro:
# - K-Means = vector quantization (VQ) - seperti di kompresi sinyal audio
# - Centroid = codebook vectors
# - Assignment = encoding ke nearest codeword

X_blobs, y_true = make_blobs(n_samples=300, centers=4,
                              cluster_std=0.8, random_state=42)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_blobs)

print("=== K-Means Clustering ===")
print(f"Inertia: {kmeans.inertia_:.2f}")
# Inertia = within-cluster sum of squares (WCSS)
# Semakin kecil -> cluster semakin compact
# TIPS: Inertia selalu turun seiring k meningkat, jadi tidak bisa
#   digunakan untuk memilih k secara langsung (makanya Elbow Method).

print(f"Silhouette Score: {silhouette_score(X_blobs, labels):.4f}")
# Silhouette = (-1, 1). 1 = perfect clustering, 0 = overlapping, -1 = wrong
# Formula: s(i) = (b(i) - a(i)) / max(a(i), b(i))
#   a(i) = avg distance ke point di cluster yang sama
#   b(i) = avg distance ke point di cluster terdekat
# Silhouette mengukur seberapa "mirip" suatu point dengan clusternya
# sendiri dibanding cluster terdekat.

print(f"Adjusted Rand Index: {adjusted_rand_score(y_true, labels):.4f}")
# ARI = (-1, 1). 1 = perfect match dengan ground truth
# ARI mengukur similarity antara dua clustering, adjusted untuk chance.
# ARI = 0 berarti clustering random (tidak lebih baik dari chance).

# Elbow Method - cara menentukan K optimal
# Plot inertia vs K, cari "siku" (elbow)
# Teori: penurunan inertia yang signifikan menunjukkan struktur baru.
# Setelah elbow, penambahan cluster hanya memecah cluster existing.
# TIPS: Elbow tidak selalu jelas. Gunakan Silhouette Score sebagai validasi.

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
print("OK Saved: 01_elbow_silhouette.png")


# ===========================================================
# BAGIAN 2: DBSCAN - Density-Based Clustering
# ===========================================================
# Keunggulan DBSCAN vs K-Means:
# - Tidak perlu tentukan K
# - Bisa mendeteksi cluster berbentuk aneh
# - Bisa mengidentifikasi outliers (noise points)
#
# Matematika DBSCAN:
# - Core point: point dengan minimal min_samples neighbors dalam radius eps.
# - Border point: point dalam radius eps dari core point tapi bukan core.
# - Noise point: point yang bukan core maupun border.
# - Density-reachable: chain of core points dengan jarak <= eps.
# - Cluster: semua points yang density-reachable dari core point yang sama.
#
# Hyperparameter kunci:
# - eps (epsilon): radius neighborhood. Ini parameter PALING KRUSIAL.
#   Terlalu kecil = semua point jadi noise. Terlalu besar = semua jadi 1 cluster.
#   TIPS: Plot k-distance graph (jarak ke k-th nearest neighbor, biasanya k=4).
#   Pilih eps di "elbow" dari k-distance plot.
# - min_samples: minimum points untuk jadi core point.
#   Default 5. Naikkan untuk data noisy. Turunkan untuk data sparse.
#
# Kenapa DBSCAN penting?
# - Tidak membuat asumsi bentuk cluster.
# - Robust terhadap outliers (noise di-label -1).
# - Hanya memerlukan 2 parameter (tapi tuning eps bisa tricky).
#
# Keterbatasan DBSCAN:
# - Sensitif terhadap densitas yang bervariasi (struggle dengan cluster
#   yang sangat dense dan sangat sparse dalam dataset yang sama).
# - Tidak scale-invariant: hasil bergantung pada unit fitur.
#   WAJIB standardize sebelum DBSCAN!
# - High-dimensional data: distance metrics kurang meaningful di dimensi tinggi
#   (curse of dimensionality).
#
# Koneksi Teknik Elektro:
# - DBSCAN = adaptive thresholding di feature space
# - Eps = detection threshold
# - Core points = reliable signal detection

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
print("\nOK Saved: 02_clustering_comparison.png")


# ===========================================================
# BAGIAN 3: PCA - Principal Component Analysis
# ===========================================================
# PCA = cari arah variance terbesar dalam data
# Mathematically: eigendecomposition dari covariance matrix
#
# Matematika PCA (langkah demi langkah):
# 1. Center data: X_centered = X - mean(X)
#    (WAJIB! PCA sensitive ke origin)
# 2. Compute covariance matrix: C = X_centered.T @ X_centered / (n-1)
#    Element C[i,j] = covariance antara fitur i dan j.
# 3. Eigendecomposition: C = V @ Lambda @ V.T
#    V = matrix eigenvectors (principal components)
#    Lambda = diagonal matrix eigenvalues (variance explained)
# 4. Sort eigenvalues descending, eigenvectors mengikuti.
# 5. Project: X_pca = X_centered @ V[:, :k]
#
# Variance Explained:
# - Setiap PC menjelaskan proporsi variance = lambda_i / Sum(lambda)
# - Cumulative variance = running sum dari variance explained.
# - Pilih k berdasarkan threshold (misal 95% variance retained).
#
# Kenapa PCA penting?
# - Dimensionality reduction: kurangi dimensi tanpa kehilangan informasi signifikan.
# - Decorrelation: PC saling orthogonal (tidak berkorelasi).
# - Noise reduction: noise biasanya ada di PC dengan eigenvalue kecil.
# - Visualization: project ke 2D/3D untuk visualisasi.
#
# Hyperparameter kunci:
# - n_components: jumlah PC. Bisa integer, float (variance ratio), atau None (semua).
# - TIPS: Jika n_components = 0.95, PCA otomatis memilih k minimal untuk 95% variance.
#
# Keterbatasan PCA:
# - Linear only: tidak bisa menangkap non-linear structure.
#   Untuk non-linear, gunakan t-SNE, UMAP, atau Kernel PCA.
# - Sensitif ke scale: fitur dengan variance besar akan mendominasi.
#   WAJIB standardize sebelum PCA!
# - Interpretabilitas: PC adalah kombinasi linear fitur asli,
#   yang kadang sulit diinterpretasikan secara domain.
#
# Koneksi Teknik Elektro:
# - PCA = Karhunen-Loeve Transform (KLT) - optimal decorrelation & compression
# - Eigenvectors = basis functions (seperti Fourier series tapi data-adaptive)
# - Eigenvalues = energy di setiap mode
# - Truncated PCA = lossy compression (sama seperti JPEG!)

# Generate high-dimensional data
from sklearn.datasets import load_digits
digits = load_digits()
X_digits = digits.data   # 64 features (8x8 pixel)
y_digits = digits.target  # 10 kelas (digit 0-9)

print(f"\n=== PCA pada Digits Dataset ===")
print(f"Original shape: {X_digits.shape}")

# Standardize
# PCA sensitive ke scale -> harus standardize terlebih dahulu
# TIPS: Untuk image data, centering cukup (std ~ sama).
# Untuk data sensor dengan unit berbeda, WAJIB standardize.
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
# PC1 selalu menjelaskan variance terbesar, PC2 kedua terbesar, dst.
# Jika PC1 mendominasi (>80%), data hampir 1-dimensional.
axes[0].plot(pca.explained_variance_ratio_[:30], 'bo-', markersize=4)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained')
axes[0].set_title('Variance per PC')
axes[0].grid(True)

# Cumulative variance
# Threshold 95% adalah tradeoff yang umum:
# - Terlalu rendah (<90%) = kehilangan informasi signifikan.
# - Terlalu tinggi (>99%) = tidak ada pengurangan dimensi yang meaningful.
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
print("OK Saved: 03_pca_variance.png")

# 2D visualization
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='tab10',
                     s=10, alpha=0.7)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Digits Dataset - PCA 2D Projection')
plt.colorbar(scatter)
plt.savefig('04_pca_2d.png', dpi=100, bbox_inches='tight')
plt.close()
print("OK Saved: 04_pca_2d.png")


# ===========================================================
# BAGIAN 4: t-SNE - Non-linear Dimensionality Reduction
# ===========================================================
# PCA = linear -> preserves global structure
# t-SNE = non-linear -> preserves LOCAL structure (neighborhood)
#
# Matematika t-SNE:
# 1. Compute pairwise similarity di high-dim space:
#    p_{j|i} = exp(-||x_i - x_j||**2 / 2*sigma_i**2) / Sum_{k != i} exp(...)
#    sigma_i ditentukan oleh perplexity (seberapa banyak neighbors yang dipertimbangkan).
# 2. Compute pairwise similarity di low-dim space:
#    q_{ij} = (1 + ||y_i - y_j||**2)**(-1) / Sum_{k != l} (1 + ||y_k - y_l||**2)**(-1)
#    Ini adalah Student-t distribution (hence "t" in t-SNE).
# 3. Minimize KL-divergence antara P dan Q menggunakan gradient descent.
#
# Hyperparameter kunci:
# - perplexity: kira-kira jumlah neighbors yang dipertimbangkan.
#   Biasanya 5-50. Default 30.
#   Perplexity kecil = local structure, perplexity besar = global structure.
#   PERINGATAN: perplexity tidak boleh lebih besar dari n_samples - 1.
# - learning_rate: default "auto" (sekitar n/12). Terlalu besar = blob,
#   terlalu kecil = clustered terlalu rapat.
# - n_iter: default 1000. Naikkan untuk convergence yang lebih baik.
# - early_exaggeration: default 12. Memisahkan cluster di awal training.
#
# Kenapa t-SNE penting?
# - Sangat bagus untuk visualisasi cluster di high-dimensional data.
# - Menangkap non-linear relationships yang PCA lewatkan.
# - Wajib gunakan untuk EDA data high-dimensional.
#
# Keterbatasan t-SNE:
# - Stochastic: hasil berbeda setiap run (gunakan random_state).
# - Tidak preservasi global structure (jarak antar cluster tidak bermakna).
# - Tidak bisa transform data baru (no transform method, hanya fit_transform).
# - Mahal komputasi: O(n^2) sampai O(n log n) dengan Barnes-Hut.
# - Parameters sensitive: perplexity dan learning_rate perlu di-tune.
#
# TIPS:
# - Jalankan t-SNE berkali-kali dengan parameter berbeda.
# - Jangan interpretasi jarak antar cluster (hanya local structure).
# - Preprocessing dengan PCA ke ~50 dimensi bisa mempercepat t-SNE.

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
print("OK Saved: 05_pca_vs_tsne.png")
print("-> t-SNE biasanya memisahkan cluster lebih jelas untuk visualisasi")


# ===========================================================
# BAGIAN 5: Anomaly Detection
# ===========================================================
# Sangat relevan untuk Teknik Elektro:
# - Predictive maintenance (deteksi anomali mesin)
# - Power quality monitoring
# - Network intrusion detection
#
# Isolation Forest:
# - Prinsip: outliers lebih "isolated" dan lebih mudah dipisahkan.
# - Algoritma: secara random pilih fitur dan split threshold.
#   Ulangi sampai setiap point terisolasi.
# - Outliers membutuhkan fewer splits untuk diisolasi.
# - Anomaly score = avg path length (shorter = more anomalous).
#
# Kenapa Isolation Forest bagus?
# - Tidak memerlukan model "normal" secara eksplisit.
# - Efisien: O(n log n) untuk training dan inference.
# - Robust untuk high-dimensional data.
# - Tidak perlu label anomaly (unsupervised).
#
# Hyperparameter kunci:
# - contamination: proporsi outliers yang diharapkan (0.0 - 0.5).
#   Default 'auto' (= 0.1). Ini MEMPENGARUHI threshold.
#   PERINGATAN: contamination harus di-set berdasarkan domain knowledge.
#   Terlalu rendah = banyak false negatives. Terlalu tinggi = banyak false positives.
# - n_estimators: jumlah trees. Default 100.
#
# Alternatif methods:
# - Local Outlier Factor (LOF): density-based. Bagus untuk local anomalies.
# - One-Class SVM: mempelajari boundary data normal. Bagus untuk boundary complex.
# - Elliptic Envelope: asumsikan Gaussian. Bagus untuk data Gaussian.

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
print("\nOK Saved: 06_anomaly_detection.png")


# ===========================================================
# LATIHAN 9: Unsupervised Analysis
# ===========================================================
"""
TARGET Learning Objectives:
   - Mengimplementasikan K-Means dari nol
   - Mengimplementasikan PCA dari nol
   - Mengaplikasikan unsupervised methods ke real dataset

PANDUAN LANGKAH-LANGKAH:

STEP 1: Implementasi K-Means FROM SCRATCH (NumPy only)
------------------------------------------------------
Buat class KMeansScratch dengan algoritma Lloyd:

   a) __init__(self, k, max_iters=100, tol=1e-4)
   b) fit(X):
      - Random initialization centroid (pilih k random points dari X)
      - For iter in range(max_iters):
        * Assign: labels = argmin ||X - centroids||**2
        * Update: centroids = mean(X[labels==i]) untuk setiap i
        * Check convergence: if change < tol, break
        
   c) predict(X): return argmin ||X - centroids||**2
   
   TIPS KENAPA from scratch?
     - Memahami algoritma secara mendalam
     - Memahami sensitivity ke inisialisasi
     - Memahami convergence criteria

   TEST Verification:
     - Compare dengan sklearn KMeans pada dataset yang sama
     - Inertia harus mendekati
     - Labels bisa berbeda (permutation) tapi clustering sama


STEP 2: Implementasi K-Means++ Initialization
---------------------------------------------
K-Means++ = smart initialization untuk centroid.

   Algoritma:
   a) Pilih centroid pertama secara random
   b) Untuk setiap point, hitung D(x)**2 = jarak ke centroid terdekat
   c) Pilih centroid baru dengan probabilitas proporsional D(x)**2
   d) Ulangi sampai k centroid
   
   TIPS KENAPA K-Means++?
     - Menghindari poor initialization
     - Convergence lebih cepat
     - Hasil lebih konsisten
     - Default di sklearn


STEP 3: Implementasi PCA FROM SCRATCH
--------------------------------------
Buat class PCAScratch:

   a) fit(X):
      - Center data: X_centered = X - mean
      - Covariance matrix: C = X_centered.T @ X_centered / (n-1)
      - Eigendecomposition: eigenvalues, eigenvectors = np.linalg.eigh(C)
      - Sort by eigenvalues descending
      
   b) transform(X, n_components):
      - Project X ke top-k eigenvectors
      - return X @ eigenvectors[:, :k]
      
   TIPS KENAPA from scratch?
     - Memahami bahwa PCA = eigendecomposition of covariance
     - Memahami bahwa PC = eigenvectors
     - Memahami bahwa variance explained = eigenvalues

   TEST Verification:
     - Compare dengan sklearn PCA
     - Eigenvalues harus sama (ordering bisa berbeda untuk degenerate)
     - Transformasi harus sama (sign bisa berbeda)


STEP 4: Gunakan PCA + K-Means pada Real Dataset
-----------------------------------------------
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


TIPS HINTS:
   - np.linalg.eigh untuk symmetric matrix (covariance)
   - np.argsort untuk sorting eigenvalues
   - np.linalg.norm(X[:, None] - centroids, axis=2) untuk distance matrix
   - np.argmin(distance_matrix, axis=1) untuk labels

PERINGATAN COMMON MISTAKES:
   - Tidak center data sebelum PCA
   - Lupa sort eigenvalues descending
   - K-Means tanpa multiple init -> stuck di local minimum
   - Menghitung covariance dengan n bukan n-1 (bias correction)

TARGET EXPECTED OUTPUT:
   - KMeansScratch yang matching dengan sklearn
   - PCAScratch yang matching dengan sklearn
   - Analysis: optimal PC untuk digit clustering
   - Insight: digit mana yang sering "tercluster bersama"
"""


# ===========================================================
# 🔥 CHALLENGE: Anomaly Detection untuk Power Quality
# ===========================================================
"""
TARGET Learning Objectives:
   - Membangun anomaly detection system untuk domain power systems
   - Menggabungkan time domain dan frequency domain features
   - Membandingkan multiple unsupervised methods

PANDUAN LANGKAH-LANGKAH:

STEP 1: Generate Data Normal dan Anomali
-----------------------------------------
Simulasi monitoring kualitas daya listrik:

   a) Data normal (800 samples):
      - Sinyal 50Hz sinusoidal, THD < 5%
      - Voltage: 220V +/- 5%
      - Duration: 1 detik per sample
      - Sampling rate: 1000 Hz
      
   b) Inject anomalies (200 samples):
      - Voltage sag (tegangan turun > 10%)
      - Voltage swell (tegangan naik > 10%)
      - Harmonic distortion (THD > 8%)
      - Transient spikes (impulse noise)
      - Frequency deviation (49-51 Hz -> 48 atau 52 Hz)

   TIPS KENAPA anomalies ini?
     - Realistic untuk power systems
     - Setiap anomaly punya signature yang berbeda
     - Penting untuk protective relaying


STEP 2: Extract Features
------------------------
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
   
   TIPS KENAPA features ini?
     - Voltage sag/swell terdeteksi di time domain
     - Harmonic distortion terdeteksi di frequency domain
     - Crest factor sensitif terhadap transients


STEP 3: Apply Unsupervised Methods
-----------------------------------
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
---------------------------
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
-----------------------
   a) Method mana yang terbaik untuk setiap jenis anomaly?
   b) Fitur mana yang paling diskriminatif?
   c) Apakah ada anomaly yang tidak tertangkap? Kenapa?
   d) Bagaimana handle false alarms di production?


TIPS HINTS:
   - np.fft.rfft untuk frequency domain analysis
   - THD = sqrt(sum(harmonics**2)) / fundamental
   - IsolationForest(contamination=0.2) untuk 20% anomaly
   - One-Class SVM nu parameter ~ expected anomaly ratio

PERINGATAN COMMON MISTAKES:
   - Training anomaly detector pada data dengan anomaly -> overfit
   - Tidak scale features -> distance-based methods bias
   - Mengabaikan temporal patterns -> anomaly mungkin sequential
   - Threshold terlalu strict -> banyak false negatives

TARGET EXPECTED OUTPUT:
   - Anomaly detection system dengan F1 > 0.85
   - Per-method comparison table
   - Feature importance analysis
   - Recommendation untuk deployment

Ini SANGAT relevan untuk power systems engineer!
"""

print("\n" + "="*50)
print("OK Modul selesai! Lanjut ke: 03-classical-ml/03_feature_engineering.py")
print("="*50)
