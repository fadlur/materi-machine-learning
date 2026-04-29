"""
=============================================================
FASE 1 - MODUL 3: VISUALISASI DATA
=============================================================
"If you can't see it, you can't understand it."

Visualisasi bukan cuma untuk presentasi - ini tool DEBUGGING
paling powerful di ML. Kamu HARUS bisa visualisasi:
1. Distribusi data (histogram, KDE, box plot)
2. Relasi antar fitur (scatter, correlation, pair plot)
3. Model performance (learning curves, confusion matrix, ROC)
4. Error analysis (residual plots, prediction vs actual)

Koneksi Teknik Elektro:
- Histogram = probability density function (PDF) estimator
- Correlation = coherence antara sinyal
- Learning curve = step response dari adaptive system
- Spectrogram = time-frequency representation seperti STFT

Durasi target: 2-3 jam
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ===========================================================
# BAGIAN 1: Distribusi Data
# ===========================================================
# Distribusi data = fundamental untuk memahami karakteristik dataset.
# Kenapa penting?
# - Banyak algoritma ML mengasumsikan normal distribution
#   (Linear Regression: residual normal, Gaussian Naive Bayes: likelihood normal)
# - Skewed distribution -> perlu transformasi (log, sqrt, Box-Cox)
# - Outlier terlihat jelas di histogram dan box plot
# - Bimodal/multimodal -> indikasi adanya subpopulasi yang berbeda

np.random.seed(42)
data = pd.DataFrame({
    'normal': np.random.normal(0, 1, 1000),
    # Distribusi normal: bell curve, mean=median=mode
    # Properties: symmetric, 68-95-99.7 rule
    'skewed': np.random.exponential(2, 1000),
    # Distribusi eksponensial: skewed ke kanan (tail panjang di kanan)
    # Contoh nyata: waktu antar kejadian (failure intervals, arrival times)
    'bimodal': np.concatenate([
        np.random.normal(-2, 0.5, 500),
        np.random.normal(2, 0.5, 500)
    ])
    # Bimodal: dua peak -> menunjukkan ada 2 grup/subpopulasi
    # Contoh: tinggi badan populasi campuran pria dan wanita
})

# Membuat 3 subplot horizontally
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# figsize=(width, height) dalam inches
# 15x4 = lebar 15 inci, tinggi 4 inci -> cocok untuk 3 plot berdampingan

for i, col in enumerate(data.columns):
    # Histogram: membagi data ke bins dan menghitung frekuensi
    # bins=50: jumlah interval (semakin banyak = semakin halus, tapi bisa noisy)
    # alpha=0.7: transparansi agar overlapping bisa terlihat
    # edgecolor: warna border bar untuk memisahkan antar bin
    # density=False -> sumbu y = count/frequency
    # density=True -> sumbu y = probability density (area total = 1)
    axes[i].hist(data[col], bins=50, alpha=0.7, edgecolor='black')
    
    axes[i].set_title(f'Distribusi: {col}')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    
    # Vertical line untuk mean (garis putus-putus merah)
    axes[i].axvline(data[col].mean(), color='red', linestyle='--', label='mean')
    # axvline = vertical line di posisi tertentu pada sumbu x
    
    # Vertical line untuk median (garis putus-putus hijau)
    axes[i].axvline(data[col].median(), color='green', linestyle='--', label='median')
    
    axes[i].legend()

plt.tight_layout()
# tight_layout() menyesuaikan spacing antar subplot agar tidak tumpang tindih
# Ini penting karena label axis bisa saling bertabrakan
plt.savefig('01_distribusi.png', dpi=100, bbox_inches='tight')
# dpi=100: resolution 100 dots per inch -> balance antara kualitas dan ukuran file
# bbox_inches='tight': memotong whitespace berlebih di sekitar figure
# format='png': bisa juga 'pdf', 'svg', 'jpg'
plt.close()
# close() untuk membebaskan memory figure dari backend matplotlib
print("Saved: 01_distribusi.png")

# INSIGHT: Kalau mean != median -> distribusi skewed
# Ini penting karena banyak model ML mengasumsikan distribusi normal
# Contoh: Linear Regression mengasumsikan residual normally distributed.
# Transformasi untuk skewed data:
# - Right skewed (tail kanan panjang): log(x), sqrt(x), Box-Cox
# - Left skewed (tail kiri panjang): x^2, x^3, exponential
# Box-Cox: transformasi parameterik yang mencari lambda optimal
#          untuk membuat data sedekat mungkin dengan normal

# --- Density Plot (KDE) ---
# KDE (Kernel Density Estimate) = smooth estimate dari PDF.
# Histogram bisa terlihat berbeda tergantung bins -> KDE lebih stabil.
# KDE bekerja dengan meletakkan kernel (biasanya Gaussian) di setiap data point,
# lalu menjumlahkan semua kernel -> hasil smooth curve.
# Parameter bandwidth: semakin besar = semakin smooth, semakin kecil = semakin detail.
fig, ax = plt.subplots(figsize=(10, 5))
for col in data.columns:
    sns.kdeplot(data[col], ax=ax, label=col, fill=True, alpha=0.3)
# fill=True -> area di bawah curve diisi warna
# alpha=0.3 -> transparan agar overlapping terlihat
ax.set_title('Kernel Density Estimate (KDE)')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.savefig('01_kde.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 01_kde.png")

# --- Box Plot ---
# Box plot (box-and-whisker plot) menunjukkan 5-number summary:
# - Bottom whisker: min atau Q1 - 1.5*IQR (mana yang lebih besar)
# - Bottom box line: Q1 (25th percentile)
# - Middle box line: median (50th percentile)
# - Top box line: Q3 (75th percentile)
# - Top whisker: max atau Q3 + 1.5*IQR (mana yang lebih kecil)
# - Points outside whiskers: outliers
# Box plot lebih efisien untuk membandingkan distribusi antar grup.
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=data, ax=ax)
ax.set_title('Box Plot Comparison')
plt.tight_layout()
plt.savefig('01_boxplot.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 01_boxplot.png")

# --- Q-Q Plot (Quantile-Quantile) ---
# Q-Q plot membandingkan quantile data dengan quantile distribusi teoritis (normal).
# Jika data normal, titik-titik akan membentuk garis lurus diagonal.
# Deviasi di ujung-ujung (tails) -> masalah skewness atau heavy tails.
# Ini adalah cara terbaik untuk mengecek normalitas secara visual.
from scipy import stats
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(data.columns):
    stats.probplot(data[col], dist="norm", plot=axes[i])
    axes[i].set_title(f'Q-Q Plot: {col}')
plt.tight_layout()
plt.savefig('01_qqplot.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 01_qqplot.png")



# ===========================================================
# BAGIAN 2: Relasi Antar Fitur
# ===========================================================
# Correlation analysis = fundamental untuk feature selection dan EDA.
# Fitur yang sangat berkorelasi satu sama lain bisa redundan (multicollinearity).
# Fitur yang berkorelasi kuat dengan target adalah kandidat feature yang baik.
#
# Tipe-tipe correlation:
# 1. Pearson correlation: linear relationship (-1 sampai +1)
#    - Mengukur kekuatan dan arah hubungan LINEAR
#    - Sensitif terhadap outlier
#    - Asumsi: hubungan linear dan distribusi normal
# 2. Spearman correlation: monotonic relationship (rank-based)
#    - Mengukur hubungan monotonic (bisa non-linear tapi konsisten naik/turun)
#    - Robust terhadap outlier
#    - Tidak memerlukan asumsi distribusi
# 3. Kendall's tau: concordance-based (untuk sample kecil)
#
# Interpretasi Pearson r:
# - |r| > 0.7 -> strong correlation
# - 0.3 < |r| < 0.7 -> moderate correlation
# - |r| < 0.3 -> weak correlation
# - r > 0 -> positive (naik bersama)
# - r < 0 -> negative (salah satu naik, lainnya turun)
#
# PERINGATAN: Correlation != Causation!
# Korelasi tinggi tidak berarti sebab-akibat.
# Contoh klasik: ice cream sales berkorelasi dengan drowning incidents
# (keduanya dipengaruhi oleh suhu panas, bukan saling menyebabkan).

# Buat dataset dengan relasi yang jelas
n = 200
X1 = np.random.randn(n)
# X2 berkorelasi positif dengan X1 (koefisien 0.5)
# + noise (0.5 * np.random.randn) untuk membuat tidak perfect correlation
# Koefisien 0.5 artinya: varians X2 yang dijelaskan oleh X1 = 0.5^2 = 25%
# Sisanya 75% adalah noise (varians independen)
X2 = 0.5 * X1 + np.random.randn(n) * 0.5  # korrelasi positif ~0.7

# X3 berkorelasi negatif dengan X1 (koefisien -0.8)
# Semakin besar X1, semakin kecil X3
X3 = -0.8 * X1 + np.random.randn(n) * 0.3  # korrelasi negatif ~-0.9

# X4 independent (tidak berkorelasi dengan X1)
# Tidak ada hubungan linear maupun non-linear
X4 = np.random.randn(n)  # tidak berkorelasi ~0

# Target y = kombinasi linear dari X1, X2, X3 + noise
# Threshold 0 untuk mengubah ke klasifikasi biner
y = (X1 + X2 - X3 + np.random.randn(n) * 0.5 > 0).astype(int)
# y berkorelasi dengan X1, X2, X3 tapi tidak dengan X4

df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'target': y})

# Correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Heatmap ---
# Correlation matrix = Pearson correlation coefficient antar semua pasangan kolom.
# Range: -1 (perfect negative) sampai +1 (perfect positive).
# 0 = no linear correlation.
# Method='pearson' -> default, bisa juga 'spearman' atau 'kendall'
corr = df.corr()
# annot=True: menampilkan nilai correlation di setiap cell
# fmt='.2f': format 2 desimal
# cmap='RdBu_r': Red (positive) - Blue (negative), reversed
# center=0: membuat 0 sebagai warna netral (putih)
# vmin=-1, vmax=1: memastikan skala warna konsisten
diverging_cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, fmt='.2f', ax=axes[0],
            square=True, linewidths=0.5)
# square=True -> cell berbentuk persegi
# linewidths -> garis pemisah antar cell
axes[0].set_title('Correlation Matrix (Pearson)')

# --- Pair plot (scatter matrix) ---
# Pair plot = pairwise scatter plot antar semua fitur.
# Sangat berguna untuk melihat relasi non-linear dan clustering.
# Diagonal bisa diisi dengan KDE atau histogram.
# c=y: mewarnai point berdasarkan target class.
# cmap='RdYlBu': Red-Yellow-Blue color map.
# Koneksi Teknik Elektro: mirip dengan scatter plot di instrument calibration
# untuk melihat linearity dan hysteresis
pd.plotting.scatter_matrix(df[['X1', 'X2', 'X3', 'X4']], 
                           c=y, cmap='RdYlBu', alpha=0.5,
                           figsize=(10, 10), diagonal='kde')
plt.savefig('02_scatter_matrix.png', dpi=100, bbox_inches='tight')
plt.close()

# Save correlation heatmap separately
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=ax,
            square=True, linewidths=0.5, fmt='.2f')
ax.set_title('Correlation Matrix')
plt.tight_layout()
plt.savefig('02_correlation.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 02_correlation.png, 02_scatter_matrix.png")

# --- Joint plot (scatter + marginal distributions) ---
# Joint plot menggabungkan scatter plot dengan histogram/KDE di margin.
# Memberikan informasi relasi + distribusi marginal secara bersamaan.
sns.jointplot(data=df, x='X1', y='X2', kind='reg', height=6)
# kind='reg' -> menambahkan regression line + confidence interval
# kind='kde' -> density contour plot
plt.tight_layout()
plt.savefig('02_jointplot.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 02_jointplot.png")


# ===========================================================
# BAGIAN 3: Visualisasi untuk ML
# ===========================================================
# Decision boundary = batas pemisah antar kelas di feature space.
# Memvisualisasikan decision boundary membantu kita memahami:
# - Apakah model linear sudah cukup atau perlu non-linear?
# - Di mana model membuat kesalahan (misclassification regions)?
# - Seberapa kompleks boundary yang dibutuhkan?
#
# Decision boundary bisa linear (garis lurus, hyperplane) atau non-linear
# (kurva, ensemble boundaries, neural network manifolds).
# Visualisasi ini hanya feasible untuk 2D atau 3D feature space.

from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model_predict_fn, title="Decision Boundary"):
    """
    Memvisualisasi decision boundary untuk model 2D.
    
    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, 2)
        Feature matrix. HARUS 2 dimensi agar bisa di-plot.
        
    y : np.ndarray, shape (n_samples,)
        Target labels. Bisa binary atau multi-class.
        
    model_predict_fn : callable
        Function yang menerima array (n_points, 2) dan mengembalikan
        prediksi (n_points,). Harus bisa dipanggil seperti:
        predictions = model_predict_fn(points_array)
        
    title : str, optional
        Judul plot. Default "Decision Boundary".
        
    Returns:
    --------
    plt : matplotlib.pyplot module
        Object plt yang sudah di-configure. Panggil plt.show()
        atau plt.savefig() setelah fungsi ini.
        
    Notes:
    ------
    - Function ini membuat mesh grid di seluruh feature space
    - Setiap point di grid diprediksi oleh model
    - Hasil prediksi di-contour plot untuk menunjukkan region
    - Koneksi ke Teknik Elektro: mirip dengan plotting
      magnitude response di 2D filter analysis
    """
    # h = resolusi grid (semakin kecil = semakin halus, tapi semakin lambat)
    h = 0.02
    # x_min, x_max = batas horizontal plot (dengan margin 1 unit)
    # Margin memberi ruang agar boundary tidak terpotong di tepi
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = batas vertikal plot (dengan margin 1 unit)
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # np.meshgrid membuat grid coordinates
    # np.arange(x_min, x_max, h) = array dari x_min ke x_max dengan step h
    # xx dan yy adalah matrix 2D yang merepresentasikan grid
    # xx[i,j] = koordinat x, yy[i,j] = koordinat y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # np.c_ menggabungkan xx.ravel() dan yy.ravel() menjadi (n_points, 2)
    # ravel() meng-flatten matrix 2D menjadi 1D (row-major order)
    # Jumlah points = len(xx.ravel()) = (x_max - x_min)/h * (y_max - y_min)/h
    Z = model_predict_fn(np.c_[xx.ravel(), yy.ravel()])
    # Reshape hasil prediksi kembali ke shape grid agar bisa di-contour plot
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    # contourf = filled contour plot (region diisi warna)
    # alpha=0.3: transparan biar data points tetap terlihat di atasnya
    # cmap='RdYlBu': warna merah dan biru untuk 2 kelas
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    # contour (tanpa 'f') = garis boundary saja (opsional)
    plt.contour(xx, yy, Z, colors='black', linewidths=0.5)
    
    # scatter plot data asli
    # edgecolors='black': border putih agar point terlihat jelas
    # s=50: size marker
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black', s=50)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    return plt


# Contoh: linear decision boundary
# X_2d = 2 fitur pertama dari df
X_2d = df[['X1', 'X2']].values

# Function untuk classifier linear sederhana
def simple_linear_classifier(X):
    """
    Classifier linear sederhana untuk demo decision boundary.
    
    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, 2)
        Input features.
        
    Returns:
    --------
    np.ndarray, shape (n_samples,)
        Binary predictions (0 atau 1).
        
    Notes:
    ------
    - Decision boundary: X[:, 0] + X[:, 1] = 0
    - Ini adalah garis lurus dengan slope -1 dan intercept 0
    - Di atas garis -> prediksi 1, di bawah garis -> prediksi 0
    """
    return (X[:, 0] + X[:, 1] > 0).astype(int)


fig = plot_decision_boundary(X_2d, y, simple_linear_classifier,
                             "Contoh: Linear Decision Boundary")
plt.savefig('03_decision_boundary.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 03_decision_boundary.png")

# --- Contoh: non-linear decision boundary ---
# Decision boundary berbentuk lingkaran (radius = 1.5)
def circle_classifier(X):
    """Classifier dengan boundary berbentuk lingkaran."""
    return (X[:, 0]**2 + X[:, 1]**2 > 2).astype(int)

# Generate data concentric untuk demo non-linear
np.random.seed(42)
theta_inner = np.random.uniform(0, 2*np.pi, 50)
theta_outer = np.random.uniform(0, 2*np.pi, 50)
r_inner = np.random.normal(1, 0.2, 50)
r_outer = np.random.normal(2.5, 0.3, 50)
X_circle = np.vstack([
    np.c_[r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)],
    np.c_[r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)]
])
y_circle = np.array([0]*50 + [1]*50)

fig = plot_decision_boundary(X_circle, y_circle, circle_classifier,
                             "Contoh: Non-Linear (Circular) Boundary")
plt.savefig('03_decision_boundary_nonlinear.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 03_decision_boundary_nonlinear.png")



# ===========================================================
# BAGIAN 4: Visualisasi Model Performance
# ===========================================================
# Confusion matrix dan learning curve adalah visualisasi WAJIB
# untuk evaluasi model classification.
# Untuk regression: residual plot dan prediction vs actual plot.

# --- Confusion matrix plot ---
def plot_confusion_matrix(y_true, y_pred, classes=['Class 0', 'Class 1']):
    """
    Memvisualisasi confusion matrix secara manual.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth labels.
        
    y_pred : np.ndarray
        Predicted labels dari model.
        
    classes : list of str, optional
        Nama kelas untuk axis labels. Default ['Class 0', 'Class 1'].
        
    Returns:
    --------
    fig : matplotlib Figure
        Figure object yang berisi confusion matrix plot.
        
    Notes:
    ------
    - Confusion matrix C[i][j] = count of samples with true label i
      that were predicted as label j.
    - Diagonal = correct predictions
    - Off-diagonal = misclassifications
    - Untuk binary classification:
      * C[0][0] = True Negative (TN)
      * C[0][1] = False Positive (FP) -> Type I error
      * C[1][0] = False Negative (FN) -> Type II error
      * C[1][1] = True Positive (TP)
    - Metrics dari confusion matrix:
      * Accuracy = (TP + TN) / (TP + TN + FP + FN)
      * Precision = TP / (TP + FP) -> dari yang diprediksi positif, berapa yang benar?
      * Recall = TP / (TP + FN) -> dari yang sebenarnya positif, berapa yang ketangkap?
      * F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
      * Specificity = TN / (TN + FP)
    - Koneksi ke Teknik Elektro: mirip dengan error matrix
      di communication systems (BER analysis)
    """
    # Hitung confusion matrix manual
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    # Loop melalui setiap pasangan (true, predicted)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    
    fig, ax = plt.subplots(figsize=(6, 5))
    # annot=True: tampilkan angka di setiap cell
    # fmt='d': format integer
    # cmap='Blues': gradasi biru
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    return fig


# Contoh confusion matrix
y_true = np.array([0]*50 + [1]*50)
y_pred = np.array([0]*45 + [1]*5 + [0]*10 + [1]*40)  # some misclassifications
# TN=45, FP=5, FN=10, TP=40
# Accuracy = (45+40)/100 = 85%
# Precision = 40/45 = 88.9%
# Recall = 40/50 = 80%
# F1 = 2*(0.889*0.8)/(0.889+0.8) = 84.2%
fig = plot_confusion_matrix(y_true, y_pred)
plt.savefig('04_confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 04_confusion_matrix.png")

# --- Normalized Confusion Matrix ---
# Normalized per row -> menunjukkan recall per kelas
# Normalized per column -> menunjukkan precision per kelas
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm = np.array([[45, 5], [10, 40]])
# Normalize by row (true labels)
cm_row = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_row, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'], ax=axes[0])
axes[0].set_title('Normalized by Row (Recall)')
# Normalize by column (predicted labels)
cm_col = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
sns.heatmap(cm_col, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'], ax=axes[1])
axes[1].set_title('Normalized by Column (Precision)')
plt.tight_layout()
plt.savefig('04_confusion_matrix_norm.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 04_confusion_matrix_norm.png")


# --- ROC Curve (Receiver Operating Characteristic) ---
# ROC curve = plot True Positive Rate (TPR) vs False Positive Rate (FPR)
# pada berbagai threshold classification.
# TPR = Recall = TP / (TP + FN)
# FPR = FP / (FP + TN)
# AUC (Area Under Curve) = metrik aggregate (0.5 = random, 1.0 = perfect)
# ROC curve berguna untuk memilih threshold optimal.
# Model yang bagus memiliki ROC curve mendekati sudut kiri atas.

from sklearn.metrics import roc_curve, auc

# Simulasi predicted probabilities
np.random.seed(42)
y_true_roc = np.array([0]*100 + [1]*100)
# Class 0: prob ~ low, Class 1: prob ~ high (dengan overlap)
y_scores = np.concatenate([
    np.random.normal(0.3, 0.2, 100),  # class 0 scores
    np.random.normal(0.7, 0.2, 100)   # class 1 scores
])
y_scores = np.clip(y_scores, 0, 1)

fpr, tpr, thresholds = roc_curve(y_true_roc, y_scores)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
# Diagonal = random classifier (AUC = 0.5)
ax.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR)')
ax.set_title('Receiver Operating Characteristic (ROC)')
ax.legend(loc='lower right')
ax.grid(True)
plt.tight_layout()
plt.savefig('04_roc_curve.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 04_roc_curve.png")

# --- Precision-Recall Curve ---
# PR curve = plot Precision vs Recall pada berbagai threshold.
# Lebih informatif daripada ROC untuk imbalanced datasets.
# Di dataset imbalanced, ROC bisa terlihat bagus padahal precision rendah.
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, pr_thresholds = precision_recall_curve(y_true_roc, y_scores)
ap = average_precision_score(y_true_roc, y_scores)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.2f})')
# Baseline = proporsi kelas positif
baseline = np.sum(y_true_roc) / len(y_true_roc)
ax.axhline(baseline, color='red', linestyle='--', label=f'Baseline ({baseline:.2f})')
ax.fill_between(recall, precision, alpha=0.2, color='blue')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend(loc='lower left')
ax.grid(True)
plt.tight_layout()
plt.savefig('04_pr_curve.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 04_pr_curve.png")


# --- Learning curve plot ---
def plot_learning_curve(train_sizes, train_scores, val_scores, title="Learning Curve"):
    """
    Memvisualisasi learning curve untuk diagnosa overfitting/underfitting.
    
    Parameters:
    -----------
    train_sizes : array-like
        Ukuran training set untuk setiap titik (x-axis).
        
    train_scores : array-like
        Training score untuk setiap train_size.
        
    val_scores : array-like
        Validation score untuk setiap train_size.
        
    title : str, optional
        Judul plot. Default "Learning Curve".
        
    Returns:
    --------
    fig : matplotlib Figure
        Figure object yang berisi learning curve plot.
        
    Notes:
    ------
    - Learning curve menunjukkan bagaimana model performance
      berubah seiring bertambahnya data training.
    - High training score + low validation score = overfitting (high variance)
      -> butuh lebih banyak data, regularization, atau model lebih sederhana
    - Low training score + low validation score = underfitting (high bias)
      -> butuh model lebih kompleks, lebih banyak fitur, atau lebih lama training
    - Training dan validation score converge = good fit
      -> model sudah cukup kompleks dan data cukup banyak
    - Koneksi ke Teknik Elektro: mirip dengan convergence plot
      di adaptive filtering (LMS algorithm)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    # Plot training score
    ax.plot(train_sizes, train_scores, 'o-', color='blue', label='Training Score')
    # Plot validation score
    ax.plot(train_sizes, val_scores, 'o-', color='green', label='Validation Score')
    # Fill between untuk menunjukkan variance (shaded area)
    # Simulasi confidence interval +/- std
    ax.fill_between(train_sizes,
                    train_scores - 0.02, train_scores + 0.02, alpha=0.1, color='blue')
    ax.fill_between(train_sizes,
                    val_scores - 0.05, val_scores + 0.05, alpha=0.1, color='green')
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Score (Accuracy / R^2)')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0.5, 1.05)
    ax.grid(True)
    return fig


# Contoh learning curve (simulated)
sizes = np.array([50, 100, 200, 400, 800])
# Overfitting case: training tinggi, validation rendah, gap besar
train_acc_overfit = np.array([0.99, 0.97, 0.95, 0.93, 0.92])
val_acc_overfit = np.array([0.70, 0.78, 0.83, 0.87, 0.89])

# Underfitting case: training dan validation sama-sama rendah
train_acc_underfit = np.array([0.65, 0.66, 0.67, 0.68, 0.68])
val_acc_underfit = np.array([0.63, 0.64, 0.65, 0.66, 0.66])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Overfitting
axes[0].plot(sizes, train_acc_overfit, 'o-', label='Training Score')
axes[0].plot(sizes, val_acc_overfit, 'o-', label='Validation Score')
axes[0].fill_between(sizes, train_acc_overfit - 0.02, train_acc_overfit + 0.02, alpha=0.1)
axes[0].fill_between(sizes, val_acc_overfit - 0.05, val_acc_overfit + 0.05, alpha=0.1)
axes[0].set_title('Overfitting (High Variance)')
axes[0].set_xlabel('Training Size')
axes[0].set_ylabel('Score')
axes[0].legend()
axes[0].set_ylim(0.5, 1.05)
axes[0].annotate('Large gap = overfitting', xy=(400, 0.90), fontsize=10, color='red')

# Underfitting
axes[1].plot(sizes, train_acc_underfit, 'o-', label='Training Score')
axes[1].plot(sizes, val_acc_underfit, 'o-', label='Validation Score')
axes[1].fill_between(sizes, train_acc_underfit - 0.02, train_acc_underfit + 0.02, alpha=0.1)
axes[1].fill_between(sizes, val_acc_underfit - 0.05, val_acc_underfit + 0.05, alpha=0.1)
axes[1].set_title('Underfitting (High Bias)')
axes[1].set_xlabel('Training Size')
axes[1].set_ylabel('Score')
axes[1].legend()
axes[1].set_ylim(0.5, 1.05)
axes[1].annotate('Both low = underfitting', xy=(400, 0.68), fontsize=10, color='red')

plt.tight_layout()
plt.savefig('05_learning_curve.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 05_learning_curve.png")

# --- Residual Plot (untuk regression) ---
# Residual = actual - predicted
# Residual plot membantu diagnosa masalah model regression:
# - Random scatter around 0 -> model OK (homoscedastic)
# - Pattern/funnel shape -> model misspecified (heteroscedastic)
# - Curve pattern -> butuh fitur non-linear
# - Outliers di residual -> data points yang sulit diprediksi
np.random.seed(42)
x_reg = np.linspace(0, 10, 100)
y_true_reg = 2 * x_reg + 1 + np.random.randn(100) * 2  # y = 2x + 1 + noise
y_pred_reg = 2.1 * x_reg + 0.8  # prediksi model (sedikit bias)
residuals = y_true_reg - y_pred_reg

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Residual vs Predicted
axes[0].scatter(y_pred_reg, residuals, alpha=0.6, edgecolors='black')
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('Predicted Values')
axes[0].set_ylabel('Residuals (Actual - Predicted)')
axes[0].set_title('Residual Plot')
axes[0].grid(True)

# Histogram of residuals
axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(0, color='red', linestyle='--', label='Zero residual')
axes[1].set_xlabel('Residual Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals')
axes[1].legend()
plt.tight_layout()
plt.savefig('05_residuals.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 05_residuals.png")



# ===========================================================
# BAGIAN 5: Signal Processing Visualization (EE-Relevant!)
# ===========================================================
# Sinyal + FFT - familiar territory untuk background EE!
# FFT (Fast Fourier Transform) mengubah sinyal dari time domain
# ke frequency domain menggunakan algoritma divide-and-conquer O(N log N).
#
# Sampling Theorem (Nyquist-Shannon):
# - Untuk merekonstruksi sinyal kontinu, sampling rate fs harus > 2 * f_max
# - f_max = frekuensi tertinggi di sinyal
# - f_nyquist = fs / 2 = frekuensi maksimum yang bisa diwakili
# - Aliasing terjadi jika ada komponen frekuensi > f_nyquist
#
# Koneksi Teknik Elektro:
# - FFT = DFT yang dioptimasi, same as DSP course
# - Frequency domain analysis = spectrum analyzer principle
# - Power spectral density (PSD) = |FFT|^2 / (fs * N)
# - Windowing = mengurangi spectral leakage (sama seperti di DAS)

fs = 1000  # sampling frequency 1kHz
# fs = 1000 Hz -> bisa merepresentasikan frekuensi sampai 500 Hz (Nyquist)
t = np.arange(0, 1, 1/fs)
# np.arange(0, 1, 1/1000) = 1000 titik dari 0 sampai hampir 1 detik
# Step = 1/fs = 1ms (sampling period T_s)
# Total samples N = 1000

# Sinyal campuran: 50Hz + 120Hz + noise
# Ini mirip dengan sinyal power system (50Hz fundamental + harmonics)
# 50 Hz = frekuensi standar power grid di Indonesia/Eropa
# 120 Hz = harmonic ke-2.4 (bisa dari rectifier/non-linear load)
# Noise = white noise dari sensor/ADC quantization
signal = (np.sin(2 * np.pi * 50 * t) +           # 50 Hz fundamental
          0.5 * np.sin(2 * np.pi * 120 * t) +      # 120 Hz harmonic
          0.3 * np.random.randn(len(t)))            # white noise

# FFT (Fast Fourier Transform)
# np.fft.fft menghitung Discrete Fourier Transform secara efisien O(N log N)
# Output: complex numbers (magnitude + phase)
fft_vals = np.fft.fft(signal)
# np.fft.fftfreq menghasilkan frequency axis yang sesuai
# freqs[i] = i * fs / N untuk i = 0, 1, ..., N/2
freqs = np.fft.fftfreq(len(t), 1/fs)

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# --- Time domain ---
axes[0].plot(t[:200], signal[:200])  # 200ms pertama
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Signal (Time Domain)')
axes[0].grid(True)
# Time domain menunjukkan amplitude terhadap waktu
# Tapi susah melihat komponen frekuensi secara langsung

# --- Frequency domain ---
# Ambil setengah spektrum (karena simetri untuk real signal)
# N = 1000 -> ambil 500 points (0 sampai 500 Hz)
# np.abs(fft_vals) = magnitude spectrum
# Faktor 2/N untuk normalisasi (karena hanya ambil half spectrum)
positive_freqs = freqs[:len(freqs)//2]
magnitude = np.abs(fft_vals[:len(fft_vals)//2]) * 2/len(t)
axes[1].plot(positive_freqs, magnitude)
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude')
axes[1].set_title('Signal (Frequency Domain)')
axes[1].set_xlim(0, 200)
axes[1].grid(True)
# Frequency domain menunjukkan komponen frekuensi yang ada di sinyal
# Peak di 50 Hz (magnitude ~1.0) dan 120 Hz (magnitude ~0.5) terlihat jelas!
# Noise terlihat sebagai baseline rendah di semua frekuensi

plt.tight_layout()
plt.savefig('06_signal_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 06_signal_analysis.png")

# --- Power Spectral Density (PSD) ---
# PSD = power per unit frequency (V^2/Hz atau dBm/Hz)
# Welch's method: membagi sinyal ke overlapping segments, hitung periodogram
# per segment, lalu rata-rata -> mengurangi variance estimate
from scipy.signal import welch
f_psd, psd = welch(signal, fs=fs, nperseg=256)
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(f_psd, psd)  # log scale untuk y-axis
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD [V**2/Hz]')
ax.set_title('Power Spectral Density (Welch)')
ax.set_xlim(0, 200)
ax.grid(True)
plt.tight_layout()
plt.savefig('06_psd.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 06_psd.png")

# KONEKSI KE ML:
# - Time domain features -> statistik (mean, std, peak, RMS, crest factor)
# - Frequency domain features -> dominant frequency, spectral energy, bandwidth
# - Time-frequency features -> spectrogram, MFCC (untuk audio)
# - Ini PERSIS yang dilakukan di audio/vibration ML!
# - Vibration analysis: frequency peak di bearing fault frequencies


# ===========================================================
# BAGIAN 6: Feature Importance Visualization
# ===========================================================
# Feature importance menunjukkan kontribusi setiap fitur terhadap prediksi.
# Metode untuk mendapatkan feature importance:
# 1. Tree-based: impurity decrease (Gini importance) atau permutation importance
# 2. Linear models: absolute coefficient values (setelah scaling!)
# 3. Permutation importance: acak nilai satu fitur, lihat penurunan performance

# Simulasi feature importance
feature_names = ['X1', 'X2', 'X3', 'X4', 'temp', 'volt', 'humidity']
importance_scores = np.array([0.35, 0.25, 0.20, 0.02, 0.10, 0.05, 0.03])
# X1, X2, X3 penting; X4 tidak penting (sesuai correlation analysis di atas)

# Sort untuk visualisasi yang bagus
sorted_idx = np.argsort(importance_scores)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(range(len(importance_scores)), importance_scores[sorted_idx], color='steelblue')
ax.set_yticks(range(len(importance_scores)))
ax.set_yticklabels([feature_names[i] for i in sorted_idx])
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importance')
ax.grid(axis='x')
plt.tight_layout()
plt.savefig('07_feature_importance.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 07_feature_importance.png")

# INSIGHT: Feature importance membantu:
# - Feature selection: buang fitur dengan importance ~0
# - Model interpretability: jelaskan kenapa model memprediksi X
# - Debugging: cek apakah fitur yang penting sesuai domain knowledge
# PERINGATAN: Correlation-based importance bisa misleading jika ada multicollinearity!


# ===========================================================
# LATIHAN 3: Visualization Dashboard
# ===========================================================
"""
TARGET Learning Objectives:
   - Membuat comprehensive visualization dashboard
   - Menggabungkan multiple plot types dalam satu figure
   - Mengembangkan kemampuan storytelling dengan data visual

PANDUAN LANGKAH-LANGKAH:

STEP 1: Pilih Dataset
---------------------
Gunakan dataset dari Exercise 2 (Titanic atau Power Consumption)
atau generate synthetic data yang relevan.


STEP 2: Buat Figure dengan 6 Subplot (2x3 layout)
-------------------------------------------------
Gunakan plt.subplots(2, 3, figsize=(18, 12)) untuk membuat grid 2x3.

   a) Subplot [0,0]: Distribusi setiap fitur numerik (histogram + KDE)
      - Gunakan sns.histplot(data=df, x='col', kde=True)
      - KDE (Kernel Density Estimate) = smooth PDF estimate
      - Perhatikan apakah distribusi normal, skewed, atau bimodal
      
   b) Subplot [0,1]: Correlation heatmap
      - Gunakan sns.heatmap(df.corr(), annot=True)
      - Fokus pada correlations dengan target variable
      - Cari fitur yang redundan (correlation > 0.9 antar fitur)
      
   c) Subplot [0,2]: Box plot per kategori target
      - Gunakan sns.boxplot(data=df, x='target', y='numeric_col')
      - Box plot menunjukkan median, quartiles, dan outliers
      - Lihat apakah ada perbedaan distribusi yang signifikan antar kelas
      
   d) Subplot [1,0]: Time series plot (kalau ada komponen waktu)
      - Gunakan plt.plot(df['timestamp'], df['value'])
      - Atau plt.plot(df.index, df['value']) jika index = time
      - Tambahkan rolling average untuk melihat trend
      
   e) Subplot [1,1]: Pie chart distribusi kelas
      - Gunakan plt.pie(df['target'].value_counts(), labels=...)
      - Atau bar plot jika terlalu banyak kelas
      - Periksa class imbalance
      
   f) Subplot [1,2]: Scatter plot 2 fitur paling berkorelasi, diwarnai target
      - Pilih 2 fitur dengan highest absolute correlation dengan target
      - Gunakan plt.scatter(x=df['f1'], y=df['f2'], c=df['target'])
      - Tambahkan colorbar untuk interpretasi


STEP 3: Styling dan Layout
--------------------------
   a) Beri judul utama pada figure: plt.suptitle('EDA Dashboard', fontsize=16, y=1.02)
   b) Beri judul pada setiap subplot yang deskriptif
   c) Gunakan plt.tight_layout() agar tidak tumpang tindih
   d) Gunakan warna palette yang konsisten di semua plot
   e) Simpan sebagai 'my_eda_dashboard.png' dengan dpi=150
   f) Tambahkan text annotation untuk insight penting


STEP 4: Interpretasi
--------------------
   Tulis 3-5 insight dari dashboard yang kamu buat.
   Contoh:
   - "Feature X memiliki bimodal distribution, menunjukkan 2 subpopulasi"
   - "Class 0 dan Class 1 terpisah dengan jelas di scatter plot f1 vs f2"
   - "Feature Y memiliki 3 outliers ekstrem yang perlu diinvestigasi"
   - "Korelasi tinggi antara f1 dan f2 -> multicollinearity, pertimbangkan drop salah satu"
   - "Class imbalance 90:10 -> perlu stratified sampling atau class weighting"


TIPS:
   - sns.histplot(kde=True) menambahkan density curve
   - plt.colorbar() menambahkan color scale untuk scatter plot
   - plt.xticks(rotation=45) memutar label jika terlalu panjang
   - Gunakan consistent color palette di semua plot (sns.set_palette)
   - sns.set_context('talk') untuk presentasi, 'paper' untuk publikasi

PERINGATAN COMMON MISTAKES:
   - Subplot yang terlalu kecil -> gunakan figsize yang besar (minimal 18x12)
   - Tumpang tindih label -> gunakan tight_layout() atau adjust hspace/wspace
   - Tidak memberi judul -> audience tidak tahu apa yang dilihat
   - Warna yang tidak konsisten -> membingungkan interpretasi
   - DPI terlalu rendah -> gambar blur saat dicetak/presentasi

TARGET EXPECTED OUTPUT:
   - File 'my_eda_dashboard.png' dengan 6 plot berkualitas
   - Minimal 3 insight tertulis dengan supporting evidence dari plot
   - Dashboard yang bisa dipresentasikan ke stakeholder non-teknis
"""


# ===========================================================
# CHALLENGE: Spectrogram Visualization
# ===========================================================
"""
TARGET Learning Objectives:
   - Memahami Short-Time Fourier Transform (STFT)
   - Memvisualisasikan sinyal non-stationary
   - Menyadari koneksi signal processing dengan CNN

PANDUAN LANGKAH-LANGKAH:

STEP 1: Generate Sinyal Non-Stationary
---------------------------------------
Buat sinyal chirp: frekuensi naik linear dari 10Hz ke 200Hz dalam 2 detik.

   Formula: y(t) = sin(2 * pi * f(t) * t)
   dimana f(t) = f_start + (f_end - f_start) * t / T
   
   fs = 1000 Hz
   T = 2 detik
   f_start = 10 Hz
   f_end = 200 Hz
   
   TIPS KENAPA chirp?
     - Chirp = sinyal dengan frekuensi yang berubah seiring waktu
     - Non-stationary = statistik berubah seiring waktu
     - Contoh nyata: sonar, radar, bird calls, power system transients
     - Chirp signal digunakan untuk testing frequency response dari system


STEP 2: Implementasi STFT Manual
---------------------------------
STFT = DFT yang dihitung pada window sliding (time-localized FFT).

   Parameters:
   - window_size = 256 samples (panjang window dalam sample)
   - hop_size = 128 samples (overlap = 50%)
   - window_function = Hamming atau Hanning (mengurangi spectral leakage)
   
   TIPS Apa yang harus dilakukan:
     a) Bagi sinyal menjadi overlapping windows
        for start in range(0, len(signal) - window_size, hop_size):
            window = signal[start:start + window_size]
     b) Apply window function ke setiap window
        window = window * np.hamming(window_size)
     c) Hitung FFT untuk setiap window
        spectrum = np.fft.fft(window)
     d) Stack hasil FFT menjadi matrix 2D (spectrogram)
        spectrogram[:, frame_idx] = np.abs(spectrum[:window_size//2])
     
   PERINGATAN:
     - Window function mengurangi spectral leakage (sidelobe suppression)
     - Overlap (hop_size < window_size) meningkatkan time resolution
     - Trade-off: window besar -> frequency resolution bagus, time resolution buruk
     - Trade-off: window kecil -> time resolution bagus, frequency resolution buruk
       (Heisenberg uncertainty principle dalam signal processing!)


STEP 3: Plot Spectrogram
------------------------
   Gunakan plt.imshow() atau plt.pcolormesh() untuk plot spectrogram.
   
   a) X-axis = time (frame index * hop_size / fs)
   b) Y-axis = frequency (0 sampai fs/2)
   c) Color = magnitude (dB scale lebih baik: 20*log10(mag + epsilon))
   
   TIPS KENAPA spectrogram?
     - Menunjukkan frekuensi yang dominan di setiap waktu
     - Untuk chirp: harusnya terlihat diagonal line (freq naik seiring waktu)
     - Untuk speech: terlihat formant frequencies yang berubah seiring waktu
     - Untuk vibration: terlihat fault frequencies yang muncul saat kerusakan


STEP 4: Bandingkan dengan Librosa/Matplotlib
-------------------------------------------
   a) Gunakan plt.specgram() dari matplotlib
   b) Atau librosa.feature.melspectrogram() (kalau librosa terinstall)
      - Mel-spectrogram menggunakan Mel scale (perceptual frequency scale)
   c) Bandingkan hasil manual vs library -> seharusnya mirip


STEP 5: Koneksi ke CNN
----------------------
   Tulis analisis:
   - Spectrogram = representasi 2D dari sinyal 1D (time-frequency image)
   - CNN bisa "melihat" pattern frekuensi-waktu seperti image
   - Contoh aplikasi: speech recognition, music genre classification
   - Vibration analysis untuk predictive maintenance (bearing fault detection)
   - Time-frequency representation memungkinkan CNN menangkap
     both temporal dynamics dan spectral content secara bersamaan


TIPS:
   - np.hamming(window_size) menghasilkan Hamming window
   - range(0, len(signal) - window_size, hop_size) untuk sliding window
   - 20 * np.log10(np.abs(STFT) + 1e-10) untuk dB scale (hindari log(0))
   - origin='lower' di imshow agar frekuensi rendah di bawah (konvensi)
   - aspect='auto' agar tidak terlalu terdistorsi

PERINGATAN COMMON MISTAKES:
   - Tidak apply window function -> spectral leakage yang parah (sidelobe tinggi)
   - Salah axis orientation di imshow (default origin='upper')
   - Tidak normalisasi magnitude -> dynamic range tidak terlihat
   - Lupa half spectrum (real signal -> simetri)
   - Hop_size terlalu besar -> time resolution jelek, spectrogram terlihat "patah-patah"

TARGET EXPECTED OUTPUT:
   - Plot spectrogram dengan diagonal line (chirp signature)
   - Perbandingan manual vs library (plt.specgram)
   - Analisis koneksi ke CNN (2-3 paragraf)
   - Pemahaman trade-off time-frequency resolution
   
Ini juga preview dari bagaimana CNN akan "melihat" data audio/sinyal!
Spectrogram = representasi 2D dari sinyal -> bisa diproses seperti image.
Di industrial IoT, spectrogram vibration digunakan untuk bearing fault diagnosis
dengan CNN (achieving >95% accuracy untuk inner race, outer race, dan ball faults).
"""

print("\n" + "="*50)
print("OK Fase 1 selesai!")
print("Sebelum lanjut, pastikan:")
print("  - Semua exercise selesai")
print("  - Challenge minimal 1 sudah dicoba")
print("  - Bisa jelaskan konsep tanpa melihat kode")
print("\nLanjut ke: 02-ml-dari-nol/01_linear_regression_scratch.py")
print("="*50)
