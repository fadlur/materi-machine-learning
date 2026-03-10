"""
=============================================================
FASE 1 — MODUL 3: VISUALISASI DATA
=============================================================
"If you can't see it, you can't understand it."

Visualisasi bukan cuma untuk presentasi — ini tool DEBUGGING
paling powerful di ML. Kamu HARUS bisa visualisasi:
1. Distribusi data (histogram, KDE)
2. Relasi antar fitur (scatter, correlation)
3. Model performance (learning curves, confusion matrix)

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
# 📖 BAGIAN 1: Distribusi Data
# ===========================================================

np.random.seed(42)
data = pd.DataFrame({
    'normal': np.random.normal(0, 1, 1000),
    'skewed': np.random.exponential(2, 1000),
    'bimodal': np.concatenate([
        np.random.normal(-2, 0.5, 500),
        np.random.normal(2, 0.5, 500)
    ])
})

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, col in enumerate(data.columns):
    axes[i].hist(data[col], bins=50, alpha=0.7, edgecolor='black')
    axes[i].set_title(f'Distribusi: {col}')
    axes[i].axvline(data[col].mean(), color='red', linestyle='--', label='mean')
    axes[i].axvline(data[col].median(), color='green', linestyle='--', label='median')
    axes[i].legend()

plt.tight_layout()
plt.savefig('01_distribusi.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 01_distribusi.png")

# INSIGHT: Kalau mean ≠ median → distribusi skewed
# Ini penting karena banyak model ML mengasumsikan distribusi normal


# ===========================================================
# 📖 BAGIAN 2: Relasi Antar Fitur
# ===========================================================

# Buat dataset dengan relasi yang jelas
n = 200
X1 = np.random.randn(n)
X2 = 0.5 * X1 + np.random.randn(n) * 0.5  # korrelasi positif
X3 = -0.8 * X1 + np.random.randn(n) * 0.3  # korrelasi negatif
X4 = np.random.randn(n)  # tidak berkorelasi
y = (X1 + X2 - X3 + np.random.randn(n) * 0.5 > 0).astype(int)

df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'target': y})

# Correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title('Correlation Matrix')

# Pair plot alternative: scatter matrix
pd.plotting.scatter_matrix(df[['X1', 'X2', 'X3', 'X4']], 
                           c=y, cmap='RdYlBu', alpha=0.5,
                           figsize=(10, 10))
plt.savefig('02_scatter_matrix.png', dpi=100, bbox_inches='tight')
plt.close()

# Save correlation heatmap separately
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Correlation Matrix')
plt.tight_layout()
plt.savefig('02_correlation.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 02_correlation.png, 02_scatter_matrix.png")


# ===========================================================
# 📖 BAGIAN 3: Visualisasi untuk ML
# ===========================================================

# Decision boundary visualization
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model_predict_fn, title="Decision Boundary"):
    """Visualisasi decision boundary — akan sering dipakai nanti"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model_predict_fn(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black', s=50)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    return plt

# Contoh: linear decision boundary
X_2d = df[['X1', 'X2']].values

def simple_linear_classifier(X):
    return (X[:, 0] + X[:, 1] > 0).astype(int)

fig = plot_decision_boundary(X_2d, y, simple_linear_classifier,
                             "Contoh: Linear Decision Boundary")
plt.savefig('03_decision_boundary.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 03_decision_boundary.png")


# ===========================================================
# 📖 BAGIAN 4: Visualisasi Model Performance
# ===========================================================

# Confusion matrix plot
def plot_confusion_matrix(y_true, y_pred, classes=['Class 0', 'Class 1']):
    """Plot confusion matrix — WAJIB tahu untuk evaluasi model"""
    # Hitung confusion matrix manual
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    return fig

# Contoh
y_true = np.array([0]*50 + [1]*50)
y_pred = np.array([0]*45 + [1]*5 + [0]*10 + [1]*40)  # some misclassifications
fig = plot_confusion_matrix(y_true, y_pred)
plt.savefig('04_confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 04_confusion_matrix.png")


# Learning curve plot
def plot_learning_curve(train_sizes, train_scores, val_scores, title="Learning Curve"):
    """Plot learning curve — untuk diagnosa overfitting/underfitting"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores, 'o-', label='Training Score')
    ax.plot(train_sizes, val_scores, 'o-', label='Validation Score')
    ax.fill_between(train_sizes,
                    train_scores - 0.02, train_scores + 0.02, alpha=0.1)
    ax.fill_between(train_sizes,
                    val_scores - 0.05, val_scores + 0.05, alpha=0.1)
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0.5, 1.05)
    return fig

# Contoh learning curve (simulated)
sizes = np.array([50, 100, 200, 400, 800])
train_acc = np.array([0.99, 0.97, 0.95, 0.93, 0.92])  # menurun (less overfit)
val_acc = np.array([0.70, 0.78, 0.83, 0.87, 0.89])     # meningkat

fig = plot_learning_curve(sizes, train_acc, val_acc)
plt.savefig('05_learning_curve.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 05_learning_curve.png")


# ===========================================================
# 📖 BAGIAN 5: Signal Processing Visualization (EE-Relevant!)
# ===========================================================

# Sinyal + FFT — familiar territory!
fs = 1000  # sampling frequency 1kHz
t = np.arange(0, 1, 1/fs)

# Sinyal campuran
signal = (np.sin(2 * np.pi * 50 * t) +           # 50 Hz
          0.5 * np.sin(2 * np.pi * 120 * t) +      # 120 Hz
          0.3 * np.random.randn(len(t)))            # noise

# FFT
fft_vals = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(t), 1/fs)

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Time domain
axes[0].plot(t[:200], signal[:200])  # 200ms
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Signal (Time Domain)')

# Frequency domain
positive_freqs = freqs[:len(freqs)//2]
magnitude = np.abs(fft_vals[:len(fft_vals)//2]) * 2/len(t)
axes[1].plot(positive_freqs, magnitude)
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude')
axes[1].set_title('Signal (Frequency Domain)')
axes[1].set_xlim(0, 200)

plt.tight_layout()
plt.savefig('06_signal_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 06_signal_analysis.png")

# KONEKSI KE ML:
# - Time domain features → statistik (mean, std, peak)
# - Frequency domain features → dominant frequency, spectral energy
# - Ini PERSIS yang dilakukan di audio/vibration ML!


# ===========================================================
# 🏋️ EXERCISE 3: Visualization Dashboard
# ===========================================================
"""
Buat visualization dashboard untuk dataset yang kamu pakai di Exercise 2.
Dashboard harus berisi (minimal 6 plot dalam 1 figure):

1. Distribusi setiap fitur numerik (histogram + KDE)
2. Correlation heatmap
3. Box plot per kategori target
4. Time series plot (kalau ada komponen waktu)
5. Pie chart distribusi kelas
6. Scatter plot 2 fitur paling berkorelasi, diwarnai berdasarkan target

Simpan sebagai 'my_eda_dashboard.png'
"""


# ===========================================================
# 🔥 CHALLENGE: Spectrogram Visualization
# ===========================================================
"""
Buat fungsi yang:
1. Generate sinyal non-stationary (frekuensi berubah seiring waktu)
   → chirp signal: frekuensi naik dari 10Hz ke 200Hz
2. Hitung dan plot spectrogram (Short-Time Fourier Transform / STFT)
3. Ini SANGAT relevan untuk:
   - Audio classification (speech, music genre)
   - Vibration analysis (predictive maintenance)
   - EEG/ECG signal analysis

Ini juga preview dari bagaimana CNN akan "melihat" data audio/sinyal!
Spectrogram = representasi 2D dari sinyal → bisa diproses seperti image.

Hint: matplotlib punya plt.specgram() tapi coba implementasi STFT manual dulu.
"""

print("\n" + "="*50)
print("✅ Fase 1 selesai!")
print("Sebelum lanjut, pastikan:")
print("  ✓ Semua exercise selesai")
print("  ✓ Challenge minimal 1 sudah dicoba")
print("  ✓ Bisa jelaskan konsep tanpa melihat kode")
print("\nLanjut ke: 02-ml-dari-nol/01_linear_regression_scratch.py")
print("="*50)
