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

Koneksi Teknik Elektro:
- Histogram = probability density function (PDF) estimator
- Correlation = coherence antara sinyal
- Learning curve = step response dari adaptive system

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
# Distribusi data = fundamental untuk memahami karakteristik dataset.
# Kenapa penting?
# - Banyak algoritma ML mengasumsikan normal distribution
# - Skewed distribution → perlu transformasi (log, sqrt)
# - Outlier terlihat jelas di histogram

np.random.seed(42)
data = pd.DataFrame({
    'normal': np.random.normal(0, 1, 1000),
    # Distribusi normal: bell curve, mean=median=mode
    'skewed': np.random.exponential(2, 1000),
    # Distribusi eksponensial: skewed ke kanan (tail panjang di kanan)
    'bimodal': np.concatenate([
        np.random.normal(-2, 0.5, 500),
        np.random.normal(2, 0.5, 500)
    ])
    # Bimodal: dua peak → menunjukkan ada 2 grup/subpopulasi
})

# Membuat 3 subplot horizontally
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# figsize=(width, height) dalam inches

for i, col in enumerate(data.columns):
    # Histogram: membagi data ke bins dan menghitung frekuensi
    axes[i].hist(data[col], bins=50, alpha=0.7, edgecolor='black')
    # bins=50: jumlah interval
    # alpha=0.7: transparansi
    # edgecolor: warna border bar
    
    axes[i].set_title(f'Distribusi: {col}')
    
    # Vertical line untuk mean (garis putus-putus merah)
    axes[i].axvline(data[col].mean(), color='red', linestyle='--', label='mean')
    # axvline = vertical line di posisi tertentu
    
    # Vertical line untuk median (garis putus-putus hijau)
    axes[i].axvline(data[col].median(), color='green', linestyle='--', label='median')
    
    axes[i].legend()

plt.tight_layout()
# tight_layout() menyesuaikan spacing antar subplot agar tidak tumpang tindih
plt.savefig('01_distribusi.png', dpi=100, bbox_inches='tight')
# dpi=100: resolution 100 dots per inch
# bbox_inches='tight': memotong whitespace berlebih
plt.close()
# close() untuk membebaskan memory
print("📊 Saved: 01_distribusi.png")

# INSIGHT: Kalau mean ≠ median → distribusi skewed
# Ini penting karena banyak model ML mengasumsikan distribusi normal
# Contoh: Linear Regression mengasumsikan residual normally distributed.


# ===========================================================
# 📖 BAGIAN 2: Relasi Antar Fitur
# ===========================================================
# Correlation analysis = fundamental untuk feature selection.
# Fitur yang sangat berkorelasi satu sama lain bisa redundan.

# Buat dataset dengan relasi yang jelas
n = 200
X1 = np.random.randn(n)
# X2 berkorelasi positif dengan X1 (koefisien 0.5)
# + noise (0.5 * np.random.randn) untuk membuat tidak perfect correlation
X2 = 0.5 * X1 + np.random.randn(n) * 0.5  # korrelasi positif
# X3 berkorelasi negatif dengan X1 (koefisien -0.8)
X3 = -0.8 * X1 + np.random.randn(n) * 0.3  # korrelasi negatif
# X4 independent (tidak berkorelasi dengan X1)
X4 = np.random.randn(n)  # tidak berkorelasi
# Target y = kombinasi linear dari X1, X2, X3 + noise
y = (X1 + X2 - X3 + np.random.randn(n) * 0.5 > 0).astype(int)

df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'target': y})

# Correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Heatmap ---
# Correlation matrix = Pearson correlation coefficient antar semua pasangan kolom.
# Range: -1 (perfect negative) sampai +1 (perfect positive).
# 0 = no linear correlation.
corr = df.corr()
# annot=True: menampilkan nilai correlation di setiap cell
# cmap='RdBu_r': Red (positive) - Blue (negative), reversed
# center=0: membuat 0 sebagai warna netral (putih)
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title('Correlation Matrix')

# --- Scatter matrix ---
# Scatter matrix = pairwise scatter plot antar semua fitur.
# Sangat berguna untuk melihat relasi non-linear dan clustering.
# c=y: mewarnai point berdasarkan target class.
# cmap='RdYlBu': Red-Yellow-Blue color map.
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
# Decision boundary = batas pemisah antar kelas.
# Memvisualisasikan decision boundary membantu kita memahami
# apakah model linear sudah cukup atau perlu non-linear.

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
    # h = resolusi grid (semakin kecil = semakin halus)
    h = 0.02
    # x_min, x_max = batas horizontal plot (dengan margin 1 unit)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = batas vertikal plot (dengan margin 1 unit)
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # np.meshgrid membuat grid coordinates
    # np.arange(x_min, x_max, h) = array dari x_min ke x_max dengan step h
    # xx dan yy adalah matrix 2D yang merepresentasikan grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # np.c_ menggabungkan xx.ravel() dan yy.ravel() menjadi (n_points, 2)
    # ravel() meng-flatten matrix 2D menjadi 1D
    Z = model_predict_fn(np.c_[xx.ravel(), yy.ravel()])
    # Reshape hasil prediksi kembali ke shape grid
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    # contourf = filled contour plot
    # alpha=0.3: transparan biar data points tetap terlihat
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    # scatter plot data asli
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
    - Ini adalah garis lurus dengan slope -1
    """
    return (X[:, 0] + X[:, 1] > 0).astype(int)


fig = plot_decision_boundary(X_2d, y, simple_linear_classifier,
                             "Contoh: Linear Decision Boundary")
plt.savefig('03_decision_boundary.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 03_decision_boundary.png")


# ===========================================================
# 📖 BAGIAN 4: Visualisasi Model Performance
# ===========================================================
# Confusion matrix dan learning curve adalah visualisasi WAJIB
# untuk evaluasi model.

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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    return fig


# Contoh confusion matrix
y_true = np.array([0]*50 + [1]*50)
y_pred = np.array([0]*45 + [1]*5 + [0]*10 + [1]*40)  # some misclassifications
fig = plot_confusion_matrix(y_true, y_pred)
plt.savefig('04_confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 04_confusion_matrix.png")


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
    - High training score + low validation score = overfitting
    - Low training score + low validation score = underfitting
    - Training dan validation score converge = good fit
    - Koneksi ke Teknik Elektro: mirip dengan convergence plot
      di adaptive filtering (LMS algorithm)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    # Plot training score
    ax.plot(train_sizes, train_scores, 'o-', label='Training Score')
    # Plot validation score
    ax.plot(train_sizes, val_scores, 'o-', label='Validation Score')
    # Fill between untuk menunjukkan variance (shaded area)
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
# Gap antara training dan validation = overfitting indicator

fig = plot_learning_curve(sizes, train_acc, val_acc)
plt.savefig('05_learning_curve.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 05_learning_curve.png")


# ===========================================================
# 📖 BAGIAN 5: Signal Processing Visualization (EE-Relevant!)
# ===========================================================
# Sinyal + FFT — familiar territory!
# FFT (Fast Fourier Transform) mengubah sinyal dari time domain
# ke frequency domain.

fs = 1000  # sampling frequency 1kHz
t = np.arange(0, 1, 1/fs)
# np.arange(0, 1, 1/1000) = 1000 titik dari 0 sampai hampir 1 detik
# Step = 1/fs = 1ms (sampling period)

# Sinyal campuran: 50Hz + 120Hz + noise
# Ini mirik dengan sinyal power system (50Hz fundamental + harmonics)
signal = (np.sin(2 * np.pi * 50 * t) +           # 50 Hz fundamental
          0.5 * np.sin(2 * np.pi * 120 * t) +      # 120 Hz harmonic
          0.3 * np.random.randn(len(t)))            # noise

# FFT (Fast Fourier Transform)
# np.fft.fft menghitung Discrete Fourier Transform secara efisien O(N log N)
fft_vals = np.fft.fft(signal)
# np.fft.fftfreq menghasilkan frequency axis yang sesuai
freqs = np.fft.fftfreq(len(t), 1/fs)
# freqs[i] = frequency yang sesuai dengan fft_vals[i]

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# --- Time domain ---
axes[0].plot(t[:200], signal[:200])  # 200ms
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Signal (Time Domain)')
# Time domain menunjukkan amplitude terhadap waktu
# Tapi susah melihat komponen frekuensi

# --- Frequency domain ---
# Ambil setengah spektrum (karena simetri untuk real signal)
positive_freqs = freqs[:len(freqs)//2]
# Magnitude spectrum = 2/N * |FFT|
# Faktor 2 karena kita hanya ambil setengah spektrum
magnitude = np.abs(fft_vals[:len(fft_vals)//2]) * 2/len(t)
axes[1].plot(positive_freqs, magnitude)
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude')
axes[1].set_title('Signal (Frequency Domain)')
axes[1].set_xlim(0, 200)
# Frequency domain menunjukkan komponen frekuensi yang ada di sinyal
# Peak di 50 Hz dan 120 Hz terlihat jelas!

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
🎯 Learning Objectives:
   - Membuat comprehensive visualization dashboard
   - Menggabungkan multiple plot types dalam satu figure
   - Mengembangkan kemampuan storytelling dengan data visual

📋 LANGKAH-LANGKAH:

STEP 1: Pilih Dataset
─────────────────────
Gunakan dataset dari Exercise 2 (Titanic atau Power Consumption)
atau generate synthetic data yang relevan.


STEP 2: Buat Figure dengan 6 Subplot (2x3 layout)
─────────────────────────────────────────────────
Gunakan plt.subplots(2, 3, figsize=(18, 12)) untuk membuat grid 2x3.

   a) Subplot [0,0]: Distribusi setiap fitur numerik (histogram + KDE)
      - Gunakan sns.histplot(data=df, x='col', kde=True)
      - KDE (Kernel Density Estimate) = smooth PDF estimate
      
   b) Subplot [0,1]: Correlation heatmap
      - Gunakan sns.heatmap(df.corr(), annot=True)
      - Fokus pada correlations dengan target variable
      
   c) Subplot [0,2]: Box plot per kategori target
      - Gunakan sns.boxplot(data=df, x='target', y='numeric_col')
      - Box plot menunjukkan median, quartiles, dan outliers
      
   d) Subplot [1,0]: Time series plot (kalau ada komponen waktu)
      - Gunakan plt.plot(df['timestamp'], df['value'])
      - Atau plt.plot(df.index, df['value']) jika index = time
      
   e) Subplot [1,1]: Pie chart distribusi kelas
      - Gunakan plt.pie(df['target'].value_counts(), labels=...)
      - Atau bar plot jika terlalu banyak kelas
      
   f) Subplot [1,2]: Scatter plot 2 fitur paling berkorelasi, diwarnai target
      - Pilih 2 fitur dengan highest absolute correlation dengan target
      - Gunakan plt.scatter(x=df['f1'], y=df['f2'], c=df['target'])


STEP 3: Styling dan Layout
──────────────────────────
   a) Beri judul utama pada figure: plt.suptitle('EDA Dashboard', fontsize=16)
   b) Beri judul pada setiap subplot
   c) Gunakan plt.tight_layout() agar tidak tumpang tindih
   d) Simpan sebagai 'my_eda_dashboard.png' dengan dpi=150


STEP 4: Interpretasi
────────────────────
   Tulis 3-5 insight dari dashboard yang kamu buat.
   Contoh:
   - "Feature X memiliki bimodal distribution, menunjukkan 2 subpopulasi"
   - "Class 0 dan Class 1 terpisah dengan jelas di scatter plot f1 vs f2"
   - "Feature Y memiliki 3 outliers ekstrem yang perlu diinvestigasi"


💡 HINTS:
   - sns.histplot(kde=True) menambahkan density curve
   - plt.colorbar() menambahkan color scale untuk scatter plot
   - plt.xticks(rotation=45) memutar label jika terlalu panjang
   - Gunakan consistent color palette di semua plot

⚠️ COMMON MISTAKES:
   - Subplot yang terlalu kecil → gunakan figsize yang besar
   - Tumpang tindih label → gunakan tight_layout()
   - Tidak memberi judul → audience tidak tahu apa yang dilihat
   - Warna yang tidak konsisten → membingungkan interpretasi

🎯 EXPECTED OUTPUT:
   - File 'my_eda_dashboard.png' dengan 6 plot
   - Minimal 3 insight tertulis
   - Dashboard yang bisa dipresentasikan ke stakeholder
"""


# ===========================================================
# 🔥 CHALLENGE: Spectrogram Visualization
# ===========================================================
"""
🎯 Learning Objectives:
   - Memahami Short-Time Fourier Transform (STFT)
   - Memvisualisasikan sinyal non-stationary
   - Menyadari koneksi signal processing dengan CNN

📋 LANGKAH-LANGKAH:

STEP 1: Generate Sinyal Non-Stationary
───────────────────────────────────────
Buat sinyal chirp: frekuensi naik linear dari 10Hz ke 200Hz dalam 2 detik.

   Formula: y(t) = sin(2π * f(t) * t)
   dimana f(t) = f_start + (f_end - f_start) * t / T
   
   fs = 1000 Hz
   T = 2 detik
   f_start = 10 Hz
   f_end = 200 Hz
   
   💡 KENAPA chirp?
     - Chirp = sinyal dengan frekuensi yang berubah seiring waktu
     - Non-stationary = statistik berubah seiring waktu
     - Contoh nyata: sonar, radar, bird calls


STEP 2: Implementasi STFT Manual
─────────────────────────────────
STFT = DFT yang dihitung pada window sliding.

   Parameters:
   - window_size = 256 samples
   - hop_size = 128 samples (overlap = 50%)
   - window_function = Hamming atau Hanning
   
   💡 Apa yang harus dilakukan:
     a) Bagi sinyal menjadi overlapping windows
     b) Apply window function ke setiap window
     c) Hitung FFT untuk setiap window
     d) Stack hasil FFT menjadi matrix 2D (spectrogram)
     
   ⚠️ Hati-hati:
     - Window function mengurangi spectral leakage
     - Overlap (hop_size < window_size) meningkatkan time resolution


STEP 3: Plot Spectrogram
────────────────────────
   Gunakan plt.imshow() atau plt.pcolormesh() untuk plot spectrogram.
   
   a) X-axis = time
   b) Y-axis = frequency
   c) Color = magnitude (dB scale lebih baik: 20*log10(mag))
   
   💡 KENAPA spectrogram?
     - Menunjukkan frekuensi yang dominan di setiap waktu
     - Untuk chirp: harusnya terlihat diagonal line (freq naik seiring waktu)


STEP 4: Bandingkan dengan Librosa/Matplotlib
───────────────────────────────────────────
   a) Gunakan plt.specgram() dari matplotlib
   b) Atau librosa.feature.melspectrogram() (kalau librosa terinstall)
   c) Bandingkan hasil manual vs library


STEP 5: Koneksi ke CNN
──────────────────────
   Tulis analisis:
   - Spectrogram = representasi 2D dari sinyal 1D
   - CNN bisa "melihat" pattern frekuensi-waktu seperti image
   - Contoh aplikasi: speech recognition, music genre classification
   - Vibration analysis untuk predictive maintenance


💡 HINTS:
   - np.hamming(window_size) menghasilkan Hamming window
   - range(0, len(signal) - window_size, hop_size) untuk sliding window
   - 20 * np.log10(np.abs(STFT) + 1e-10) untuk dB scale
   - origin='lower' di imshow agar frekuensi rendah di bawah

⚠️ COMMON MISTAKES:
   - Tidak apply window function → spectral leakage yang parah
   - Salah axis orientation di imshow
   - Tidak normalisasi magnitude → dynamic range tidak terlihat

🎯 EXPECTED OUTPUT:
   - Plot spectrogram dengan diagonal line (chirp signature)
   - Perbandingan manual vs library
   - Analisis koneksi ke CNN (2-3 paragraf)
   
Ini juga preview dari bagaimana CNN akan "melihat" data audio/sinyal!
Spectrogram = representasi 2D dari sinyal → bisa diproses seperti image.
"""

print("\n" + "="*50)
print("✅ Fase 1 selesai!")
print("Sebelum lanjut, pastikan:")
print("  ✓ Semua exercise selesai")
print("  ✓ Challenge minimal 1 sudah dicoba")
print("  ✓ Bisa jelaskan konsep tanpa melihat kode")
print("\nLanjut ke: 02-ml-dari-nol/01_linear_regression_scratch.py")
print("="*50)
