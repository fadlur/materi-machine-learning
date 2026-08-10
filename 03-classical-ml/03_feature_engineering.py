"""
=============================================================
FASE 3 - MODUL 3: FEATURE ENGINEERING
=============================================================
"Data scientists spend 80% of their time on data preparation."

Feature engineering = mengubah raw data menjadi representasi
yang lebih mudah dipelajari oleh model ML.

Ini adalah SKILL yang paling membedakan:
- Pemula: pakai fitur apa adanya
- Expert: create fitur yang "membuka" pattern tersembunyi

Background EE memberikan keuntungan BESAR di sini:
- Signal processing -> feature extraction dari time series
- Domain knowledge -> meaningful engineered features
- Physics-based features -> fitur yang punya interpretasi fisik

Durasi target: 3-4 jam
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PolynomialFeatures, LabelEncoder, OneHotEncoder
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

np.random.seed(42)


# ===========================================================
# BAGIAN 1: Numerical Transformations
# ===========================================================
# Scaling adalah langkah preprocessing yang PENTING.
# Banyak algoritma (SVM, KNN, Neural Network, PCA) sensitive ke scale.
#
# Matematika Scaling:
# 1. StandardScaler (Z-score normalization):
#    z = (x - mu) / sigma
#    Hasil: mean ~ 0, std ~ 1.
#    Cocok untuk: data dengan distribusi Gaussian, outlier tidak terlalu ekstrem.
#    PERINGATAN: tidak robust terhadap outlier (outlier bisa memindahkan mean).
#
# 2. MinMaxScaler:
#    x_scaled = (x - min) / (max - min)
#    Hasil: range [0, 1] (atau [-1, 1] jika ada negatif).
#    Cocok untuk: data dengan batas yang jelas (image pixel 0-255).
#    PERINGATAN: sangat sensitif terhadap outlier (outlier bisa "compress"
#    data normal ke range kecil).
#
# 3. RobustScaler:
#    x_scaled = (x - median) / IQR
#    IQR = Q3 - Q1 (interquartile range, 75th - 25th percentile).
#    Cocok untuk: data dengan outlier yang signifikan.
#    Lebih robust karena median dan IQR tidak dipengaruhi outlier ekstrem.
#
# Kapan menggunakan yang mana?
# - Data Gaussian tanpa outlier: StandardScaler.
# - Data dengan outlier: RobustScaler.
# - Data dengan batas known: MinMaxScaler.
# - Algoritma berbasis jarak (KNN, SVM): WAJIB scaling.
# - Tree-based (Random Forest, XGBoost): tidak perlu scaling.
#
# Log Transform:
# - Gunakan untuk distribusi right-skewed (ekor panjang ke kanan).
# - log1p(x) = log(1+x) untuk handle zeros.
# - Mengubah multiplicative relationships menjadi additive.
# - Contoh: income, population, sensor readings dengan decay.
#
# Edge case:
# - Jangan standardize sebelum split data! Fit scaler hanya pada training data.
# - Log transform pada nilai negatif: tambahkan offset atau gunakan signed log.
# - Jika data sudah normalized (misal 0-1), scaling tambahan tidak perlu.

# Generate sample data
n = 1000
data = pd.DataFrame({
    'voltage': np.random.normal(220, 10, n),
    'current': np.abs(np.random.normal(5, 2, n)),
    'temperature': np.random.exponential(20, n) + 25,
    'rpm': np.random.uniform(500, 3000, n),
    'vibration': np.abs(np.random.normal(0, 1, n)) ** 2,  # skewed
    'hours_running': np.random.exponential(500, n),
})

# Scaling comparison
print("=== Scaling Comparison ===")
scalers = {
    'StandardScaler': StandardScaler(),    # z-score: (x - mean) / std
    'MinMaxScaler': MinMaxScaler(),         # scale ke [0, 1]
    'RobustScaler': RobustScaler(),         # pakai median & IQR (robust terhadap outlier)
}

fig, axes = plt.subplots(1, 4, figsize=(16, 3))
axes[0].hist(data['vibration'], bins=50, edgecolor='black')
axes[0].set_title('Original (skewed)')

for i, (name, scaler) in enumerate(scalers.items()):
    scaled = scaler.fit_transform(data[['vibration']])
    axes[i+1].hist(scaled, bins=50, edgecolor='black')
    axes[i+1].set_title(name)

plt.tight_layout()
plt.savefig('01_scaling.png', dpi=100, bbox_inches='tight')
plt.close()
print("OK Saved: 01_scaling.png")

# Log transform untuk distribusi skewed
# Log transform mengurangi skewness dengan "mengepaskan" ekor panjang.
# Rumus: log1p(x) = ln(1+x). Mengapa +1? Untuk handle x=0 (ln(0) undefined).
# Transformasi ini membuat distribusi lebih mendekati Gaussian,
# yang membantu algoritma berbasis jarak dan linear models.
data['vibration_log'] = np.log1p(data['vibration'])  # log(1+x) untuk handle zeros
data['hours_log'] = np.log1p(data['hours_running'])

print("\nVibration - Skewness before log: "
      f"{data['vibration'].skew():.2f}")
print("Vibration - Skewness after log:  "
      f"{data['vibration_log'].skew():.2f}")


# ===========================================================
# BAGIAN 2: Domain-Specific Features (EE-Based!)
# ===========================================================
# Fitur yang dibuat berdasarkan domain knowledge SELALU lebih
# powerful dari fitur statistik generik.
#
# Mengapa domain features lebih baik?
# - Interpretability: fitur berbasis fisika bisa dijelaskan ke engineer lain.
# - Discriminative power: hubungan fisika seringkali memang ada di data.
# - Robustness: fitur fisika biasanya lebih stabil terhadap noise.
# - Transferability: knowledge dari satu sistem bisa dipakai ke sistem serupa.
#
# Contoh domain features untuk motor listrik:
# - Power (P = V * I): indikator beban. Perubahan power bisa menandakan fault.
# - Power factor (cos(theta)): indikator efisiensi. PF rendah = reactive power tinggi.
# - Thermal risk (I * T): kombinasi arus tinggi dan suhu tinggi = bahaya.
# - Vibration per RPM: vibrasi tinggi di RPM rendah lebih serius (resonance).
#
# TIPS: Domain knowledge adalah competitive advantage untuk EE engineer.
# Jangan hanya mengandalkan fitur statistik generik!

# Power (dari Teknik Elektro: P = V * I)
data['power'] = data['voltage'] * data['current']

# Apparent power, Power factor (jika punya phase angle)
# Matematika 3-phase power:
# - Apparent power S = V_rms * I_rms (VA)
# - Real power P = V_rms * I_rms * cos(theta) (Watt)
# - Reactive power Q = V_rms * I_rms * sin(theta) (VAR)
# - Power factor = P / S = cos(theta)
# PF mendekati 1 berarti efisien. PF rendah berarti banyak reactive power.
data['phase_angle'] = np.random.uniform(0, np.pi/4, n)
data['real_power'] = data['voltage'] * data['current'] * np.cos(data['phase_angle'])
data['reactive_power'] = data['voltage'] * data['current'] * np.sin(data['phase_angle'])
data['power_factor'] = np.cos(data['phase_angle'])

# Efficiency indicator
# Power per RPM mengindikasikan seberapa efisien motor bekerja.
# Motor yang sehat memiliki rasio power/RPM yang konsisten.
data['power_per_rpm'] = data['power'] / (data['rpm'] + 1)

# Thermal indicator (arus tinggi + suhu tinggi = bahaya!)
# Hubungan termal: P_loss = I**2 * R. Panas proporsional dengan kuadrat arus.
# Ini adalah indikator early warning untuk overheating.
data['thermal_risk'] = data['current'] * data['temperature'] / 100

# Vibration normalized by RPM (vibrasi tinggi di RPM rendah = lebih serius)
# Resonance sering terjadi pada frekuensi tertentu. Normalisasi RPM
# membantu mendeteksi abnormal vibration terlepas dari kecepatan.
data['vibration_per_rpm'] = data['vibration'] / (data['rpm'] + 1) * 1000

# Operational age indicator
# Binning continuous ke categorical bisa membantu model menangkap
# non-linear relationships. Contoh: degradasi tidak linear dengan waktu.
# pd.cut menggunakan interval sama lebar. pd.qcut menggunakan quantile
# (sama jumlah sample per bin).
data['age_category'] = pd.cut(data['hours_running'],
                               bins=[0, 200, 1000, 5000, np.inf],
                               labels=['new', 'running-in', 'normal', 'aging'])

print("\n=== Domain Features Created ===")
print(data[['power', 'power_factor', 'thermal_risk',
            'vibration_per_rpm']].describe().round(2))


# ===========================================================
# BAGIAN 3: Time Series Features
# ===========================================================
# Untuk data yang punya komponen waktu (sensor, monitoring)
#
# Mengapa time series features penting?
# - Single sample tidak cukup: fault biasanya terlihat dari pattern di window.
# - Temporal dynamics: trend, seasonality, transients.
# - Statistical aggregation: reduce noise dengan summary statistics.
#
# Fitur yang umum diekstrak:
# - Basic stats: mean, std, min, max, range, skewness, kurtosis
#   Kurtosis > 3 menandakan impulsive noise (penting untuk bearing fault detection).
# - Percentiles: q25, q50, q75. Robust terhadap outliers.
# - Rolling stats: moving average, moving std. Menangkap local trends.
# - Rate of change: diff mean, max diff. Deteksi abrupt changes.
# - Zero crossings: jumlah perubahan tanda. Related ke frekuensi dasar.
#
# Koneksi Teknik Elektro:
# - Zero crossings = fundamental frequency estimation
# - RMS = effective value dari sinyal AC
# - Crest factor = peak/RMS, indicator of impulsive noise
# - Kurtosis = sensitif terhadap transient/spike

def extract_time_features(series, window_sizes=[5, 10, 20]):
    """
    Extract statistical features dari time series.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data.
    window_sizes : list, default [5, 10, 20]
        Window sizes untuk rolling features.
        
    Returns:
    --------
    dict
        Dictionary feature names dan values.
        
    Notes:
    ------
    Features extracted:
    - Basic stats: mean, std, min, max, range, skewness, kurtosis
    - Percentiles: q25, q50, q75
    - Rolling stats: moving average, moving std
    - Rate of change: diff mean, max diff
    - Zero crossings: jumlah perubahan tanda
    
    Koneksi Teknik Elektro:
    - Zero crossings = fundamental frequency estimation
    - RMS = effective value dari sinyal AC
    - Crest factor = peak/RMS, indicator of impulsive noise
    
    TIPS:
    - Window size harus sesuai dengan timescale fenomena yang dipantau.
    - Untuk fault detection, gunakan window yang mencakup beberapa siklus.
    - Rolling features di ujung window bisa NaN -> handle dengan care.
    """
    features = {}
    
    # Basic statistics
    features['mean'] = series.mean()
    features['std'] = series.std()
    features['min'] = series.min()
    features['max'] = series.max()
    features['range'] = series.max() - series.min()
    features['skewness'] = series.skew()
    features['kurtosis'] = series.kurtosis()
    
    # Percentiles
    for q in [25, 50, 75]:
        features[f'q{q}'] = series.quantile(q/100)
    
    # Rolling features
    for w in window_sizes:
        features[f'rolling_mean_{w}'] = series.rolling(w).mean().iloc[-1]
        features[f'rolling_std_{w}'] = series.rolling(w).std().iloc[-1]
    
    # Rate of change
    features['mean_diff'] = series.diff().mean()
    features['max_diff'] = series.diff().abs().max()
    
    # Zero crossings (relevant untuk signal analysis!)
    # Zero crossing = titik di mana sinyal melewati mean-nya.
    # Jumlah zero crossings dalam window berkorelasi dengan frekuensi dominan.
    zero_crossings = np.sum(np.diff(np.sign(series - series.mean())) != 0)
    features['zero_crossings'] = zero_crossings
    
    return features


# Demo: extract features dari sinyal simulasi
t = np.arange(1000)
signal = pd.Series(
    np.sin(2 * np.pi * t / 100) + 0.3 * np.random.randn(1000)
)

features = extract_time_features(signal)
print("\n=== Time Series Features ===")
for name, value in list(features.items())[:10]:
    print(f"  {name}: {value:.4f}")


# ===========================================================
# BAGIAN 4: Frequency Domain Features
# ===========================================================
# FFT (Fast Fourier Transform) memetakan sinyal dari time domain
# ke frequency domain. Fitur di frequency domain sering lebih
# diskriminatif untuk fault detection.
#
# Mengapa frequency domain?
# - Fault mekanis menghasilkan signature frekuensi yang khas.
# - Contoh: bearing fault menghasilkan frekuensi karakteristik
#   (BPFO, BPFI, BSF) yang bisa dideteksi di spektrum.
# - Noise di time domain bisa di-filter di frequency domain.
#
# Fitur Frequency Domain:
# - dominant_freq: frekuensi dengan magnitude tertinggi.
# - spectral_energy: total energy di frequency domain (Parseval's theorem).
# - spectral_entropy: seberapa "flat" spektrum. Tinggi = noise-like.
#   Rendah = tonal (dominant frequency jelas).
# - spectral_centroid: "center of mass" spectrum. Tinggi = sinyal "bright".
# - spectral_bandwidth: spread spectrum. Lebar = banyak frekuensi.
# - energy_band: energy ratio per frequency band (low/mid/high).
# - thd: Total Harmonic Distortion = sqrt(sum(harmonics**2)) / fundamental.
#   THD > 5% menandakan power quality issue.
#
# Koneksi Teknik Elektro:
# - FFT = transformasi ke frequency domain
# - THD = ukuran harmonic distortion (power quality)
# - Spectral centroid = "brightness" dari sinyal
# - Band energy = filter bank analysis (seperti equalizer)

def extract_frequency_features(signal, fs=1000):
    """
    Extract fitur dari frequency domain - familiar buat EE!
    
    Parameters:
    -----------
    signal : np.ndarray
        Time domain signal.
    fs : int, default 1000
        Sampling frequency (Hz).
        
    Returns:
    --------
    dict
        Dictionary frequency domain features.
        
    Notes:
    ------
    Features:
    - dominant_freq: frekuensi dengan magnitude tertinggi
    - spectral_energy: total energy di frequency domain
    - spectral_entropy: entropy dari power spectral density
    - spectral_centroid: "center of mass" dari spectrum
    - spectral_bandwidth: spread dari spectrum
    - energy_band: energy ratio per band
    - thd: Total Harmonic Distortion (klasik EE!)
    
    Koneksi Teknik Elektro:
    - FFT = transformasi ke frequency domain
    - THD = ukuran harmonic distortion (power quality)
    - Spectral centroid = "brightness" dari sinyal
    
    TIPS:
    - Gunakan windowing (Hanning/Hamming) sebelum FFT untuk mengurangi leakage.
    - fs harus sesuai dengan Nyquist: fs > 2 * f_max.
    - Untuk sinyal non-stasioner, gunakan STFT (Short-Time Fourier Transform).
    """
    features = {}
    
    # FFT
    # np.fft.rfft hanya menghitung positive frequencies (efisien untuk real signal).
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    magnitude = np.abs(fft_vals)
    power = magnitude ** 2
    
    # Spectral features
    # +1 pada argmax untuk skip DC component (freq=0) karena biasanya tidak informatif.
    features['dominant_freq'] = freqs[np.argmax(magnitude[1:]) + 1]
    features['spectral_energy'] = np.sum(power)
    
    # Spectral entropy: mengukur seberapa "teratur" spektrum.
    # Entropi tinggi = spektrum flat (white noise).
    # Entropi rendah = spektrum terpusat (tonal).
    psd_norm = power / (power.sum() + 1e-12)
    features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    
    features['spectral_centroid'] = np.sum(freqs * magnitude) / np.sum(magnitude)
    features['spectral_bandwidth'] = np.sqrt(
        np.sum((freqs - features['spectral_centroid'])**2 * magnitude) / np.sum(magnitude)
    )
    
    # Band energy ratios
    # Membagi spektrum ke band frekuensi untuk analisis multi-band.
    for low, high, name in [(0, 50, 'low'), (50, 200, 'mid'), (200, fs//2, 'high')]:
        mask = (freqs >= low) & (freqs < high)
        features[f'energy_{name}'] = np.sum(power[mask]) / np.sum(power)
    
    # THD (Total Harmonic Distortion) - klasik EE!
    # THD mengukur seberapa "murni" sinyal sinusoidal.
    # THD = 0 berarti sinyal murni sinusoidal (tidak ada harmonik).
    # THD tinggi menandakan nonlinear load atau fault.
    fundamental_idx = np.argmax(magnitude[1:]) + 1
    fundamental_mag = magnitude[fundamental_idx]
    harmonics_mag = np.sqrt(np.sum(magnitude[2*fundamental_idx::fundamental_idx]**2))
    features['thd'] = harmonics_mag / (fundamental_mag + 1e-12)
    
    return features


# Demo
signal_demo = np.sin(2 * np.pi * 50 * np.arange(1000) / 1000)  # 50 Hz
signal_demo += 0.3 * np.sin(2 * np.pi * 150 * np.arange(1000) / 1000)  # 3rd harmonic
signal_demo += 0.1 * np.random.randn(1000)  # noise

freq_features = extract_frequency_features(signal_demo, fs=1000)
print("\n=== Frequency Domain Features ===")
for name, value in freq_features.items():
    print(f"  {name}: {value:.4f}")


# ===========================================================
# BAGIAN 5: Feature Selection
# ===========================================================
# Mengapa feature selection penting?
# - Curse of dimensionality: semakin banyak fitur, semakin banyak data yang dibutuhkan.
# - Overfitting: fitur noise bisa menyebabkan model overfit.
# - Interpretability: model dengan fewer fitur lebih mudah dijelaskan.
# - Training time: fewer fitur = training lebih cepat.
# - Inference time: fewer fitur = prediksi lebih cepat (penting untuk real-time).
#
# Tiga kategori feature selection:
# 1. Filter methods: seleksi berdasarkan statistik, independent dari model.
# 2. Wrapper methods: seleksi dengan evaluasi model (RFE, forward/backward selection).
# 3. Embedded methods: seleksi selama training (L1 regularization, tree importance).
#
# Method 1: Univariate (F-test / ANOVA)
# - F-test = ANOVA F-value antara feature dan target.
# - Mengukur seberapa besar variance antar kelas dibanding variance dalam kelas.
# - Cocok untuk: linear relationships, continuous features.
# - PERINGATAN: hanya menangkap univariate relationships (bisa miss
#   fitur yang penting hanya dalam kombinasi).
#
# Method 2: Mutual Information (MI)
# - MI = ukuran ketergantungan (informasi bersama) antara feature dan target.
# - Bisa menangkap non-linear relationships.
# - MI = 0 berarti independent. MI tinggi berarti strongly dependent.
# - PERINGATAN: estimasi MI untuk continuous data bisa noisy (perlu banyak sample).
#
# Method 3: Tree-based Importance
# - Importance = decrease in impurity (Gini/Entropy) ketika feature digunakan untuk split.
# - Di-average di seluruh tree dan split.
# - Bisa menangkap interactions antar fitur (karena tree bisa split berulang).
# - PERINGATAN: biased ke fitur dengan banyak nilai unik (high cardinality).
#   Gunakan permutation importance untuk hasil lebih reliable.

# Generate dataset dengan fitur informative dan noisy
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=500, n_features=20,
                            n_informative=5, n_redundant=5,
                            n_repeated=2, random_state=42)

feature_names = [f'feat_{i}' for i in range(20)]

print("\n=== Feature Selection Methods ===")

# Method 1: Univariate (F-test)
selector_f = SelectKBest(f_classif, k=10)
X_selected_f = selector_f.fit_transform(X, y)
scores_f = selector_f.scores_
print("\nF-test scores (top 5):")
top_f = np.argsort(scores_f)[::-1][:5]
for idx in top_f:
    print(f"  {feature_names[idx]}: {scores_f[idx]:.2f}")

# Method 2: Mutual Information
selector_mi = SelectKBest(mutual_info_classif, k=10)
selector_mi.fit(X, y)
scores_mi = selector_mi.scores_
print("\nMutual Information (top 5):")
top_mi = np.argsort(scores_mi)[::-1][:5]
for idx in top_mi:
    print(f"  {feature_names[idx]}: {scores_mi[idx]:.4f}")

# Method 3: Random Forest importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
print("\nRandom Forest Importance (top 5):")
top_rf = np.argsort(importances)[::-1][:5]
for idx in top_rf:
    print(f"  {feature_names[idx]}: {importances[idx]:.4f}")

# Visualisasi perbandingan
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
methods = [('F-test', scores_f), ('Mutual Info', scores_mi),
           ('RF Importance', importances)]

for ax, (name, scores) in zip(axes, methods):
    sorted_idx = np.argsort(scores)[::-1]
    ax.barh(range(len(scores)), scores[sorted_idx])
    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
    ax.set_title(name)
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('02_feature_selection.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nOK Saved: 02_feature_selection.png")


# ===========================================================
# BAGIAN 6: Dimensionality Reduction Impact
# ===========================================================
# Mengapa dimensionality reduction bisa meningkatkan performance?
# - Mengurangi noise: fitur dengan variance rendah seringnya noise.
# - Mengurangi multicollinearity: fitur berkorelasi tinggi redundant.
# - Mengurangi overfitting: fewer dimensions = simpler model.
# - Mempercepat training: fewer fitur = computation lebih cepat.
#
# Tradeoff:
# - Terlalu sedikit fitur = underfitting (kehilangan informasi penting).
# - Terlalu banyak fitur = overfitting (curse of dimensionality).
# - Optimal: biasanya ada "sweet spot" yang bisa ditemukan dengan CV.
#
# Cara menentukan jumlah fitur optimal:
# - Plot performance vs jumlah fitur.
# - Pilih titik di mana penambahan fitur tidak lagi meningkatkan performance.
# - Gunakan nested CV untuk unbiased estimate.

print("\n=== Effect of Feature Selection on Model Performance ===")
n_features_list = [3, 5, 10, 15, 20]
scores_by_n = []

for n_feat in n_features_list:
    selector = SelectKBest(f_classif, k=n_feat)
    pipeline = Pipeline([
        ('select', selector),
        ('scale', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    scores_by_n.append(cv_scores.mean())
    print(f"  {n_feat} features: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")


# ===========================================================
# LATIHAN 10: Feature Engineering Pipeline
# ===========================================================
"""
TARGET Learning Objectives:
   - Membangun automated feature engineering pipeline
   - Mengimplementasikan feature selection strategies
   - Membuat reusable feature engineering class

PANDUAN LANGKAH-LANGKAH:

STEP 1: Automatic Feature Generation
-------------------------------------
Buat class FeatureEngineer yang bisa generate features otomatis:

   a) Polynomial interactions:
      - X1*X2, X1**2, X2**2 untuk semua pasangan fitur
      - Gunakan PolynomialFeatures dari sklearn
      
   b) Log transforms:
      - Deteksi skewness > 1 -> apply log1p
      - Simpan mapping untuk transformasi inverse
      
   c) Ratio features:
      - X1/X2 untuk pasangan fitur yang meaningful
      - Handle division by zero dengan epsilon
      
   d) Binning:
      - Binning continuous features ke categories
      - Gunakan pd.cut() atau pd.qcut()
      - Binning mengubah linear relationship jadi non-linear


STEP 2: Automatic Feature Selection
------------------------------------
Implementasi multi-stage selection:

   a) Remove low-variance features:
      - threshold = 0.01 * variance max
      - Fitur dengan variance ~0 = konstan -> tidak informatif
      
   b) Remove highly correlated features:
      - correlation > 0.95 -> redundant
      - Keep satu dari setiap pasangan correlated
      
   c) Select top K by mutual information:
      - K = 10 atau 20 (configurable)
      - Gunakan SelectKBest(mutual_info_classif)
      
   d) Recursive Feature Elimination (RFE):
      - Train model dengan semua fitur
      - Remove fitur dengan importance terendah
      - Ulangi sampai tersisa K fitur
      
   TIPS KENAPA multi-stage?
     - Setiap stage menghilangkan tipe redundancy yang berbeda
     - Low-variance -> constant features
     - High-correlation -> linear redundancy
     - MI/RFE -> relevance to target


STEP 3: Pipeline Integration
----------------------------
   a) fit_transform(X_train, y_train):
      - Fit semua transformers pada training data
      - Transform training data
      - Simpan state (scalers, selectors, etc.)
      
   b) transform(X_test):
      - HANYA transform, tanpa fitting!
      - Gunakan state yang disimpan saat fit_transform
      - Ini mencegah data leakage
      
   c) get_feature_names():
      - Return nama fitur setelah engineering
      - Berguna untuk interpretability dan debugging


STEP 4: Report Generation
--------------------------
   Buat report yang menjelaskan:
   
   a) Feature importance ranking (top 10)
   b) Removed features & alasan:
      - Low variance: [list]
      - High correlation: [list]
      - Low mutual information: [list]
      
   c) Before/after performance comparison:
      - Model dengan semua fitur vs selected features
      - Metrics: accuracy, training time, inference time
      
   d) Transformations applied:
      - Log transforms: [list fitur]
      - Polynomial features: [list]
      - Binned features: [list]


TIPS HINTS:
   - Gunakan sklearn.base.BaseEstimator untuk compatibility
   - Simpan semua transformers di dictionary: self.transformers_
   - Gunakan joblib untuk serialize pipeline
   - Document setiap transformation dengan docstring

PERINGATAN COMMON MISTAKES:
   - Fit pada test data -> data leakage
   - Tidak handle fitur baru di test data
   - Polynomial degree terlalu tinggi -> explosion dimensionality
   - Tidak inverse transform untuk interpretasi

TARGET EXPECTED OUTPUT:
   - Reusable FeatureEngineer class
   - Comprehensive report
   - Performance improvement dari feature engineering
   - Reduced dimensionality tanpa significant performance loss
"""


# ===========================================================
# 🔥 CHALLENGE: End-to-End Feature Engineering
# ===========================================================
"""
TARGET Learning Objectives:
   - Mengaplikasikan feature engineering ke dataset realistis
   - Menggabungkan domain knowledge EE dengan ML
   - Membangun production-ready feature pipeline

PANDUAN LANGKAH-LANGKAH:

STEP 1: Generate Dataset
------------------------
Konteks: Predictive Maintenance untuk Motor Listrik

   Dataset (generate synthetic):
   - 500 windows of sensor data, masing-masing 1000 samples
   - Sensors: voltage, current, vibration (3-axis), temperature
   - Labels: 0=healthy, 1=degrading, 2=failing
   
   TIPS KENAPA window-based?
     - Fault biasanya terdeteksi dari pattern di window
     - Single sample tidak cukup informatif
     - Window = 1 detik dengan fs=1000 Hz


STEP 2: Extract Features dari Setiap Window
-------------------------------------------
   Time domain (per window):
   - mean, std, RMS, peak, crest factor, kurtosis
   - Zero crossings, peak-to-peak amplitude
   - Trend (slope dari linear regression pada window)
   
   Frequency domain (per window):
   - Dominant frequency, THD, spectral energy
   - Band ratios: low/mid/high frequency energy
   - Spectral entropy, spectral flatness
   
   Cross-sensor:
   - Correlation between voltage & current
   - Vibration RMS (sqrt(vx**2 + vy**2 + vz**2))
   - Cross-correlation lag antara sensors


STEP 3: Domain Knowledge Feature Engineering
---------------------------------------------
Gunakan domain knowledge dari Teknik Elektro:

   a) Power features:
      - Apparent power = V_rms * I_rms
      - Real power = V_rms * I_rms * cos(theta)
      - Power factor = cos(theta)
      
   b) Thermal aging:
      - I**2*t (thermal aging indicator)
      - Temperature rise above ambient
      
   c) Mechanical health:
      - Vibration severity per ISO 10816
      - Bearing condition indicators (BPFO, BPFI)
      
   d) Power quality:
      - Voltage unbalance (negatif sequence)
      - Harmonic content (THD-V, THD-I)
      - Flicker (Pst)


STEP 4: Feature Selection
-------------------------
   a) Bandingkan 3+ metode selection:
      - Univariate (F-test, MI)
      - Tree-based importance
      - RFE dengan Random Forest
      
   b) Analisis:
      - Fitur mana yang paling diskriminatif per kelas?
      - Apakah domain features lebih penting dari statistical features?
      - Berapa fitur optimal? (plot performance vs n_features)


STEP 5: Train & Evaluate
------------------------
   a) Dengan semua fitur
   b) Dengan selected features (top 10, 20, 30)
   c) Bandingkan:
      - Accuracy, precision, recall, F1 per kelas
      - Training time
      - Model complexity
      
   d) Visualisasi:
      - Feature importance plot
      - Confusion matrix untuk best model
      - t-SNE dari features (lihat separability)


TIPS HINTS:
   - ISO 10816 vibration severity bisa diapproximate dengan RMS
   - THD = sqrt(sum(V_h**2)) / V_1 untuk h=2..N
   - I**2*t = integral dari I**2 dt (discretize dengan sum)
   - Cross-correlation: np.correlate(sensor1, sensor2, mode='full')

PERINGATAN COMMON MISTAKES:
   - Feature extraction sebelum train/test split -> leakage
   - Tidak normalisasi frequency domain features
   - Mengabaikan class imbalance di evaluation
   - Terlalu banyak fitur -> curse of dimensionality

TARGET EXPECTED OUTPUT:
   - 50+ features extracted per window
   - Selected features: 15-20 dengan performance >= 95% dari all features
   - Domain features dominan di top importance
   - Clear narrative: "fitur X penting karena ..."

Simpan hasilnya di projects/project_02_klasifikasi_sinyal/
"""

print("\n" + "="*50)
print("OK FASE 3 SELESAI!")
print("="*50)
print("""
Kamu sekarang bisa:
OK Menggunakan sklearn untuk supervised learning
OK Clustering, PCA, anomaly detection
OK Feature engineering (termasuk domain-specific EE features!)
OK Proper feature selection

Sebelum lanjut:
1. Selesaikan Project 2: Klasifikasi Sinyal
2. Pastikan semua exercise selesai

Lanjut ke: 04-deep-learning/01_neural_net_scratch.py
""")
