"""
=============================================================
FASE 3 — MODUL 3: FEATURE ENGINEERING
=============================================================
"Data scientists spend 80% of their time on data preparation."

Feature engineering = mengubah raw data menjadi representasi
yang lebih mudah dipelajari oleh model ML.

Ini adalah SKILL yang paling membedakan:
- Pemula: pakai fitur apa adanya
- Expert: create fitur yang "membuka" pattern tersembunyi

Background EE memberikan keuntungan BESAR di sini:
- Signal processing → feature extraction dari time series
- Domain knowledge → meaningful engineered features
- Physics-based features → fitur yang punya interpretasi fisik

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
# 📖 BAGIAN 1: Numerical Transformations
# ===========================================================

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
print("📊 Saved: 01_scaling.png")

# Log transform untuk distribusi skewed
data['vibration_log'] = np.log1p(data['vibration'])  # log(1+x) untuk handle zeros
data['hours_log'] = np.log1p(data['hours_running'])

print("\nVibration - Skewness before log: "
      f"{data['vibration'].skew():.2f}")
print("Vibration - Skewness after log:  "
      f"{data['vibration_log'].skew():.2f}")


# ===========================================================
# 📖 BAGIAN 2: Domain-Specific Features (EE-Based!)
# ===========================================================
# Fitur yang dibuat berdasarkan domain knowledge SELALU lebih
# powerful dari fitur statistik generik.

# Power (dari Teknik Elektro: P = V * I)
data['power'] = data['voltage'] * data['current']

# Apparent power, Power factor (jika punya phase angle)
data['phase_angle'] = np.random.uniform(0, np.pi/4, n)
data['real_power'] = data['voltage'] * data['current'] * np.cos(data['phase_angle'])
data['reactive_power'] = data['voltage'] * data['current'] * np.sin(data['phase_angle'])
data['power_factor'] = np.cos(data['phase_angle'])

# Efficiency indicator
data['power_per_rpm'] = data['power'] / (data['rpm'] + 1)

# Thermal indicator (arus tinggi + suhu tinggi = bahaya!)
data['thermal_risk'] = data['current'] * data['temperature'] / 100

# Vibration normalized by RPM (vibrasi tinggi di RPM rendah = lebih serius)
data['vibration_per_rpm'] = data['vibration'] / (data['rpm'] + 1) * 1000

# Operational age indicator
data['age_category'] = pd.cut(data['hours_running'],
                               bins=[0, 200, 1000, 5000, np.inf],
                               labels=['new', 'running-in', 'normal', 'aging'])

print("\n=== Domain Features Created ===")
print(data[['power', 'power_factor', 'thermal_risk',
            'vibration_per_rpm']].describe().round(2))


# ===========================================================
# 📖 BAGIAN 3: Time Series Features
# ===========================================================
# Untuk data yang punya komponen waktu (sensor, monitoring)

def extract_time_features(series, window_sizes=[5, 10, 20]):
    """Extract statistical features dari time series"""
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
# 📖 BAGIAN 4: Frequency Domain Features
# ===========================================================

def extract_frequency_features(signal, fs=1000):
    """Extract fitur dari frequency domain — familiar buat EE!"""
    features = {}

    # FFT
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    magnitude = np.abs(fft_vals)
    power = magnitude ** 2

    # Spectral features
    features['dominant_freq'] = freqs[np.argmax(magnitude[1:]) + 1]
    features['spectral_energy'] = np.sum(power)
    features['spectral_entropy'] = -np.sum(
        (power / power.sum()) * np.log2(power / power.sum() + 1e-12)
    )
    features['spectral_centroid'] = np.sum(freqs * magnitude) / np.sum(magnitude)
    features['spectral_bandwidth'] = np.sqrt(
        np.sum((freqs - features['spectral_centroid'])**2 * magnitude) / np.sum(magnitude)
    )

    # Band energy ratios
    for low, high, name in [(0, 50, 'low'), (50, 200, 'mid'), (200, fs//2, 'high')]:
        mask = (freqs >= low) & (freqs < high)
        features[f'energy_{name}'] = np.sum(power[mask]) / np.sum(power)

    # THD (Total Harmonic Distortion) — klasik EE!
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
# 📖 BAGIAN 5: Feature Selection
# ===========================================================

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
print("\n📊 Saved: 02_feature_selection.png")


# ===========================================================
# 📖 BAGIAN 6: Dimensionality Reduction Impact
# ===========================================================

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
    print(f"  {n_feat} features: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ===========================================================
# 🏋️ EXERCISE 10: Feature Engineering Pipeline
# ===========================================================
"""
Buat class FeatureEngineer yang bisa:

1. Automatic feature generation:
   - Polynomial interactions (X1*X2, X1², etc.)
   - Log transforms untuk skewed features
   - Ratio features (X1/X2)
   - Binning continuous features

2. Automatic feature selection:
   - Remove low-variance features
   - Remove highly correlated features (correlation > 0.95)
   - Select top K by mutual information
   - Recursive Feature Elimination (RFE)

3. Pipeline integration:
   - fit_transform(X_train, y_train)
   - transform(X_test) — HANYA transform, tanpa fitting!
   - get_feature_names() — tracking nama fitur

4. Report:
   - Feature importance ranking
   - Removed features & alasan
   - Before/after performance comparison
"""


# ===========================================================
# 🔥 CHALLENGE: End-to-End Feature Engineering
# ===========================================================
"""
Konteks: Predictive Maintenance untuk Motor Listrik

Dataset (generate synthetic):
- 500 windows of sensor data, masing-masing 1000 samples
- Sensors: voltage, current, vibration (3-axis), temperature
- Labels: 0=healthy, 1=degrading, 2=failing

Tugas:
1. Extract features dari setiap window:
   - Time domain: mean, std, RMS, peak, crest factor, kurtosis
   - Frequency domain: dominant freq, THD, spectral energy, band ratios
   - Cross-sensor: correlation between voltage & current, vibration RMS

2. Feature engineering:
   - Gunakan DOMAIN KNOWLEDGE dari Teknik Elektro
   - Contoh: I²t (thermal aging), vibration severity per ISO 10816
   - Power quality features: voltage unbalance, harmonic content

3. Feature selection:
   - Bandingkan 3+ metode selection
   - Analisis: fitur mana yang paling diskriminatif per kelas?

4. Train & evaluate model:
   - Dengan semua fitur vs selected features
   - Apakah feature engineering meningkatkan performance?

Simpan hasilnya di projects/project_02_klasifikasi_sinyal/
"""

print("\n" + "="*50)
print("🎉 FASE 3 SELESAI!")
print("="*50)
print("""
Kamu sekarang bisa:
✅ Menggunakan sklearn untuk supervised learning
✅ Clustering, PCA, anomaly detection
✅ Feature engineering (termasuk domain-specific EE features!)
✅ Proper feature selection

Sebelum lanjut:
1. Selesaikan Project 2: Klasifikasi Sinyal
2. Pastikan semua exercise selesai

Lanjut ke: 04-deep-learning/01_neural_net_scratch.py
""")
