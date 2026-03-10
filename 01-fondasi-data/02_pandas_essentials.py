"""
=============================================================
FASE 1 — MODUL 2: PANDAS ESSENTIALS
=============================================================
Pandas = tool utama untuk data manipulation di ML pipeline.

Kamu perlu Pandas untuk:
- Load & inspect dataset
- Clean data (missing values, outliers, tipe data)
- Transform & feature engineering
- Split data untuk training/testing

Durasi target: 2-3 jam
=============================================================
"""

import numpy as np
import pandas as pd

# ===========================================================
# 📖 BAGIAN 1: Membuat & Membaca Data
# ===========================================================

# Buat DataFrame dari dictionary
sensor_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
    'temperature': np.random.normal(25, 5, 100),
    'humidity': np.random.normal(60, 10, 100),
    'voltage': np.random.normal(220, 5, 100),
    'status': np.random.choice(['normal', 'warning', 'fault'], 100, p=[0.7, 0.2, 0.1])
})

# Sengaja tambahkan missing values (realistis!)
mask = np.random.random(100) < 0.05
sensor_data.loc[mask, 'temperature'] = np.nan
sensor_data.loc[np.random.random(100) < 0.03, 'voltage'] = np.nan

print("📊 Dataset sensor:")
print(sensor_data.head(10))
print(f"\nShape: {sensor_data.shape}")
print(f"\nInfo:")
print(sensor_data.info())
print(f"\nStatistik deskriptif:")
print(sensor_data.describe())


# ===========================================================
# 📖 BAGIAN 2: Data Inspection & Cleaning
# ===========================================================

# Cek missing values
print("\n--- Missing Values ---")
print(sensor_data.isnull().sum())
print(f"Total missing: {sensor_data.isnull().sum().sum()}")

# Handle missing values — beberapa strategi
# Strategi 1: Drop rows (kehilangan data, tapi simpel)
# df_clean = sensor_data.dropna()

# Strategi 2: Fill dengan mean (paling umum untuk numerik)
# df_clean = sensor_data.fillna(sensor_data.mean(numeric_only=True))

# Strategi 3: Interpolate (BAGUS untuk time series / sensor data!)
df_clean = sensor_data.copy()
df_clean['temperature'] = df_clean['temperature'].interpolate(method='linear')
df_clean['voltage'] = df_clean['voltage'].interpolate(method='linear')

print(f"\nSetelah interpolasi, missing: {df_clean.isnull().sum().sum()}")

# Deteksi outliers (IQR method)
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)

outlier_mask = detect_outliers_iqr(df_clean['temperature'])
print(f"\nOutliers di temperature: {outlier_mask.sum()}")


# ===========================================================
# 📖 BAGIAN 3: Filtering, Grouping, Aggregation
# ===========================================================

# Filtering
faults = df_clean[df_clean['status'] == 'fault']
print(f"\n--- Fault records: {len(faults)} ---")

high_temp_faults = df_clean[
    (df_clean['status'] == 'fault') &
    (df_clean['temperature'] > 30)
]
print(f"Fault dengan temp > 30°C: {len(high_temp_faults)}")

# Groupby — analisis per kategori
print("\n--- Statistik per Status ---")
grouped = df_clean.groupby('status').agg({
    'temperature': ['mean', 'std', 'min', 'max'],
    'voltage': ['mean', 'std'],
    'humidity': 'mean'
}).round(2)
print(grouped)

# Time-based analysis
df_clean['hour'] = df_clean['timestamp'].dt.hour
hourly_avg = df_clean.groupby('hour')['temperature'].mean()
print(f"\n--- Rata-rata suhu per jam (sample) ---")
print(hourly_avg.head())


# ===========================================================
# 📖 BAGIAN 4: Feature Engineering dengan Pandas
# ===========================================================
# Ini KUNCI untuk ML — model hanya sebagus fitur-fiturnya!

# Rolling statistics (moving average, moving std)
df_clean['temp_rolling_mean'] = df_clean['temperature'].rolling(window=5).mean()
df_clean['temp_rolling_std'] = df_clean['temperature'].rolling(window=5).std()

# Lag features (untuk time series prediction)
df_clean['temp_lag_1'] = df_clean['temperature'].shift(1)
df_clean['temp_lag_3'] = df_clean['temperature'].shift(3)

# Rate of change
df_clean['temp_diff'] = df_clean['temperature'].diff()

# Encode categorical variable
df_clean['status_encoded'] = df_clean['status'].map({
    'normal': 0, 'warning': 1, 'fault': 2
})

# One-hot encoding
status_dummies = pd.get_dummies(df_clean['status'], prefix='status')
df_featured = pd.concat([df_clean, status_dummies], axis=1)

print("\n--- DataFrame dengan features baru ---")
print(df_featured[['temperature', 'temp_rolling_mean', 'temp_lag_1',
                    'temp_diff', 'status_encoded']].head(10))


# ===========================================================
# 📖 BAGIAN 5: Persiapan Data untuk ML
# ===========================================================

# Feature matrix (X) dan target (y)
feature_cols = ['temperature', 'humidity', 'voltage',
                'temp_rolling_mean', 'temp_rolling_std',
                'temp_lag_1', 'temp_lag_3', 'temp_diff']

# Drop rows dengan NaN (dari rolling/lag features)
df_ml = df_featured.dropna(subset=feature_cols)

X = df_ml[feature_cols].values  # convert ke numpy array
y = df_ml['status_encoded'].values

print(f"\n--- Data siap untuk ML ---")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Distribusi kelas: {np.bincount(y.astype(int))}")

# Train-test split (manual, nanti pakai sklearn)
n = len(X)
indices = np.random.permutation(n)
train_size = int(0.8 * n)

X_train = X[indices[:train_size]]
X_test = X[indices[train_size:]]
y_train = y[indices[:train_size]]
y_test = y[indices[train_size:]]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ===========================================================
# 🏋️ EXERCISE 2: Eksplorasi Dataset Publik
# ===========================================================
"""
Download salah satu dataset ini dan lakukan full EDA:

1. Opsi A (EE-related): UCI Power Consumption Dataset
   https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

2. Opsi B (General): Titanic Dataset
   import seaborn as sns
   df = sns.load_dataset('titanic')

Tugas:
a) Load data dan inspect (shape, dtypes, missing values)
b) Bersihkan data (handle missing, outliers)
c) Buat minimal 5 fitur baru yang meaningful
d) Visualisasi (di modul selanjutnya, tapi coba dulu)
e) Siapkan X dan y untuk ML

PENTING: Tulis INSIGHT, bukan cuma kode!
Contoh insight: "Voltage drop > 10V berkorelasi dengan status 'fault' —
ini masuk akal karena fault biasanya menyebabkan voltage sag."
"""


# ===========================================================
# 🔥 CHALLENGE: Pipeline Otomatis
# ===========================================================
"""
Buat class DataPipeline yang:
1. __init__(self, df) — terima raw DataFrame
2. .inspect() — print summary (shape, missing, dtypes)
3. .clean(strategy='interpolate') — handle missing values
4. .add_rolling_features(columns, windows) — tambah rolling stats
5. .add_lag_features(columns, lags) — tambah lag features
6. .encode_categorical(columns) — encode categorical columns
7. .prepare_ml(target_col, feature_cols) — return X_train, X_test, y_train, y_test

Class ini akan berguna di semua project selanjutnya!

Kenapa bikin class sendiri? Karena di dunia nyata, data pipeline = 80% waktu ML.
Lebih baik punya pipeline yang solid daripada model yang fancy.
"""

print("\n" + "="*50)
print("✅ Modul 2 selesai! Lanjut ke: 01-fondasi-data/03_visualisasi.py")
print("="*50)
