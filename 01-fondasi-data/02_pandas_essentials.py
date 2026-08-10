"""
=============================================================
FASE 1 - MODUL 2: PANDAS ESSENTIALS
=============================================================
Pandas = tool utama untuk data manipulation di ML pipeline.

Kamu perlu Pandas untuk:
- Load & inspect dataset (CSV, Excel, SQL, JSON)
- Clean data (missing values, outliers, tipe data yang salah)
- Transform & feature engineering (aggregate, group, pivot)
- Split data untuk training/testing
- Time series analysis dan resampling

Koneksi Teknik Elektro:
- DataFrame = tabel measurement dari multiple sensor dalam satu waktu
- Time series = logged data dari SCADA/DAS (Data Acquisition System)
- Groupby = aggregate statistics per equipment/motor/zone
- Missing values = sensor failure atau communication loss

Durasi target: 2-3 jam
=============================================================
"""

import numpy as np
import pandas as pd

# ===========================================================
# BAGIAN 1: Membuat & Membaca Data
# ===========================================================
# DataFrame adalah struktur data 2D labeled (mirip spreadsheet atau SQL table).
# Komponen utama DataFrame:
# - Index: label baris (default: integer 0, 1, 2, ...)
# - Columns: label kolom
# - Values: data mentah (stored sebagai NumPy array di backend)
#
# Setiap kolom di DataFrame adalah Series (1D labeled array).
# Series bisa memiliki tipe data berbeda per kolom, tapi dalam satu kolom
# tipe data harus seragam (konsep yang sama dengan NumPy array).
#
# Pandas dibangun di atas NumPy, jadi semua operasi numerik
# di Pandas menggunakan NumPy di backend untuk kecepatan.

# --- Membuat DataFrame dari dictionary ---
# Dictionary keys menjadi nama kolom.
# Dictionary values menjadi data kolom (list, array, atau Series).
# Semua values harus memiliki panjang yang sama (kecuali scalar yang di-broadcast).
#
# pd.date_range() menghasilkan sequence datetime yang evenly spaced.
# Parameter penting:
# - start: tanggal/waktu awal
# - periods: jumlah periode yang dihasilkan
# - freq: frekuensi ('h'=hourly, 'D'=daily, 'W'=weekly, 'M'=monthly,
#                     'min'=minutely, 'S'=secondly, 'ms'=millisecond)
# - tz: timezone (opsional, contoh: 'UTC', 'Asia/Jakarta')
#
# Frekuensi yang sering dipakai di time series ML:
# - 'h' : sensor logging per jam (energy consumption, temperature)
# - '15min' : high-resolution industrial sensor
# - 'D' : daily aggregation (sales, weather)
# - 'MS' : month start (monthly reports)
sensor_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
    # freq='h' = hourly frequency
    # Periode 100 jam = ~4.17 hari data
    'temperature': np.random.normal(25, 5, 100),
    # np.random.normal(mean, std, size) -> Gaussian distribution
    # mean=25 (suhu ruangan), std=5 (variasi normal)
    'humidity': np.random.normal(60, 10, 100),
    # mean=60% RH, std=10%
    'voltage': np.random.normal(220, 5, 100),
    # mean=220V, std=5V (toleransi tegangan normal)
    'status': np.random.choice(['normal', 'warning', 'fault'], 100, p=[0.7, 0.2, 0.1])
    # np.random.choice dengan probabilities p
    # p=[0.7, 0.2, 0.1] -> 70% normal, 20% warning, 10% fault
    # Di ML imbalanced classification, class weights perlu disesuaikan
})

# --- Menambahkan missing values secara sengaja ---
# Kenapa? Karena real-world data SELALU punya missing values!
# Tidak ada dataset nyata yang 100% lengkap.
#
# Penyebab missing values di sensor data:
# - Sensor failure atau kalibrasi ulang
# - Communication loss antara sensor dan data logger
# - Buffer overflow di DAS (Data Acquisition System)
# - Human error (tidak mencatat manual reading)
#
# Masking: memilih subset data berdasarkan kondisi boolean.
# np.random.random(100) < 0.05 menghasilkan array boolean
# dengan ~5% elemen True (random uniform).
# Hasil True/False tidak predictable, simulasi real-world randomness.
np.random.seed(42)
mask = np.random.random(100) < 0.05
sensor_data.loc[mask, 'temperature'] = np.nan
# .loc[indexer, column] = access by label (recommended untuk assignment)
# np.nan = Not a Number (representasi missing value di NumPy/Pandas)
# np.nan adalah float, jadi kolom yang awalnya integer akan jadi float jika ada NaN
sensor_data.loc[np.random.random(100) < 0.03, 'voltage'] = np.nan

print("Dataset sensor:")
print(sensor_data.head(10))
# .head(n) = menampilkan n baris pertama (default n=5)
# Berguna untuk quick sanity check format data

print(f"\nShape: {sensor_data.shape}")
# .shape = (n_rows, n_columns), sama seperti NumPy array

print(f"\nInfo:")
print(sensor_data.info())
# .info() = summary komprehensif:
# - Class type (DataFrame)
# - RangeIndex (start, stop, step)
# - Total columns dan nama-namanya
# - Non-null count per kolom (kritis untuk deteksi missing!)
# - Dtype per kolom (int64, float64, object, datetime64, dll)
# - Memory usage (penting untuk dataset besar)
# Perhatikan: kolom dengan NaN akan memiliki non-null count < total rows.

print(f"\nStatistik deskriptif:")
print(sensor_data.describe())
# .describe() = statistik untuk kolom numerik:
# - count: jumlah non-null values
# - mean: rata-rata (sensitif terhadap outlier)
# - std: standard deviation (penyebaran data)
# - min: nilai minimum
# - 25%: first quartile (Q1)
# - 50%: median / second quartile (Q2)
# - 75%: third quartile (Q3)
# - max: nilai maximum
#
# Insight dari .describe():
# - Jika min/max sangat jauh dari mean -> kemungkinan outlier
# - Jika std sangat besar -> data sangat bervariasi atau noisy
# - Jika count < total rows -> ada missing values
# - Bandingkan mean dan median: jika mean >> median, distribusi right-skewed

# --- .describe() untuk kategorikal ---
print(f"\nStatistik kategorikal:")
print(sensor_data['status'].describe())
# Untuk Series non-numerik, .describe() menghasilkan:
# - count: total non-null
# - unique: jumlah kategori unik
# - top: kategori paling sering (mode)
# - freq: frekuensi kategori top

# --- Value counts ---
print(f"\nDistribusi status:")
print(sensor_data['status'].value_counts())
# .value_counts() = menghitung frekuensi setiap kategori
# Default: descending order (kategori paling sering di atas)
# Parameter normalize=True -> proporsi, bukan count
print(f"Proporsi status:")
print(sensor_data['status'].value_counts(normalize=True).round(3))


# ===========================================================
# BAGIAN 2: Data Inspection & Cleaning
# ===========================================================
# Data cleaning = 60-80% waktu di project ML nyata!
# Quality data > fancy model.
# Garbage in, garbage out (GIGO) -> model tidak bisa belajar dari data jelek.
#
# Workflow inspection:
# 1. Cek shape -> berapa banyak data yang tersedia
# 2. Cek dtypes -> apakah tipe data sudah benar?
# 3. Cek missing -> berapa banyak dan di kolom mana?
# 4. Cek duplicates -> apakah ada data duplikat?
# 5. Cek distribusi -> apakah ada outlier atau skewness?

# --- Cek missing values secara detail ---
# isnull() menghasilkan DataFrame boolean (True jika NaN/None/NaT).
# .sum() pada DataFrame boolean menghitung jumlah True per kolom.
# .sum().sum() menghitung total True di seluruh DataFrame.
print("\n--- Missing Values ---")
missing_count = sensor_data.isnull().sum()
missing_pct = (missing_count / len(sensor_data) * 100).round(2)
missing_df = pd.DataFrame({
    'count': missing_count,
    'percentage': missing_pct
})
print(missing_df)
print(f"Total missing di seluruh dataset: {sensor_data.isnull().sum().sum()}")

# --- Tipe-tipe missing values di Pandas ---
# 1. np.nan (Not a Number) -> untuk float
# 2. None -> untuk object (string) dan datetimes
# 3. pd.NaT (Not a Time) -> khusus untuk datetime
# 4. Inf / -Inf -> terkadang dianggap missing di beberapa konteks
#
# PENTING: np.nan adalah float, sehingga kolom integer yang mengandung
# np.nan akan otomatis di-cast ke float64. Ini bisa membesarkan memory usage.
# Solusi: gunakan pd.Int64Dtype() (nullable integer) jika perlu integer + missing.

# --- Handle missing values - beberapa strategi ---
# Strategi pemilihan tergantung pada:
# - Jumlah missing (<5% vs >30%)
# - Tipe data (numerik vs kategorikal vs datetime)
# - Konteks domain (time series vs cross-sectional)
# - Mekanisme missing (MCAR, MAR, MNAR)

# Strategi 1: Drop rows (listwise deletion)
# Kapan dipakai:
# - Missing < 5% dan data cukup besar (tidak kehilangan banyak informasi)
# - Data missing completely at random (MCAR)
# Kapan TIDAK dipakai:
# - Missing > 10% (kehilangan terlalu banyak data)
# - Data small sample (N < 1000)
# - Data missing not at random (MNAR) -> bisa bias
# df_clean = sensor_data.dropna()
# Parameter subset=['col1', 'col2'] -> hanya drop jika NaN di kolom tertentu
# Parameter how='all' -> drop hanya jika SEMUA kolom NaN

# Strategi 2: Fill dengan mean/median (univariate imputation)
# Kapan dipakai:
# - Data numerik, distribusi tidak terlalu skewed -> mean
# - Data numerik, distribusi skewed atau ada outlier -> median (robust)
# - Data cross-sectional (bukan time series)
# Kelemahan:
# - Mengurangi varians (varians turun karena banyak nilai jadi sama)
# - Tidak mempertahankan korelasi antar fitur
# - Bias jika data tidak MCAR
# df_clean = sensor_data.fillna(sensor_data.mean(numeric_only=True))
# numeric_only=True -> hanya kolom numerik yang dihitung mean-nya

# Strategi 3: Interpolate (BAGUS untuk time series / sensor data!)
# Interpolasi memperkirakan nilai missing berdasarkan tetangga terdekat.
# Method 'linear' = garis lurus antara 2 titik known (first-order approximation).
# Method 'time' = interpolasi berdasarkan jarak waktu (untuk irregular time series).
# Method 'polynomial', order=2 -> fitting parabola (lebih smooth).
# Method 'nearest' -> ambil nilai tetangga terdekat (step function).
#
# Koneksi Teknik Elektro:
# - Interpolasi linear = zero-order hold / first-order hold di DAS
# - Sensor sampling rate tidak selalu konstan -> perlu time-based interpolation
# - Missing data di SCADA sering di-interpolate sebelum masuk ke historian
df_clean = sensor_data.copy()
# Selalu .copy() sebelum modifikasi untuk preserve original data!
df_clean['temperature'] = df_clean['temperature'].interpolate(method='linear')
# Linear interpolasi: jika t_i missing, diisi dengan (t_{i-1} + t_{i+1}) / 2
df_clean['voltage'] = df_clean['voltage'].interpolate(method='linear')
# Limit=direction: 'forward', 'backward', 'both'
# Jika NaN di awal atau akhir series, interpolasi linear tidak bisa mengisi
# (karena tidak punya neighbor di satu sisi). Solusi: gunakan limit_direction.

print(f"\nSetelah interpolasi, missing: {df_clean.isnull().sum().sum()}")

# --- Deteksi outliers (IQR method) ---
# Outlier = nilai yang sangat berbeda dari mayoritas data.
# Outlier bisa:
# - Noise / error measurement (sensor glitch)
# - Event yang sebenarnya terjadi (fault, anomaly) -> JANGAN dihapus!
# - Data entry error (human mistake)
#
# IQR (Interquartile Range) = Q3 - Q1
# Q1 = 25th percentile (kuartil bawah)
# Q3 = 75th percentile (kuartil atas)
# Outlier = nilai < Q1 - 1.5*IQR atau > Q3 + 1.5*IQR
# Rumus ini berasal dari boxplot (Tukey's fences).
#
# Koneksi Teknik Elektro:
# - Mirip dengan threshold detection di fault monitoring systems
# - Protection relay menggunakan threshold (overcurrent, undervoltage)
# - BSCB (Breaker Status Change) detection menggunakan threshold-based logic

def detect_outliers_iqr(series):
    """
    Mendeteksi outliers menggunakan metode Interquartile Range (IQR).
    
    Parameters:
    -----------
    series : pd.Series
        Kolom data numerik yang akan diperiksa outliers-nya.
        
    Returns:
    --------
    pd.Series (boolean)
        Series boolean dengan True untuk outlier, False untuk normal.
        
    Notes:
    ------
    - IQR = Q3 - Q1 (range antara 75th dan 25th percentile)
    - Lower bound = Q1 - 1.5 * IQR
    - Upper bound = Q3 + 1.5 * IQR
    - Nilai di luar bounds dianggap outlier (Tukey's fences)
    - 1.5 * IQR adalah konvensi standar, bisa diubah ke 3.0 untuk lebih konservatif
    - Koneksi ke Teknik Elektro: mirip dengan threshold detection
      di fault monitoring systems dan protection relay
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)


outlier_mask = detect_outliers_iqr(df_clean['temperature'])
print(f"\nOutliers di temperature: {outlier_mask.sum()}")
# .sum() pada boolean Series = menghitung True (karena True=1, False=0)

# --- Alternatif deteksi outlier: Z-score method ---
# Z-score = (x - mean) / std
# Jika |z| > 3, dianggap outlier (asumsi distribusi normal).
# Lebih sensitif terhadap mean dan std -> tidak robust jika ada outlier besar.
# Kelebihan: lebih intuitif (berapa sigma dari mean).
# Kekurangan: asumsikan normal distribution (tidak selalu valid).
z_scores = np.abs((df_clean['temperature'] - df_clean['temperature'].mean()) / df_clean['temperature'].std())
outliers_z = z_scores > 3
print(f"Outliers di temperature (Z-score > 3): {outliers_z.sum()}")

# --- Duplikasi data ---
# Duplikat bisa terjadi karena:
# - Double logging dari sensor
# - ETL pipeline yang berjalan 2x
# - Merge/join yang salah
# - Data scraping yang redundant
print(f"\nJumlah duplikat: {df_clean.duplicated().sum()}")
# .duplicated() -> True jika baris identik dengan baris sebelumnya
# Parameter subset=['col1'] -> cek duplikat hanya berdasarkan kolom tertentu
# Parameter keep='last' -> anggap duplikat yang pertama, simpan yang terakhir



# ===========================================================
# BAGIAN 3: Filtering, Grouping, Aggregation
# ===========================================================
# Operasi-operasi ini fundamental untuk EDA (Exploratory Data Analysis).
# EDA = proses memahami data sebelum membangun model.
# Tujuan EDA: menemukan pattern, anomaly, dan insight yang akan
# membimbing feature engineering dan model selection.

# --- Filtering ---
# Memilih baris berdasarkan kondisi boolean.
# Hasil filtering adalah DataFrame baru (copy, bukan view di semua kasus).
# Gunakan & untuk AND, | untuk OR, ~ untuk NOT.
# SETIAP kondisi harus di-wrap dalam parentheses karena precedence operator!
# Tanpa parentheses, Python akan mengevaluasi & dan | sebelum comparison.

# Single condition
faults = df_clean[df_clean['status'] == 'fault']
print(f"\n--- Fault records: {len(faults)} ---")
# len(faults) = jumlah baris yang memenuhi kondisi
# Alternatif: faults.shape[0]

# Multiple conditions (AND)
high_temp_faults = df_clean[
    (df_clean['status'] == 'fault') &
    (df_clean['temperature'] > 30)
]
print(f"Fault dengan temp > 30C: {len(high_temp_faults)}")

# Multiple conditions (OR)
abnormal = df_clean[
    (df_clean['status'] == 'fault') |
    (df_clean['status'] == 'warning')
]
print(f"Total abnormal (fault OR warning): {len(abnormal)}")

# Negasi (NOT)
normal_only = df_clean[~(df_clean['status'] == 'fault')]
print(f"Non-fault records: {len(normal_only)}")

# .query() method - alternatif syntax SQL-like
# Lebih readable untuk kondisi kompleks
# Hati-hati: .query() tidak selalu secepat boolean indexing untuk dataset besar
high_voltage = df_clean.query("voltage > 225 and status == 'normal'")
print(f"High voltage normal: {len(high_voltage)}")

# .between() untuk range selection
# Inclusive secara default (left <= value <= right)
temp_mid = df_clean[df_clean['temperature'].between(20, 30)]
print(f"Temperature 20-30C: {len(temp_mid)}")

# .isin() untuk multiple value matching
selected_status = df_clean[df_clean['status'].isin(['normal', 'warning'])]
print(f"Status normal atau warning: {len(selected_status)}")

# --- Groupby - analisis per kategori ---
# Split-Apply-Combine strategy (Wickham, 2011):
# 1. Split: data dipecah ke groups berdasarkan key (status)
# 2. Apply: fungsi diterapkan ke setiap group independently
# 3. Combine: hasil digabung menjadi DataFrame baru
#
# groupby() menghasilkan GroupBy object (lazy evaluation).
# Tidak ada komputasi yang terjadi sampai aggregation function dipanggil.
# Ini efisien karena tidak membuat copy data.
#
# Koneksi Teknik Elektro:
# - Aggregate statistics per motor/equipment -> reliability analysis
# - Mean vibration per machine type -> predictive maintenance
# - Energy consumption per zone -> load balancing analysis

print("\n--- Statistik per Status ---")
grouped = df_clean.groupby('status').agg({
    'temperature': ['mean', 'std', 'min', 'max'],
    'voltage': ['mean', 'std'],
    'humidity': 'mean'
}).round(2)
# .agg() = aggregate dengan multiple functions
# Dictionary format: {column: function_or_list_of_functions}
# Hasilnya adalah MultiIndex columns: (column_name, agg_function)
print(grouped)

# --- groupby dengan custom functions ---
# .agg() bisa menerima custom lambda atau named function
print("\n--- Custom aggregation ---")
custom_agg = df_clean.groupby('status').agg({
    'temperature': ['mean', lambda x: x.quantile(0.9)],
    'voltage': ['min', 'max', 'count']
})
# Lambda di atas menghitung 90th percentile (P90)
# P90 = nilai di mana 90% data berada di bawahnya
# Di reliability engineering: P90 load = design load untuk most cases
print(custom_agg)

# --- groupby dengan transform (broadcast hasil ke original shape) ---
# .transform() mengembalikan Series dengan PANJANG YANG SAMA seperti input.
# Berguna untuk normalisasi per group atau menghitung group-relative metrics.
df_clean['temp_group_mean'] = df_clean.groupby('status')['temperature'].transform('mean')
df_clean['temp_vs_group_mean'] = df_clean['temperature'] - df_clean['temp_group_mean']
# Contoh: seberapa jauh suhu titik ini dari rata-rata status yang sama
print("\n--- Group-relative deviation ---")
print(df_clean[['status', 'temperature', 'temp_group_mean', 'temp_vs_group_mean']].head())

# --- Time-based analysis ---
# dt accessor untuk operasi datetime pada Series.
# .dt adalah namespace yang berisi banyak properties dan methods untuk datetime.
# Properties: .year, .month, .day, .hour, .minute, .second, .dayofweek, .weekofyear
# Methods: .strftime(), .to_period(), .floor(), .ceil()
#
# Koneksi Teknik Elektro:
# - .dt.hour -> analisis load profile (peak vs off-peak hours)
# - .dt.dayofweek -> weekday vs weekend consumption pattern
# - .dt.month -> seasonal variation in renewable energy output

df_clean['hour'] = df_clean['timestamp'].dt.hour
df_clean['day_of_week'] = df_clean['timestamp'].dt.dayofweek
# dayofweek: 0=Monday, 6=Sunday

hourly_avg = df_clean.groupby('hour')['temperature'].mean()
print(f"\n--- Rata-rata suhu per jam (sample) ---")
print(hourly_avg.head())

# --- Resampling time series ---
# .resample() adalah groupby khusus untuk time series.
# Mengelompokkan data berdasarkan frekuensi waktu, lalu apply aggregation.
# 'D' = daily, 'H' = hourly, 'W' = weekly, 'M' = monthly
# .mean(), .sum(), .std(), .first(), .last(), .ohlc() (open-high-low-close)
#
# Koneksi Teknik Elektro:
# - Resampling 1-min data ke 15-min -> SCADA historian compression
# - Daily average power -> energy billing calculation
# - Monthly peak demand -> contract capacity planning
print("\n--- Resampling ke daily ---")
daily_avg = df_clean.resample('D', on='timestamp')['temperature'].mean()
print(daily_avg.head())
# 'D' = calendar day, '6H' = 6-hourly, '15min' = 15-minutely
# on='timestamp' -> kolom datetime yang menjadi acuan resampling

# --- Pivot table ---
# Pivot table = spreadsheet-style aggregation.
# Mengubah data dari long format ke wide format.
# index = baris, columns = kolom, values = data, aggfunc = aggregation
pivot = df_clean.pivot_table(
    values='temperature',
    index='hour',
    columns='status',
    aggfunc='mean'
).round(2)
print("\n--- Pivot: rata-rata suhu per jam dan status ---")
print(pivot)
# Hasil: rows=hour (0-23), cols=status (fault, normal, warning), values=mean temp


# ===========================================================
# BAGIAN 4: Feature Engineering dengan Pandas
# ===========================================================
# Ini KUNCI untuk ML - model hanya sebagus fitur-fiturnya!
# Feature engineering = mengubah raw data menjadi representasi
# yang lebih informatif untuk model.
#
# Prinsip-prinsip feature engineering:
# 1. Domain knowledge -> fitur yang meaningful
# 2. Interaksi fitur -> kombinasi dari existing features
# 3. Transformasi matematis -> log, sqrt, power, dll.
# 4. Aggregasi temporal -> rolling, expanding, lag
# 5. Encoding -> mengubah kategorikal menjadi numerik
#
# Koneksi Teknik Elektro:
# - Rolling stats = moving average filter di signal processing
# - Lag features = tapped delay line (FIR filter structure)
# - Rate of change = derivative approximation (numerical differentiation)
# - RMS = root mean square (standard power measurement)

# --- Rolling statistics (moving average, moving std) ---
# Rolling window menghitung statistik pada window yang bergeser (sliding).
# Konsep ini identik dengan Moving Average (MA) filter di DSP.
# window=5 artinya setiap nilai adalah statistik dari 5 data terakhir.
#
# Parameter rolling:
# - window: ukuran window (int) atau time period (offset string seperti '2H')
# - min_periods: minimum observations yang dibutuhkan untuk menghasilkan hasil
# - center: jika True, window di-center di titik saat ini (bukan trailing)
# - win_type: jenis window ('boxcar'=uniform, 'triang', 'gaussian')
#
# Rolling vs Expanding:
# - Rolling: fixed window size (hanya N data terakhir)
# - Expanding: growing window (semua data dari awal sampai sekarang)
#
# Aplikasi di ML:
# - Smoothing noisy sensor data
# - Menangkap trend lokal (short-term behavior)
# - Features untuk time series forecasting
df_clean['temp_rolling_mean_5'] = df_clean['temperature'].rolling(window=5, min_periods=1).mean()
df_clean['temp_rolling_std_5'] = df_clean['temperature'].rolling(window=5, min_periods=1).std()
df_clean['temp_rolling_max_5'] = df_clean['temperature'].rolling(window=5, min_periods=1).max()
# min_periods=1 -> menghasilkan output meski data kurang dari window size
# Tanpa min_periods, baris pertama < window akan NaN

# Window berbeda untuk multi-scale analysis
df_clean['temp_rolling_mean_10'] = df_clean['temperature'].rolling(window=10, min_periods=1).mean()
# Multiple window sizes -> capture different time scales
# Short window (3-5): fast response, noisy
# Long window (20-50): slow response, smooth

# Expanding window (semua data dari start sampai current)
df_clean['temp_expanding_mean'] = df_clean['temperature'].expanding().mean()
# Expanding mean = cumulative average
# Di control systems: ini mirip integral action (semua history mempengaruhi output)

# --- Lag features (untuk time series prediction) ---
# Lag feature = nilai dari waktu sebelumnya.
# Ini memungkinkan model "melihat" history untuk prediksi.
# shift(1) = nilai dari 1 timestep sebelumnya (t-1)
# shift(3) = nilai dari 3 timestep sebelumnya (t-3)
# shift(-1) = nilai dari 1 timestep KE DEPAN (t+1) -> jangan dipakai sebagai fitur (data leakage!)
#
# Koneksi Teknik Elektro:
# - Lag features = tapped delay line di digital filter (FIR/IIR)
# - shift(1), shift(2), ... = z^(-1), z^(-2), ... di Z-domain
# - Autoregressive model: y(t) = a1*y(t-1) + a2*y(t-2) + ... + noise

df_clean['temp_lag_1'] = df_clean['temperature'].shift(1)
df_clean['temp_lag_3'] = df_clean['temperature'].shift(3)
df_clean['volt_lag_1'] = df_clean['voltage'].shift(1)

# --- Rate of change (derivative approximation) ---
# diff() = selisih dengan baris sebelumnya (first difference).
# diff(1) = x(t) - x(t-1) -> discrete derivative (backward difference)
# diff(2) = x(t) - x(t-2) -> second-order difference
# diff() / dt = approximate derivative jika dt konstan
#
# Koneksi Teknik Elektro:
# - diff() = backward difference approximation: dy/dt ~ (y[n] - y[n-1]) / T_s
# - Di control systems: rate limiter menggunakan derivative
# - Di protection: rate-of-change of frequency (ROCOF) relay
df_clean['temp_diff'] = df_clean['temperature'].diff()
df_clean['temp_diff_2'] = df_clean['temperature'].diff(2)
# diff_2 lebih smooth (less noise-sensitive) tapi lebih lambat (more delay)

# --- Percentage change ---
# pct_change() = (x(t) - x(t-1)) / x(t-1) -> relative change
# Berguna untuk data dengan skala berbeda (persentase lebih comparable)
df_clean['temp_pct_change'] = df_clean['temperature'].pct_change()

# --- Encode categorical variable ---
# Model ML membutuhkan input numerik.
# Encoding strategies:
# 1. Label encoding: kategori -> integer (0, 1, 2, ...)
#    - Baik untuk ordinal categories (low, medium, high)
#    - Buruk untuk nominal categories (red, green, blue) -> false ordering
# 2. One-hot encoding: kategori -> kolom biner (0/1)
#    - Baik untuk nominal categories
#    - Menambah dimensionalitas (N kategori -> N kolom)
#    - Drop_first=True untuk menghindari multicollinearity (dummy variable trap)
# 3. Target encoding: kategori -> mean target value per kategori
#    - Baik untuk high cardinality (banyak kategori unik)
#    - Risk: overfitting jika tidak regularized

# Label encoding dengan .map()
df_clean['status_encoded'] = df_clean['status'].map({
    'normal': 0, 'warning': 1, 'fault': 2
})
# .map() menerjemahkan setiap nilai sesuai dictionary.
# Nilai yang tidak ada di dictionary akan jadi NaN (hati-hati!).
# Alternatif: pd.factorize() untuk auto-label encoding.

# One-hot encoding dengan pd.get_dummies()
# prefix='status' -> nama kolom jadi status_normal, status_warning, status_fault
# drop_first=False -> semua kategori dibuat kolom (default)
status_dummies = pd.get_dummies(df_clean['status'], prefix='status')
df_featured = pd.concat([df_clean, status_dummies], axis=1)
# pd.concat() menggabungkan DataFrame secara horizontal (axis=1) atau vertikal (axis=0)
# ignore_index=True -> reset index setelah concat

print("\n--- DataFrame dengan features baru ---")
display_cols = ['temperature', 'temp_rolling_mean_5', 'temp_lag_1',
                'temp_diff', 'status_encoded', 'status_normal', 'status_warning']
print(df_featured[display_cols].head(10))

# --- Binning / Discretization ---
# Mengubah variabel kontinu menjadi kategorikal (interval).
# Berguna untuk:
# - Mengurangi noise (smoothing)
# - Menangkap non-linear relationship
# - Membuat interpretasi lebih mudal
# - Decision trees bekerja lebih baik dengan binned features (kadang)
#
# pd.cut(): equal-width bins (range sama)
# pd.qcut(): equal-frequency bins (jumlah sample sama per bin)
df_featured['temp_binned'] = pd.cut(
    df_featured['temperature'],
    bins=[-np.inf, 20, 25, 30, np.inf],
    labels=['cold', 'cool', 'warm', 'hot']
)
# bins: batas-batas interval (left edge, right edge, ...]
# labels: nama untuk setiap bin
# right=True -> interval (a, b] (a excluded, b included)
print(f"\n--- Temperature bins ---")
print(df_featured['temp_binned'].value_counts())



# ===========================================================
# BAGIAN 5: Persiapan Data untuk ML
# ===========================================================
# Setelah feature engineering, kita perlu memisahkan
# features (X) dan target (y), lalu split train/test.
#
# Prinsip-prinsip penting:
# 1. Feature matrix X: semua kolom yang menjadi input model
# 2. Target vector y: kolom yang diprediksi (label)
# 3. Train/test split: pisahkan SEBELUM training
# 4. No data leakage: jangan gunakan informasi dari test set saat training
#
# --- Feature matrix (X) dan target (y) ---
# Feature cols = kolom yang akan dipakai sebagai input model.
# Biasanya kita drop kolom yang:
# - Tidak informatif (ID, timestamp raw, nama)
# - Redundant (multicollinearity tinggi)
# - Target leakage (informasi dari masa depan)
# - Terlalu banyak missing values (>50%)
feature_cols = ['temperature', 'humidity', 'voltage',
                'temp_rolling_mean_5', 'temp_rolling_std_5',
                'temp_lag_1', 'temp_lag_3', 'temp_diff',
                'hour', 'status_encoded']

# Drop rows dengan NaN (dari rolling/lag features)
# subset=feature_cols artinya hanya cek NaN di kolom tersebut.
# Kalau ada NaN di kolom lain yang tidak dipakai, tidak masalah.
df_ml = df_featured.dropna(subset=feature_cols + ['status_encoded'])
# how='any' -> drop jika ADA NaN di salah satu kolom subset

# X = feature matrix (input)
X = df_ml[feature_cols].values  # .values mengkonversi ke numpy array
# .values menghasilkan NumPy array tanpa index dan column labels
# Untuk retain column names, gunakan .to_numpy()

# y = target vector (output)
y = df_ml['status_encoded'].values

print(f"\n--- Data siap untuk ML ---")
print(f"X shape: {X.shape}")
print(f"  -> {X.shape[0]} samples, {X.shape[1]} features")
print(f"y shape: {y.shape}")
print(f"Distribusi kelas: {np.bincount(y.astype(int))}")
# np.bincount menghitung frekuensi setiap integer value.
# Output: [count_class_0, count_class_1, count_class_2, ...]
# Contoh: [50, 30, 20] artinya 50 normal, 30 warning, 20 fault

# Class imbalance check
class_counts = np.bincount(y.astype(int))
class_props = class_counts / len(y)
print(f"Proporsi kelas:")
for i, (count, prop) in enumerate(zip(class_counts, class_props)):
    print(f"  Kelas {i}: {count} samples ({prop:.1%})")

# --- Train-test split (manual, nanti pakai sklearn) ---
# Kenapa split? Untuk evaluasi model pada data yang belum pernah dilihat.
# Jika kita evaluasi pada data training, hasilnya terlalu optimistik (overfitting).
#
# Random permutation = mengacak urutan data.
# Ini penting jika data ada ordering (time series, sorted by target).
# Jika data tidak di-shuffle, test set mungkin hanya berisi kelas tertentu.
#
# Stratification = memastikan proporsi kelas di train dan test sama.
# Penting untuk imbalanced datasets!
# Manual split di bawah ini TIDAK stratified -> untuk production, gunakan sklearn.

np.random.seed(42)
n = len(X)
indices = np.random.permutation(n)
# np.random.permutation(n) menghasilkan array [0, 1, ..., n-1] yang diacak

train_size = int(0.8 * n)  # 80% training, 20% testing
# Biasanya 70/30, 80/20, atau 90/10 tergantung ukuran dataset
# Dataset kecil -> lebih banyak training (90/10)
# Dataset besar -> 80/20 atau 70/30 cukup

X_train = X[indices[:train_size]]
X_test = X[indices[train_size:]]
y_train = y[indices[:train_size]]
y_test = y[indices[train_size:]]

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# Verifikasi proporsi kelas di train dan test
print(f"Train class distribution: {np.bincount(y_train.astype(int))}")
print(f"Test class distribution:  {np.bincount(y_test.astype(int))}")

# --- Feature scaling (Z-score normalization) ---
# Normalisasi HARUS dilakukan SETELAH split!
# Mean dan std dihitung HANYA dari training data.
# Test data di-transform menggunakan training statistics.
# Ini mencegah data leakage (test data tidak boleh mempengaruhi training).
#
# Kenapa scaling penting?
# - Gradient descent lebih cepat konvergen
# - Regularization bekerja merata
# - Distance-based algorithms tidak bias

train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
# axis=0 -> per kolom (fitur)

# Add epsilon untuk menghindari division by zero
eps = 1e-8
X_train_scaled = (X_train - train_mean) / (train_std + eps)
X_test_scaled = (X_test - train_mean) / (train_std + eps)

print(f"\n--- Setelah scaling ---")
print(f"Train mean (harus ~0): {X_train_scaled.mean(axis=0).round(4)}")
print(f"Train std  (harus ~1): {X_train_scaled.std(axis=0).round(4)}")
# Test set mungkin tidak exactly mean=0, std=1 karena dihitung dari train stats

# --- Cross-validation (konsep) ---
# K-fold CV = membagi training set ke K fold, training di K-1 fold, validate di 1 fold.
# Ulang K kali sehingga setiap fold menjadi validation sekali.
# Hasil: K metric scores -> rata-rata + std untuk estimasi performa yang robust.
# Lebih reliable daripada single train/validation split.
# Nanti di sklearn: from sklearn.model_selection import cross_val_score, KFold


# ===========================================================
# LATIHAN 2: Eksplorasi Dataset Publik
# ===========================================================
"""
TARGET Learning Objectives:
   - Melakukan full EDA (Exploratory Data Analysis) pada dataset nyata
   - Menerapkan data cleaning, feature engineering, dan preparation
   - Mengembangkan intuisi data dengan menulis insight

PANDUAN LANGKAH-LANGKAH:

STEP 1: Pilih Dataset
---------------------
Download salah satu dataset ini:

   Opsi A (EE-related): UCI Power Consumption Dataset
   https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
   
   Opsi B (General): Titanic Dataset
   ```python
   import seaborn as sns
   df = sns.load_dataset('titanic')
   ```
   
   TIPS: Rekomendasi: Mulai dengan Titanic (lebih simpel, dokumentasi banyak)


STEP 2: Load dan Inspect Data
-----------------------------
Lakukan initial inspection secara sistematis:

   a) Cek shape: df.shape -> berapa baris dan kolom?
      - Dataset kecil (<1000): semua algoritma bisa jalan
      - Dataset medium (1000-100k): standard ML algorithms
      - Dataset besar (>100k): perlu optimasi dan scalable methods
   
   b) Cek tipe data: df.dtypes -> apakah ada yang salah tipe?
      - Object yang seharusnya datetime -> pd.to_datetime()
      - Object yang seharusnya category -> .astype('category')
      - Numeric yang disimpan sebagai string -> .astype(float)
   
   c) Cek missing values: df.isnull().sum() -> berapa persen missing?
      - <5%: bisa drop atau simple imputation
      - 5-30%: perlu strategi imputation yang thoughtful
      - >50%: pertimbangkan drop kolom
   
   d) Cek duplicate rows: df.duplicated().sum()
      - Duplikat bisa mendistorsi statistik dan overfitting
   
   e) Lihat sample data: df.head(10) dan df.tail(10)
      - Head: apakah format data sesuai harapan?
      - Tail: apakah ada data yang anomali di akhir?
      - Cek apakah ada index yang tidak monoton -> sorting issue
   
   TIPS KENAPA penting?
     - Shape memberi gambaran ukuran dataset
     - dtypes menunjukkan apakah perlu konversi tipe data
     - Missing values menentukan strategi cleaning
     - Duplicates = data quality issue yang sering terlewat


STEP 3: Bersihkan Data
----------------------
   a) Handle missing values:
      - Numerik: mean, median, atau interpolate (pilih yang paling sesuai)
        * Mean: data symmetric, tidak ada outlier ekstrem
        * Median: data skewed atau ada outlier (robust)
        * Interpolate: time series data
      - Kategorikal: mode (nilai paling sering) atau 'Unknown'
        * Mode: jika ada kategori dominan
        * 'Unknown': jika missing itu sendiri informatif
      - Drop kolom jika missing > 50% (kecuali kolom sangat penting)
      
   b) Handle outliers:
      - Gunakan IQR method (seperti contoh di atas)
      - Atau Z-score method: |z| > 3 dianggap outlier
      - Jangan hapus outlier tanpa paham kenapa ada outlier!
        * Outlier bisa jadi noise -> hapus
        * Outlier bisa jadi event penting -> pertahankan dan investigasi
      
   c) Fix tipe data:
      - Convert string dates ke datetime: pd.to_datetime()
      - Convert categorical ke category dtype: df['col'].astype('category')
        * Category dtype hemat memory untuk data dengan banyak repetisi
      
   TIPS KENAPA penting?
     - Missing values bisa menyebabkan error di model (sklearn tidak terima NaN)
     - Outlier bisa mendistorsi statistik dan model (terutama mean-based)
     - Tipe data yang salah = komputasi tidak efisien dan hasil salah


STEP 4: Buat Minimal 5 Fitur Baru yang Meaningful
-------------------------------------------------
   Contoh untuk Titanic:
   a) FamilySize = SibSp + Parch + 1 (total family members)
      - Insight: orang dengan keluarga besar mungkin berbeda survival rate
   b) IsAlone = 1 jika FamilySize == 1, else 0
      - Insight: traveling alone vs with family
   c) AgeGroup = binning age ke 'Child', 'Adult', 'Senior'
      - Insight: kebijakan 'women and children first'
   d) FarePerPerson = Fare / FamilySize
      - Insight: harga per orang lebih informatif dari total fare
   e) Title = extract dari Name (Mr, Mrs, Miss, etc.)
      - Insight: Title mengindikasikan social status dan gender
   
   TIPS KENAPA penting?
     - Fitur baru bisa menangkap pattern yang tersembunyi
     - Domain knowledge -> meaningful features
     - Contoh: 'Title' dari nama bisa mengindikasikan social status
     - Feature engineering sering lebih impactful daripada model tuning


STEP 5: Visualisasi
-------------------
   Buat minimal 4 visualisasi:
   a) Distribusi target variable (count plot / pie chart)
      - Cek class imbalance
   b) Correlation heatmap antar fitur numerik
      - Identifikasi multicollinearity (fitur yang redundant)
      - Korelasi tinggi dengan target = fitur penting
   c) Box plot fitur numerik vs target kategorikal
      - Lihat perbedaan distribusi per kelas
      - Identifikasi fitur yang bisa memisahkan kelas
   d) Bar plot fitur kategorikal vs target
      - Proporsi target per kategori
      - Cari kategori yang strongly correlated dengan target
   
   TIPS KENAPA penting?
     - Visualisasi menunjukkan pattern yang tidak terlihat di angka
     - Correlation menunjukkan redundancy atau multicollinearity
     - Box plot menunjukkan perbedaan median dan spread antar kelas


STEP 6: Siapkan X dan y untuk ML
---------------------------------
   a) Pilih feature columns (drop ID, name, raw timestamps, dll.)
   b) Encode categorical variables (one-hot atau label encoding)
      - One-hot untuk nominal (tidak ada urutan)
      - Label encoding untuk ordinal (ada urutan alami)
   c) Pisahkan X (features) dan y (target)
   d) Split train/test (80/20) dengan random shuffle
      - Stratified split jika imbalanced (gunakan sklearn)
   
   TIPS KENAPA penting?
     - Data harus dalam format numerik untuk model ML
     - Train/test split mencegah overfitting pada data training
     - Stratification memastikan representasi kelas sama di train dan test


TIPS:
   - Gunakan df['col'].fillna() untuk mengisi missing
   - Gunakan pd.cut() untuk binning continuous variable
   - Gunakan .groupby('target').mean() untuk lihat perbedaan per kelas
   - Simpan visualisasi dengan plt.savefig()
   - Gunakan df['col'].astype('category') untuk hemat memory

PERINGATAN COMMON MISTAKES:
   - Melakukan imputation SEBELUM train/test split -> data leakage!
     Imputation statistics (mean, median) harus dihitung dari training data saja.
   - Mengabaikan missing values -> error saat training (sklearn raise error)
   - Menggunakan label encoding untuk nominal categories -> false ordering
     Contoh: [red=0, green=1, blue=2] -> model pikir blue > green > red
   - Tidak random shuffle sebelum split -> bias jika data sorted
   - Melakukan feature scaling sebelum split -> data leakage!

PENTING: Tulis INSIGHT, bukan cuma kode!
Contoh insight yang baik:
   "Voltage drop > 10V berkorelasi dengan status 'fault' -
   ini masuk akal karena fault biasanya menyebabkan voltage sag."
   
   "Passengers with Title 'Master' (male children) have higher survival
   rate than adult men, suggesting 'women and children first' policy."
   
   "Rolling mean temperature dengan window 6 jam menunjukkan trend
   yang lebih smooth daripada raw data, mengurangi noise dari sensor."
"""


# ===========================================================
# 🔥 CHALLENGE: Pipeline Otomatis
# ===========================================================
"""
TARGET Learning Objectives:
   - Membangun reusable data pipeline class
   - Memahami workflow end-to-end dari raw data ke ML-ready
   - Menyiapkan infrastructure yang bisa dipakai di semua project

PANDUAN LANGKAH-LANGKAH:

STEP 1: Design Class Structure
------------------------------
Buat class DataPipeline dengan interface berikut:

   class DataPipeline:
       def __init__(self, df):
           '''Terima raw DataFrame sebagai input'''
           self.df_raw = df.copy()
           self.df = df.copy()
           
       def inspect(self):
           '''Print summary: shape, missing, dtypes, duplicates'''
           
       def clean(self, strategy='interpolate'):
           '''Handle missing values berdasarkan strategi'''
           # Support: 'drop', 'mean', 'median', 'interpolate', 'mode'
           
       def add_rolling_features(self, columns, windows):
           '''Tambah rolling stats (mean, std, min, max)'''
           
       def add_lag_features(self, columns, lags):
           '''Tambah lag features'''
           
       def encode_categorical(self, columns):
           '''One-hot encode categorical columns'''
           
       def prepare_ml(self, target_col, feature_cols, test_ratio=0.2):
           '''Return X_train, X_test, y_train, y_test'''
           # JANGAN LUPA: split dulu, baru normalize/impute!


STEP 2: Implementasi Method .inspect()
--------------------------------------
   Print informasi berikut secara terstruktur:
   - Shape (baris, kolom)
   - Missing values per kolom (count dan percentage)
   - Tipe data setiap kolom
   - Jumlah duplicate rows
   - Statistik dasar (mean, std, min, max) untuk numerik
   - Jumlah kategori unik untuk kolom kategorikal
   
   TIPS KENAPA penting?
     - Setiap kali dapat dataset baru, inspect dulu!
     - Menghindari asumsi yang salah tentang data
     - Mencegah error di tengah pipeline karena tipe data salah


STEP 3: Implementasi Method .clean()
------------------------------------
   Support multiple strategies dengan parameter:
   - 'drop': drop rows dengan missing values
   - 'mean': isi dengan mean per kolom (numerik)
   - 'median': isi dengan median per kolom (numerik)
   - 'interpolate': interpolasi linear (untuk time series)
   - 'mode': isi dengan nilai paling sering (untuk kategorikal)
   - 'ffill': forward fill (isi dengan nilai sebelumnya)
   - 'bfill': backward fill (isi dengan nilai sesudahnya)
   
   Implementasi tips:
   - Pisahkan handling untuk numerik dan kategorikal
   - Untuk numerik: mean/median/interpolate
   - Untuk kategorikal: mode/ffill/bfill
   - Untuk datetime: ffill/bfill atau interpolate(method='time')
   
   TIPS KENAPA penting?
     - Strategi yang berbeda untuk tipe data yang berbeda
     - Time series -> interpolate, kategorikal -> mode
     - FFill cocok untuk data yang jarang berubah (status on/off)


STEP 4: Implementasi Feature Engineering Methods
-------------------------------------------------
   .add_rolling_features(columns, windows):
   - Untuk setiap kolom di 'columns', buat rolling features
   - Untuk setiap window di 'windows', buat rolling_mean dan rolling_std
   - Contoh: columns=['temperature'], windows=[3, 6, 12]
     -> temp_rolling_mean_3, temp_rolling_std_3, temp_rolling_mean_6, ...
   - Gunakan min_periods=1 agar tidak ada NaN di awal
   
   .add_lag_features(columns, lags):
   - Untuk setiap kolom dan setiap lag, buat lag feature
   - Contoh: columns=['temperature'], lags=[1, 3, 6]
     -> temp_lag_1, temp_lag_3, temp_lag_6
   - Pertimbangkan: untuk prediksi t-step ahead, lag harus >= t
     (jangan pakai lag 1 untuk prediksi 3-step ahead -> data leakage!)


STEP 5: Implementasi .prepare_ml()
----------------------------------
   a) Validasi: pastikan target_col dan feature_cols ada di DataFrame
      - Raise ValueError jika tidak ada
   b) Drop rows dengan NaN di feature_cols atau target_col
   c) Split train/test menggunakan random permutation
      - Gunakan np.random.permutation untuk shuffle
      - Parameter stratify (optional): jika imbalanced, gunakan sklearn
   d) Return X_train, X_test, y_train, y_test sebagai numpy arrays
   
   PERINGATAN: JANGAN normalize di sini!
     Normalisasi harus dilakukan SETELAH split, dan mean/std
     dihitung HANYA dari training data untuk mencegah data leakage.
     Pipeline ini hanya bertugas split, bukan scaling.


STEP 6: Testing
---------------
   Test pipeline kamu dengan dataset sintetis:
   ```python
   df_test = pd.DataFrame({
       'timestamp': pd.date_range('2024-01-01', periods=50, freq='h'),
       'A': [1, 2, np.nan, 4, 5] * 10,
       'B': ['x', 'y', 'x', 'y', 'x'] * 10,
       'target': [0, 1, 0, 1, 0] * 10
   })
   
   pipeline = DataPipeline(df_test)
   pipeline.inspect()
   pipeline.clean(strategy='mean')
   pipeline.add_rolling_features(['A'], [3, 5])
   pipeline.add_lag_features(['A'], [1, 2])
   pipeline.encode_categorical(['B'])
   X_train, X_test, y_train, y_test = pipeline.prepare_ml(
       'target', ['A', 'A_rolling_mean_3', 'A_lag_1', 'B_x', 'B_y']
   )
   print(f"Train: {X_train.shape}, Test: {X_test.shape}")
   ```


TIPS:
   - Gunakan self.df = df.copy() di __init__ untuk avoid modifying original
   - Gunakan getattr() untuk dynamic method call berdasarkan strategy
     Contoh: strategy_func = getattr(self.df, strategy) untuk 'ffill', 'bfill'
   - Simpan state di self (misal: self.df_clean, self.df_features)
   - Return self dari setiap method untuk method chaining (fluent interface)
     Contoh: pipeline.clean().add_features().prepare_ml()

PERINGATAN COMMON MISTAKES:
   - Modifying original DataFrame (selalu pakai .copy())
   - Data leakage: normalize sebelum split
   - Lupa handle NaN yang baru muncul dari rolling/lag features
   - Tidak validasi input (kolom yang tidak ada di DataFrame)
   - Lupa random seed -> hasil tidak reproducible

TARGET EXPECTED OUTPUT:
   - Pipeline yang bisa dipakai ulang di semua project
   - Code yang clean dan well-documented
   - Test cases yang passing
   - Method chaining yang smooth

Kenapa bikin class sendiri? Karena di dunia nyata, data pipeline = 80% waktu ML.
Lebih baik punya pipeline yang solid daripada model yang fancy.
Pipeline yang baik bisa menghemat waktu berjam-jam di setiap project baru.
"""

print("\n" + "="*50)
print("OK Modul 2 selesai! Lanjut ke: 01-fondasi-data/03_visualisasi.py")
print("="*50)
