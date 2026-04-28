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

Koneksi Teknik Elektro:
- DataFrame = tabel measurement dari multiple sensor
- Time series = logged data dari SCADA/DAS
- Groupby = aggregate statistics per equipment/motor

Durasi target: 2-3 jam
=============================================================
"""

import numpy as np
import pandas as pd

# ===========================================================
# 📖 BAGIAN 1: Membuat & Membaca Data
# ===========================================================
# DataFrame adalah struktur data 2D labeled (mirip spreadsheet).
# Label pada baris = index, label pada kolom = columns.
# Setiap kolom bisa memiliki tipe data yang berbeda.

# --- Membuat DataFrame dari dictionary ---
# Dictionary keys menjadi nama kolom.
# Dictionary values menjadi data kolom (list atau array).
# pd.date_range() menghasilkan sequence datetime yang evenly spaced.
sensor_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
    # freq='h' = hourly frequency
    'temperature': np.random.normal(25, 5, 100),
    # np.random.normal(mean, std, size) → Gaussian distribution
    'humidity': np.random.normal(60, 10, 100),
    'voltage': np.random.normal(220, 5, 100),
    'status': np.random.choice(['normal', 'warning', 'fault'], 100, p=[0.7, 0.2, 0.1])
    # np.random.choice dengan probabilities p
})

# --- Menambahkan missing values secara sengaja ---
# Kenapa? Karena real-world data SELALU punya missing values!
# Masking: memilih subset data berdasarkan kondisi boolean.
# np.random.random(100) < 0.05 menghasilkan array boolean
# dengan ~5% elemen True (random).
mask = np.random.random(100) < 0.05
sensor_data.loc[mask, 'temperature'] = np.nan
# .loc[indexer, column] = access by label
# np.nan = Not a Number (representasi missing value di NumPy/Pandas)
sensor_data.loc[np.random.random(100) < 0.03, 'voltage'] = np.nan

print("📊 Dataset sensor:")
print(sensor_data.head(10))
# .head(n) = menampilkan n baris pertama
print(f"\nShape: {sensor_data.shape}")
# .shape = (n_rows, n_columns)
print(f"\nInfo:")
print(sensor_data.info())
# .info() = summary tipe data dan non-null count per kolom
print(f"\nStatistik deskriptif:")
print(sensor_data.describe())
# .describe() = statistik untuk kolom numerik (count, mean, std, min, 25%, 50%, 75%, max)


# ===========================================================
# 📖 BAGIAN 2: Data Inspection & Cleaning
# ===========================================================
# Data cleaning = 60-80% waktu di project ML!
# Quality data > fancy model.

# --- Cek missing values ---
# isnull() menghasilkan DataFrame boolean (True jika NaN).
# .sum() menghitung jumlah True per kolom.
print("\n--- Missing Values ---")
print(sensor_data.isnull().sum())
print(f"Total missing: {sensor_data.isnull().sum().sum()}")

# --- Handle missing values — beberapa strategi ---
# Strategi 1: Drop rows (kehilangan data, tapi simpel)
# Gunakan jika missing < 5% dan data cukup besar.
# df_clean = sensor_data.dropna()

# Strategi 2: Fill dengan mean (paling umum untuk numerik)
# Gunakan jika data missing random dan distribusi normal.
# df_clean = sensor_data.fillna(sensor_data.mean(numeric_only=True))

# Strategi 3: Interpolate (BAGUS untuk time series / sensor data!)
# Interpolasi memperkirakan nilai missing berdasarkan tetangga.
# Method 'linear' = garis lurus antar 2 titik known.
# Ini sangat cocok untuk sensor data karena sinyal umumnya smooth.
df_clean = sensor_data.copy()
df_clean['temperature'] = df_clean['temperature'].interpolate(method='linear')
df_clean['voltage'] = df_clean['voltage'].interpolate(method='linear')

print(f"\nSetelah interpolasi, missing: {df_clean.isnull().sum().sum()}")

# --- Deteksi outliers (IQR method) ---
# Outlier = nilai yang sangat berbeda dari mayoritas.
# IQR (Interquartile Range) = Q3 - Q1
# Outlier = nilai < Q1 - 1.5*IQR atau > Q3 + 1.5*IQR
# Rumus ini berasal dari boxplot (Tukey's fences).

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
    - Koneksi ke Teknik Elektro: mirip dengan threshold detection
      di fault monitoring systems
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)


outlier_mask = detect_outliers_iqr(df_clean['temperature'])
print(f"\nOutliers di temperature: {outlier_mask.sum()}")
# .sum() pada boolean Series = menghitung True


# ===========================================================
# 📖 BAGIAN 3: Filtering, Grouping, Aggregation
# ===========================================================
# Operasi-operasi ini fundamental untuk EDA (Exploratory Data Analysis).

# --- Filtering ---
# Memilih baris berdasarkan kondisi boolean.
# Gunakan & untuk AND, | untuk OR, ~ untuk NOT.
# Setiap kondisi harus di-wrap dalam parentheses.
faults = df_clean[df_clean['status'] == 'fault']
print(f"\n--- Fault records: {len(faults)} ---")

# Multiple conditions
high_temp_faults = df_clean[
    (df_clean['status'] == 'fault') &
    (df_clean['temperature'] > 30)
]
print(f"Fault dengan temp > 30°C: {len(high_temp_faults)}")

# --- Groupby — analisis per kategori ---
# Split-Apply-Combine strategy:
# 1. Split data into groups berdasarkan key (status)
# 2. Apply function to each group (mean, std, min, max)
# 3. Combine results into new DataFrame
print("\n--- Statistik per Status ---")
grouped = df_clean.groupby('status').agg({
    'temperature': ['mean', 'std', 'min', 'max'],
    'voltage': ['mean', 'std'],
    'humidity': 'mean'
}).round(2)
# .agg() = aggregate dengan multiple functions
# Hasilnya adalah MultiIndex columns
print(grouped)

# --- Time-based analysis ---
# dt accessor untuk operasi datetime pada Series.
# .dt.hour mengambil jam dari timestamp.
df_clean['hour'] = df_clean['timestamp'].dt.hour
hourly_avg = df_clean.groupby('hour')['temperature'].mean()
print(f"\n--- Rata-rata suhu per jam (sample) ---")
print(hourly_avg.head())


# ===========================================================
# 📖 BAGIAN 4: Feature Engineering dengan Pandas
# ===========================================================
# Ini KUNCI untuk ML — model hanya sebagus fitur-fiturnya!
# Feature engineering = mengubah raw data menjadi representasi
# yang lebih informatif untuk model.

# --- Rolling statistics (moving average, moving std) ---
# Rolling window menghitung statistik pada window sliding.
# Ini penting untuk time series karena menangkap trend lokal.
# window=5 artinya setiap nilai adalah average dari 5 data terakhir.
df_clean['temp_rolling_mean'] = df_clean['temperature'].rolling(window=5).mean()
df_clean['temp_rolling_std'] = df_clean['temperature'].rolling(window=5).std()
# Rolling menghasilkan NaN untuk baris pertama < window size

# --- Lag features (untuk time series prediction) ---
# Lag feature = nilai dari waktu sebelumnya.
# Ini memungkinkan model "melihat" history.
# shift(1) = nilai dari 1 timestep sebelumnya.
# shift(3) = nilai dari 3 timestep sebelumnya.
df_clean['temp_lag_1'] = df_clean['temperature'].shift(1)
df_clean['temp_lag_3'] = df_clean['temperature'].shift(3)

# --- Rate of change ---
# diff() = selisih dengan baris sebelumnya.
# Ini menangkap kecepatan perubahan (derivative approximation).
df_clean['temp_diff'] = df_clean['temperature'].diff()

# --- Encode categorical variable ---
# Model ML membutuhkan input numerik.
# Label encoding = mengubah kategori menjadi integer.
df_clean['status_encoded'] = df_clean['status'].map({
    'normal': 0, 'warning': 1, 'fault': 2
})
# .map() menerjemahkan setiap nilai sesuai dictionary.

# --- One-hot encoding ---
# One-hot = representasi biner untuk kategori.
# Setiap kategori menjadi kolom tersendiri (0 atau 1).
# pd.get_dummies() otomatis membuat kolom baru.
status_dummies = pd.get_dummies(df_clean['status'], prefix='status')
df_featured = pd.concat([df_clean, status_dummies], axis=1)
# pd.concat() menggabungkan DataFrame secara horizontal (axis=1)

print("\n--- DataFrame dengan features baru ---")
print(df_featured[['temperature', 'temp_rolling_mean', 'temp_lag_1',
                    'temp_diff', 'status_encoded']].head(10))


# ===========================================================
# 📖 BAGIAN 5: Persiapan Data untuk ML
# ===========================================================
# Setelah feature engineering, kita perlu memisahkan
# features (X) dan target (y), lalu split train/test.

# --- Feature matrix (X) dan target (y) ---
# Feature cols = kolom yang akan dipakai sebagai input model.
# Biasanya kita drop kolom yang tidak informatif atau redundant.
feature_cols = ['temperature', 'humidity', 'voltage',
                'temp_rolling_mean', 'temp_rolling_std',
                'temp_lag_1', 'temp_lag_3', 'temp_diff']

# Drop rows dengan NaN (dari rolling/lag features)
# subset=feature_cols artinya hanya cek NaN di kolom tersebut.
df_ml = df_featured.dropna(subset=feature_cols)

# X = feature matrix (input)
X = df_ml[feature_cols].values  # .values mengkonversi ke numpy array
# y = target vector (output)
y = df_ml['status_encoded'].values

print(f"\n--- Data siap untuk ML ---")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Distribusi kelas: {np.bincount(y.astype(int))}")
# np.bincount menghitung frekuensi setiap integer value.

# --- Train-test split (manual, nanti pakai sklearn) ---
# Kenapa split? Untuk evaluasi model pada data yang belum pernah dilihat.
# Random permutation = mengacak urutan data.
# Ini penting jika data ada ordering (time series, sorted).
n = len(X)
indices = np.random.permutation(n)
train_size = int(0.8 * n)  # 80% training, 20% testing

X_train = X[indices[:train_size]]
X_test = X[indices[train_size:]]
y_train = y[indices[:train_size]]
y_test = y[indices[train_size:]]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ===========================================================
# 🏋️ EXERCISE 2: Eksplorasi Dataset Publik
# ===========================================================
"""
🎯 Learning Objectives:
   - Melakukan full EDA (Exploratory Data Analysis) pada dataset nyata
   - Menerapkan data cleaning, feature engineering, dan preparation
   - Mengembangkan intuisi data dengan menulis insight

📋 LANGKAH-LANGKAH:

STEP 1: Pilih Dataset
─────────────────────
Download salah satu dataset ini:

   Opsi A (EE-related): UCI Power Consumption Dataset
   https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
   
   Opsi B (General): Titanic Dataset
   ```python
   import seaborn as sns
   df = sns.load_dataset('titanic')
   ```
   
   💡 Rekomendasi: Mulai dengan Titanic (lebih simpel, dokumentasi banyak)


STEP 2: Load dan Inspect Data
─────────────────────────────
Lakukan initial inspection:

   a) Cek shape: df.shape → berapa baris dan kolom?
   b) Cek tipe data: df.dtypes → apakah ada yang salah tipe?
   c) Cek missing values: df.isnull().sum() → berapa persen missing?
   d) Cek duplicate rows: df.duplicated().sum()
   e) Lihat sample data: df.head(10) dan df.tail(10)
   
   💡 KENAPA penting? 
     - Shape memberi gambaran ukuran dataset
     - dtypes menunjukkan apakah perlu konversi tipe data
     - Missing values menentukan strategi cleaning


STEP 3: Bersihkan Data
──────────────────────
   a) Handle missing values:
      - Numerik: mean, median, atau interpolate (pilih yang paling sesuai)
      - Kategorikal: mode (nilai paling sering) atau 'Unknown'
      - Drop kolom jika missing > 50%
      
   b) Handle outliers:
      - Gunakan IQR method (seperti contoh di atas)
      - Atau Z-score method: |z| > 3 dianggap outlier
      - Jangan hapus outlier tanpa paham kenapa ada outlier!
      
   c) Fix tipe data:
      - Convert string dates ke datetime: pd.to_datetime()
      - Convert categorical ke category dtype: df['col'].astype('category')
      
   💡 KENAPA penting?
     - Missing values bisa menyebabkan error di model
     - Outlier bisa mendistorsi statistik dan model
     - Tipe data yang salah = komputasi tidak efisien


STEP 4: Buat Minimal 5 Fitur Baru yang Meaningful
──────────────────────────────────────────────────
   Contoh untuk Titanic:
   a) FamilySize = SibSp + Parch + 1 (total family members)
   b) IsAlone = 1 jika FamilySize == 1, else 0
   c) AgeGroup = binning age ke 'Child', 'Adult', 'Senior'
   d) FarePerPerson = Fare / FamilySize
   e) Title = extract dari Name (Mr, Mrs, Miss, etc.)
   
   💡 KENAPA penting?
     - Fitur baru bisa menangkap pattern yang tersembunyi
     - Domain knowledge → meaningful features
     - Contoh: 'Title' dari nama bisa mengindikasikan social status


STEP 5: Visualisasi
───────────────────
   Buat minimal 4 visualisasi:
   a) Distribusi target variable
   b) Correlation heatmap antar fitur numerik
   c) Box plot fitur numerik vs target kategorikal
   d) Bar plot fitur kategorikal vs target
   
   💡 KENAPA penting?
     - Visualisasi menunjukkan pattern yang tidak terlihat di angka
     - Correlation menunjukkan redundancy atau multicollinearity


STEP 6: Siapkan X dan y untuk ML
─────────────────────────────────
   a) Pilih feature columns (drop ID, name, dll.)
   b) Encode categorical variables (one-hot atau label encoding)
   c) Pisahkan X dan y
   d) Split train/test (80/20)
   
   💡 KENAPA pentik?
     - Data harus dalam format numerik untuk model ML
     - Train/test split mencegah overfitting pada data training


💡 HINTS:
   - Gunakan df['col'].fillna() untuk mengisi missing
   - Gunakan pd.cut() untuk binning continuous variable
   - Gunakan .groupby('target').mean() untuk lihat perbedaan per kelas
   - Simpan visualisasi dengan plt.savefig()

⚠️ COMMON MISTAKES:
   - Melakukan imputation SEBELUM train/test split → data leakage!
   - Mengabaikan missing values → error saat training
   - Menggunakan label encoding untuk nominal categories → false ordering
   - Tidak random shuffle sebelum split → bias jika data sorted

📝 PENTING: Tulis INSIGHT, bukan cuma kode!
Contoh insight yang baik:
   "Voltage drop > 10V berkorelasi dengan status 'fault' —
   ini masuk akal karena fault biasanya menyebabkan voltage sag."
   
   "Passengers with Title 'Master' (male children) have higher survival
   rate than adult men, suggesting 'women and children first' policy."
"""


# ===========================================================
# 🔥 CHALLENGE: Pipeline Otomatis
# ===========================================================
"""
🎯 Learning Objectives:
   - Membangun reusable data pipeline class
   - Memahami workflow end-to-end dari raw data ke ML-ready
   - Menyiapkan infrastructure yang bisa dipakai di semua project

📋 LANGKAH-LANGKAH:

STEP 1: Design Class Structure
──────────────────────────────
Buat class DataPipeline dengan interface berikut:

   class DataPipeline:
       def __init__(self, df):
           '''Terima raw DataFrame sebagai input'''
           
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
──────────────────────────────────────
   Print informasi berikut:
   - Shape (baris, kolom)
   - Missing values per kolom (count dan percentage)
   - Tipe data setiap kolom
   - Jumlah duplicate rows
   - Statistik dasar (mean, std, min, max) untuk numerik
   
   💡 KENAPA penting?
     - Setiap kali dapat dataset baru, inspect dulu!
     - Menghindari asumsi yang salah tentang data


STEP 3: Implementasi Method .clean()
────────────────────────────────────
   Support multiple strategies:
   - 'drop': drop rows dengan missing values
   - 'mean': isi dengan mean per kolom
   - 'median': isi dengan median per kolom
   - 'interpolate': interpolasi linear (untuk time series)
   - 'mode': isi dengan nilai paling sering (untuk kategorikal)
   
   💡 KENAPA pentik?
     - Strategi yang berbeda untuk tipe data yang berbeda
     - Time series → interpolate, kategorikal → mode


STEP 4: Implementasi Feature Engineering Methods
─────────────────────────────────────────────────
   .add_rolling_features(columns, windows):
   - Untuk setiap kolom di 'columns', buat rolling features
   - Untuk setiap window di 'windows', buat rolling_mean dan rolling_std
   - Contoh: columns=['temperature'], windows=[3, 6, 12]
     → temp_rolling_mean_3, temp_rolling_std_3, temp_rolling_mean_6, ...
     
   .add_lag_features(columns, lags):
   - Untuk setiap kolom dan setiap lag, buat lag feature
   - Contoh: columns=['temperature'], lags=[1, 3, 6]
     → temp_lag_1, temp_lag_3, temp_lag_6


STEP 5: Implementasi .prepare_ml()
──────────────────────────────────
   a) Validasi: pastikan target_col dan feature_cols ada di DataFrame
   b) Drop rows dengan NaN di feature_cols atau target_col
   c) Split train/test menggunakan random permutation
   d) Return X_train, X_test, y_train, y_test sebagai numpy arrays
   
   ⚠️ Hati-hati: JANGAN normalize di sini!
     Normalisasi harus dilakukan SETELAH split, dan mean/std
     dihitung HANYA dari training data untuk mencegah data leakage.


STEP 6: Testing
───────────────
   Test pipeline kamu dengan dataset sintetis:
   ```python
   df_test = pd.DataFrame({
       'A': [1, 2, np.nan, 4, 5],
       'B': ['x', 'y', 'x', 'y', 'x'],
       'target': [0, 1, 0, 1, 0]
   })
   
   pipeline = DataPipeline(df_test)
   pipeline.inspect()
   pipeline.clean(strategy='mean')
   pipeline.encode_categorical(['B'])
   X_train, X_test, y_train, y_test = pipeline.prepare_ml('target', ['A', 'B_x', 'B_y'])
   ```


💡 HINTS:
   - Gunakan self.df = df.copy() di __init__ untuk avoid modifying original
   - Gunakan getattr() untuk dynamic method call berdasarkan strategy
   - Simpan state di self (misal: self.df_clean, self.df_features)
   - Return self dari setiap method untuk method chaining

⚠️ COMMON MISTAKES:
   - Modifying original DataFrame (selalu pakai .copy())
   - Data leakage: normalize sebelum split
   - Lupa handle NaN yang baru muncul dari rolling/lag features
   - Tidak validasi input (kolom yang tidak ada di DataFrame)

🎯 EXPECTED OUTPUT:
   - Pipeline yang bisa dipakai ulang di semua project
   - Code yang clean dan well-documented
   - Test cases yang passing

Kenapa bikin class sendiri? Karena di dunia nyata, data pipeline = 80% waktu ML.
Lebih baik punya pipeline yang solid daripada model yang fancy.
"""

print("\n" + "="*50)
print("✅ Modul 2 selesai! Lanjut ke: 01-fondasi-data/03_visualisasi.py")
print("="*50)
