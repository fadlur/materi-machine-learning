"""
=============================================================
FASE 1 — MODUL 1: NUMPY ESSENTIALS
=============================================================
Kenapa NumPy dulu?
- Semua library ML di Python dibangun di atas NumPy
- Memahami array operations = memahami cara kerja internal ML
- Dengan background Teknik Elektro, kamu sudah paham matrix algebra
  → ini tinggal mapping ke sintaks Python

Durasi target: 2-3 jam
=============================================================
"""

import numpy as np

# ===========================================================
# 📖 BAGIAN 1: Array Creation & Basic Operations
# ===========================================================
# Sebagai engineer, kamu sudah familiar dengan vektor dan matrix.
# NumPy array = representasi efisien dari struktur data ini.
# 
# Koneksi Teknik Elektro:
# - Vektor = sinyal 1D (contoh: tegangan terhadap waktu)
# - Matrix = sistem multi-channel (contoh: data dari multiple sensor)
# - Identity matrix = sistem tanpa cross-coupling

# --- Membuat array dari list Python ---
# np.array() mengkonversi list Python menjadi NumPy array.
# Array NumPy lebih efisien karena:
# 1. Typed (semua elemen sama tipe) → memory compact
# 2. Stored contiguously in memory → cache friendly
# 3. Supports vectorized operations → no Python loop overhead
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Vektor x: {x}")
print(f"Shape: {x.shape}, Dtype: {x.dtype}")
# shape → dimensi array, (5,) artinya 1D dengan 5 elemen
# dtype → tipe data elemen, float64 = 8 bytes per elemen

# --- Matrix 2D ---
# Matrix 2D = array dengan shape (baris, kolom)
# Di ML, setiap baris biasanya adalah satu sample,
# dan setiap kolom adalah satu fitur.
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(f"\nMatrix A:\n{A}")
print(f"Shape: {A.shape}")
# Shape (3, 3) = 3 baris, 3 kolom

# --- Array generators — sering dipakai untuk inisialisasi ---
# np.zeros: matrix nol → useful untuk inisialisasi accumulator
zeros = np.zeros((3, 4))          # matrix 3x4 dengan semua elemen 0
# np.ones: matrix satu → useful untuk bias initialization
ones = np.ones((2, 3))            # matrix 2x3 dengan semua elemen 1
# np.eye: identity matrix → fundamental di linear algebra & control theory
identity = np.eye(4)              # matrix identitas 4x4
# np.random.randn: random dari distribusi normal (Gaussian)
# → Sering dipakai untuk weight initialization di neural networks
random_normal = np.random.randn(3, 3)

# --- Linspace — familiar dari MATLAB/signal processing ---
# np.linspace(start, stop, num) menghasilkan 'num' titik yang equally spaced
# dari 'start' sampai 'stop' (inklusif).
# Ini sangat berguna untuk:
# - Sampling waktu (time vector)
# - Frequency axis untuk FFT
# - Plotting smooth curves
t = np.linspace(0, 2 * np.pi, 100)  # 100 titik dari 0 sampai 2π
signal = np.sin(t)  # sinyal sinusoidal — kamu pasti sering pakai ini

print(f"\nSinyal sinusoidal: {signal[:5]}...")  # 5 sample pertama


# ===========================================================
# 📖 BAGIAN 2: Broadcasting & Vectorization
# ===========================================================
# INI KUNCI PENTING untuk ML!
# Broadcasting = operasi antara array dengan shape berbeda
# Vectorization = hindari loop, gunakan operasi array
#
# Koneksi Teknik Elektro:
# - Broadcasting = automatic impedance matching di circuit
# - Vectorization = parallel processing di FPGA/DSP

# Contoh: normalisasi data (z-score)
# Rumus: z = (x - mean) / std
# Di statistik, ini disebut standard score atau z-score.
# Z-score mengukur berapa banyak standard deviation sebuah nilai
# berada dari mean. Ini penting karena banyak algoritma ML
# mengasumsikan data terdistribusi normal dengan mean=0, std=1.
data = np.random.randn(1000, 5)  # 1000 samples, 5 features

# CARA BURUK (loop) — jangan lakukan ini
# Kenapa buruk?
# 1. Python loops are SLOW (interpreted, not compiled)
# 2. Tidak bisa di-optimize oleh NumPy's C backend
# 3. Code lebih panjang dan sulit dibaca
# for i in range(data.shape[1]):
#     data[:, i] = (data[:, i] - data[:, i].mean()) / data[:, i].std()

# CARA BAIK (vectorized + broadcasting)
# data.shape = (1000, 5)
# mean.shape = (5,) → di-expand otomatis ke (1, 5) lalu (1000, 5)
# std.shape = (5,) → di-expand otomatis ke (1, 5) lalu (1000, 5)
# Operasi (1000, 5) - (5,) → broadcasting → (1000, 5)
mean = data.mean(axis=0)  # mean per kolom → shape (5,)
std = data.std(axis=0)    # std per kolom → shape (5,)
data_normalized = (data - mean) / std  # broadcasting: (1000,5) - (5,) → otomatis!

print(f"\nNormalized mean (harus ~0): {data_normalized.mean(axis=0).round(4)}")
print(f"Normalized std  (harus ~1): {data_normalized.std(axis=0).round(4)}")


# ===========================================================
# 📖 BAGIAN 3: Linear Algebra Operations
# ===========================================================
# Ini yang paling relevan untuk ML. Hampir semua model ML
# pada dasarnya adalah operasi linear algebra.
#
# Koneksi Teknik Elektro:
# - Matrix multiplication = cascading linear systems
# - Eigenvalue decomposition = modal analysis of dynamic systems
# - SVD = optimal low-rank approximation (source coding!)
# - Solve linear system = circuit analysis (KCL/KVL)

# --- Matrix multiplication ---
# A @ B = np.dot(A, B)
# Shape rule: (m, n) @ (n, p) → (m, p)
# Ini adalah operasi paling fundamental di ML:
# - Linear regression: y = Xw
# - Neural network layer: z = Wx + b
# - Attention mechanism: Attention = QK^T
A = np.random.randn(3, 4)
B = np.random.randn(4, 2)
C = A @ B  # atau np.dot(A, B)
print(f"\nMatrix multiplication: ({A.shape}) @ ({B.shape}) = {C.shape}")

# --- Transpose ---
# A.T membalik baris dan kolom.
# Di ML, transpose sering dipakai untuk:
# - Mengubah orientation data (samples as rows vs columns)
# - Menyesuaikan dimensi untuk matrix multiplication
print(f"A^T shape: {A.T.shape}")
# (3, 4) → (4, 3)

# --- Eigenvalue decomposition — pasti familiar dari kuliah ---
# M = V Λ V^(-1), dimana:
# - Λ adalah diagonal matrix of eigenvalues
# - V adalah matrix of eigenvectors
# Aplikasi di ML: PCA, spectral clustering, PageRank
M = np.random.randn(3, 3)
M = M @ M.T  # buat symmetric positive definite (agar eigendecomposition real)
# M @ M.T selalu menghasilkan matrix symmetric
# Symmetric matrix memiliki eigenvalues real dan eigenvectors orthogonal
eigenvalues, eigenvectors = np.linalg.eigh(M)
print(f"\nEigenvalues: {eigenvalues}")
# Eigenvalues menggambarkan "magnitude" dari setiap mode

# --- SVD — nanti akan dipakai di PCA ---
# A = U Σ V^T, dimana:
# - U: left singular vectors (orthogonal)
# - Σ: singular values (diagonal, non-negative)
# - V^T: right singular vectors (orthogonal)
# Aplikasi: PCA, image compression, collaborative filtering
U, S, Vt = np.linalg.svd(A)
print(f"SVD: U{U.shape}, S{S.shape}, Vt{Vt.shape}")
# S adalah vector singular values, bukan matrix diagonal

# --- Solve linear system: Ax = b ---
# Mencari x yang memenuhi persamaan Ax = b.
# Di circuit analysis: x = node voltages, b = current sources.
# Di ML: x = model parameters, b = targets, A = design matrix.
A_square = np.random.randn(3, 3)
b = np.random.randn(3)
x = np.linalg.solve(A_square, b)
print(f"\nSolusi Ax=b: x = {x.round(4)}")
print(f"Verifikasi (Ax harus = b): {(A_square @ x).round(4)}")
print(f"b asli:                     {b.round(4)}")


# ===========================================================
# 📖 BAGIAN 4: Indexing & Slicing (Penting untuk Data Processing)
# ===========================================================
# Indexing & slicing memungkinkan kita mengakses subset data.
# Ini fundamental untuk:
# - Train/test split
# - Batch processing
# - Feature selection

data = np.random.randn(100, 5)

# --- Basic slicing ---
# Syntax: array[start:stop:step]
# Slicing di NumPy creates a VIEW (bukan copy) → memory efficient
first_10_rows = data[:10]          # 10 baris pertama, shape (10, 5)
last_column = data[:, -1]          # kolom terakhir, shape (100,)
subset = data[20:30, 1:3]         # baris 20-29, kolom 1-2, shape (10, 2)

# --- Boolean indexing — SANGAT sering dipakai ---
# Memilih elemen berdasarkan kondisi boolean.
# Shape boolean mask harus sama dengan array yang di-index.
mask = data[:, 0] > 0             # di mana kolom pertama positif, shape (100,)
positive_rows = data[mask]        # hanya baris dengan kolom pertama > 0
print(f"\nBaris dengan kolom pertama > 0: {positive_rows.shape[0]} dari {data.shape[0]}")

# --- Fancy indexing ---
# Menggunakan array of indices untuk memilih baris/kolom tertentu.
# Fancy indexing SELALU membuat copy (bukan view).
indices = np.array([0, 5, 10, 50, 99])
selected = data[indices]
print(f"Selected rows shape: {selected.shape}")


# ===========================================================
# 📖 BAGIAN 5: Practical ML Operations
# ===========================================================
# Function-function ini sering muncul di implementasi ML.

# --- Softmax function — nanti dipakai di classification ---
# Softmax: σ(z)_i = exp(z_i) / Σ_j exp(z_j)
# Mengubah vector logits menjadi distribusi probabilitas.
# Properties:
# - Output selalu positif
# - Sum of outputs = 1
# - Monotonic: higher input → higher output

def softmax(z):
    """
    Mengubah vector logits menjadi distribusi probabilitas menggunakan softmax.
    
    Parameters:
    -----------
    z : np.ndarray
        Vector atau matrix logits (output sebelum normalisasi).
        Shape bisa (n_classes,) untuk single sample atau
        (n_samples, n_classes) untuk multiple samples.
        
    Returns:
    --------
    np.ndarray
        Array dengan shape yang sama seperti input, tapi setiap elemen
        sudah di-normalisasi menjadi probabilitas (0-1) dan jumlah = 1.
        
    Notes:
    ------
    - z - z.max() adalah numerical stability trick.
      Tanpa ini, exp(z) bisa overflow jika z besar.
    - exp(z - z.max()) = exp(z) / exp(z.max())
    - exp(z.max()) akan di-cancel di numerator dan denominator.
    - Koneksi ke Teknik Elektro: mirip dengan normalisasi power 
      spectral density di signal processing.
    """
    exp_z = np.exp(z - z.max())  # dikurangi max untuk numerical stability
    return exp_z / exp_z.sum()


logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"\nSoftmax: {logits} → {probs.round(4)} (sum = {probs.sum():.4f})")

# --- Euclidean distance matrix — dipakai di KNN, clustering ---
# Distance matrix D[i,j] = jarak antara point i dan point j.
# Formula: ||a - b||² = ||a||² + ||b||² - 2*a·b
# Trick ini memungkinkan perhitungan O(n²) tanpa explicit loops.

def pairwise_distance(X):
    """
    Menghitung matrix jarak Euclidean antar semua pasangan titik.
    
    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features)
        Matrix data points. Setiap baris adalah satu titik.
        
    Returns:
    --------
    np.ndarray, shape (n_samples, n_samples)
        Distance matrix symmetric. D[i,j] = jarak antara X[i] dan X[j].
        Diagonal D[i,i] = 0.
        
    Notes:
    ------
    - Menggunakan trigonometric identity: ||a-b||² = ||a||² + ||b||² - 2a·b
    - Ini menghindari explicit loop → vectorized & C-speed
    - np.maximum(distances, 0) menghindari numerical errors yang bisa
      menghasilkan nilai negatif kecil
    - Koneksi ke Teknik Elektro: mirip dengan auto-correlation matrix
      di adaptive filtering (LMS/RLS)
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    sq_norms = np.sum(X**2, axis=1, keepdims=True)  # shape (n, 1)
    # keepdims=True menjaga dimensi agar broadcasting berfungsi
    distances = sq_norms + sq_norms.T - 2 * X @ X.T
    # sq_norms.T → shape (1, n)
    # sq_norms + sq_norms.T → broadcasting → shape (n, n)
    # X @ X.T → shape (n, n)
    return np.sqrt(np.maximum(distances, 0))


points = np.random.randn(5, 2)
dist_matrix = pairwise_distance(points)
print(f"\nDistance matrix (5 points):\n{dist_matrix.round(3)}")


# ===========================================================
# 🏋️ EXERCISE 1: Implementasi Fungsi-fungsi Berikut
# ===========================================================
"""
🎯 Learning Objectives:
   - Setelah exercise ini, kamu akan menguasai operasi NumPy fundamental
   - Kamu akan bisa mengimplementasikan normalisasi, similarity, dan encoding
   - Kamu akan paham broadcasting dan vectorization secara praktis

📋 LANGKAH-LANGKAH:

STEP 1: Implementasi batch_normalize(X)
─────────────────────────────────────────
Buat function batch_normalize(X) yang menerima array 2D dan mengembalikan
array yang sudah di-normalize per kolom (mean=0, std=1).

   - Input: array (N, D) → N samples, D features
   - Output: array (N, D) yang sudah di-normalize per kolom
   - Rumus: X_norm = (X - mean) / std

   💡 Apa yang harus dilakukan:
     a) Hitung mean per kolom menggunakan X.mean(axis=0)
     b) Hitung std per kolom menggunakan X.std(axis=0)
     c) Lakukan broadcasting: (X - mean) / std
     
   ⚠️ Hati-hati: std bisa 0 untuk kolom konstan!
     Solusi: tambahkan epsilon kecil (1e-8) ke std.
     
   Verification setelah implementasi:
     X_test = np.random.randn(50, 3)
     X_norm = batch_normalize(X_test)
     print(X_norm.mean(axis=0))  # Harus [~0, ~0, ~0]
     print(X_norm.std(axis=0))   # Harus [~1, ~1, ~1]


STEP 2: Implementasi cosine_similarity(a, b)
─────────────────────────────────────────────
Buat function cosine_similarity(a, b) yang menghitung cosine similarity
antara dua vektor 1D.

   - Input: dua vektor 1D (array 1D)
   - Output: skalar float antara -1 dan 1
   - Rumus: cos(θ) = (a · b) / (||a|| * ||b||)

   💡 Apa yang harus dilakukan:
     a) Hitung dot product: np.dot(a, b) atau a @ b
     b) Hitung norm (magnitude) masing-masing vektor: np.linalg.norm(a)
     c) Bagi dot product dengan product of norms
     
   ⚠️ Hati-hati: jika salah satu vektor adalah zero vector,
     denominator akan 0 → division by zero!
     Solusi: tambahkan epsilon kecil atau handle dengan if.
     
   Verification setelah implementasi:
     a = np.array([1, 0, 0])
     b = np.array([0, 1, 0])
     print(cosine_similarity(a, b))  # Harus ~0 (orthogonal)
     print(cosine_similarity(a, a))  # Harus ~1 (identical)


STEP 3: Implementasi one_hot_encode(labels, num_classes)
──────────────────────────────────────────────────────────
Buat function one_hot_encode(labels, num_classes) yang mengubah
array label integer menjadi matrix one-hot.

   - Input: array 1D berisi integer label, dan jumlah kelas
   - Output: array (N, num_classes) one-hot encoded
   - Contoh: [0, 2, 1] dengan 3 kelas → [[1,0,0], [0,0,1], [0,1,0]]

   💡 Apa yang harus dilakukan:
     a) Buat matrix nol dengan shape (len(labels), num_classes)
     b) Set elemen [i, labels[i]] = 1 untuk setiap i
     
   ⚠️ Hati-hati: labels harus integer 0..num_classes-1.
     Jika labels mencakup nilai >= num_classes, akan index out of bounds.
     
   Verification setelah implementasi:
     labels = np.array([0, 2, 1, 0])
     oh = one_hot_encode(labels, 3)
     print(oh)
     # Expected:
     # [[1, 0, 0],
     #  [0, 0, 1],
     #  [0, 1, 0],
     #  [1, 0, 0]]


💡 HINTS:
   - Gunakan np.zeros() untuk inisialisasi matrix nol
   - Gunakan np.dot() atau @ untuk dot product
   - Gunakan np.linalg.norm() untuk menghitung magnitude vektor
   - Broadcasting di NumPy sangat powerful — manfaatkan!

⚠️ COMMON MISTAKES:
   - Lupa menambahkan axis=0 di mean/std → menghasilkan scalar, bukan per kolom
   - Tidak menangani division by zero → NaN atau inf
   - Lupa reshape labels untuk indexing 2D di one-hot encoding

🧪 TEST CASES (uncomment setelah implementasi):
"""

# def batch_normalize(X):
#     # TODO: implementasi di sini
#     pass

# def cosine_similarity(a, b):
#     # TODO: implementasi di sini
#     pass

# def one_hot_encode(labels, num_classes):
#     # TODO: implementasi di sini
#     pass

# --- Test (uncomment setelah implementasi) ---
# X_test = np.random.randn(50, 3)
# X_norm = batch_normalize(X_test)
# assert np.allclose(X_norm.mean(axis=0), 0, atol=1e-10), "Mean harus ~0"
# assert np.allclose(X_norm.std(axis=0), 1, atol=1e-10), "Std harus ~1"
# print("✅ batch_normalize passed!")

# a = np.array([1, 0, 0])
# b = np.array([0, 1, 0])
# assert abs(cosine_similarity(a, b)) < 1e-10, "Orthogonal vectors harus cos=0"
# assert abs(cosine_similarity(a, a) - 1.0) < 1e-10, "Same vector harus cos=1"
# print("✅ cosine_similarity passed!")

# labels = np.array([0, 2, 1, 0])
# oh = one_hot_encode(labels, 3)
# expected = np.array([[1,0,0],[0,0,1],[0,1,0],[1,0,0]])
# assert np.array_equal(oh, expected), f"Expected:\n{expected}\nGot:\n{oh}"
# print("✅ one_hot_encode passed!")


# ===========================================================
# 🔥 CHALLENGE: Signal Processing dengan NumPy
# ===========================================================
"""
🎯 Learning Objectives:
   - Menggabungkan konsep signal processing (EE) dengan NumPy
   - Memahami DFT dari rumus dasar (bukan hanya API call)
   - Membandingkan performa implementasi manual vs optimized library

Dengan background Teknik Elektro, ini harusnya menyenangkan!

📋 LANGKAH-LANGKAH:

STEP 1: Generate Sinyal Campuran
─────────────────────────────────
Buat function generate_signal(t) yang menghasilkan:
   y(t) = sin(2π*10*t) + 0.5*sin(2π*50*t) + noise
   
   - t: time vector (gunakan np.linspace)
   - noise: Gaussian white noise (np.random.randn)
   - Sampling rate: 1000 Hz
   - Duration: 1 detik

   💡 KENAPA sinyal ini?
     - 10 Hz = fundamental frequency
     - 50 Hz = 5th harmonic
     - Noise = menguji robustness algoritma


STEP 2: Implementasi DFT Manual
───────────────────────────────
Implementasi Discrete Fourier Transform (DFT) MANUAL dengan NumPy.
JANGAN pakai np.fft — bangun dari rumus DFT.

   Rumus DFT: X[k] = Σ_{n=0}^{N-1} x[n] * exp(-j*2π*k*n/N)
   
   💡 Apa yang harus dilakukan:
     a) Buat matrix twiddle factors W[k,n] = exp(-j*2π*k*n/N)
        Gunakan np.outer untuk membuat matrix ini efisien.
     b) Kalikan dengan signal: X = W @ x
     c) Return magnitude spectrum: np.abs(X)
     
   ⚠️ Hati-hati: twiddle factors adalah complex numbers!
     Gunakan np.exp() dengan argumen kompleks (1j * ...)

   💡 KENAPA manual? 
     - Memahami apa yang sebenarnya dilakukan np.fft
     - DFT adalah dot product dengan basis functions (sinusoids)
     - Ini mirip dengan matched filter di radar/sonar


STEP 3: Plot Sinyal dan Spektrum
─────────────────────────────────
Buat figure dengan 2 subplot:
   - Subplot 1: sinyal di time domain (200 ms pertama saja, biar jelas)
   - Subplot 2: magnitude spectrum di frequency domain
   
   💡 KENAPA plot?
     - Visual inspection adalah debugging paling powerful
     - Harusnya ada 2 peak: di 10 Hz dan 50 Hz
     - Noise terlihat sebagai baseline di seluruh frekuensi


STEP 4: Bandingkan Kecepatan
─────────────────────────────
Bandingkan waktu eksekusi DFT manual vs np.fft.fft.

   💡 Apa yang diuji:
     a) Buat signal dengan panjang yang sama
     b) Jalankan DFT manual dan catat waktu (time.time())
     c) Jalankan np.fft.fft dan catat waktu
     d) Print speedup factor
     
   ⚠️ Expected result: np.fft akan JAUH lebih cepat (100x+)
     Karena np.fft menggunakan FFT algorithm O(N log N),
     sedangkan DFT manual adalah O(N²).


💡 HINTS:
   - np.linspace(0, 1, 1000) → time vector dengan fs=1000 Hz
   - np.exp(-1j * 2 * np.pi * np.outer(k, n) / N) → twiddle matrix
   - np.abs(fft_vals[:N//2]) → magnitude (ambil setengah karena simetri)
   - np.fft.fftfreq(N, 1/fs) → frequency axis

⚠️ COMMON MISTAKES:
   - Lupa memakai hanya setengah spektrum (real signal → simetri)
   - Salah menghitung frequency axis
   - Tidak normalisasi magnitude (np.abs(X) * 2/N)

🎯 EXPECTED OUTPUT:
   - Plot menunjukkan 2 peak tajam di 10 Hz dan 50 Hz
   - np.fft ~100x+ lebih cepat dari DFT manual
   - Insight: manual DFT berguna untuk pemahaman, tapi FFT untuk production

Koneksi ke ML:
- Fourier transform → basis dari CNN (konvolusi di time domain = multiplication di freq domain!)
- Signal decomposition → sama dengan feature extraction di ML
- Pemahaman frekuensi domain → berguna untuk time series analysis
"""

print("\n" + "="*50)
print("✅ Modul 1 selesai! Lanjut ke: 01-fondasi-data/02_pandas_essentials.py")
print("="*50)
