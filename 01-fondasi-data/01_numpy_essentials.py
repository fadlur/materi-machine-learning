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

# Membuat array dari list
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Vektor x: {x}")
print(f"Shape: {x.shape}, Dtype: {x.dtype}")

# Matrix 2D
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(f"\nMatrix A:\n{A}")
print(f"Shape: {A.shape}")

# Array generators — sering dipakai untuk inisialisasi
zeros = np.zeros((3, 4))          # matrix nol
ones = np.ones((2, 3))            # matrix satu
identity = np.eye(4)              # identity matrix (familiar dari linear algebra)
random_normal = np.random.randn(3, 3)  # random dari distribusi normal

# Linspace — familiar dari MATLAB/signal processing
t = np.linspace(0, 2 * np.pi, 100)  # 100 titik dari 0 sampai 2π
signal = np.sin(t)  # sinyal sinusoidal — kamu pasti sering pakai ini

print(f"\nSinyal sinusoidal: {signal[:5]}...")  # 5 sample pertama


# ===========================================================
# 📖 BAGIAN 2: Broadcasting & Vectorization
# ===========================================================
# INI KUNCI PENTING untuk ML!
# Broadcasting = operasi antara array dengan shape berbeda
# Vectorization = hindari loop, gunakan operasi array

# Contoh: normalisasi data (z-score)
# Rumus: z = (x - mean) / std
data = np.random.randn(1000, 5)  # 1000 samples, 5 features

# CARA BURUK (loop) — jangan lakukan ini
# for i in range(data.shape[1]):
#     data[:, i] = (data[:, i] - data[:, i].mean()) / data[:, i].std()

# CARA BAIK (vectorized + broadcasting)
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

A = np.random.randn(3, 4)
B = np.random.randn(4, 2)

# Matrix multiplication
C = A @ B  # atau np.dot(A, B)
print(f"\nMatrix multiplication: ({A.shape}) @ ({B.shape}) = {C.shape}")

# Transpose
print(f"A^T shape: {A.T.shape}")

# Eigenvalue decomposition — pasti familiar dari kuliah
M = np.random.randn(3, 3)
M = M @ M.T  # buat symmetric positive definite
eigenvalues, eigenvectors = np.linalg.eigh(M)
print(f"\nEigenvalues: {eigenvalues}")

# SVD — nanti akan dipakai di PCA
U, S, Vt = np.linalg.svd(A)
print(f"SVD: U{U.shape}, S{S.shape}, Vt{Vt.shape}")

# Solve linear system: Ax = b
A_square = np.random.randn(3, 3)
b = np.random.randn(3)
x = np.linalg.solve(A_square, b)
print(f"\nSolusi Ax=b: x = {x.round(4)}")
print(f"Verifikasi (Ax harus = b): {(A_square @ x).round(4)}")
print(f"b asli:                     {b.round(4)}")


# ===========================================================
# 📖 BAGIAN 4: Indexing & Slicing (Penting untuk Data Processing)
# ===========================================================

data = np.random.randn(100, 5)

# Basic slicing
first_10_rows = data[:10]          # 10 baris pertama
last_column = data[:, -1]          # kolom terakhir
subset = data[20:30, 1:3]         # baris 20-29, kolom 1-2

# Boolean indexing — SANGAT sering dipakai
mask = data[:, 0] > 0             # di mana kolom pertama positif
positive_rows = data[mask]
print(f"\nBaris dengan kolom pertama > 0: {positive_rows.shape[0]} dari {data.shape[0]}")

# Fancy indexing
indices = np.array([0, 5, 10, 50, 99])
selected = data[indices]
print(f"Selected rows shape: {selected.shape}")


# ===========================================================
# 📖 BAGIAN 5: Practical ML Operations
# ===========================================================

# Softmax function — nanti dipakai di classification
def softmax(z):
    """Softmax: ubah logits jadi probabilitas"""
    exp_z = np.exp(z - z.max())  # dikurangi max untuk numerical stability
    return exp_z / exp_z.sum()

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"\nSoftmax: {logits} → {probs.round(4)} (sum = {probs.sum():.4f})")

# Euclidean distance matrix — dipakai di KNN, clustering
def pairwise_distance(X):
    """Hitung jarak antar semua pasangan titik"""
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    sq_norms = np.sum(X**2, axis=1, keepdims=True)
    distances = sq_norms + sq_norms.T - 2 * X @ X.T
    return np.sqrt(np.maximum(distances, 0))

points = np.random.randn(5, 2)
dist_matrix = pairwise_distance(points)
print(f"\nDistance matrix (5 points):\n{dist_matrix.round(3)}")


# ===========================================================
# 🏋️ EXERCISE 1: Implementasi Fungsi-fungsi Berikut
# ===========================================================
"""
Instruksi: Implementasi fungsi-fungsi di bawah ini TANPA melihat contoh
di atas. Jalankan test di bawahnya untuk verifikasi.

1. batch_normalize(X):
   Input: array (N, D)
   Output: array (N, D) yang sudah di-normalize per kolom (mean=0, std=1)

2. cosine_similarity(a, b):
   Input: dua vektor 1D
   Output: cosine similarity (skalar antara -1 dan 1)
   Rumus: cos(θ) = (a · b) / (||a|| * ||b||)

3. one_hot_encode(labels, num_classes):
   Input: array 1D berisi integer label, jumlah kelas
   Output: array (N, num_classes) one-hot encoded
   Contoh: [0, 2, 1] dengan 3 kelas → [[1,0,0], [0,0,1], [0,1,0]]
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
Dengan background Teknik Elektro, ini harusnya menyenangkan!

Buat fungsi yang:
1. Generate sinyal campuran: y(t) = sin(2π*10*t) + 0.5*sin(2π*50*t) + noise
2. Implementasi DFT (Discrete Fourier Transform) MANUAL dengan NumPy
   (bukan np.fft — bangun dari rumus DFT)
3. Plot sinyal dan spektrum frekuensinya
4. Bandingkan kecepatan DFT manual vs np.fft.fft

Ini relevan karena:
- Fourier transform → basis dari CNN (konvolusi!)
- Signal decomposition → sama dengan feature extraction di ML
- Pemahaman frekuensi domain → berguna untuk time series analysis

Hint: DFT formula → X[k] = Σ x[n] * exp(-j*2π*k*n/N)
"""

print("\n" + "="*50)
print("✅ Modul 1 selesai! Lanjut ke: 01-fondasi-data/02_pandas_essentials.py")
print("="*50)
