"""
=============================================================
FASE 1 - MODUL 1: NUMPY ESSENTIALS
=============================================================
Kenapa NumPy dulu?
- Semua library ML di Python dibangun di atas NumPy (pandas, scikit-learn, PyTorch)
- Memahami array operations = memahami cara kerja internal ML
- Operasi numerik di Python native lambat; NumPy memberikan performa C-level
  melalui vectorization dan memory-contiguous arrays

Durasi target: 2-3 jam
=============================================================
"""

import numpy as np

# ===========================================================
# BAGIAN 1: Array Creation & Basic Operations
# ===========================================================
# Vektor dan matrix adalah struktur data fundamental di ML.
# NumPy array (ndarray) adalah representasi efisien dari struktur data ini.
#
# Perbedaan List Python vs NumPy Array:
# 1. List Python: array of pointers ke objek Python di heap.
#    - Setiap elemen adalah pointer (8 bytes di 64-bit system)
#    - Pointer menunjuk ke objek float/int yang sebenarnya (24+ bytes)
#    - Total overhead besar, fragmentasi memory
#    - Tipe campuran dimungkinkan (heterogeneous)
#    - Akses per elemen melalui dereference pointer
#
# 2. NumPy Array: typed array yang contiguous di memory.
#    - Tidak ada pointer per elemen, data mentah tersimpan langsung
#    - Semua elemen harus tipe sama (homogeneous)
#    - Layout contiguous -> CPU cache-friendly -> akses sangat cepat
#    - Operasi vectorized menggunakan SIMD (Single Instruction Multiple Data)
#
# Contoh konkret:
#   List [1.0, 2.0, 3.0] di memory: 3 pointer -> 3 objek float terpisah
#   np.array([1.0, 2.0, 3.0]) di memory: [0x3FF0000000000000, 0x4000000000000000, 0x4008000000000000]
#   (3 nilai float64, total 24 bytes, contiguous)

# --- Membuat array dari list Python ---
# np.array() mengkonversi list/tuple Python menjadi ndarray.
#
# Parameter dtype mengontrol tipe data:
# - float64 (default di most systems): 64-bit double precision
#   Range: ~+/-1.79769e308
#   Presisi: ~15-17 digit desimal
#   Digunakan: scientific computing, default di NumPy
#
# - float32: 32-bit single precision
#   Range: ~+/-3.40282e38
#   Presisi: ~6-9 digit desimal
#   Digunakan: deep learning (hemat memory 2x, cukup untuk neural nets)
#
# - int64, int32, int16, int8: signed integers dengan range berbeda
#   int8: -128 sampai 127
#   int16: -32768 sampai 32767
#   int32: ~-2.1e9 sampai ~2.1e9
#   int64: ~-9.2e18 sampai ~9.2e18
#
# - uint8, uint16, ...: unsigned integers (hanya positif)
#   uint8: 0 sampai 255 -> sering untuk gambar (pixel values)
#
# - bool_: 1 byte (0 atau 1) -> untuk mask dan flag
#
# - complex64, complex128: untuk bilangan kompleks (rarely used di ML)

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Vektor x: {x}")
print(f"Shape: {x.shape}, Dtype: {x.dtype}")
print(f"Size: {x.size} elemen, Itemsize: {x.itemsize} bytes")
print(f"Total memory: {x.nbytes} bytes")
print(f"Ndim: {x.ndim}")
# shape  -> tuple yang mendeskripsikan dimensi array, (5,) artinya 1D dengan 5 elemen
# dtype  -> tipe data setiap elemen, float64 = 8 bytes per elemen
# ndim   -> jumlah dimensi (axis), 1 untuk vektor, 2 untuk matrix, dst.
# size   -> total jumlah elemen (product dari semua dimensi shape)
# itemsize -> bytes per elemen
# nbytes -> total bytes = size * itemsize

# --- Matrix 2D ---
# Matrix 2D = array dengan shape (baris, kolom) atau (samples, features)
# Di supervised learning, konvensi yang paling umum:
# - Setiap baris = satu observasi / sample / data point
# - Setiap kolom = satu fitur / variabel / attribute
#
# Contoh: dataset 3 siswa dengan nilai [Matematika, Fisika, Kimia]
#         Baris 0: siswa pertama, Kolom 0: nilai Matematika
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(f"\nMatrix A:\n{A}")
print(f"Shape: {A.shape}")
print(f"ndim: {A.ndim}, size: {A.size}")
# Shape (3, 3) = 3 baris, 3 kolom
# A[0, 1] = elemen baris ke-0, kolom ke-1 -> 2

# --- Array generators - sering dipakai untuk inisialisasi di ML ---
# np.zeros: membuat matrix dengan semua elemen 0
#   Penggunaan di ML:
#   - Inisialisasi accumulator (misal: gradient accumulator di optimizer)
#   - Padding dalam NLP/CV (zero-padding untuk sequence/gambar)
#   - Membuat attention mask (0 = mask, 1 = attend)
#   - Bias initialization sebelum training (meski jarang)
zeros = np.zeros((3, 4))          # matrix 3x4 dengan semua elemen 0
print(f"\nnp.zeros((3,4)) dtype default: {zeros.dtype}")
print(zeros)

# np.ones: membuat matrix dengan semua elemen 1
#   Penggunaan di ML:
#   - Inisialisasi bias di neural network (sebelum training dimulai)
#   - Membuat uniform weight untuk averaging
#   - Menghitung mean melalui dot product dengan ones
ones = np.ones((2, 3))            # matrix 2x3 dengan semua elemen 1
print(f"\nnp.ones((2,3)):\n{ones}")

# np.eye: identity matrix I_n (1 di diagonal, 0 di lainnya)
#   Properti fundamental: A @ I = A untuk semua A yang shape-compatible
#   Penggunaan di ML:
#   - Regularization L2: menambahkan lambdaI ke matrix (ridge regression)
#   - Orthogonal initialization untuk weight matrix
#   - Covariance matrix dari white noise
#   - Inverse dari I adalah I sendiri
identity = np.eye(4)              # matrix identitas 4x4
print(f"\nnp.eye(4):\n{identity}")
# np.eye(n, m) bisa membuat matrix non-square dengan 1 di diagonal utama

# np.full: membuat array dengan nilai konstan tertentu
#   Penggunaan: inisialisasi dengan nilai spesifik (misal: fill value untuk missing data)
full_five = np.full((2, 2), 5.0)
print(f"\nnp.full((2,2), 5.0):\n{full_five}")

# np.arange: seperti range() tapi menghasilkan array
#   Format: np.arange(start, stop, step)
#   stop adalah eksklusif (sama seperti range Python)
#   Berguna untuk membuat index array
ar = np.arange(0, 10, 2)
print(f"\nnp.arange(0, 10, 2): {ar}")

# np.random.randn: random dari distribusi normal standard (mu=0, sigma=1)
#   "randn" = RANDom Normal
#   Penggunaan di ML:
#   - Weight initialization di neural networks
#     * Xavier/Glorot init: W = randn * sqrt(2/(fan_in + fan_out))
#     * He init: W = randn * sqrt(2/fan_in)
#   - Menambahkan Gaussian noise untuk data augmentation
#   - Simulasi data untuk testing
#   - Sampling dari posterior di Bayesian ML
#
#   Distribusi normal muncul secara alami di banyak fenomena karena
#   Central Limit Theorem: jumlah banyak variabel independen cenderung normal.
np.random.seed(42)                # seed untuk reproducibility
random_normal = np.random.randn(3, 3)
print(f"\nnp.random.randn(3,3):\n{random_normal.round(4)}")

# --- Linspace - generator titik equally spaced ---
# np.linspace(start, stop, num) menghasilkan 'num' titik yang equally spaced
# dari 'start' sampai 'stop' (inklusif di kedua ujung secara default).
# Parameter endpoint=False bisa dipakai untuk membuat stop eksklusif.
#
# Perbedaan arange vs linspace:
# - arange: kontrol step size, jumlah elemen tergantung range
# - linspace: kontrol jumlah elemen, step size dihitung otomatis
#
# Berguna untuk:
# - Membuat grid untuk plotting (x-axis values)
# - Sampling parameter space (hyperparameter search grid)
# - Membuat time vector untuk time series
# - Domain untuk fungsi kontinu
N = 100
t = np.linspace(0, 2 * np.pi, N)  # 100 titik dari 0 sampai 2*pi
# Total range = 2*pi - 0 = 2*pi
# Jumlah interval = N - 1 = 99
# Step size = 2*pi / 99 ~ 0.063466
actual_step = t[1] - t[0]
expected_step = (2 * np.pi) / (N - 1)
print(f"\nlinspace dari 0 sampai 2*pi dengan {N} titik:")
print(f"  Step aktual: {actual_step:.6f}")
print(f"  Step teori:  {expected_step:.6f}")
print(f"  Match: {np.isclose(actual_step, expected_step)}")

# Menggunakan linspace untuk mengevaluasi fungsi matematika
signal = np.sin(t)
print(f"\nSinusoidal signal:")
print(f"  Nilai di t=0:      {signal[0]:.4f}  (sin(0) = 0)")
print(f"  Nilai di t=pi/2:   {signal[N//4]:.4f}  (sin(pi/2) = 1)")
print(f"  Nilai di t=pi:     {signal[N//2]:.4f}  (sin(pi) = 0)")
print(f"  Nilai di t=3pi/2:  {signal[3*N//4]:.4f}  (sin(3pi/2) = -1)")
print(f"  Nilai di t=2*pi:   {signal[-1]:.4f}  (sin(2*pi) ~ 0)")

# --- Reshape, ravel, dan transpose ---
# reshape: mengubah shape tanpa mengubah data (returns view jika possible)
# -1 di satu dimensi berarti "infer dari size dan dimensi lain"
# reshape(3, 4) artinya 3 baris, 4 kolom
flat = np.arange(12)              # [0, 1, 2, ..., 11], shape (12,)
matrix_3x4 = flat.reshape(3, 4)   # shape (3, 4)
matrix_2x6 = flat.reshape(2, -1)  # shape (2, 6), -1 diinfer = 12/2 = 6
print(f"\nReshape:")
print(f"  {flat.shape} -> reshape(3,4) -> {matrix_3x4.shape}")
print(f"  {flat.shape} -> reshape(2,-1) -> {matrix_2x6.shape}")
# reshape returns VIEW jika memory layout contiguous, COPY jika tidak.
# View berbagi data yang sama di memory -> perubahan pada view memodifikasi original.

# ravel: meng-flatten array menjadi 1D (selalu mencoba return view)
flattened = matrix_3x4.ravel()
print(f"  {matrix_3x4.shape} -> ravel() -> {flattened.shape}")

# transpose (atau .T): menukar baris dan kolom
# (m, n) -> (n, m)
# Di ML, transpose sering dipakai untuk:
# - Menyesuaikan dimensi untuk matrix multiplication
# - Mengubah orientation data (features-as-rows -> features-as-cols)
print(f"  {matrix_3x4.shape} -> .T -> {matrix_3x4.T.shape}")



# ===========================================================
# BAGIAN 2: Broadcasting & Vectorization
# ===========================================================
# INI ADALAH KUNCI PENTING untuk ML!
# Broadcasting dan vectorization adalah dua fitur NumPy yang membuat
# operasi numerik di Python secepat C/Fortran.
#
# Vectorization:
# - Menghindari explicit Python for-loops
# - Operasi diterapkan pada seluruh array sekaligus
# - NumPy meneruskan operasi ke backend C yang di-compile
# - Backend C bisa memanfaatkan SIMD (SSE/AVX) untuk parallelism di CPU level
# - Cache locality jauh lebih baik karena data contiguous
#
# Broadcasting:
# - Mekanisme yang memungkinkan operasi antara array dengan shape berbeda
# - Aturan broadcasting (dari dokumentasi NumPy):
#   1. Jika dua array berbeda ndim, shape dari array dengan ndim lebih kecil
#      di-pad dengan 1 di depan (left-padded).
#   2. Jika di suatu dimensi ukurannya tidak sama, dan salah satunya adalah 1,
#      maka yang 1 di-stretch (di-duplicate secara virtual) untuk menyesuaikan.
#   3. Jika di suatu dimensi ukurannya tidak sama dan keduanya bukan 1,
#      maka error: ValueError: operands could not be broadcast together.
#
# Contoh visual broadcasting:
#   Array A shape: (3, 4)
#   Array B shape: (4,)   -> pad jadi (1, 4) -> stretch dim 0 -> (3, 4)
#   Hasil: (3, 4) operasi dengan (3, 4)
#
#   Array A shape: (3, 1)
#   Array B shape: (1, 5)
#   Stretch dim 1 dari A: (3, 1) -> (3, 5)
#   Stretch dim 0 dari B: (1, 5) -> (3, 5)
#   Hasil: (3, 5)

# Contoh: normalisasi data (z-score standardization)
# Rumus: z = (x - mu) / sigma
# Di statistik, ini disebut standard score.
# Z-score mengukur berapa banyak standard deviation sebuah nilai
# berada dari mean. Ini penting karena banyak algoritma ML
# mengasumsikan data terdistribusi normal dengan mean=0, std=1.
#
# Alasan normalisasi penting di ML:
# 1. Gradient descent lebih cepat konvergen (contours lebih circular)
# 2. Regularization bekerja secara merata di semua fitur
# 3. Distance-based algorithms (KNN, K-Means, SVM) tidak bias ke fitur besar
# 4. Numerical stability: menghindari overflow/underflow
np.random.seed(42)
data = np.random.randn(1000, 5)  # 1000 samples, 5 features

# CARA BURUK (explicit Python loop) - jangan lakukan ini
# Kenapa buruk?
# 1. Python loops adalah interpreted, bukan compiled -> overhead besar
# 2. Setiap iterasi melakukan Python bytecode execution
# 3. Tidak bisa di-optimize oleh compiler
# 4. Tidak memanfaatkan SIMD atau cache locality
# 5. Code lebih verbose dan sulit dibaca
# for i in range(data.shape[1]):
#     data[:, i] = (data[:, i] - data[:, i].mean()) / data[:, i].std()

# CARA BAIK (vectorized + broadcasting)
# data.shape = (1000, 5)
# mean.shape = (5,) -> di-pad otomatis jadi (1, 5) oleh broadcasting
# lalu di-stretch di axis 0 menjadi virtual shape (1000, 5)
# std.shape = (5,) -> sama, di-pad jadi (1, 5) lalu stretch ke (1000, 5)
# Operasi (1000, 5) - (5,) -> broadcasting -> (1000, 5)
mean = data.mean(axis=0)  # mean per kolom -> shape (5,)
std = data.std(axis=0)    # std per kolom -> shape (5,)
data_normalized = (data - mean) / std  # broadcasting otomatis!

print(f"\nZ-score normalization:")
print(f"  Original data: mean={data.mean(axis=0).round(4)}, std={data.std(axis=0).round(4)}")
print(f"  Normalized mean (harus ~0): {data_normalized.mean(axis=0).round(10)}")
print(f"  Normalized std  (harus ~1): {data_normalized.std(axis=0).round(4)}")

# --- Broadcasting rules visual examples ---
print(f"\n--- Broadcasting Examples ---")

# Example 1: Scalar broadcasting
# (3, 3) + scalar -> scalar di-stretch ke (3, 3)
mat = np.ones((3, 3))
mat_plus_5 = mat + 5
print(f"(3,3) + 5 -> shape: {mat_plus_5.shape}, semua elemen = {mat_plus_5[0,0]}")

# Example 2: Row vector broadcasting
# (3, 4) + (4,) -> (4,) di-pad jadi (1,4) lalu stretch ke (3,4)
mat = np.zeros((3, 4))
row = np.array([1, 2, 3, 4])  # shape (4,)
result = mat + row
print(f"(3,4) + (4,) -> shape: {result.shape}")
print(f"  Row 0: {result[0]}")
print(f"  Row 1: {result[1]}")
print(f"  Row 2: {result[2]}")

# Example 3: Column vector broadcasting
# (3, 4) + (3, 1) -> (3,1) di-stretch di axis 1 ke (3,4)
mat = np.zeros((3, 4))
col = np.array([[1], [2], [3]])  # shape (3, 1)
result = mat + col
print(f"(3,4) + (3,1) -> shape: {result.shape}")
print(f"  Col 0: {result[:, 0]}")
print(f"  Col 1: {result[:, 1]}")

# Example 4: Broadcasting failure
# (3, 4) + (3,) -> (3,) di-pad jadi (1,3)
# (3, 4) vs (1, 3) -> dim 1: 4 vs 3, keduanya bukan 1 -> ERROR!
try:
    mat = np.zeros((3, 4))
    vec = np.array([1, 2, 3])  # shape (3,)
    result = mat + vec
except ValueError as e:
    print(f"(3,4) + (3,) -> ERROR: {e}")


# ===========================================================
# BAGIAN 3: Linear Algebra Operations
# ===========================================================
# Ini yang paling relevan untuk ML. Hampir semua model ML
# pada dasarnya adalah operasi linear algebra yang di-stack dan di-nonlinear-kan.
#
# Beberapa contoh operasi linear algebra di ML:
# - Linear Regression: y_hat = Xbeta, solve for beta
# - Neural Network: z = Wx + b, a = activation(z)
# - Attention: Attention(Q,K,V) = softmax(QK^T/sqrtd_k)V
# - PCA: eigendecomposition atau SVD dari covariance matrix
# - PageRank: eigenvector dari transition matrix

# --- Matrix multiplication ---
# A @ B (atau np.dot(A, B) untuk 2D) = matrix product
# Shape rule: (m, n) @ (n, p) -> (m, p)
# Inner dimensions (n) harus sama; outer dimensions (m, p) menentukan output shape.
#
# Ini adalah operasi paling fundamental di ML:
# - Linear regression: y_hat = Xw  (X: data, w: weights)
# - Neural network layer: z = Wx + b  (W: weight matrix, x: input, b: bias)
# - Attention mechanism: scores = Q @ K.T
#
# Kompleksitas: O(m * n * p)
# Untuk matrix persegi nxn: O(n**3) untuk algoritma naive,
# tapi library optimized seperti BLAS bisa lebih cepat.
np.random.seed(42)
A = np.random.randn(3, 4)
B = np.random.randn(4, 2)
C = A @ B  # atau np.dot(A, B)
print(f"\nMatrix multiplication: ({A.shape}) @ ({B.shape}) = {C.shape}")
print(f"  C[0,0] = dot(A[0,:], B[:,0]) = {C[0,0]:.4f}")

# --- Transpose ---
# A.T membalik baris dan kolom.
# (m, n) -> (n, m)
# Di ML, transpose sering dipakai untuk:
# - Mengubah orientation data (samples as rows vs columns)
# - Menyesuaikan dimensi untuk matrix multiplication (K.T di attention)
# - Covariance matrix: X.T @ X / (n-1)
print(f"A^T shape: {A.T.shape}")
# (3, 4) -> (4, 3)

# --- Inverse matrix ---
# A_inv = np.linalg.inv(A) sedemikian sehingga A @ A_inv = I
# Hanya berlaku untuk matrix square yang non-singular (det != 0)
# Penggunaan di ML:
# - Normal equation untuk linear regression: beta = (X^T X)^(-1) X^T y
# - Gaussian Process: K^(-1) untuk posterior computation
# - Newton's method: menggunakan Hessian inverse
A_square = np.array([[4, 7], [2, 6]], dtype=float)
A_inv = np.linalg.inv(A_square)
print(f"\nInverse matrix:")
print(f"  A:\n{A_square}")
print(f"  A_inv:\n{A_inv.round(4)}")
print(f"  A @ A_inv (harus I):\n{(A_square @ A_inv).round(10)}")

# --- Eigenvalue decomposition ---
# Untuk matrix square M: M = V Lambda V^(-1), dimana:
# - Lambda adalah diagonal matrix of eigenvalues (lambda_1, lambda_2, ...)
# - V adalah matrix of eigenvectors (kolom-kolomnya adalah eigenvectors)
# - Eigenvector v memenuhi: Mv = lambdav
#
# Aplikasi di ML:
# - PCA: eigendecomposition dari covariance matrix
# - Spectral clustering: eigendecomposition dari Laplacian matrix
# - PageRank: eigenvector dengan eigenvalue terbesar
# - Stability analysis di optimization
#
# Untuk symmetric matrix (M = M.T), eigendecomposition selalu real
# dan eigenvectors orthogonal: V^T V = I.
M = np.random.randn(3, 3)
M = M @ M.T  # buat symmetric positive semi-definite
# M @ M.T selalu menghasilkan matrix symmetric
# Symmetric matrix memiliki eigenvalues real dan eigenvectors orthogonal
eigenvalues, eigenvectors = np.linalg.eigh(M)
# np.linalg.eigh khusus untuk Hermitian/symmetric (lebih efisien dan stabil)
print(f"\nEigenvalue decomposition:")
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Eigenvectors shape: {eigenvectors.shape}")
# Verifikasi: M @ v ~ lambda * v
v0 = eigenvectors[:, 0]
l0 = eigenvalues[0]
print(f"  Verifikasi Mv ~ lambdav: {np.allclose(M @ v0, l0 * v0)}")

# --- SVD (Singular Value Decomposition) ---
# Untuk matrix arbitrary A (bisa non-square): A = U Sum V^T, dimana:
# - U: left singular vectors, shape (m, m), orthogonal (U^T U = I)
# - Sum: singular values, shape (m, n), diagonal matrix dengan nilai non-negative
#   Di NumPy, S adalah vector 1D berisi singular values (bukan matrix diagonal)
# - V^T: right singular vectors transposed, shape (n, n), orthogonal
#
# Hubungan dengan eigendecomposition:
# - U adalah eigenvectors dari A @ A.T
# - V adalah eigenvectors dari A.T @ A
# - Sum**2 adalah eigenvalues dari A.T @ A (atau A @ A.T)
#
# Aplikasi di ML:
# - PCA: SVD dari data matrix (lebih stabil dari eigendecomposition langsung)
# - Image compression: retain top-k singular values
# - Collaborative filtering (matrix factorization)
# - Latent Semantic Analysis (LSA) di NLP
# - Low-rank approximation: A_k = U_k Sum_k V_k^T
U, S, Vt = np.linalg.svd(A)
print(f"\nSVD:")
print(f"  U shape: {U.shape} (left singular vectors)")
print(f"  S shape: {S.shape} (singular values vector)")
print(f"  Vt shape: {Vt.shape} (right singular vectors transposed)")
print(f"  Singular values: {S.round(4)}")
# Rekonstruksi: U @ diag(S) @ Vt ~ A
S_diag = np.zeros_like(A)
S_diag[:len(S), :len(S)] = np.diag(S)
A_reconstructed = U @ S_diag @ Vt
print(f"  Reconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")

# --- Solve linear system: Ax = b ---
# Mencari x yang memenuhi persamaan Ax = b.
# np.linalg.solve menggunakan LU decomposition (dengan partial pivoting)
# yang lebih stabil dan cepat daripada menghitung A^(-1) lalu kalikan b.
#
# Kompleksitas: O(n**3) untuk factorization + O(n**2) untuk solve
# Aplikasi di ML:
# - Linear regression closed form: x = (A^T A)^(-1) A^T b
# - Gaussian Process prediction
# - Constrained optimization (KKT systems)
A_square = np.array([[3, 1], [1, 2]], dtype=float)
b = np.array([9, 8], dtype=float)
x = np.linalg.solve(A_square, b)
print(f"\nSolve Ax = b:")
print(f"  A = {A_square.tolist()}")
print(f"  b = {b.tolist()}")
print(f"  x = {x.round(4).tolist()}")
print(f"  Verifikasi Ax = { (A_square @ x).round(4).tolist() } (harus = b)")

# --- Pseudo-inverse (Moore-Penrose) ---
# A_pinv = np.linalg.pinv(A) untuk matrix non-square atau singular
# A_pinv memenuhi 4 kondisi Moore-Penrose
# Digunakan untuk least-squares solution: x = A_pinv @ b
A_rect = np.random.randn(3, 2)
A_pinv = np.linalg.pinv(A_rect)
print(f"\nPseudo-inverse:")
print(f"  A shape: {A_rect.shape}")
print(f"  A_pinv shape: {A_pinv.shape}")
# A @ A_pinv @ A ~ A
print(f"  A @ A_pinv @ A ~ A ? {np.allclose(A_rect @ A_pinv @ A_rect, A_rect)}")



# ===========================================================
# BAGIAN 4: Indexing & Slicing (Penting untuk Data Processing)
# ===========================================================
# Indexing & slicing memungkinkan kita mengakses subset data.
# Ini fundamental untuk:
# - Train/validation/test split
# - Batch processing (mini-batch gradient descent)
# - Feature selection
# - Data cleaning (memilih/membuang baris/kolom tertentu)
#
# Konsep kritis: VIEW vs COPY
# - View: berbagi data yang sama di memory dengan array asli
#         perubahan pada view memodifikasi array asli
#         slicing basic ([:, :], [1:5]) biasanya menghasilkan view
# - Copy: membuat salinan independen di memory
#         perubahan tidak mempengaruhi array asli
#         fancy indexing, np.copy(), .copy() menghasilkan copy
#
# Mengapa ini penting?
# - View hemat memory (tidak duplikasi data)
# - Tapi hati-hati: memodifikasi view bisa merusak data original secara tidak sengaja

np.random.seed(42)
data = np.random.randn(100, 5)

# --- Basic slicing ---
# Syntax: array[start:stop:step]
# start inklusif, stop eksklusif, step default=1
# Jika start/stop diabaikan, diambil dari awal/akhir array
# Negative index: -1 = elemen terakhir, -2 = elemen kedua terakhir, dst.
first_10_rows = data[:10]          # 10 baris pertama, shape (10, 5)
last_column = data[:, -1]          # kolom terakhir, shape (100,)
subset = data[20:30, 1:3]         # baris 20-29, kolom 1-2, shape (10, 2)
step_slice = data[::2, ::2]       # setiap baris genap, setiap kolom genap

print(f"\nIndexing & Slicing:")
print(f"  data shape: {data.shape}")
print(f"  data[:10] shape: {first_10_rows.shape}")
print(f"  data[:, -1] shape: {last_column.shape}")
print(f"  data[20:30, 1:3] shape: {subset.shape}")
print(f"  data[::2, ::2] shape: {step_slice.shape}")

# View demonstration
view = data[:10]
view[0, 0] = 999.0
print(f"  Setelah modifikasi view, data[0,0] = {data[0,0]:.1f} (berubah!)")
# Restore
view[0, 0] = data[0, 0]  # revert (tapi data sudah berubah, ini tidak benar-benar revert)
# Lebih baik buat copy jika ingin modifikasi independen
copy = data[:10].copy()
copy[0, 0] = 888.0
print(f"  Setelah modifikasi copy, data[0,0] tetap = {data[0,0]:.1f}")

# --- Boolean indexing - SANGAT sering dipakai ---
# Memilih elemen berdasarkan kondisi boolean.
# Boolean mask harus shape yang sama dengan array yang di-index (di axis yang relevan).
# Hasil boolean indexing SELALU 1D (flattened) karena tidak tahu shape output.
# Penggunaan di ML:
# - Filter outlier (ambil data di mana |z-score| < 3)
# - Pilih sample dari kelas tertentu
# - Data cleaning (ambil baris di mana fitur tidak missing)
mask = data[:, 0] > 0             # di mana kolom pertama positif, shape (100,)
positive_rows = data[mask]        # hanya baris dengan kolom pertama > 0
print(f"\nBoolean indexing:")
print(f"  Total baris: {data.shape[0]}")
print(f"  Baris dengan kolom pertama > 0: {positive_rows.shape[0]}")
print(f"  Proporsi: {positive_rows.shape[0] / data.shape[0]:.2%}")

# Boolean indexing dengan multiple conditions
# Gunakan & (AND), | (OR), ~ (NOT) - jangan lupa kurung karena precedence operator!
condition = (data[:, 0] > 0) & (data[:, 1] < 0)
selected = data[condition]
print(f"  Baris dengan col0>0 DAN col1<0: {selected.shape[0]}")

# np.where: mengembalikan index di mana kondisi True
# Bisa juga dipakai untuk conditional assignment: np.where(condition, value_if_true, value_if_false)
indices_where = np.where(data[:, 0] > 0)[0]
print(f"  np.where indices shape: {indices_where.shape}")

# --- Fancy indexing ---
# Menggunakan array of indices untuk memilih baris/kolom tertentu.
# Fancy indexing SELALU membuat copy (bukan view).
# Penggunaan di ML:
# - Shuffle data: data[permutation_indices]
# - K-fold cross validation: data[train_indices], data[val_indices]
# - Random sampling: data[np.random.choice(n, size=k, replace=False)]
indices = np.array([0, 5, 10, 50, 99])
selected = data[indices]
print(f"\nFancy indexing:")
print(f"  Indices: {indices}")
print(f"  Selected shape: {selected.shape}")

# np.clip: membatasi nilai dalam range [min, max]
# Berguna untuk:
# - Gradient clipping di RNN/LSTM (mencegah exploding gradients)
# - Membatasi pixel values [0, 255]
# - Winsorization untuk robust statistics
clipped = np.clip(data[:5, 0], -0.5, 0.5)
print(f"\n  Original first 5 of col0: {data[:5, 0].round(4)}")
print(f"  Clipped [-0.5, 0.5]:      {clipped.round(4)}")

# --- np.argmin, np.argmax, np.argsort ---
# argmin/argmax: mengembalikan INDEX dari nilai minimum/maximum
# Penggunaan: mencari prediksi kelas (class dengan logit tertinggi)
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"\n  Array: {arr}")
print(f"  Argmax (index max): {np.argmax(arr)} (value={arr[np.argmax(arr)]})")
print(f"  Argmin (index min): {np.argmin(arr)} (value={arr[np.argmin(arr)]})")

# argsort: mengembalikan index yang akan mengurutkan array
sorted_indices = np.argsort(arr)
print(f"  Argsort: {sorted_indices}")
print(f"  Array sorted: {arr[sorted_indices]}")
# Penggunaan: mengurutkan prediksi berdasarkan confidence score


# ===========================================================
# BAGIAN 5: Practical ML Operations
# ===========================================================
# Function-function ini sering muncul di implementasi ML.
# Memahami implementasinya dari nol membantu debugging dan optimasi.

# --- Softmax function - nanti dipakai di classification ---
# Softmax: sigma(z)_i = exp(z_i) / Sum_j exp(z_j)
# Mengubah vector logits menjadi distribusi probabilitas.
#
# Properties:
# - Output selalu positif (karena exp)
# - Sum of outputs = 1 (karena dibagi total)
# - Monotonic: higher input -> higher output
# - Differentiable (penting untuk backpropagation)
#
# Aplikasi di ML:
# - Output layer untuk multi-class classification
# - Attention mechanism (menghitung attention weights)
# - Cross-entropy loss membutuhkan probabilitas dari softmax

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
    - exp(z.max()) akan di-cancel di numerator dan denominator
      karena pembagi juga menggunakan exp yang sama.
    - Untuk matrix input, max dihitung per row (axis=1) karena
      setiap row adalah satu sample dengan logits independen.
    """
    # numerical stability: shift by max
    exp_z = np.exp(z - z.max())
    return exp_z / exp_z.sum()


logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"\nSoftmax:")
print(f"  Logits:  {logits}")
print(f"  Probs:   {probs.round(4)}")
print(f"  Sum:     {probs.sum():.4f} (harus = 1.0)")
print(f"  Max idx: {np.argmax(probs)} (kelas dengan probabilitas tertinggi)")

# --- Euclidean distance matrix - dipakai di KNN, clustering ---
# Distance matrix D[i,j] = jarak Euclidean antara point i dan point j.
# Formula: ||a - b||**2 = ||a||**2 + ||b||**2 - 2*a*b
# Identity ini memungkinkan perhitungan O(n**2) tanpa explicit loops.
#
# Kenapa ini efisien?
# - ||a||**2 dihitung sekali per point: O(n)
# - a*b dihitung via matrix multiplication: O(n**2) tapi optimized BLAS
# - Total lebih cepat daripada n**2 loop di Python
#
# Aplikasi di ML:
# - K-Nearest Neighbors: cari k tetangga terdekat
# - K-Means: hitung jarak ke centroid
# - DBSCAN: hitung jarak untuk density-based clustering
# - Gaussian kernel: exp(-||x-y||**2 / 2sigma**2)

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
    - Menggunakan identity: ||a-b||**2 = ||a||**2 + ||b||**2 - 2a*b
    - Ini menghindari explicit loop -> vectorized & C-speed
    - np.maximum(distances, 0) menghindari numerical errors yang bisa
      menghasilkan nilai negatif kecil akibat floating point precision.
    - keepdims=True menjaga dimensi agar broadcasting berfungsi dengan benar.
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a*b
    sq_norms = np.sum(X**2, axis=1, keepdims=True)  # shape (n, 1)
    # X @ X.T -> shape (n, n), setiap elemen [i,j] = dot(X[i], X[j])
    distances = sq_norms + sq_norms.T - 2 * X @ X.T
    # sq_norms.T -> shape (1, n)
    # sq_norms + sq_norms.T -> broadcasting -> shape (n, n)
    # Hasil symmetric, diagonal teoritis = 0 (tapi mungkin ada floating point noise)
    return np.sqrt(np.maximum(distances, 0))


points = np.random.randn(5, 2)
dist_matrix = pairwise_distance(points)
print(f"\nPairwise distance matrix (5 points):")
print(f"{dist_matrix.round(3)}")
print(f"  Symmetric? {np.allclose(dist_matrix, dist_matrix.T)}")
print(f"  Diagonal ~0? {np.allclose(np.diag(dist_matrix), 0)}")

# --- Cross-entropy loss (informasi) ---
# Cross-entropy mengukur perbedaan antara distribusi true (y) dan prediksi (p).
# H(y, p) = -Sum y_i * log(p_i)
# Untuk binary classification: H = -[y*log(p) + (1-y)*log(1-p)]
# Untuk multi-class: H = -log(p_{true_class})
#
# Numerical stability:
# - Clip probabilitas agar tidak tepat 0 atau 1 (log(0) = -inf)
# - Biasanya di-clip ke [epsilon, 1-epsilon] dengan epsilon ~1e-7

def binary_cross_entropy(y_true, y_pred, eps=1e-7):
    """
    Menghitung binary cross-entropy loss.
    
    Parameters:
    -----------
    y_true : np.ndarray, shape (n,)
        Label biner (0 atau 1).
    y_pred : np.ndarray, shape (n,)
        Probabilitas prediksi (0 sampai 1).
    eps : float
        Epsilon untuk clipping agar log(0) tidak terjadi.
    
    Returns:
    --------
    float
        Rata-rata cross-entropy loss.
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Demo cross-entropy
y_true = np.array([1, 0, 1, 1, 0])
y_pred_perfect = np.array([0.99, 0.01, 0.99, 0.99, 0.01])
y_pred_bad = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
print(f"\nBinary Cross-Entropy:")
print(f"  Perfect pred loss: {binary_cross_entropy(y_true, y_pred_perfect):.4f} (semakin kecil semakin baik)")
print(f"  Random pred loss:  {binary_cross_entropy(y_true, y_pred_bad):.4f}")



# ===========================================================
# 🏋️ EXERCISE 1: Implementasi Fungsi-fungsi Berikut
# ===========================================================
"""
TARGET Learning Objectives:
   - Setelah exercise ini, kamu akan menguasai operasi NumPy fundamental
   - Kamu akan bisa mengimplementasikan normalisasi, similarity, dan encoding
   - Kamu akan paham broadcasting dan vectorization secara praktis

PANDUAN LANGKAH-LANGKAH:

STEP 1: Implementasi batch_normalize(X)

Buat function batch_normalize(X) yang menerima array 2D dan mengembalikan
array yang sudah di-normalize per kolom (mean=0, std=1).

   - Input: array (N, D) -> N samples, D features
   - Output: array (N, D) yang sudah di-normalize per kolom
   - Rumus: X_norm = (X - mean) / std

   TIPS Apa yang harus dilakukan:
     a) Hitung mean per kolom menggunakan X.mean(axis=0)
     b) Hitung std per kolom menggunakan X.std(axis=0)
     c) Lakukan broadcasting: (X - mean) / std
     
   PERINGATAN Hati-hati: std bisa 0 untuk kolom konstan!
     Solusi: tambahkan epsilon kecil (1e-8) ke std.
     Jika std ~ 0, artinya fitur tersebut tidak memiliki variasi
     dan tidak memberikan informasi untuk model.
     
   Verification setelah implementasi:
     X_test = np.random.randn(50, 3)
     X_norm = batch_normalize(X_test)
     print(X_norm.mean(axis=0))  # Harus [~0, ~0, ~0]
     print(X_norm.std(axis=0))   # Harus [~1, ~1, ~1]

   INSIGHT Mengapa normalisasi penting?
     - Algoritma berbasis gradient (neural nets, linear/logistic regression)
       konvergen lebih cepat karena loss landscape lebih simetris.
     - Distance-based algorithms (KNN, K-Means, SVM dengan RBF kernel)
       tidak akan bias terhadap fitur dengan skala besar.
     - Regularization (L1/L2) memberikan penalty yang merata ke semua fitur.


STEP 2: Implementasi cosine_similarity(a, b)

Buat function cosine_similarity(a, b) yang menghitung cosine similarity
antara dua vektor 1D.

   - Input: dua vektor 1D (array 1D)
   - Output: skalar float antara -1 dan 1
   - Rumus: cos(theta) = (a * b) / (||a|| * ||b||)

   TIPS Apa yang harus dilakukan:
     a) Hitung dot product: np.dot(a, b) atau a @ b
     b) Hitung norm (magnitude) masing-masing vektor: np.linalg.norm(a)
     c) Bagi dot product dengan product of norms
     
   PERINGATAN Hati-hati: jika salah satu vektor adalah zero vector,
     denominator akan 0 -> division by zero!
     Solusi: tambahkan epsilon kecil atau handle dengan if.
     
   Verification setelah implementasi:
     a = np.array([1, 0, 0])
     b = np.array([0, 1, 0])
     print(cosine_similarity(a, b))  # Harus ~0 (orthogonal)
     print(cosine_similarity(a, a))  # Harus ~1 (identical)

   INSIGHT Interpretasi nilai:
     - cos(theta) = 1  -> vektor searah (identical direction)
     - cos(theta) = 0  -> vektor orthogonal (tidak berkorelasi)
     - cos(theta) = -1 -> vektor berlawanan arah
     - Di NLP: cosine similarity digunakan untuk mencari dokumen/keyword
       yang paling mirip dalam vector space (TF-IDF, word embeddings).
     - Di recommender systems: mengukur kesamaan preferensi user.


STEP 3: Implementasi one_hot_encode(labels, num_classes)

Buat function one_hot_encode(labels, num_classes) yang mengubah
array label integer menjadi matrix one-hot.

   - Input: array 1D berisi integer label, dan jumlah kelas
   - Output: array (N, num_classes) one-hot encoded
   - Contoh: [0, 2, 1] dengan 3 kelas -> [[1,0,0], [0,0,1], [0,1,0]]

   TIPS Apa yang harus dilakukan:
     a) Buat matrix nol dengan shape (len(labels), num_classes)
     b) Set elemen [i, labels[i]] = 1 untuk setiap i
        Gunakan integer array indexing: result[np.arange(N), labels] = 1
     
   PERINGATAN Hati-hati: labels harus integer 0..num_classes-1.
     Jika labels mencakup nilai >= num_classes, akan index out of bounds.
     Selalu validasi: assert np.all(labels < num_classes)
     
   Verification setelah implementasi:
     labels = np.array([0, 2, 1, 0])
     oh = one_hot_encode(labels, 3)
     print(oh)
     # Expected:
     # [[1, 0, 0],
     #  [0, 0, 1],
     #  [0, 1, 0],
     #  [1, 0, 0]]

   INSIGHT Mengapa one-hot encoding?
     - Model ML bekerja dengan numerik, bukan kategori string.
     - Untuk label klasifikasi multi-class, cross-entropy loss mengharapkan
       distribusi probabilitas (one-hot) atau index kelas.
     - Alternatif: label encoding (0, 1, 2) bisa menimbulkan ordinal
       relationship yang tidak ada (kelas 2 tidak "lebih besar" dari kelas 1).
     - Untuk fitur kategori: one-hot mencegah model mengasumsikan
       urutan/antara kategori (untuk tree-based models, label encoding OK).


TIPS HINTS:
   - Gunakan np.zeros() untuk inisialisasi matrix nol
   - Gunakan np.dot() atau @ untuk dot product
   - Gunakan np.linalg.norm() untuk menghitung magnitude vektor
   - Broadcasting di NumPy sangat powerful - manfaatkan!
   - Untuk one-hot, indexing 2D dengan array: arr[row_indices, col_indices] = values

PERINGATAN COMMON MISTAKES:
   - Lupa menambahkan axis=0 di mean/std -> menghasilkan scalar, bukan per kolom
   - Tidak menangani division by zero -> NaN atau inf
   - Lupa reshape labels untuk indexing 2D di one-hot encoding
   - Menggunakan (labels,) indexing yang salah -> gunakan np.arange(len(labels))

   CARA MENJALANKAN TEST:  python 01_numpy_essentials.py --exercise
"""


# --- Implementasi kamu (isi bagian TODO di bawah ini) ---
def batch_normalize(X):
    """
    Normalisasi per kolom: X_norm = (X - mean) / std.
    Input (N, D), output (N, D) dengan mean=0, std=1 per kolom.

    Tips: X.mean(axis=0), X.std(axis=0), tambahkan epsilon kecil
    agar tidak terjadi division by zero saat std = 0.
    """
    # TODO: implementasi di sini
    raise NotImplementedError("Implementasi batch_normalize dulu!")


def cosine_similarity(a, b):
    """
    cos(theta) = (a . b) / (||a|| * ||b||). Output skalar antara -1 dan 1.

    Tips: np.dot(a, b) dan np.linalg.norm(a).
    Hati-hati division by zero jika salah satu vektor adalah zero vector.
    """
    # TODO: implementasi di sini
    raise NotImplementedError("Implementasi cosine_similarity dulu!")


def one_hot_encode(labels, num_classes):
    """
    Ubah label integer -> matrix one-hot dengan shape (N, num_classes).

    Tips: result[np.arange(N), labels] = 1
    """
    # TODO: implementasi di sini
    raise NotImplementedError("Implementasi one_hot_encode dulu!")


# ===========================================================
# 🧪 TEST RUNNER: Cek jawaban secara otomatis (PASS/FAIL)
# ===========================================================
# Jalankan:  python 01_numpy_essentials.py --exercise
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "00-setup"))
from exercise_utils import safe_check, summary, run_exercise_mode


def run_exercise_tests():
    print("\n" + "=" * 50)
    print("🧪 Menjalankan test exercise...")
    print("=" * 50)

    X_test = np.random.randn(50, 3)
    safe_check("batch_normalize: mean ~ 0 per kolom",
               lambda: np.allclose(batch_normalize(X_test).mean(axis=0), 0, atol=1e-10))
    safe_check("batch_normalize: std ~ 1 per kolom",
               lambda: np.allclose(batch_normalize(X_test).std(axis=0), 1, atol=1e-10))

    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    safe_check("cosine_similarity: vektor orthogonal ~ 0",
               lambda: abs(cosine_similarity(a, b)) < 1e-10)
    safe_check("cosine_similarity: vektor identik ~ 1",
               lambda: abs(cosine_similarity(a, a) - 1.0) < 1e-10)

    labels = np.array([0, 2, 1, 0])
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    safe_check("one_hot_encode: output benar",
               lambda: np.array_equal(one_hot_encode(labels, 3), expected))

    summary()


if run_exercise_mode():
    run_exercise_tests()


# ===========================================================
# 🔥 CHALLENGE: Numerical Computing dengan NumPy
# ===========================================================
"""
TARGET Learning Objectives:
   - Memahami perbedaan implementasi manual vs library optimized
   - Mengimplementasikan algoritma numerik fundamental dari nol
   - Memahami kompleksitas komputasi dan trade-off
   - Mempersiapkan pemahaman untuk optimasi di PyTensor/PyTorch

PANDUAN LANGKAH-LANGKAH:

STEP 1: Implementasi Matrix Multiplication Manual

Implementasi matrix multiplication dari nol menggunakan loop Python.
JANGAN pakai @ atau np.dot - bangun dari definisi:

   C[i,j] = Sum_k A[i,k] * B[k,j]
   
   TIPS Apa yang harus dilakukan:
     a) Buat matrix C kosong dengan shape (m, p)
     b) Loop i dari 0 sampai m-1
     c) Loop j dari 0 sampai p-1
     d) Loop k dari 0 sampai n-1, akumulasi A[i,k] * B[k,j]
     
   PERINGATAN Hati-hati: matrix multiplication hanya valid jika
     A.shape[1] == B.shape[0] (inner dimensions harus sama).
     
   TIPS KENAPA manual?
     - Memahami kompleksitas O(m*n*p)
     - Memahami mengapa vectorized implementation JAUH lebih cepat
     - Memahami cache locality (loop order matters!)


STEP 2: Bandingkan Kecepatan Manual vs NumPy

Bandingkan waktu eksekusi matrix multiplication manual vs A @ B.

   TIPS Apa yang diuji:
     a) Buat matrix A dan B dengan shape (200, 200) atau lebih besar
     b) Jalankan manual multiplication dan catat waktu (time.time())
     c) Jalankan A @ B dan catat waktu
     d) Print speedup factor
     
   PERINGATAN Expected result: NumPy akan JAUH lebih cepat (50x-200x+)
     Karena:
     1. NumPy menggunakan BLAS/LAPACK (Fortran/C optimized)
     2. SIMD vectorization (AVX instructions)
     3. Cache-optimized blocking algorithms
     4. No Python interpreter overhead per element
     
   TIPS Bonus: coba ubah urutan loop (i-j-k vs i-k-j vs j-i-k)
     dan amati perbedaan performa. Loop i-k-j biasanya lebih cepat
     karena better cache locality untuk B dan C.


STEP 3: Implementasi Numerical Gradient

Implementasi numerical gradient untuk function f: R^n -> R.
Numerical gradient digunakan untuk:
- Verifikasi implementasi backpropagation (gradient checking)
- Kasus di mana analytical gradient sulit diturunkan

   Rumus: df/dx_i ~ (f(x + epsilon*e_i) - f(x - epsilon*e_i)) / (2epsilon)
   
   Dimana e_i adalah basis vector ke-i (1 di posisi i, 0 di lainnya).
   
   TIPS Apa yang harus dilakukan:
     a) Buat function f(x) yang menerima vector dan mengembalikan scalar
        Contoh: f(x) = Sum x_i**2 (sum of squares)
     b) Untuk setiap dimensi i:
        - Buat perturbasi positif: x_plus = x.copy(); x_plus[i] += eps
        - Buat perturbasi negatif: x_minus = x.copy(); x_minus[i] -= eps
        - Hitung (f(x_plus) - f(x_minus)) / (2*eps)
     c) Return vector gradient dengan shape sama seperti x
     
   PERINGATAN Pilih epsilon yang tepat:
     - Terlalu kecil: floating point cancellation error
     - Terlalu besar: aproksimasi tidak akurat
     - Sweet spot: sqrt(machine epsilon) ~ 1e-8 untuk float64

   Verification:
     f(x) = Sum x_i**2
     Analytical gradient: gradf = 2x
     Numerical gradient harus ~ 2x (check dengan np.allclose)


STEP 4: Efficient Batch Operations

Implementasi batch softmax dan batch cross-entropy.

   TIPS Apa yang harus dilakukan:
     a) Batch softmax: menerima matrix (N, C) dan mengembalikan (N, C)
        - Setiap row di-normalisasi independently
        - Gunakan axis=1 untuk max dan sum
        - Perhatikan keepdims=True untuk broadcasting
        
     b) Batch cross-entropy: menerima y_true (N, C) one-hot dan y_pred (N, C)
        - Clip y_pred untuk numerical stability
        - Hitung -Sum y_true * log(y_pred) per sample
        - Return mean loss
        
   TIPS KENAPA batch?
     - ML training memproses data dalam batch (mini-batch gradient descent)
     - Implementasi batch lebih efisien daripada loop per sample
     - Hardware (GPU) di-optimasi untuk operasi batch besar


TIPS HINTS:
   - np.zeros((m, p)) untuk inisialisasi matrix hasil
   - time.time() untuk timing
   - np.eye(n) untuk basis vectors
   - np.clip(x, min_val, max_val) untuk numerical stability
   - np.sum(axis=1, keepdims=True) untuk operasi per row

PERINGATAN COMMON MISTAKES:
   - Lupa .copy() saat membuat perturbasi -> memodifikasi original
   - Salah axis di operasi batch (axis=0 vs axis=1)
   - Tidak keepdims -> broadcasting gagal
   - Epsilon terlalu kecil/besar di numerical gradient

TARGET EXPECTED OUTPUT:
   - Speedup factor NumPy vs manual: 50x-500x (tergantung ukuran matrix)
   - Numerical gradient match dengan analytical: allclose dengan atol=1e-5
   - Batch loss menghasilkan skalar positif yang masuk akal

Insight utama:
- Python loops lambat karena interpreter overhead
- NumPy vectorization menggunakan C/Fortran backend
- Di PyTorch/TensorFlow, kita akan melihat konsep yang sama tapi dengan GPU
"""

print("\n" + "="*50)
print("OK Modul 1 selesai! Lanjut ke: 01-fondasi-data/02_pandas_essentials.py")
print("="*50)
