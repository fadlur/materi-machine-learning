"""
=============================================================
FASE 2 — MODUL 1: LINEAR REGRESSION FROM SCRATCH
=============================================================
Ini INTI dari pendekatan anti-tutorial-hell.
Kamu akan membangun Linear Regression dari NOL — hanya NumPy.

Kenapa? Karena Linear Regression mengandung SEMUA konsep fundamental:
- Loss function
- Gradient descent
- Overfitting & regularization
- Bias-variance tradeoff

Kalau kamu benar-benar paham ini, sisanya tinggal variasi.

Dengan background Teknik Elektro:
- Least squares → pasti sudah kenal dari curve fitting
- Gradient descent → mirip iterative optimization di control theory
- Matrix inverse → sudah jadi makanan sehari-hari

Durasi target: 3-4 jam (take your time, ini fondasi!)
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ===========================================================
# 📖 BAGIAN 1: Closed-Form Solution (Normal Equation)
# ===========================================================
# Linear Regression: y = Xw + b
# Atau dalam bentuk augmented: y = X̃θ, dimana X̃ = [X, 1] dan θ = [w, b]
#
# Solusi optimal (minimize MSE): θ* = (X̃ᵀX̃)⁻¹ X̃ᵀy
# Ini pasti familiar — ini cuma least squares!
#
# Koneksi Teknik Elektro:
# - Normal equation = least squares solution untuk overdetermined system
# - Mirip dengan pseudo-inverse di system identification

def generate_linear_data(n_samples=100, n_features=1, noise=0.3):
    """
    Generate data linear dengan noise Gaussian.
    
    Parameters:
    -----------
    n_samples : int, default 100
        Jumlah data points yang akan di-generate.
        
    n_features : int, default 1
        Jumlah fitur (dimensi X).
        
    noise : float, default 0.3
        Standard deviation dari noise Gaussian yang ditambahkan.
        Semakin besar → data semakin "berisik" dan sulit di-fit.
        
    Returns:
    --------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix. Setiap baris adalah satu sample.
        
    y : np.ndarray, shape (n_samples,)
        Target vector. y = X @ w + b + noise.
        
    true_weights : np.ndarray, shape (n_features,)
        True weights yang digunakan untuk generate data.
        Berguna untuk membandingkan dengan hasil learning.
        
    true_bias : float
        True bias yang digunakan untuk generate data.
        
    Notes:
    ------
    - np.random.randn(n_samples, n_features) menghasilkan X dari N(0,1)
    - Noise di-generate dari N(0, noise²)
    - true_weights juga di-generate random dari N(0,1)
    - Koneksi ke Teknik Elektro: mirip dengan pengukuran sensor
      dengan additive white Gaussian noise (AWGN channel)
    """
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    true_bias = 2.0
    y = X @ true_weights + true_bias + noise * np.random.randn(n_samples)
    return X, y, true_weights, true_bias


class LinearRegressionClosedForm:
    """
    Linear Regression dengan Closed-Form Solution (Normal Equation).
    
    Model: y = Xw + b
    Solusi: θ = (X̃ᵀX̃)⁻¹ X̃ᵀy, dimana X̃ = [X, 1], θ = [w, b]
    
    Attributes:
    -----------
    theta : np.ndarray
        Parameter gabungan [weights, bias].
    weights : np.ndarray
        Koefisien regresi (slope).
    bias : float
        Intercept.
        
    Notes:
    ------
    - Closed-form = solusi langsung tanpa iterasi
    - Kompleksitas: O(n²d + nd² + d³) untuk n samples, d features
    - Lebih cepat untuk data kecil, tapi mahal untuk data besar
    - Koneksi Teknik Elektro: mirip dengan direct solution
      di least squares estimation
    """
    
    def __init__(self):
        self.theta = None
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """
        Melatih model menggunakan Normal Equation.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data features.
            
        y : np.ndarray, shape (n_samples,)
            Training data targets.
            
        Returns:
        --------
        self : LinearRegressionClosedForm
            Object model yang sudah di-fit (untuk method chaining).
            
        Notes:
        ------
        - X di-augment dengan kolom 1 untuk menghandle bias
        - np.column_stack([X, np.ones(len(X))]) membuat X̃
        - np.linalg.solve(A, b) menghitung x dari Ax = b
        - Kenapa pakai solve bukan inverse?
          → Lebih stabil secara numerik. np.linalg.inv bisa amplify error.
          → solve menggunakan LU decomposition yang lebih stable.
        - Koneksi Teknik Elektro: mirip dengan solving linear system
          di nodal analysis (matrix admittance × node voltages = currents)
        """
        # Augment X dengan kolom 1 (untuk bias)
        # np.ones(len(X)) membuat vektor 1 dengan panjang n_samples
        X_aug = np.column_stack([X, np.ones(len(X))])
        
        # Normal equation: θ = (XᵀX)⁻¹ Xᵀy
        # Tapi kita pakai solve untuk numerical stability:
        # solve(XᵀX, Xᵀy) = (XᵀX)⁻¹ Xᵀy
        self.theta = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
        # @ = matrix multiplication operator (Python 3.5+)
        # X_aug.T @ X_aug menghasilkan matrix (d+1) × (d+1)
        # X_aug.T @ y menghasilkan vektor (d+1,)
        
        # Extract weights dan bias dari theta
        self.weights = self.theta[:-1]  # Semua kecuali elemen terakhir
        self.bias = self.theta[-1]      # Elemen terakhir = bias
        return self
    
    def predict(self, X):
        """
        Memprediksi target untuk data baru.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data yang akan diprediksi.
            
        Returns:
        --------
        np.ndarray, shape (n_samples,)
            Prediksi untuk setiap sample.
            
        Notes:
        ------
        - y_pred = X̃ @ θ = X @ w + b
        - Augment X dengan kolom 1 terlebih dahulu
        """
        X_aug = np.column_stack([X, np.ones(len(X))])
        return X_aug @ self.theta
    
    def score(self, X, y):
        """
        Menghitung R² score (coefficient of determination).
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data features.
            
        y : np.ndarray, shape (n_samples,)
            True targets.
            
        Returns:
        --------
        float
            R² score. Range: (-∞, 1].
            - 1.0 = perfect prediction
            - 0.0 = model seburuk memprediksi mean
            - Negatif = model lebih buruk dari memprediksi mean
            
        Notes:
        ------
        R² = 1 - SS_res / SS_tot
        SS_res = Σ(y - ŷ)² (sum of squared residuals)
        SS_tot = Σ(y - ȳ)² (total sum of squares)
        
        Koneksi Teknik Elektro: mirip dengan SNR (Signal-to-Noise Ratio)
        SS_tot = "signal power", SS_res = "noise power"
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - (ss_res / ss_tot)


# Test closed-form
X, y, true_w, true_b = generate_linear_data(200, 1, noise=0.5)

model_cf = LinearRegressionClosedForm()
model_cf.fit(X, y)

print("=== Closed-Form Solution ===")
print(f"True weights: {true_w}, True bias: {true_b}")
print(f"Learned weights: {model_cf.weights}, Learned bias: {model_cf.bias:.4f}")
print(f"R² score: {model_cf.score(X, y):.4f}")


# ===========================================================
# 📖 BAGIAN 2: Gradient Descent Solution
# ===========================================================
# Kenapa gradient descent kalau closed-form ada?
# 1. Closed-form butuh inverse matrix → O(n³) → lambat untuk data besar
# 2. Gradient descent bisa dipakai untuk SEMUA model, bukan cuma linear
# 3. Semua deep learning pakai gradient descent
#
# Dari Teknik Elektro: gradient descent = steepest descent optimization.
# Mirip dengan iterasi di adaptive filtering (LMS algorithm).

class LinearRegressionGD:
    """
    Linear Regression dengan Gradient Descent.
    
    Model: y = Xw + b
    Loss: MSE = (1/n) Σ(y - ŷ)²
    Update: w = w - lr * ∂L/∂w
    
    Attributes:
    -----------
    weights : np.ndarray
        Koefisien regresi (terupdate selama training).
    bias : float
        Intercept (terupdate selama training).
    loss_history : list
        Riwayat loss di setiap iterasi (untuk plotting convergence).
        
    Notes:
    ------
    - Gradient descent = iterative optimization
    - Learning rate menentukan step size
    - Convergence bergantung pada learning rate dan inisialisasi
    - Koneksi Teknik Elektro: mirip LMS adaptive filter
      w(n+1) = w(n) + μ * e(n) * x(n)
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Parameters:
        -----------
        learning_rate : float, default 0.01
            Step size untuk gradient descent.
            Terlalu kecil → convergence lambat.
            Terlalu besar → divergence (oscillate atau NaN).
            
        n_iterations : int, default 1000
            Jumlah iterasi gradient descent.
        """
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        """
        Melatih model menggunakan Gradient Descent.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data features.
            
        y : np.ndarray, shape (n_samples,)
            Training data targets.
            
        Returns:
        --------
        self : LinearRegressionGD
            Object model yang sudah di-fit.
            
        Notes:
        ------
        Algoritma per iterasi:
        1. Forward pass: ŷ = Xw + b
        2. Hitung loss: MSE = (1/n) Σ(ŷ - y)²
        3. Hitung gradient:
           ∂L/∂w = (2/n) Xᵀ(ŷ - y)
           ∂L/∂b = (2/n) Σ(ŷ - y)
        4. Update parameters: w -= lr * ∂L/∂w, b -= lr * ∂L/∂b
        
        Koneksi Teknik Elektro:
        - Gradient = arah steepest ascent (naik tercepat)
        - Negative gradient = arah steepest descent (turun tercepat)
        - Learning rate = step size (μ di adaptive filtering)
        """
        n_samples, n_features = X.shape
        
        # Inisialisasi random (atau zeros)
        # Random * 0.01 untuk membuat nilai awal kecil
        # Ini membantu convergence dan menghindari symmetry breaking issues
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        
        for i in range(self.n_iter):
            # === Forward pass: prediksi ===
            # ŷ = X @ w + b
            # Broadcasting: (n, d) @ (d,) → (n,) lalu + scalar → (n,)
            y_pred = X @ self.weights + self.bias
            
            # === Hitung loss (Mean Squared Error) ===
            # MSE = rata-rata dari squared errors
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
            # === Hitung gradient ===
            # error = ŷ - y → shape (n_samples,)
            error = y_pred - y
            # ∂L/∂w = (2/n) Xᵀ(error)
            # X.T @ error: (d, n) @ (n,) → (d,)
            dw = (2 / n_samples) * (X.T @ error)
            # ∂L/∂b = (2/n) Σ(error)
            db = (2 / n_samples) * np.sum(error)
            
            # === Update parameters (gradient descent step) ===
            # w_new = w_old - lr * gradient
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Print progress setiap 200 iterasi
            if (i + 1) % 200 == 0:
                print(f"  Iteration {i+1}/{self.n_iter}, Loss: {loss:.6f}")
        
        return self
    
    def predict(self, X):
        """
        Memprediksi target untuk data baru.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data yang akan diprediksi.
            
        Returns:
        --------
        np.ndarray, shape (n_samples,)
            Prediksi untuk setiap sample.
        """
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """
        Menghitung R² score.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data features.
            
        y : np.ndarray, shape (n_samples,)
            True targets.
            
        Returns:
        --------
        float
            R² score.
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - (ss_res / ss_tot)


# Test gradient descent
print("\n=== Gradient Descent Solution ===")
model_gd = LinearRegressionGD(learning_rate=0.1, n_iterations=1000)
model_gd.fit(X, y)

print(f"Learned weights: {model_gd.weights}, Learned bias: {model_gd.bias:.4f}")
print(f"R² score: {model_gd.score(X, y):.4f}")

# Plot loss curve — ini PENTING, harus selalu dilihat
# Loss curve menunjukkan apakah model belajar dengan baik
plt.figure(figsize=(8, 4))
plt.plot(model_gd.loss_history)
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.yscale('log')  # Log scale untuk melihat detail di awal training
plt.grid(True)
plt.savefig('01_loss_curve.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 01_loss_curve.png")


# ===========================================================
# 📖 BAGIAN 3: Polynomial Regression & Overfitting
# ===========================================================
# Ini demonstrasi konsep TERPENTING di ML: overfitting vs underfitting.
# Underfitting = model terlalu simpel, tidak menangkap pattern.
# Overfitting = model terlalu kompleks, menghapal noise.
# Good fit = model menangkap pattern tanpa menghapal noise.

def create_polynomial_features(X, degree):
    """
    Membuat fitur polynomial dari data input.
    
    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, 1)
        Data 1D (single feature).
        
    degree : int
        Derajat polynomial tertinggi.
        Contoh: degree=3 → [x, x², x³]
        
    Returns:
    --------
    np.ndarray, shape (n_samples, degree)
        Feature matrix dengan polynomial terms.
        
    Notes:
    ------
    - Polynomial features memungkinkan linear model untuk fit non-linear data
    - Ini adalah feature engineering sederhana
    - Semakin tinggi degree → semakin fleksibel (tapi risk overfitting)
    - Koneksi Teknik Elektro: mirik dengan Taylor series expansion
      untuk approximate non-linear functions
    """
    features = [X]  # x^1
    for d in range(2, degree + 1):
        features.append(X ** d)  # x^d
    return np.column_stack(features)


# Generate nonlinear data
np.random.seed(42)
# np.sort untuk membuat data terurut (lebih mudah di-plot)
X_nl = np.sort(np.random.uniform(-3, 3, 30)).reshape(-1, 1)
# y = sin(x) + noise → non-linear relationship
y_nl = np.sin(X_nl.ravel()) + 0.3 * np.random.randn(30)

# Fit dengan berbagai degree
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
degrees = [1, 4, 15]
titles = ['Underfitting (degree=1)', 'Good fit (degree=4)', 'Overfitting (degree=15)']

# X_plot untuk smooth curve visualization
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)

for ax, degree, title in zip(axes, degrees, titles):
    X_poly = create_polynomial_features(X_nl, degree)
    X_poly_plot = create_polynomial_features(X_plot, degree)
    
    # Gunakan closed-form untuk fitting cepat
    model = LinearRegressionClosedForm()
    model.fit(X_poly, y_nl)
    y_plot = model.predict(X_poly_plot)
    
    # Plot data points
    ax.scatter(X_nl, y_nl, color='blue', s=30, label='Data')
    # Plot fitted curve
    ax.plot(X_plot, y_plot, color='red', linewidth=2, label=f'Degree {degree}')
    # Plot true function (sinus)
    ax.plot(X_plot, np.sin(X_plot.ravel()), color='green',
            linestyle='--', alpha=0.5, label='True function')
    ax.set_title(title)
    ax.set_ylim(-2, 2)
    ax.legend()

plt.tight_layout()
plt.savefig('02_overfitting_demo.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 02_overfitting_demo.png")
print("\n🎯 Lihat gambar: degree 15 fit sempurna di training data,")
print("   tapi prediksi di antara titik jadi liar → OVERFITTING!")


# ===========================================================
# 📖 BAGIAN 4: Regularization (Ridge & Lasso)
# ===========================================================
# Solusi overfitting: tambahkan penalty ke loss function.
# Regularization membatasi magnitude weights → model lebih simpel.
#
# Ridge (L2): Loss = MSE + λ * ||w||²  → weights dikecilkan, tidak jadi 0
# Lasso (L1): Loss = MSE + λ * ||w||₁  → beberapa weights jadi PERSIS 0 (feature selection!)
#
# Koneksi Teknik Elektro:
# - L2 regularization = damping term di dynamic systems
# - L1 regularization = sparsity constraint (compressed sensing)

class RidgeRegressionGD:
    """
    Ridge Regression (L2 regularization) from scratch.
    
    Model: y = Xw + b
    Loss: MSE + α * ||w||²
    Update: w = w - lr * (∂MSE/∂w + 2αw)
    
    Attributes:
    -----------
    weights : np.ndarray
        Koefisien regresi.
    bias : float
        Intercept.
    loss_history : list
        Riwayat loss.
        
    Notes:
    ------
    - Ridge menambahkan penalty terhadap squared magnitude weights
    - Ini mendorong weights menjadi kecil, tapi tidak persis 0
    - α (alpha) = regularization strength
    - Semakin besar α → semakin kuat regularization → weights semakin kecil
    - Bias TIDAK diregularisasi (hanya weights)
    - Koneksi Teknik Elektro: mirip dengan adding damping
      untuk menghindari oscillation di control systems
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=1.0):
        """
        Parameters:
        -----------
        learning_rate : float, default 0.01
            Step size gradient descent.
            
        n_iterations : int, default 1000
            Jumlah iterasi.
            
        alpha : float, default 1.0
            Regularization strength (λ).
            alpha=0 → tidak ada regularization (ordinary least squares)
            alpha besar → weights sangat kecil (strong regularization)
        """
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.alpha = alpha
        self.loss_history = []
    
    def fit(self, X, y):
        """
        Melatih model dengan Ridge Regression.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.
            
        y : np.ndarray, shape (n_samples,)
            Training targets.
            
        Returns:
        --------
        self : RidgeRegressionGD
            Object model yang sudah di-fit.
            
        Notes:
        ------
        Gradient dengan regularization:
        ∂L/∂w = (2/n) Xᵀ(ŷ - y) + 2αw
        ∂L/∂b = (2/n) Σ(ŷ - y)  ← bias tidak diregularisasi!
        
        Kenapa bias tidak diregularisasi?
        - Bias hanya menggeser output, tidak mempengaruhi kompleksitas model
        - Regularizing bias bisa menyebabkan underfitting
        """
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        
        for i in range(self.n_iter):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y
            
            # Loss = MSE + alpha * ||w||²
            mse = np.mean(error ** 2)
            reg_loss = self.alpha * np.sum(self.weights ** 2)
            loss = mse + reg_loss
            self.loss_history.append(loss)
            
            # Gradient dengan regularization term
            # ∂L/∂w = ∂MSE/∂w + ∂(α||w||²)/∂w
            #       = (2/n)Xᵀ(error) + 2αw
            dw = (2/n_samples) * (X.T @ error) + 2 * self.alpha * self.weights
            # bias tidak diregularisasi!
            db = (2/n_samples) * np.sum(error)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
        return self
    
    def predict(self, X):
        """
        Memprediksi target untuk data baru.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data yang akan diprediksi.
            
        Returns:
        --------
        np.ndarray, shape (n_samples,)
            Prediksi untuk setiap sample.
        """
        return X @ self.weights + self.bias


# Demo: Ridge vs No Regularization pada polynomial degree 15
print("\n=== Regularization Demo ===")
X_poly = create_polynomial_features(X_nl, 15)
X_poly_plot = create_polynomial_features(X_plot, 15)

# Normalize features (penting untuk polynomial!)
# Polynomial features bisa memiliki scale yang sangat berbeda
# (x=10 → x^15 = 10^15!). Normalisasi mencegah numerical issues.
mean = X_poly.mean(axis=0)
std = X_poly.std(axis=0) + 1e-8  # +epsilon untuk menghindari division by zero
X_poly_norm = (X_poly - mean) / std
X_poly_plot_norm = (X_poly_plot - mean) / std

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
alphas = [0, 0.01, 1.0]
titles = ['No Regularization', 'Ridge α=0.01', 'Ridge α=1.0']

for ax, alpha, title in zip(axes, alphas, titles):
    if alpha == 0:
        model = LinearRegressionClosedForm()
        model.fit(X_poly_norm, y_nl)
    else:
        model = RidgeRegressionGD(learning_rate=0.01, n_iterations=5000, alpha=alpha)
        model.fit(X_poly_norm, y_nl)
    
    y_plot = model.predict(X_poly_plot_norm)
    
    ax.scatter(X_nl, y_nl, color='blue', s=30)
    ax.plot(X_plot, y_plot, color='red', linewidth=2)
    ax.set_title(title)
    ax.set_ylim(-2, 2)

plt.tight_layout()
plt.savefig('03_regularization_demo.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 03_regularization_demo.png")


# ===========================================================
# 📖 BAGIAN 5: Train/Test Split & Cross-Validation
# ===========================================================
# Train/test split = membagi data untuk training dan evaluasi.
# Cross-validation = evaluasi yang lebih robust dengan multiple splits.

def train_test_split(X, y, test_ratio=0.2, seed=42):
    """
    Manual train-test split dengan random permutation.
    
    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
        
    y : np.ndarray, shape (n_samples,)
        Target vector.
        
    test_ratio : float, default 0.2
        Proporsi data untuk test set (0.2 = 20%).
        
    seed : int, default 42
        Random seed untuk reproducibility.
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : np.ndarray
        Data yang sudah di-split.
        
    Notes:
    ------
    - np.random.permutation(n) membuat array [0, 1, ..., n-1] yang diacak
    - test_idx = 20% pertama dari array yang diacak
    - train_idx = sisanya (80%)
    - Seed memastikan split yang sama setiap kali di-run
    - Koneksi Teknik Elektro: mirip dengan random sampling
      di Monte Carlo simulations
    """
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    test_size = int(n * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def k_fold_cross_validation(X, y, model_class, model_params, k=5):
    """
    K-Fold Cross Validation — manual implementation.
    
    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
        
    y : np.ndarray, shape (n_samples,)
        Target vector.
        
    model_class : class
        Class model yang akan dievaluasi (misal LinearRegressionClosedForm).
        
    model_params : dict
        Dictionary parameter untuk constructor model_class.
        Contoh: {'learning_rate': 0.1, 'n_iterations': 1000}
        
    k : int, default 5
        Jumlah fold.
        
    Returns:
    --------
    scores : list of float
        List score untuk setiap fold.
        
    Notes:
    ------
    Algoritma K-Fold CV:
    1. Bagi data menjadi k subset (folds) yang sama besar
    2. Untuk setiap fold i:
       a. Fold i = test set
       b. Semua fold lainnya = train set
       c. Train model, evaluasi pada test
    3. Return average score
    
    Keunggulan CV:
    - Semua data digunakan untuk training DAN testing
    - Hasil lebih robust (variance lebih rendah)
    - Mendeteksi overfitting lebih baik
    
    Koneksi Teknik Elektro: mirip dengan averaging multiple
    measurements untuk mengurangi noise
    """
    n = len(X)
    indices = np.random.permutation(n)
    fold_size = n // k
    scores = []
    
    for i in range(k):
        # Split
        test_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size],
                                     indices[(i + 1) * fold_size:]])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train & evaluate
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
        print(f"  Fold {i+1}: R² = {score:.4f}")
    
    print(f"  Average R²: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores


print("\n=== 5-Fold Cross Validation ===")
X_multi, y_multi, _, _ = generate_linear_data(200, 5, noise=0.5)
scores = k_fold_cross_validation(
    X_multi, y_multi,
    LinearRegressionClosedForm, {},
    k=5
)


# ===========================================================
# 🏋️ EXERCISE 4: Implementasi dari Nol
# ===========================================================
"""
🎯 Learning Objectives:
   - Memahami perbedaan L1 dan L2 regularization secara praktis
   - Mengimplementasikan varian gradient descent (SGD, Mini-batch)
   - Membandingkan convergence behavior berbagai optimizer

📋 LANGKAH-LANGKAH:

STEP 1: Implementasi Lasso Regression (L1 regularization)
──────────────────────────────────────────────────────────
Lasso = Least Absolute Shrinkage and Selection Operator.

   Loss: MSE + λ * Σ|w_i|
   
   💡 Apa yang harus dilakukan:
     a) Buat class LassoRegressionGD mirip RidgeRegressionGD
     b) Gradient dari |w| tidak smooth di w=0
     c) Gunakan subgradient:
        - sign(w) di w ≠ 0
        - 0 di w = 0
        
     Pseudocode untuk gradient:
     ```
     dw = (2/n) * X.T @ error + alpha * np.sign(weights)
     ```
     
   ⚠️ Hati-hati:
     - np.sign(0) = 0, yang benar untuk subgradient
     - Lasso bisa menghasilkan weights yang persis 0 (sparse solution)
     - Ini berarti automatic feature selection!
     
   💡 KENAPA Lasso?
     - Feature selection: fitur tidak penting → weight = 0
     - Interpretable model: hanya fitur penting yang tersisa
     - Berguna untuk high-dimensional data


STEP 2: Implementasi Stochastic Gradient Descent (SGD)
──────────────────────────────────────────────────────
SGD = update weights per SAMPLE (bukan per batch).

   💡 Apa yang harus dilakukan:
     a) Buat class LinearRegressionSGD
     b) Di setiap epoch, shuffle data
     c) Untuk setiap sample:
        - Hitung prediksi untuk 1 sample
        - Hitung gradient untuk 1 sample
        - Update weights
        
     Pseudocode:
     ```
     for epoch in range(n_epochs):
         indices = np.random.permutation(n)
         for i in indices:
             xi = X[i:i+1]  # shape (1, d)
             yi = y[i]       # scalar
             y_pred = xi @ weights + bias
             error = y_pred - yi
             dw = 2 * xi.T @ error  # note: tidak dibagi n
             db = 2 * error
             weights -= lr * dw.ravel()
             bias -= lr * db
     ```
     
   d) Tambahkan learning rate decay:
      lr = lr_initial / (1 + decay * epoch)
      
   💡 KENAPA SGD?
     - Lebih cepat per epoch (update lebih sering)
     - Noise dalam gradient bisa membantu escape local minima
     - Standard untuk training neural networks
     
   ⚠️ Hati-hati:
     - Gradient per sample sangat noisy
     - Loss curve akan oscillate
     - Learning rate lebih kecil biasanya diperlukan


STEP 3: Implementasi Mini-batch Gradient Descent
─────────────────────────────────────────────────
Mini-batch = kompromi antara full-batch dan SGD.

   💡 Apa yang harus dilakukan:
     a) Buat class LinearRegressionMiniBatch
     b) Parameter: batch_size (default 32)
     c) Di setiap epoch:
        - Shuffle data
        - Bagi menjadi batches: for i in range(0, n, batch_size)
        - Update per batch
        
     Pseudocode:
     ```
     for epoch in range(n_epochs):
         indices = np.random.permutation(n)
         for start in range(0, n, batch_size):
             end = min(start + batch_size, n)
             batch_idx = indices[start:end]
             X_batch = X[batch_idx]
             y_batch = y[batch_idx]
             # Compute gradient for batch
             # Update weights
     ```
     
   💡 KENAPA Mini-batch?
     - Lebih stabil dari SGD (gradient dari batch lebih smooth)
     - Lebih cepat dari full-batch (update lebih sering)
     - Bisa di-parallelize (GPU-friendly)
     - Batch size = tradeoff antara stability dan speed


STEP 4: Bandingkan Semua Implementasi
─────────────────────────────────────
Gunakan dataset yang sama dan bandingkan:

   a) Convergence speed (plot loss curve)
      - Full-batch GD: smooth curve, tapi lambat per epoch
      - SGD: noisy curve, tapi cepat converge
      - Mini-batch: middle ground
      
   b) Final performance (R² score)
      - Semua seharusnya mencapai performance yang sama
      - Jika tidak, cek learning rate atau jumlah iterasi
      
   c) Computational time
      - SGD: paling cepat per epoch
      - Full-batch: paling lambat per epoch
      - Tapi total time to convergence bisa berbeda!


💡 HINTS:
   - Untuk Lasso, gunakan np.sign() untuk subgradient
   - Untuk SGD, learning rate lebih kecil (~0.01 atau lebih kecil)
   - Untuk Mini-batch, batch_size = 32 atau 64 adalah standar
   - Simpan loss_history untuk setiap variant
   - Gunakan plt.semilogy() untuk plot loss dalam log scale

⚠️ COMMON MISTAKES:
   - Lupa shuffle data di SGD/Mini-batch
   - Learning rate terlalu besar → divergence
   - Tidak normalisasi features untuk polynomial → numerical overflow
   - Membandingkan dengan jumlah iterasi yang berbeda
   - Lupa decay learning rate → SGD tidak converge

🧪 VERIFICATION:
   ```python
   # Semua model harus mencapai R² > 0.9 pada data linear dengan noise kecil
   X_test, y_test, _, _ = generate_linear_data(100, 3, noise=0.1)
   for name, model in [('GD', model_gd), ('SGD', model_sgd), ('MiniBatch', model_mb)]:
       print(f"{name}: R² = {model.score(X_test, y_test):.4f}")
   ```

🎯 EXPECTED INSIGHTS:
   - Full-batch: paling smooth, tapi lambat untuk data besar
   - SGD: paling cepat per iterasi, tapi noisy
   - Mini-batch: best of both worlds
   - Lasso: beberapa weights akan persis 0 (feature selection)
"""


# ===========================================================
# 🔥 CHALLENGE: Multivariate Regression untuk Sensor Data
# ===========================================================
"""
🎯 Learning Objectives:
   - Mengaplikasikan semua konsep ke dataset realistis (EE domain)
   - Membangun end-to-end pipeline dari data generation sampai evaluation
   - Menganalisis feature importance dan model diagnostics

📋 LANGKAH-LANGKAH:

STEP 1: Define Problem & Generate Data
───────────────────────────────────────
Konteks: Prediksi konsumsi daya (Watt) dari sebuah motor listrik
berdasarkan sensor readings.

   Features:
   - Tegangan (V): np.random.normal(220, 10, 1000)
   - Arus (A): np.random.normal(5, 1, 1000)
   - Temperatur (°C): np.random.normal(40, 5, 1000)
   - RPM: np.random.uniform(1000, 3000, 1000)
   - Vibrasi (mm/s): np.random.exponential(2, 1000)
   - Kelembaban (%): np.random.normal(60, 10, 1000)
   
   Target (Power):
   P = V * I * pf + 0.001 * RPM² + 0.1 * vibration² - 0.5 * temperature + noise
   
   💡 KENAPA formula ini?
     - V*I = apparent power (dasar dari EE)
     - RPM² = losses naik kuadrat dengan kecepatan (friction, windage)
     - vibration² = mechanical losses
     - temperature = efisiensi menurun saat panas
     - noise = measurement uncertainty


STEP 2: Split Data & Baseline Model
────────────────────────────────────
   a) Split 80/20 (train/test)
   b) Fit LinearRegressionClosedForm (baseline)
   c) Report R², MSE, MAE
   d) Plot: Actual vs Predicted


STEP 3: Feature Engineering
───────────────────────────
   a) Tambahkan polynomial features (degree 2):
      - V², I², V*I, RPM², etc.
      
   b) Tambahkan interaction terms:
      - V*I (sudah ada di power formula!)
      - Temperature * RPM (thermal stress)
      
   c) Normalisasi features sebelum training
      - Fit scaler pada training data ONLY
      - Transform both train dan test
      
   💡 KENAPA polynomial?
     - True relationship non-linear (RPM², vibration²)
     - Linear model bisa fit non-linear dengan polynomial features


STEP 4: Compare Models
──────────────────────
   a) Baseline Linear (no polynomial)
   b) Linear + Polynomial degree 2
   c) Ridge Regression + Polynomial degree 2
   d) Lasso Regression + Polynomial degree 2
   
   Untuk setiap model:
   - Cross-validation score (5-fold)
   - Test set R²
   - Training time
   - Number of non-zero weights (untuk Lasso)


STEP 5: Hyperparameter Tuning (Ridge/Lasso Alpha)
──────────────────────────────────────────────────
   a) Coba alpha values: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
   b) Plot: Validation R² vs Alpha (log scale)
   c) Pilih alpha optimal
   d) Analisis: apa yang terjadi saat alpha terlalu kecil/besar?


STEP 6: Visualisasi & Report
────────────────────────────
   Buat figure dengan 4 subplot:
   a) Actual vs Predicted (scatter plot)
      - Points harus mendekati garis y=x
      
   b) Residual Plot (predicted vs residual)
      - Residual = y_true - y_pred
      - Harus random, no pattern
      
   c) Residual Distribution (histogram)
      - Harus normal (Gaussian)
      
   d) Feature Importance (bar plot)
      - Untuk Ridge/Lasso: |weights| sebagai importance
      - Sort dari yang paling besar


💡 HINTS:
   - Gunakan np.column_stack() untuk menggabungkan features
   - Gunakan StandardScaler dari sklearn (atau manual: (X-mean)/std)
   - Untuk residual analysis, gunakan stats.probplot untuk QQ plot
   - Feature importance = np.abs(model.weights)

⚠️ COMMON MISTAKES:
   - Normalisasi SEBELUM split → data leakage!
   - Polynomial degree terlalu tinggi → overfitting
   - Tidak handle multicollinearity (V*I dan V, I berkorelasi tinggi)
   - Mengabaikan residual pattern → model misspecification

🎯 DELIVERABLES:
   - Code yang well-documented
   - 4 visualisasi dalam satu figure
   - Comparison table (model vs R² vs training time)
   - Analisis feature importance (mana yang paling penting?)
   - Rekomendasi model terbaik untuk deployment

Ini adalah mini-project yang menggabungkan semua yang sudah dipelajari!
"""

print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 02-ml-dari-nol/02_logistic_regression_scratch.py")
print("="*50)
