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

def generate_linear_data(n_samples=100, n_features=1, noise=0.3):
    """Generate data linear dengan noise"""
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    true_bias = 2.0
    y = X @ true_weights + true_bias + noise * np.random.randn(n_samples)
    return X, y, true_weights, true_bias


class LinearRegressionClosedForm:
    """Linear Regression dengan Normal Equation"""

    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        # Augment X dengan kolom 1 (untuk bias)
        X_aug = np.column_stack([X, np.ones(len(X))])

        # Normal equation: θ = (XᵀX)⁻¹ Xᵀy
        self.theta = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
        # Kenapa pakai solve bukan inverse?
        # → Lebih stabil secara numerik. np.linalg.inv bisa amplify error.

        self.weights = self.theta[:-1]
        self.bias = self.theta[-1]
        return self

    def predict(self, X):
        X_aug = np.column_stack([X, np.ones(len(X))])
        return X_aug @ self.theta

    def score(self, X, y):
        """R² score"""
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

class LinearRegressionGD:
    """Linear Regression dengan Gradient Descent"""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Inisialisasi random (atau zeros)
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

        for i in range(self.n_iter):
            # Forward pass: prediksi
            y_pred = X @ self.weights + self.bias

            # Hitung loss (Mean Squared Error)
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)

            # Hitung gradient
            # ∂L/∂w = (2/n) * Xᵀ(ŷ - y)
            # ∂L/∂b = (2/n) * Σ(ŷ - y)
            error = y_pred - y
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            # Update parameters (gradient descent step)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if (i + 1) % 200 == 0:
                print(f"  Iteration {i+1}/{self.n_iter}, Loss: {loss:.6f}")

        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
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
plt.figure(figsize=(8, 4))
plt.plot(model_gd.loss_history)
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.yscale('log')
plt.grid(True)
plt.savefig('01_loss_curve.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 01_loss_curve.png")


# ===========================================================
# 📖 BAGIAN 3: Polynomial Regression & Overfitting
# ===========================================================
# Ini demonstrasi konsep TERPENTING di ML: overfitting vs underfitting

def create_polynomial_features(X, degree):
    """Buat fitur polynomial: [x, x², x³, ..., xⁿ]"""
    features = [X]
    for d in range(2, degree + 1):
        features.append(X ** d)
    return np.column_stack(features)


# Generate nonlinear data
X_nl = np.sort(np.random.uniform(-3, 3, 30)).reshape(-1, 1)
y_nl = np.sin(X_nl.ravel()) + 0.3 * np.random.randn(30)

# Fit dengan berbagai degree
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
degrees = [1, 4, 15]
titles = ['Underfitting (degree=1)', 'Good fit (degree=4)', 'Overfitting (degree=15)']

X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)

for ax, degree, title in zip(axes, degrees, titles):
    X_poly = create_polynomial_features(X_nl, degree)
    X_poly_plot = create_polynomial_features(X_plot, degree)

    model = LinearRegressionClosedForm()
    model.fit(X_poly, y_nl)
    y_plot = model.predict(X_poly_plot)

    ax.scatter(X_nl, y_nl, color='blue', s=30, label='Data')
    ax.plot(X_plot, y_plot, color='red', linewidth=2, label=f'Degree {degree}')
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
# Solusi overfitting: tambahkan penalty ke loss function
#
# Ridge (L2): Loss = MSE + λ * ||w||²  → weights dikecilkan, tidak jadi 0
# Lasso (L1): Loss = MSE + λ * ||w||₁  → beberapa weights jadi PERSIS 0 (feature selection!)

class RidgeRegressionGD:
    """Ridge Regression (L2 regularization) from scratch"""

    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=1.0):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.alpha = alpha  # regularization strength
        self.loss_history = []

    def fit(self, X, y):
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
            dw = (2/n_samples) * (X.T @ error) + 2 * self.alpha * self.weights
            db = (2/n_samples) * np.sum(error)  # bias tidak diregularisasi!

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X):
        return X @ self.weights + self.bias


# Demo: Ridge vs No Regularization pada polynomial degree 15
print("\n=== Regularization Demo ===")
X_poly = create_polynomial_features(X_nl, 15)
X_poly_plot = create_polynomial_features(X_plot, 15)

# Normalize features (penting untuk polynomial!)
mean = X_poly.mean(axis=0)
std = X_poly.std(axis=0) + 1e-8
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

def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Manual train-test split"""
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    test_size = int(n * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def k_fold_cross_validation(X, y, model_class, model_params, k=5):
    """K-Fold Cross Validation — manual implementation"""
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
1. Implementasi Lasso Regression (L1 regularization) from scratch.
   Hint: Gradient dari |w| tidak smooth di w=0.
   Gunakan subgradient: sign(w) di w≠0, dan 0 di w=0

2. Implementasi Stochastic Gradient Descent (SGD):
   Beda dengan GD biasa → update per SAMPLE, bukan per batch
   - Tambahkan learning rate decay: lr = lr_initial / (1 + decay * epoch)
   - Bandingkan convergence speed SGD vs full-batch GD

3. Implementasi Mini-batch Gradient Descent:
   - Batch size sebagai parameter
   - Shuffle data setiap epoch

Uji ketiga implementasi pada dataset yang sama dan bandingkan:
- Convergence speed (plot loss curve)
- Final performance (R² score)
- Computational time
"""


# ===========================================================
# 🔥 CHALLENGE: Multivariate Regression untuk Sensor Data
# ===========================================================
"""
Buat skenario realistis dari background Teknik Elektro:

Konteks: Prediksi konsumsi daya (Watt) dari sebuah motor listrik
berdasarkan sensor readings:
- Tegangan (V)
- Arus (A)
- Temperatur (°C)
- RPM
- Vibrasi (mm/s)
- Kelembaban (%)

Generate synthetic data di mana:
- Daya = f(V, I, T, RPM, V_vibration, H) + noise
- Sertakan interaksi non-linear (misal: P ∝ V*I, vibrasi naik di RPM tinggi)

Tugas:
1. Generate 1000 data points
2. Split train/test
3. Fit model linear → evaluate
4. Tambahkan polynomial features (interaksi) → evaluate
5. Coba berbagai alpha untuk Ridge → mana yang optimal?
6. Buat report visualisasi: actual vs predicted, residual plot, feature importance

Ini adalah mini-project yang menggabungkan semua yang sudah dipelajari!
"""

print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 02-ml-dari-nol/02_logistic_regression_scratch.py")
print("="*50)
