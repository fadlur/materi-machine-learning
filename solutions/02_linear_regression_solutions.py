"""
=============================================================
SOLUSI REFERENSI - 01_LINEAR_REGRESSION_SCRATCH (VARIAN GD)
=============================================================
Referensi jawaban untuk exercise gradient descent di
02-ml-dari-nol/01_linear_regression_scratch.py:
    - Full-batch Gradient Descent  (LassoRegressionGD)
    - Stochastic Gradient Descent  (LinearRegressionSGD)
    - Mini-batch Gradient Descent  (LinearRegressionMiniBatch)

⚠️  JANGAN dibuka sebelum mencoba sendiri!
    Target: semua varian mencapai R^2 > 0.9 pada data linear.

Jalankan solusi ini:
    python 02_linear_regression_solutions.py
=============================================================
"""

import numpy as np

np.random.seed(42)


def generate_linear_data(n_samples=200, n_features=3, noise=0.1):
    """Generate data linear dengan noise Gaussian kecil."""
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    true_bias = 2.0
    y = X @ true_weights + true_bias + noise * np.random.randn(n_samples)
    return X, y, true_weights, true_bias


class LassoRegressionGD:
    """
    Lasso = MSE + alpha * ||w||_1.

    Catatan penting:
    - L1 penalty membuat sebagian weights menjadi PERSIS 0
      (feature selection) karena subgradient np.sign(w) konstan.
    - Turunan |w| tidak terdefinisi di w=0, pakai subgradient:
      d|w|/dw = np.sign(w)  (dengan sign(0) = 0).
    - Alpha terlalu besar -> semua weights menuju 0 (underfit).
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=0.01):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.alpha = alpha
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

        for i in range(self.n_iter):
            y_pred = X @ self.weights + self.bias
            loss = np.mean((y_pred - y) ** 2) + self.alpha * np.sum(np.abs(self.weights))
            self.loss_history.append(loss)

            error = y_pred - y
            dw = (2 / n_samples) * (X.T @ error) + self.alpha * np.sign(self.weights)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        ss_res = np.sum((y - self.predict(X)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot


class LinearRegressionSGD:
    """
    Stochastic Gradient Descent: update per sample (satu per satu).

    Catatan penting:
    - Shuffle data di setiap epoch (np.random.permutation) agar
      urutan sample tidak menciptakan bias.
    - Gradient per sample lebih noisy -> learning rate HARUS lebih
      kecil (~0.01 atau lebih kecil) agar tidak divergence.
    - Loss yang dicatat = loss rata-rata di akhir epoch.
    """

    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            for i in indices:
                xi, yi = X[i], y[i]
                y_pred = np.dot(xi, self.weights) + self.bias
                error = y_pred - yi
                # Update per sample (faktor 2 dari turunan squared error)
                self.weights -= self.lr * 2 * error * xi
                self.bias -= self.lr * 2 * error

            # Catat loss rata-rata di akhir epoch (untuk plotting)
            y_pred_all = X @ self.weights + self.bias
            self.loss_history.append(np.mean((y_pred_all - y) ** 2))
        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        ss_res = np.sum((y - self.predict(X)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot


class LinearRegressionMiniBatch:
    """
    Mini-batch Gradient Descent: update per batch (misal 32/64 sample).

    Catatan penting:
    - Kompromi antara full-batch (stabil tapi lambat) dan SGD
      (cepat tapi noisy). Inilah varian standar di deep learning.
    - Shuffle di setiap epoch, lalu bagi jadi batch-batch.
    - Gradient dihitung per batch, bukan per sample.
    """

    def __init__(self, learning_rate=0.01, n_epochs=100, batch_size=32):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                Xb, yb = X[batch_idx], y[batch_idx]
                n_batch = len(batch_idx)

                y_pred = Xb @ self.weights + self.bias
                error = y_pred - yb
                dw = (2 / n_batch) * (Xb.T @ error)
                db = (2 / n_batch) * np.sum(error)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            y_pred_all = X @ self.weights + self.bias
            self.loss_history.append(np.mean((y_pred_all - y) ** 2))
        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        ss_res = np.sum((y - self.predict(X)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot


# ===========================================================
# Verifikasi: semua varian harus R^2 > 0.9 pada data linear
# ===========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Verifikasi varian Gradient Descent...")
    print("=" * 60)

    # PENTING: generate SATU dataset lalu split train/test.
    # Jika train & test di-generate terpisah, true_weights-nya BEDA
    # sehingga model men-fit relasi A tapi dievaluasi pada relasi B
    # (ini menghasilkan R^2 negatif — data leakage terbalik).
    X, y, true_w, true_b = generate_linear_data(300, 3, noise=0.1)
    split = 200
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    models = {
        "Lasso (alpha=0.01)": LassoRegressionGD(learning_rate=0.01, n_iterations=1000, alpha=0.01),
        "SGD": LinearRegressionSGD(learning_rate=0.005, n_epochs=100),
        "MiniBatch (batch=32)": LinearRegressionMiniBatch(learning_rate=0.01, n_epochs=100, batch_size=32),
    }

    all_pass = True
    for name, model in models.items():
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        ok = r2 > 0.9
        all_pass = all_pass and ok
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}: {name} -> R^2 = {r2:.4f}")

    print("=" * 60)
    if all_pass:
        print("🎉 Semua varian gradient descent mencapai R^2 > 0.9!")
    else:
        print("⚠️  Ada varian yang belum mencapai R^2 > 0.9.")
        print("    Cek learning rate (jangan terlalu besar -> divergence),")
        print("    dan jumlah iterasi/epoch (mungkin perlu ditambah).")
    print("=" * 60)
