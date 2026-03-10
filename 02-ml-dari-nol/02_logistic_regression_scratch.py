"""
=============================================================
FASE 2 — MODUL 2: LOGISTIC REGRESSION FROM SCRATCH
=============================================================
Dari regression ke classification.

Logistic Regression BUKAN regression — ini CLASSIFIER.
Namanya menyesatkan, tapi ini salah satu model paling penting:
- Baseline untuk semua classification task
- Building block dari Neural Networks (setiap neuron = logistic regression!)
- Probabilistic output → bisa dikalibrasi

Koneksi Teknik Elektro:
- Sigmoid function = transfer function (seperti di op-amp saturation)
- Cross-entropy = information theory (Shannon entropy)

Durasi target: 3-4 jam
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ===========================================================
# 📖 BAGIAN 1: Sigmoid Function & Binary Classification
# ===========================================================

def sigmoid(z):
    """
    Sigmoid: σ(z) = 1 / (1 + exp(-z))

    Properties:
    - Output selalu antara 0 dan 1 → interpretasi sebagai probabilitas
    - σ(0) = 0.5
    - σ'(z) = σ(z) * (1 - σ(z)) → gradient mudah dihitung!
    - Mirip dengan saturating transfer function di elektronika
    """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip untuk numerical stability


# Visualisasi sigmoid
z = np.linspace(-10, 10, 200)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Sigmoid
axes[0].plot(z, sigmoid(z), linewidth=2)
axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[0].set_title('Sigmoid Function σ(z)')
axes[0].set_xlabel('z')
axes[0].set_ylabel('σ(z)')
axes[0].grid(True)

# Gradient sigmoid
sig = sigmoid(z)
grad_sig = sig * (1 - sig)
axes[1].plot(z, grad_sig, linewidth=2, color='orange')
axes[1].set_title("Sigmoid Gradient σ'(z)")
axes[1].set_xlabel('z')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('01_sigmoid.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 01_sigmoid.png")


# ===========================================================
# 📖 BAGIAN 2: Cross-Entropy Loss
# ===========================================================
# Kenapa tidak pakai MSE untuk classification?
# → MSE pada sigmoid → non-convex → banyak local minima
# → Cross-Entropy → convex → guaranteed global minimum!
#
# Binary Cross-Entropy:
# L = -(1/n) Σ [y*log(ŷ) + (1-y)*log(1-ŷ)]
#
# Ini dari information theory — ukuran "jarak" antara dua distribusi.

def binary_cross_entropy(y_true, y_pred):
    """Binary Cross-Entropy Loss"""
    eps = 1e-15  # untuk menghindari log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Visualisasi loss landscape
y_true = 1  # label sebenarnya = 1
y_preds = np.linspace(0.01, 0.99, 100)
losses = [-np.log(p) for p in y_preds]  # loss when y=1

plt.figure(figsize=(8, 4))
plt.plot(y_preds, losses, linewidth=2)
plt.xlabel('Predicted Probability (ŷ)')
plt.ylabel('Loss')
plt.title('Cross-Entropy Loss (when y_true = 1)')
plt.grid(True)
plt.annotate('Prediksi benar → loss kecil', xy=(0.9, 0.1), fontsize=10)
plt.annotate('Prediksi salah → loss BESAR', xy=(0.1, 2.0), fontsize=10)
plt.savefig('02_cross_entropy_loss.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 02_cross_entropy_loss.png")


# ===========================================================
# 📖 BAGIAN 3: Logistic Regression — Full Implementation
# ===========================================================

class LogisticRegression:
    """
    Logistic Regression from scratch — binary classification.

    Model: P(y=1|x) = σ(w·x + b)
    Loss: Binary Cross-Entropy
    Optimization: Gradient Descent
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000, verbose=True):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iter):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = sigmoid(z)

            # Compute loss
            loss = binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            # Compute gradients
            # Kebetulan: gradient CE + sigmoid = (ŷ - y), sama seperti MSE + linear!
            # Ini bukan kebetulan — ini alasan cross-entropy dipilih untuk sigmoid.
            error = y_pred - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if self.verbose and (i + 1) % 200 == 0:
                acc = self.accuracy(X, y)
                print(f"  Iter {i+1}: loss={loss:.4f}, accuracy={acc:.4f}")

        return self

    def predict_proba(self, X):
        """Return probability P(y=1|x)"""
        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        """Return binary prediction"""
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


# ===========================================================
# 📖 BAGIAN 4: Test pada Synthetic Data
# ===========================================================

def make_classification_data(n_samples=300, n_features=2, separation=1.5):
    """Generate linearly separable data (mostly)"""
    n_half = n_samples // 2
    X0 = np.random.randn(n_half, n_features) - separation / 2
    X1 = np.random.randn(n_half, n_features) + separation / 2
    X = np.vstack([X0, X1])
    y = np.array([0] * n_half + [1] * n_half)

    # Shuffle
    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


# Generate data
X, y = make_classification_data(300, 2)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train
print("=== Training Logistic Regression ===")
model = LogisticRegression(learning_rate=0.5, n_iterations=1000)
model.fit(X_train, y_train)

# Evaluate
train_acc = model.accuracy(X_train, y_train)
test_acc = model.accuracy(X_test, y_test)
print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

# Plot decision boundary
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Decision boundary
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axes[0].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu',
                edgecolors='black', s=50)
axes[0].set_title(f'Decision Boundary (test acc: {test_acc:.2f})')

# Loss curve
axes[1].plot(model.loss_history)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Cross-Entropy Loss')
axes[1].set_title('Training Loss')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('03_logistic_regression_result.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 03_logistic_regression_result.png")


# ===========================================================
# 📖 BAGIAN 5: Multi-class → Softmax Regression
# ===========================================================
# Logistic regression = 2 kelas
# Softmax regression = N kelas (generalisasi)
#
# P(y=k|x) = exp(wₖ·x + bₖ) / Σⱼ exp(wⱼ·x + bⱼ)
# Loss = Cross-Entropy = -Σ yₖ * log(P(y=k|x))

class SoftmaxRegression:
    """Multi-class classification from scratch"""

    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.loss_history = []

    def softmax(self, z):
        exp_z = np.exp(z - z.max(axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def one_hot(self, y, n_classes):
        oh = np.zeros((len(y), n_classes))
        oh[np.arange(len(y)), y] = 1
        return oh

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))
        y_oh = self.one_hot(y, self.n_classes)

        # Weights: (n_features, n_classes)
        self.W = np.random.randn(n_features, self.n_classes) * 0.01
        self.b = np.zeros(self.n_classes)

        for i in range(self.n_iter):
            # Forward
            z = X @ self.W + self.b
            probs = self.softmax(z)

            # Loss
            loss = -np.mean(np.sum(y_oh * np.log(probs + 1e-15), axis=1))
            self.loss_history.append(loss)

            # Gradients
            error = probs - y_oh  # (n_samples, n_classes)
            dW = (1/n_samples) * (X.T @ error)
            db = (1/n_samples) * error.sum(axis=0)

            self.W -= self.lr * dW
            self.b -= self.lr * db

        return self

    def predict(self, X):
        z = X @ self.W + self.b
        return np.argmax(self.softmax(z), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


# Test multi-class
from sklearn.datasets import make_blobs
X_multi, y_multi = make_blobs(n_samples=300, centers=3, n_features=2,
                               random_state=42, cluster_std=1.5)

model_mc = SoftmaxRegression(learning_rate=0.5, n_iterations=1000)
model_mc.fit(X_multi, y_multi)
print(f"\n=== Softmax Regression (3 classes) ===")
print(f"Accuracy: {model_mc.accuracy(X_multi, y_multi):.4f}")


# ===========================================================
# 📖 BAGIAN 6: Evaluation Metrics — Beyond Accuracy
# ===========================================================
# Accuracy saja TIDAK CUKUP, terutama untuk imbalanced data!

def compute_metrics(y_true, y_pred):
    """Hitung precision, recall, F1 dari nol"""
    # True Positives, False Positives, False Negatives, True Negatives
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Confusion Matrix:")
    print(f"  TP={TP}, FP={FP}")
    print(f"  FN={FN}, TN={TN}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (dari yang diprediksi positif, berapa yang benar)")
    print(f"Recall:    {recall:.4f} (dari yang memang positif, berapa yang tertangkap)")
    print(f"F1-Score:  {f1:.4f} (harmonic mean precision & recall)")

    return {'accuracy': accuracy, 'precision': precision,
            'recall': recall, 'f1': f1}


# Contoh: imbalanced data (90% normal, 10% fault)
print("\n=== Metrics pada Imbalanced Data ===")
y_true_imb = np.array([0]*90 + [1]*10)
y_pred_always_0 = np.zeros(100)  # model malas: selalu prediksi 0

print("Model 'malas' (selalu prediksi 0):")
metrics1 = compute_metrics(y_true_imb, y_pred_always_0.astype(int))
print("\n→ Accuracy 90% tapi recall 0%! Model ini TIDAK BERGUNA.")
print("   Inilah kenapa accuracy bisa menipu di imbalanced data.")


# ===========================================================
# 🏋️ EXERCISE 5: Implementasi dari Nol
# ===========================================================
"""
1. Implementasi ROC curve & AUC dari nol:
   - Untuk berbagai threshold (0 sampai 1):
     - Hitung TPR (True Positive Rate = Recall)
     - Hitung FPR (False Positive Rate = FP / (FP + TN))
   - Plot TPR vs FPR
   - Hitung AUC menggunakan trapezoidal rule

2. Implementasi Learning Rate Scheduler:
   - Constant LR
   - Step decay: lr = lr_initial * decay_rate^(epoch / step_size)
   - Exponential decay: lr = lr_initial * exp(-decay * epoch)
   - Bandingkan convergence ketiga scheduler

3. Tambahkan L2 regularization ke LogisticRegression class di atas
"""


# ===========================================================
# 🔥 CHALLENGE: Fault Detection System
# ===========================================================
"""
Buat sistem deteksi fault untuk motor listrik (relevant dengan EE!):

1. Generate data sintetis (1000 samples):
   Features:
   - Arus (A): normal ~ N(5, 0.5), fault ~ N(8, 1.0)
   - Vibrasi (mm/s): normal ~ N(2, 0.3), fault ~ N(5, 1.0)
   - Temperatur (°C): normal ~ N(40, 3), fault ~ N(65, 5)
   - Noise level (dB): normal ~ N(60, 5), fault ~ N(75, 8)

   Data IMBALANCED: 90% normal, 10% fault (realistis!)

2. Train Logistic Regression (from scratch — pakai class di atas)

3. Evaluasi dengan:
   - Confusion matrix
   - Precision, Recall, F1
   - ROC curve & AUC
   - Pilih threshold optimal (bukan default 0.5!)
     → Di fault detection, recall lebih penting dari precision (kenapa?)

4. Eksperimen:
   - Apa efek class imbalance terhadap model?
   - Coba class weighting: kalikan loss sample minority dengan weight > 1
   - Bandingkan model dengan dan tanpa weighting
"""

print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 02-ml-dari-nol/03_gradient_descent_deep.py")
print("="*50)
