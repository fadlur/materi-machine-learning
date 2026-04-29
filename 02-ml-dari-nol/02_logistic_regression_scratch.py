"""
=============================================================
FASE 2 - MODUL 2: LOGISTIC REGRESSION FROM SCRATCH
=============================================================
Dari regression ke classification.

Logistic Regression BUKAN regression - ini CLASSIFIER.
Namanya menyesatkan, tapi ini salah satu model paling penting:
- Baseline untuk semua classification task
- Building block dari Neural Networks (setiap neuron = logistic regression!)
- Probabilistic output -> bisa dikalibrasi

Koneksi Teknik Elektro:
- Sigmoid function = transfer function (seperti di op-amp saturation)
- Cross-entropy = information theory (Shannon entropy)
- Decision boundary = threshold detector (komparator)

Durasi target: 3-4 jam
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ===========================================================
# BAGIAN 1: Sigmoid Function & Binary Classification
# ===========================================================
# Sigmoid = fungsi yang mengubah nilai real menjadi (0, 1).
# Di ML, output sigmoid diinterpretasikan sebagai probabilitas.
#
# DETAIL MATEMATIKA:
# Fungsi sigmoid: sigma(z) = 1 / (1 + exp(-z))
# Properties:
# - Domain: (-inf, inf), Range: (0, 1)
# - sigma(0) = 0.5
# - sigma'(z) = sigma(z) * (1 - sigma(z)) -> gradient mudah dihitung!
# - Asymptotic: sigma(z->inf) = 1, sigma(z->-inf) = 0
#
# Kenapa sigmoid? Karena output bisa diinterpretasikan sebagai probabilitas.
# P(y=1|x) = sigma(w.T @ x + b)
# Ini adalah model probabilistik: kita memodelkan distribusi conditional.
#
# Koneksi Teknik Elektro:
# - Mirip dengan saturating transfer function di elektronika
# - Op-amp dengan negative feedback -> sigmoid-like response
# - Membatasi output ke range tertentu (clipping)
# - Comparator dengan soft threshold (bukan hard threshold)

# PERINGATAN: Numerical Stability
# - np.clip(z, -500, 500) mencegah overflow di exp()
# - exp(500) ~ 10^217 (overflow di float64 -> inf)
# - exp(-500) ~ 0 (underflow -> 0, tidak terlalu bermasalah)
# - Kalau z = inf, exp(-z) = 0, sigma = 1 (OK)
# - Kalau z = -inf, exp(-z) = inf, sigma = 0/inf = 0 (OK)
# - Tapi z yang sangat besar negatif bisa menyebabkan overflow di exp(-z)


def sigmoid(z):
    """
    Fungsi sigmoid: sigma(z) = 1 / (1 + exp(-z))
    
    Parameters:
    -----------
    z : np.ndarray atau scalar
        Input bisa berupa scalar, vector, atau matrix.
        Biasanya z = w.T @ x + b (linear combination of features).
        
    Returns:
    --------
    np.ndarray atau scalar (same shape as z)
        Output dalam range (0, 1).
        
    Notes:
    ------
    Properties sigmoid:
    - Output selalu antara 0 dan 1 -> interpretasi sebagai probabilitas
    - sigma(0) = 0.5
    - sigma'(z) = sigma(z) * (1 - sigma(z)) -> gradient mudah dihitung!
    - Asymptotic: sigma(z->inf) = 1, sigma(z->-inf) = 0
    - Non-linear tapi smooth (differentiable everywhere)
    
    Koneksi Teknik Elektro:
    - Mirip dengan saturating transfer function di elektronika
    - Op-amp dengan negative feedback -> sigmoid-like response
    - Membatasi output ke range tertentu (clipping)
    
    Numerical Stability:
    - np.clip(z, -500, 500) mencegah overflow di exp()
    - exp(500) ~ 10^217 (overflow di float64)
    - Untuk z sangat negatif, exp(-z) -> inf, tapi 1/(1+inf) -> 0 (OK)
    - Untuk z sangat positif, exp(-z) -> 0, tapi 1/(1+0) -> 1 (OK)
    """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


# Visualisasi sigmoid
z = np.linspace(-10, 10, 200)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Sigmoid
axes[0].plot(z, sigmoid(z), linewidth=2)
axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[0].set_title('Sigmoid Function sigma(z)')
axes[0].set_xlabel('z')
axes[0].set_ylabel('sigma(z)')
axes[0].grid(True)

# Gradient sigmoid
sig = sigmoid(z)
grad_sig = sig * (1 - sig)
axes[1].plot(z, grad_sig, linewidth=2, color='orange')
axes[1].set_title("Sigmoid Gradient sigma'(z)")
axes[1].set_xlabel('z')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('01_sigmoid.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 01_sigmoid.png")


# ===========================================================
# BAGIAN 2: Cross-Entropy Loss
# ===========================================================
# Kenapa tidak pakai MSE untuk classification?
# -> MSE pada sigmoid -> non-convex -> banyak local minima
#    Gradient MSE * sigmoid' bisa sangat kecil di region tertentu (vanishing gradient).
# -> Cross-Entropy -> convex -> guaranteed global minimum!
#
# Binary Cross-Entropy:
# L = -(1/n) * Sum [y*log(y_hat) + (1-y)*log(1-y_hat)]
#
# Ini dari information theory - ukuran "jarak" antara dua distribusi.
# - y = distribusi sebenarnya (0 atau 1)
# - y_hat = distribusi prediksi (probabilitas)
#
# DETAIL MATEMATIKA:
# Cross-entropy berasal dari information theory.
# Entropy H(p) = -Sum p(x) * log(p(x)) -> ukuran ketidakpastian.
# Cross-entropy H(p, q) = -Sum p(x) * log(q(x)) -> ukuran ketidakpastian
#   jika kita menggunakan distribusi q untuk mendeskripsikan p.
# Kita ingin H(p, q) seminimal mungkin -> q mendekati p.
#
# Untuk binary classification:
# H(y, y_hat) = -[y * log(y_hat) + (1-y) * log(1-y_hat)]
# Jika y=1: loss = -log(y_hat). Semakin y_hat mendekati 1, loss semakin kecil.
# Jika y=0: loss = -log(1-y_hat). Semakin y_hat mendekati 0, loss semakin kecil.
#
# Koneksi Teknik Elektro:
# - Cross-entropy = expected code length di information theory
# - Mirip dengan entropy di thermodynamics (disorder/uncertainty)
# - Mengukur "surprise" dari prediksi yang salah
# - Di communications: cross-entropy = optimal code length


def binary_cross_entropy(y_true, y_pred):
    """
    Binary Cross-Entropy Loss.
    
    Parameters:
    -----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels (0 atau 1).
        
    y_pred : np.ndarray, shape (n_samples,)
        Predicted probabilities (0 sampai 1).
        
    Returns:
    --------
    float
        Rata-rata cross-entropy loss.
        
    Notes:
    ------
    - eps = 1e-15 untuk menghindari log(0) -> -inf
    - np.clip membatasi y_pred ke [eps, 1-eps]
    - Loss tinggi saat prediksi salah dengan confidence tinggi
      (misal: y=1 tapi y_pred=0.01 -> loss = -log(0.01) ~ 4.6)
    - Loss rendah saat prediksi benar dengan confidence tinggi
      (misal: y=1 tapi y_pred=0.99 -> loss = -log(0.99) ~ 0.01)
    - Logaritma natural (ln) digunakan, bukan log base 10
    
    Koneksi Teknik Elektro:
    - Cross-entropy = expected code length di information theory
    - Mirip dengan entropy di thermodynamics (disorder/uncertainty)
    - Mengukur "surprise" dari prediksi yang salah
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Visualisasi loss landscape
y_true = 1  # label sebenarnya = 1
y_preds = np.linspace(0.01, 0.99, 100)
losses = [-np.log(p) for p in y_preds]  # loss when y=1

plt.figure(figsize=(8, 4))
plt.plot(y_preds, losses, linewidth=2)
plt.xlabel('Predicted Probability (y_hat)')
plt.ylabel('Loss')
plt.title('Cross-Entropy Loss (when y_true = 1)')
plt.grid(True)
plt.annotate('Prediksi benar -> loss kecil', xy=(0.9, 0.1), fontsize=10)
plt.annotate('Prediksi salah -> loss BESAR', xy=(0.1, 2.0), fontsize=10)
plt.savefig('02_cross_entropy_loss.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 02_cross_entropy_loss.png")


# ===========================================================
# BAGIAN 3: Logistic Regression - Full Implementation
# ===========================================================
# Logistic Regression memodelkan P(y=1|x) = sigma(w.T @ x + b)
# Ini adalah Generalized Linear Model (GLM) dengan link function = logit.
#
# DETAIL MATEMATIKA:
# Model: P(y=1|x) = sigma(z), z = w.T @ x + b
# Log-odds (logit): log(P/(1-P)) = z = w.T @ x + b
# Ini adalah linear model dalam log-odds space.
#
# Loss function (Binary Cross-Entropy):
# L = -(1/n) * Sum [y_i * log(y_hat_i) + (1-y_i) * log(1-y_hat_i)]
#
# Gradient:
# dL/dw = (1/n) * X.T @ (y_hat - y)
# dL/db = (1/n) * Sum(y_hat - y)
#
# Kenapa gradient CE + sigmoid = (y_hat - y), sama seperti MSE + linear?
# Ini bukan kebetulan - ini alasan cross-entropy dipilih untuk sigmoid.
# Derivative chain rule:
#   dL/dz = dL/d(y_hat) * d(y_hat)/dz
#   dL/d(y_hat) = -(y/y_hat - (1-y)/(1-y_hat)) = (y_hat - y) / (y_hat*(1-y_hat))
#   d(y_hat)/dz = y_hat * (1-y_hat)
#   dL/dz = (y_hat - y) / (y_hat*(1-y_hat)) * y_hat*(1-y_hat) = y_hat - y
# Gradient menjadi sangat sederhana: error = y_hat - y!
#
# Koneksi Teknik Elektro:
# - Error = y_hat - y adalah "residual"
# - Gradient descent = menyesuaikan parameter untuk minimize error
# - Mirip dengan LMS adaptive filtering
# - Decision boundary linear: w.T @ x + b = 0 -> hyperplane

class LogisticRegression:
    """
    Logistic Regression from scratch - binary classification.
    
    Model: P(y=1|x) = sigma(w.T @ x + b)
    Loss: Binary Cross-Entropy
    Optimization: Gradient Descent
    
    Attributes:
    -----------
    weights : np.ndarray
        Koefisien fitur.
    bias : float
        Intercept.
    loss_history : list
        Riwayat loss per iterasi.
        
    Notes:
    ------
    - Setiap neuron di neural network = logistic regression unit
    - Output probabilitas bisa di-threshold untuk classification
    - Decision boundary linear (hyperplane)
    - Koneksi Teknik Elektro: mirip dengan binary classifier
      di decision-making systems
    - Model ini adalah linear classifier, meskipun decision function
      non-linear (sigmoid). Boundary-nya tetap linear.
    """
    
    def __init__(self, learning_rate=0.1, n_iterations=1000, verbose=True):
        """
        Parameters:
        -----------
        learning_rate : float, default 0.1
            Step size gradient descent.
            Biasanya lebih besar dari linear regression (0.1 - 1.0).
            
        n_iterations : int, default 1000
            Jumlah iterasi training.
            
        verbose : bool, default True
            Jika True, print progress setiap 200 iterasi.
        """
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        """
        Melatih model menggunakan Gradient Descent.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training features.
            
        y : np.ndarray, shape (n_samples,)
            Training labels (0 atau 1).
            
        Returns:
        --------
        self : LogisticRegression
            Model yang sudah di-fit.
            
        Notes:
        ------
        Algoritma per iterasi:
        1. Forward: z = Xw + b, y_hat = sigma(z)
        2. Loss: BCE(y, y_hat)
        3. Gradient: error = y_hat - y
           dw = (1/n) * X.T @ (error)
           db = (1/n) * Sum(error)
        4. Update: w -= lr*dw, b -= lr*db
        
        Kebetulan menarik:
        Gradient CE + sigmoid = (y_hat - y), sama seperti MSE + linear!
        Ini bukan kebetulan - ini alasan cross-entropy dipilih untuk sigmoid.
        Derivative chain rule menghasilkan simplifikasi yang indah.
        
        Kenapa inisialisasi weights = 0?
        - Untuk logistic regression, inisialisasi 0 adalah safe.
        - Karena loss convex, tidak ada masalah symmetry breaking.
        - Di neural networks, inisialisasi 0 menyebabkan problem
          (semua neuron belajar hal yang sama).
        
        Koneksi Teknik Elektro:
        - Error = y_hat - y adalah "residual"
        - Gradient descent = menyesuaikan parameter untuk minimize error
        - Mirip dengan LMS adaptive filtering
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for i in range(self.n_iter):
            # === Forward pass ===
            z = X @ self.weights + self.bias
            y_pred = sigmoid(z)
            
            # === Compute loss ===
            loss = binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)
            
            # === Compute gradients ===
            # error = y_hat - y -> shape (n_samples,)
            error = y_pred - y
            # dw = (1/n) * X.T @ (error) -> shape (n_features,)
            dw = (1 / n_samples) * (X.T @ error)
            # db = (1/n) * Sum(error) -> scalar
            db = (1 / n_samples) * np.sum(error)
            
            # === Update ===
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Print progress
            if self.verbose and (i + 1) % 200 == 0:
                acc = self.accuracy(X, y)
                print(f"  Iter {i+1}: loss={loss:.4f}, accuracy={acc:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Mengembalikan probabilitas P(y=1|x).
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data yang akan diprediksi.
            
        Returns:
        --------
        np.ndarray, shape (n_samples,)
            Probabilitas kelas 1 untuk setiap sample.
            Range: (0, 1).
        """
        return sigmoid(X @ self.weights + self.bias)
    
    def predict(self, X, threshold=0.5):
        """
        Mengembalikan prediksi biner (0 atau 1).
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data yang akan diprediksi.
            
        threshold : float, default 0.5
            Threshold untuk konversi probabilitas ke label.
            y_hat >= threshold -> 1, else -> 0.
            
            Penting: threshold default 0.5 belum tentu optimal!
            - Threshold rendah -> lebih banyak prediksi positif
              (high recall, low precision)
            - Threshold tinggi -> lebih sedikit prediksi positif
              (low recall, high precision)
            - Threshold optimal bergantung pada cost function bisnis.
            
        Returns:
        --------
        np.ndarray, shape (n_samples,)
            Prediksi biner (0 atau 1).
        """
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def accuracy(self, X, y):
        """
        Menghitung classification accuracy.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data features.
            
        y : np.ndarray, shape (n_samples,)
            True labels.
            
        Returns:
        --------
        float
            Accuracy = fraction of correct predictions.
            Range: [0, 1].
            
        Notes:
        ------
        - Accuracy bisa menipu untuk imbalanced data!
        - Jika 90% data kelas 0, model yang selalu prediksi 0
          mendapat accuracy 90% tapi tidak berguna.
        - Gunakan precision, recall, F1 untuk imbalanced data.
        """
        return np.mean(self.predict(X) == y)


# ===========================================================
# BAGIAN 4: Test pada Synthetic Data
# ===========================================================
# Kita generate data yang linearly separable (hampir) untuk testing.
# Data terdiri dari dua kelas yang terpisah oleh jarak tertentu.

def make_classification_data(n_samples=300, n_features=2, separation=1.5):
    """
    Generate data linearly separable (mostly) untuk binary classification.
    
    Parameters:
    -----------
    n_samples : int, default 300
        Total jumlah samples.
        
    n_features : int, default 2
        Jumlah fitur.
        
    separation : float, default 1.5
        Jarak antar center dari dua kelas.
        Semakin besar -> semakin mudah dipisahkan.
        Semakin kecil -> semakin sulit (overlap lebih besar).
        
    Returns:
    --------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
        
    y : np.ndarray, shape (n_samples,)
        Binary labels (0 atau 1).
        
    Notes:
    ------
    - Kelas 0: centered di -separation/2
    - Kelas 1: centered di +separation/2
    - Data di-shuffle untuk menghilangkan ordering
    - Noise Gaussian N(0,1) ditambahkan ke setiap kelas
    - separation = jarak antar means. separation=0 berarti overlapping penuh.
    """
    n_half = n_samples // 2
    X0 = np.random.randn(n_half, n_features) - separation / 2
    X1 = np.random.randn(n_half, n_features) + separation / 2
    X = np.vstack([X0, X1])
    y = np.array([0] * n_half + [1] * n_half)
    
    # Shuffle untuk menghilangkan ordering
    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


# Generate data
X, y = make_classification_data(300, 2)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train
print("\n=== Training Logistic Regression ===")
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
# Untuk logistic regression 2D, boundary adalah garis: w1*x1 + w2*x2 + b = 0
# x2 = -(w1*x1 + b) / w2
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
print("Saved: 03_logistic_regression_result.png")


# ===========================================================
# BAGIAN 5: Multi-class -> Softmax Regression
# ===========================================================
# Logistic regression = 2 kelas
# Softmax regression = N kelas (generalisasi)
#
# P(y=k|x) = exp(w_k.T @ x + b_k) / Sum_j exp(w_j.T @ x + b_j)
# Loss = Cross-Entropy = -Sum_k y_k * log(P(y=k|x))
#
# DETAIL MATEMATIKA:
# Softmax mengubah vector logits z = [z_1, z_2, ..., z_K]
# menjadi distribusi probabilitas: P_k = exp(z_k) / Sum_j exp(z_j)
#
# Properties softmax:
# - Output selalu positif (karena exp)
# - Output sum to 1 (karena dibagi Sum exp)
# - Semakin besar z_k relatif terhadap z lain -> P_k semakin besar
# - Jika satu z_k jauh lebih besar -> softmax mendekati one-hot
#
# Numerical Stability untuk Softmax:
# - exp(z_k) bisa overflow untuk z_k besar.
# - Solusi: subtract z_max dari semua z sebelum exp.
#   exp(z_k - z_max) / Sum_j exp(z_j - z_max)
#   = exp(z_k)/exp(z_max) / (Sum_j exp(z_j)/exp(z_max))
#   = exp(z_k) / Sum_j exp(z_j)  (sama!)
# - exp(z_k - z_max) <= 1 (tidak overflow)
# - Minimal ada satu term = exp(0) = 1 (tidak underflow semua).
#
# Koneksi Teknik Elektro:
# - Softmax = generalisasi sigmoid untuk multi-class
# - Output = distribusi probabilitas (jumlah = 1)
# - Juga disebut "Multinomial Logistic Regression"
# - Mirip dengan maximum likelihood estimation di communication theory

class SoftmaxRegression:
    """
    Multi-class classification from scratch.
    
    Model: P(y=k|x) = exp(z_k) / Sum_j exp(z_j), 
           dimana z_k = w_k.T @ x + b_k
    Loss: Categorical Cross-Entropy
    Optimization: Gradient Descent
    
    Attributes:
    -----------
    W : np.ndarray, shape (n_features, n_classes)
        Weight matrix. Setiap kolom adalah weights untuk satu kelas.
    b : np.ndarray, shape (n_classes,)
        Bias vector.
    loss_history : list
        Riwayat loss.
        
    Notes:
    ------
    - Softmax = generalisasi sigmoid untuk multi-class
    - Output = distribusi probabilitas (jumlah = 1)
    - Juga disebut "Multinomial Logistic Regression"
    - Koneksi Teknik Elektro: mirip dengan maximum likelihood
      estimation di communication theory
    - Setiap kelas memiliki weights sendiri (W[:, k] untuk kelas k)
    """
    
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        """
        Parameters:
        -----------
        learning_rate : float, default 0.1
            Step size gradient descent.
            
        n_iterations : int, default 1000
            Jumlah iterasi training.
        """
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.loss_history = []
    
    def softmax(self, z):
        """
        Softmax function dengan numerical stability.
        
        Parameters:
        -----------
        z : np.ndarray, shape (n_samples, n_classes)
            Logits (pre-activation scores).
            
        Returns:
        --------
        np.ndarray, shape (n_samples, n_classes)
            Probabilitas per kelas (setiap baris jumlah = 1).
            
        Notes:
        ------
        - z - z.max(axis=1, keepdims=True) untuk numerical stability
        - keepdims=True menjaga dimensi untuk broadcasting
        - exp(z - max) / sum(exp(z - max))
        - Teknik ini menghindari overflow tanpa mengubah hasil matematis
        """
        exp_z = np.exp(z - z.max(axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)
    
    def one_hot(self, y, n_classes):
        """
        One-hot encoding untuk labels.
        
        Parameters:
        -----------
        y : np.ndarray, shape (n_samples,)
            Integer labels (0, 1, ..., n_classes-1).
            
        n_classes : int
            Jumlah kelas.
            
        Returns:
        --------
        np.ndarray, shape (n_samples, n_classes)
            One-hot encoded matrix.
            Setiap baris hanya memiliki satu elemen = 1, sisanya 0.
        """
        oh = np.zeros((len(y), n_classes))
        oh[np.arange(len(y)), y] = 1
        return oh
    
    def fit(self, X, y):
        """
        Melatih model dengan Gradient Descent.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training features.
            
        y : np.ndarray, shape (n_samples,)
            Training labels (integer 0..n_classes-1).
            
        Returns:
        --------
        self : SoftmaxRegression
            Model yang sudah di-fit.
            
        Notes:
        ------
        Algoritma:
        1. One-hot encode labels
        2. Forward: Z = XW + b, P = softmax(Z)
        3. Loss: -mean(sum(y_oh * log(P)))
        4. Gradient: error = P - y_oh
           dW = (1/n) * X.T @ (error)
           db = (1/n) * sum(error)
           
        Kenapa gradient = P - y_oh?
        Sama seperti logistic regression, cross-entropy + softmax
        menghasilkan gradient yang sederhana:
        dL/dz_k = P_k - y_k (untuk one-hot y).
        Ini adalah salah satu "kebetulan" indah di ML.
        """
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))
        y_oh = self.one_hot(y, self.n_classes)
        
        # Weights: (n_features, n_classes)
        # Setiap kolom adalah weights untuk satu kelas
        self.W = np.random.randn(n_features, self.n_classes) * 0.01
        self.b = np.zeros(self.n_classes)
        
        for i in range(self.n_iter):
            # Forward
            z = X @ self.W + self.b  # (n_samples, n_classes)
            probs = self.softmax(z)   # (n_samples, n_classes)
            
            # Loss
            # Clip untuk menghindari log(0)
            loss = -np.mean(np.sum(y_oh * np.log(probs + 1e-15), axis=1))
            self.loss_history.append(loss)
            
            # Gradients
            error = probs - y_oh  # (n_samples, n_classes)
            dW = (1/n_samples) * (X.T @ error)  # (n_features, n_classes)
            db = (1/n_samples) * error.sum(axis=0)  # (n_classes,)
            
            self.W -= self.lr * dW
            self.b -= self.lr * db
        
        return self
    
    def predict(self, X):
        """
        Prediksi kelas untuk data baru.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data yang akan diprediksi.
            
        Returns:
        --------
        np.ndarray, shape (n_samples,)
            Predicted class labels.
            Kelas dengan probabilitas tertinggi dipilih.
        """
        z = X @ self.W + self.b
        return np.argmax(self.softmax(z), axis=1)
    
    def accuracy(self, X, y):
        """
        Menghitung classification accuracy.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data features.
            
        y : np.ndarray, shape (n_samples,)
            True labels.
            
        Returns:
        --------
        float
            Accuracy = proporsi prediksi yang benar.
        """
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
# BAGIAN 6: Evaluation Metrics - Beyond Accuracy
# ===========================================================
# Accuracy saja TIDAK CUKUP, terutama untuk imbalanced data!
# Precision, Recall, F1 memberikan gambaran lebih lengkap.
#
# DETAIL MATEMATIKA:
# Confusion Matrix:
# - TP: True Positive (prediksi 1, benar 1)
# - FP: False Positive (prediksi 1, benar 0) -> Type I error
# - FN: False Negative (prediksi 0, benar 1) -> Type II error
# - TN: True Negative (prediksi 0, benar 0)
#
# Metrics:
# - Accuracy = (TP+TN) / Total
# - Precision = TP / (TP+FP) -> "dari yang diprediksi positif, berapa yang benar"
# - Recall = TP / (TP+FN) -> "dari yang memang positif, berapa yang tertangkap"
# - F1 = 2 * P * R / (P+R) -> harmonic mean precision & recall
# - Specificity = TN / (TN+FP) -> "dari yang negatif, berapa yang benar terdeteksi"
#
# Tradeoff Precision-Recall:
# - Threshold rendah -> banyak prediksi positif -> high recall, low precision
# - Threshold tinggi -> sedikit prediksi positif -> low recall, high precision
# - F1 balance keduanya.
#
# Koneksi Teknik Elektro:
# - Precision = Pd (Probability of detection)
# - Recall = Pd (detection probability)
# - 1-Recall = Pfa (Probability of false alarm)
# - Tradeoff Pd vs Pfa = ROC curve ( Receiver Operating Characteristic)

def compute_metrics(y_true, y_pred):
    """
    Menghitung precision, recall, F1 dari nol.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth labels (0 atau 1).
        
    y_pred : np.ndarray
        Predicted labels (0 atau 1).
        
    Returns:
    --------
    dict
        Dictionary dengan keys: 'accuracy', 'precision', 'recall', 'f1'.
        
    Notes:
    ------
    Confusion Matrix:
    - TP: True Positive (prediksi 1, benar 1)
    - FP: False Positive (prediksi 1, benar 0) -> Type I error
    - FN: False Negative (prediksi 0, benar 1) -> Type II error
    - TN: True Negative (prediksi 0, benar 0)
    
    Metrics:
    - Accuracy = (TP+TN) / Total
    - Precision = TP / (TP+FP) -> "dari yang diprediksi positif, berapa yang benar"
    - Recall = TP / (TP+FN) -> "dari yang memang positif, berapa yang tertangkap"
    - F1 = 2 * P * R / (P+R) -> harmonic mean precision & recall
    
    Kenapa harmonic mean, bukan arithmetic mean?
    - Harmonic mean memberikan penalti besar jika salah satu metrik rendah.
    - Contoh: P=1.0, R=0.0 -> arithmetic mean = 0.5, F1 = 0.0
    - F1 = 0.0 lebih merepresentasikan bahwa model gagal total.
    
    Koneksi Teknik Elektro:
    - Precision = Pd (Probability of detection)
    - Recall = Pd
    - 1-Recall = Pfa (Probability of false alarm)
    - Tradeoff Pd vs Pfa = ROC curve
    """
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
print("\n-> Accuracy 90% tapi recall 0%! Model ini TIDAK BERGUNA.")
print("   Inilah kenapa accuracy bisa menipu di imbalanced data.")


# ===========================================================
# LATIHAN 5: Implementasi dari Nol
# ===========================================================
"""
TARGET Learning Objectives:
   - Memahami ROC curve dan AUC secara mendalam
   - Mengimplementasikan learning rate scheduler
   - Menambahkan regularization ke Logistic Regression

PANDUAN LANGKAH-LANGKAH:

STEP 1: Implementasi ROC curve & AUC dari nol
---------------------------------------------
ROC (Receiver Operating Characteristic) = plot TPR vs FPR
untuk berbagai threshold.

   TIPS: Apa yang harus dilakukan:
     a) Dapatkan predicted probabilities: probs = model.predict_proba(X)
     b) Untuk setiap threshold di [0, 0.01, 0.02, ..., 1.0]:
        - pred = (probs >= threshold).astype(int)
        - Hitung TP, FP, FN, TN
        - TPR = TP / (TP + FN)  -> Recall
        - FPR = FP / (FP + TN)
        - Simpan (FPR, TPR)
        
     c) Plot TPR vs FPR
     d) Hitung AUC menggunakan trapezoidal rule:
        AUC = Sum (FPR_i - FPR_{i-1}) * (TPR_i + TPR_{i-1}) / 2
        
   TIPS: KENAPA ROC/AUC?
     - ROC menunjukkan tradeoff antara sensitivity dan specificity
     - AUC = probability model ranks random positive higher than random negative
     - AUC = 0.5 -> random guessing
     - AUC = 1.0 -> perfect classifier
     - Robust terhadap class imbalance
     - Threshold-independent: evaluasi model tanpa memilih threshold

   PERINGATAN: Hati-hati:
     - Threshold harus diurutkan descending
     - Trapezoidal rule memerlukan sorted points
     - AUC selalu >= 0.5 (kalau < 0.5, flip predictions)


STEP 2: Implementasi Learning Rate Scheduler
--------------------------------------------
Learning rate constant sering bukan yang optimal.

   TIPS: Implementasi 3 scheduler:
   
   a) Constant LR:
      lr = lr_initial
      
   b) Step decay:
      lr = lr_initial * decay_rate^(epoch / step_size)
      
   c) Exponential decay:
      lr = lr_initial * exp(-decay * epoch)
      
   d) (Bonus) Cosine annealing:
      lr = lr_min + 0.5 * (lr_initial - lr_min) * (1 + cos(pi * epoch / max_epoch))

   TIPS: KENAPA scheduler?
     - LR besar di awal -> cepat converge ke region optimal
     - LR kecil di akhir -> fine-tuning untuk precision
     - Cosine annealing -> smooth decay, popular di deep learning
     - Mencegah oscillation di akhir training

   TEST: Test:
     - Train LogisticRegression dengan masing-masing scheduler
     - Plot loss curve per scheduler dalam satu figure
     - Bandingkan: final loss, convergence speed, stability


STEP 3: Tambahkan L2 regularization ke LogisticRegression
---------------------------------------------------------
   TIPS: Apa yang harus diubah:
     a) Tambahkan parameter alpha di __init__
     b) Modifikasi loss:
        loss = BCE + alpha * ||w||^2
     c) Modifikasi gradient:
        dw = (1/n) * X.T @ (error) + 2 * alpha * weights
        db = (1/n) * Sum(error)  <- bias tidak diregularisasi
        
   TEST: Test:
     - Generate data dengan banyak fitur (100 fitur, tapi hanya 5 yang relevan)
     - Train dengan dan tanpa regularization
     - Plot: ||weights|| vs alpha
     - Analisis: fitur mana yang weights-nya menjadi ~0?


TIPS HINTS:
   - Untuk ROC, gunakan np.linspace(0, 1, 101) untuk threshold
   - Untuk AUC, gunakan np.trapz(TPR, FPR) atau implementasi manual
   - Untuk scheduler, modifikasi lr di setiap epoch sebelum update
   - Simpan lr_history untuk plotting effective learning rate

PERINGATAN COMMON MISTAKES:
   - TPR dan FPR dihitung dengan denominators yang salah
   - Lupa urutkan threshold descending untuk ROC
   - Regularization diterapkan pada bias (seharusnya tidak)
   - Learning rate menjadi terlalu kecil -> stagnation

TARGET EXPECTED OUTPUT:
   - ROC curve yang mendekati sudut kiri atas (AUC > 0.8)
   - Loss curve yang lebih smooth dengan scheduler
   - Weights yang lebih sparse dengan regularization
"""


# ===========================================================
# CHALLENGE: Fault Detection System
# ===========================================================
"""
TARGET Learning Objectives:
   - Membangun end-to-end classification system untuk domain EE
   - Menangani class imbalance yang realistis
   - Mengoptimalkan threshold untuk business requirements

PANDUAN LANGKAH-LANGKAH:

STEP 1: Generate Data Sintetis (1000 samples)
---------------------------------------------
Buat dataset fault detection untuk motor listrik.

   Features:
   - Arus (A): normal ~ N(5, 0.5), fault ~ N(8, 1.0)
   - Vibrasi (mm/s): normal ~ N(2, 0.3), fault ~ N(5, 1.0)
   - Temperatur (C): normal ~ N(40, 3), fault ~ N(65, 5)
   - Noise level (dB): normal ~ N(60, 5), fault ~ N(75, 8)
   
   Data IMBALANCED: 90% normal, 10% fault (realistis!)
   
   TIPS: KENAPA imbalanced?
     - Di dunia nyata, fault jarang terjadi
     - Model cenderung "malas" dan selalu prediksi normal
     - Perlu strategi khusus untuk handle imbalance


STEP 2: Train Logistic Regression (from scratch)
------------------------------------------------
   a) Split data: 80% train, 20% test
   b) Normalize features (fit pada train, transform pada test)
   c) Train LogisticRegression class yang sudah dibuat
   d) Evaluasi dengan confusion matrix dan metrics


STEP 3: Evaluasi Lengkap
------------------------
   a) Confusion matrix
   b) Precision, Recall, F1
   c) ROC curve & AUC
   d) Precision-Recall curve (lebih informatif untuk imbalanced data)
   
   TIPS: Analisis:
     - Apa yang terjadi jika threshold = 0.5 (default)?
     - Apakah model cenderung False Negative atau False Positive?
     - Mana yang lebih berbahaya di fault detection?


STEP 4: Pilih Threshold Optimal
-------------------------------
Default threshold = 0.5, tapi belum tentu optimal!

   TIPS: Strategi pemilihan threshold:
   
   a) Maximizing F1-score:
      threshold_optimal = argmax F1(threshold)
      
   b) Cost-based approach:
      Misal: Cost(FN) = 10 * Cost(FP)
      (Missed fault lebih mahal dari false alarm)
      Pilih threshold yang minimize total cost.
      
   c) Youden's J statistic:
      J = TPR - FPR
      threshold_optimal = argmax J
      
   TIPS: KENAPA threshold penting?
     - Di fault detection, recall lebih penting dari precision
     - Lebih baik false alarm daripada missed fault
     - Threshold rendah -> lebih sensitive (lebih banyak alarm)
     - Threshold tinggi -> lebih conservative (lebih sedikit alarm)


STEP 5: Eksperimen Class Imbalance
----------------------------------
   a) Baseline: train tanpa modifikasi
   b) Class weighting: kalikan loss sample minority dengan weight > 1
      weight_minority = n_majority / n_minority
      
   c) Oversampling: duplicate minority samples
   d) Undersampling: remove majority samples
   
   TIPS: Bandingkan:
     - Metrics untuk setiap strategi
     - ROC curve (seharusnya tidak berubah karena threshold-independent)
     - Precision-Recall curve (seharusnya berubah)


TIPS HINTS:
   - Untuk class weighting, modifikasi gradient:
     dw = (1/n) * X.T @ (weights_sample * error)
   - Untuk cost-based threshold, definisikan cost matrix:
     cost = FP * cost_fp + FN * cost_fn
   - Gunakan np.bincount(y) untuk menghitung class distribution

PERINGATAN COMMON MISTAKES:
   - Evaluasi tanpa stratified split -> test set bisa tidak ada fault!
   - Normalisasi sebelum split -> data leakage
   - Tidak handle imbalance -> model tidak berguna
   - Threshold 0.5 diasumsikan optimal -> belum tentu!

TARGET EXPECTED OUTPUT:
   - Model dengan recall > 80% untuk fault class
   - ROC-AUC > 0.90
   - Analisis threshold optimal dengan justifikasi business
   - Comparison table: strategy vs precision vs recall vs F1

Ini adalah use case NYATA yang sangat dicari di industri manufaktur!
"""

print("\n" + "="*50)
print("OK Modul selesai! Lanjut ke: 02-ml-dari-nol/03_gradient_descent_deep.py")
print("="*50)
