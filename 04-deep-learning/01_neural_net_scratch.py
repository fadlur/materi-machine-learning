"""
=============================================================
FASE 4 - MODUL 1: NEURAL NETWORK FROM SCRATCH
=============================================================
Sebelum pakai PyTorch, kita bangun neural network dari NOL.

Ini akan memberikan pemahaman yang DEEP tentang:
- Forward propagation
- Backpropagation (chain rule in action!)
- Weight initialization
- Activation functions

Koneksi Teknik Elektro:
- Neural net = cascaded nonlinear systems
- Backprop = applying chain rule (seperti di control theory)
- Activation functions = transfer functions
- Weights = system parameters yang di-optimize

Durasi target: 4-5 jam (ini modul terpenting!)
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ===========================================================
# BAGIAN 1: Single Neuron (Perceptron)
# ===========================================================
# Satu neuron = Linear Regression/Logistic Regression!
# output = activation(w dot x + b)
#
# Matematika Single Neuron:
# z = Sum(w_i * x_i) + b = w dot x + b
# output = activation(z)
#
# Untuk sigmoid: output = 1 / (1 + exp(-z))
# Ini identik dengan logistic regression!
#
# Untuk linear (no activation): output = z
# Ini identik dengan linear regression!
#
# Activation Functions:
# 1. Sigmoid: f(z) = 1 / (1 + exp(-z))
#    Output: (0, 1). Cocok untuk probabilitas.
#    Derivative: f'(z) = f(z) * (1 - f(z))
#    PERINGATAN: Vanishing gradient! Untuk |z| besar, gradient ~ 0.
#
# 2. Tanh: f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
#    Output: (-1, 1). Zero-centered (lebih baik dari sigmoid untuk hidden layers).
#    Derivative: f'(z) = 1 - f(z)**2
#    PERINGATAN: Masih ada vanishing gradient, tapi tidak seburuk sigmoid.
#
# 3. ReLU: f(z) = max(0, z)
#    Output: [0, inf). Sederhana dan cepat.
#    Derivative: f'(z) = 1 if z > 0, else 0.
#    Kelebihan: tidak vanishing untuk z > 0. Computasi murah.
#    PERINGATAN: Dead neurons! Jika z <= 0, neuron tidak belajar lagi.
#
# 4. Leaky ReLU: f(z) = max(alpha*z, z) dengan alpha ~ 0.01
#    Solusi untuk dead neurons: gradient kecil tapi tidak nol untuk z < 0.
#
# Kenapa activation function penting?
# - Tanpa activation, stacked linear layers = single linear layer!
#   (W2 @ (W1 @ x) = (W2 @ W1) @ x = W' @ x)
# - Nonlinearity memungkinkan network mempelajari fungsi kompleks.
# - Activation function menentukan "shape" dari decision boundary.
#
# Koneksi Teknik Elektro:
# - Op-amp dengan saturating nonlinearity
# - Transfer function di control systems
# - Hysteresis di magnetic materials

class Neuron:
    """
    Single neuron - building block dari neural network.
    
    Model: z = w dot x + b, output = activation(z)
    
    Parameters:
    -----------
    n_inputs : int
        Jumlah input features.
    activation : str, default 'sigmoid'
        Fungsi aktivasi: 'sigmoid', 'relu', 'tanh'.
        
    Attributes:
    -----------
    w : np.ndarray
        Weights.
    b : float
        Bias.
        
    Notes:
    ------
    - Single neuron = logistic regression (sigmoid) atau
      linear regression (no activation)
    - Weights diinisialisasi kecil (*0.1) untuk stabilitas
      (menghindari large outputs di awal training)
    - Koneksi Teknik Elektro: mirip dengan op-amp dengan
      feedback dan saturating nonlinearity
    """
    
    def __init__(self, n_inputs, activation='sigmoid'):
        self.w = np.random.randn(n_inputs) * 0.1
        self.b = 0.0
        self.activation = activation
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : np.ndarray, shape (n_samples, n_inputs)
            Input data.
            
        Returns:
        --------
        np.ndarray
            Output setelah aktivasi.
        """
        z = x @ self.w + self.b
        if self.activation == 'sigmoid':
            # np.clip untuk mencegah overflow di exp
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        return z


# Visualisasi activation functions
z = np.linspace(-5, 5, 200)
activations = {
    'Sigmoid': 1 / (1 + np.exp(-z)),
    'Tanh': np.tanh(z),
    'ReLU': np.maximum(0, z),
    'Leaky ReLU': np.where(z > 0, z, 0.01 * z),
}

fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for ax, (name, values) in zip(axes, activations.items()):
    ax.plot(z, values, linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(name)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_activations.png', dpi=100, bbox_inches='tight')
plt.close()
print("OK Saved: 01_activations.png")


# ===========================================================
# BAGIAN 2: Multi-Layer Neural Network from Scratch
# ===========================================================
# Neural network dengan hidden layers mempelajari representasi hierarkis:
# - Layer 1: fitur dasar (edges, patterns sederhana)
# - Layer 2: kombinasi fitur dasar (textures, shapes)
# - Layer 3+: fitur semantik tingkat tinggi
#
# Matematika Forward Pass:
# z[l] = a[l-1] @ W[l] + b[l]
# a[l] = activation(z[l])
# Output: a[L] (L = layer terakhir)
#
# Matematika Backpropagation (Chain Rule):
# Untuk output layer: delta[L] = (a[L] - y) * activation'(z[L])
# Untuk hidden layer: delta[l] = (delta[l+1] @ W[l+1].T) * activation'(z[l])
# Gradient W[l] = a[l-1].T @ delta[l]
# Gradient b[l] = sum(delta[l], axis=0)
#
# Binary Cross-Entropy Loss:
# L = -1/N * Sum(y * log(y_hat) + (1-y) * log(1-y_hat))
# Derivative: dL/dy_hat = (y_hat - y) / (y_hat * (1-y_hat))
# Untuk sigmoid output, gradient terhadap z menyederhana menjadi:
# delta = a[L] - y (karena derivative sigmoid * derivative loss = a - y)
# Ini adalah alasan kenapa sigmoid + BCE sangat populer.
#
# Momentum:
# v_t = beta * v_{t-1} - lr * gradient
# w_t = w_{t-1} + v_t
# Momentum menambahkan "inertia" ke update, membantu melewati
# local minimum dan flat regions.
#
# Weight Initialization:
# - He initialization: W ~ N(0, sqrt(2/n_in))
#   Optimal untuk ReLU. Mencegah vanishing/exploding di deep networks.
# - Xavier/Glorot: W ~ N(0, sqrt(1/n_in))
#   Optimal untuk sigmoid/tanh.
# - Zero initialization: TIDAK BOLEH! Semua neuron jadi identik.
#
# Koneksi Teknik Elektro:
# - Forward pass = signal propagation melalui cascaded systems
# - Backprop = sensitivity analysis (adjoint method)
# - Chain rule = transfer function cascading
# - Momentum = inertia di mechanical systems

class NeuralNetwork:
    """
    Multi-layer neural network from scratch.
    
    Architecture: Input -> Hidden1 -> Hidden2 -> ... -> Output
    Activation: ReLU untuk hidden layers, Sigmoid untuk output
    Loss: Binary Cross-Entropy
    Optimizer: SGD with momentum
    
    Parameters:
    -----------
    layer_sizes : list
        List jumlah neurons per layer.
        Contoh [2, 16, 8, 1] = input(2), hidden(16,8), output(1).
    learning_rate : float, default 0.01
    momentum : float, default 0.9
    
    Attributes:
    -----------
    weights : list of np.ndarray
        Weight matrices per layer.
    biases : list of np.ndarray
        Bias vectors per layer.
    velocity_w : list of np.ndarray
        Velocity untuk momentum (weights).
    velocity_b : list of np.ndarray
        Velocity untuk momentum (biases).
    loss_history : list
        Riwayat loss.
        
    Notes:
    ------
    - Forward: simpan activations dan z_values untuk backprop
    - Backprop: chain rule dari output ke input
    - Momentum: velocity = beta*velocity - lr*gradient
    - He initialization: good untuk ReLU
    
    Koneksi Teknik Elektro:
    - Forward pass = signal propagation melalui cascaded systems
    - Backprop = sensitivity analysis (adjoint method)
    - Chain rule = transfer function cascading
    """
    
    def __init__(self, layer_sizes, learning_rate=0.01, momentum=0.9):
        self.layers = layer_sizes
        self.lr = learning_rate
        self.momentum = momentum
        self.loss_history = []
        
        # Initialize weights (Xavier/He initialization)
        self.weights = []
        self.biases = []
        self.velocity_w = []
        self.velocity_b = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization: good for ReLU
            # Var(W) = 2 / n_in mencegah variance activation tetap konstan
            # seiring kedalaman network bertambah.
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
            self.velocity_w.append(np.zeros_like(w))
            self.velocity_b.append(np.zeros_like(b))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        # Derivative ReLU: 1 jika z > 0, 0 jika z <= 0
        # Di z=0, technically undefined, tapi praktis kita set ke 0 atau 1.
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        # Clip untuk mencegah overflow/underflow di exp
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def forward(self, X):
        """
        Forward pass - simpan semua intermediate values untuk backprop.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data.
            
        Returns:
        --------
        np.ndarray
            Output predictions.
        """
        self.activations = [X]  # a[0] = input
        self.z_values = []       # pre-activation values
        
        current = X
        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            current = self.relu(z)  # Hidden layers: ReLU
            self.activations.append(current)
        
        # Output layer: Sigmoid (binary)
        z = current @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        output = self.sigmoid(z)
        self.activations.append(output)
        
        return output
    
    def backward(self, y):
        """
        Backpropagation - the CORE of neural network training.
        
        Parameters:
        -----------
        y : np.ndarray
            True labels.
            
        Notes:
        ------
        Chain rule: dL/dw = dL/da * da/dz * dz/dw
        
        Ini mirip dengan analisis cascaded systems di Teknik Elektro:
        transfer function total = product of individual transfer functions.
        Backprop = menghitung "gain" di setiap stage secara backward.
        
        Algoritma:
        1. Output layer: delta = a - y (untuk sigmoid + BCE)
        2. Hidden layer: delta = (delta_next @ W.T) * activation'(z)
        3. Gradients: dW = a_prev.T @ delta, db = sum(delta)
        """
        n = len(y)
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        
        # Output layer gradient
        # Untuk sigmoid + BCE, gradient terhadap z_pre-activation menyederhana
        # menjadi (a - y). Ini karena dL/da * da/dz = (a-y)/(a*(1-a)) * a*(1-a) = a-y
        delta = self.activations[-1] - y
        
        gradients_w = []
        gradients_b = []
        
        # Backpropagate through layers (dari output ke input)
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient untuk weights dan biases di layer i
            # dW = a[l-1].T @ delta[l] (rata-rata di-batch)
            dw = (1/n) * (self.activations[i].T @ delta)
            db = (1/n) * np.sum(delta, axis=0)
            gradients_w.insert(0, dw)
            gradients_b.insert(0, db)
            
            if i > 0:
                # Propagate delta ke layer sebelumnya
                # delta[l-1] = (delta[l] @ W[l].T) * activation'(z[l-1])
                delta = (delta @ self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
        
        # Update weights dengan momentum
        # Momentum menambahkan velocity yang terakumulasi dari gradient sebelumnya.
        # Ini membantu melewati: (1) local minimum, (2) saddle points, (3) flat regions.
        for i in range(len(self.weights)):
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.lr * gradients_w[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.lr * gradients_b[i]
            self.weights[i] += self.velocity_w[i]
            self.biases[i] += self.velocity_b[i]
    
    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        """
        Training loop dengan mini-batch.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training labels.
        epochs : int, default 100
        batch_size : int, default 32
        verbose : bool, default True
            
        Notes:
        ------
        Mini-batch training:
        - Full batch: stabil tapi lambat, bisa stuck di local minimum.
        - Stochastic (batch=1): noisy tapi bisa escape local minimum.
        - Mini-batch: kompromi. Default 32-128.
        
        Shuffle setiap epoch:
        - Mencegah learning order-dependent.
        - Membantu generalisasi.
        """
        n = len(X)
        
        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                self.forward(X_batch)
                self.backward(y_batch)
            
            # Compute epoch loss
            output = self.forward(X)
            y_col = y.reshape(-1, 1) if y.ndim == 1 else y
            # Tambahkan epsilon untuk mencegah log(0)
            loss = -np.mean(y_col * np.log(output + 1e-15) +
                           (1 - y_col) * np.log(1 - output + 1e-15))
            self.loss_history.append(loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                acc = self.accuracy(X, y)
                print(f"  Epoch {epoch+1}: loss={loss:.4f}, accuracy={acc:.4f}")
    
    def predict(self, X):
        proba = self.forward(X)
        return (proba >= 0.5).astype(int).ravel()
    
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


# ===========================================================
# BAGIAN 3: Test Neural Network
# ===========================================================
# Dataset make_moons: non-linearly separable, cocok untuk demo NN.
# Mengapa make_moons? Karena linear model (logistic regression) GAGAL,
# tapi neural network dengan hidden layer BERHASIL.

from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# Normalize
# WAJIB normalize sebelum training neural network!
# Alasan: gradient descent konvergen lebih cepat pada data ter-scale.
# Jika fitur memiliki scale berbeda, loss surface menjadi elongated,
# menyebabkan oscillation dan convergence lambat.
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train_n = (X_train - mean) / std
X_test_n = (X_test - mean) / std

# Train
print("=== Training Neural Network [2, 32, 16, 1] ===")
nn = NeuralNetwork([2, 32, 16, 1], learning_rate=0.1, momentum=0.9)
nn.train(X_train_n, y_train, epochs=200, batch_size=32)

print(f"\nTrain accuracy: {nn.accuracy(X_train_n, y_train):.4f}")
print(f"Test accuracy:  {nn.accuracy(X_test_n, y_test):.4f}")

# Visualize decision boundary
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

h = 0.02
x_min, x_max = X_train_n[:, 0].min() - 1, X_train_n[:, 0].max() + 1
y_min, y_max = X_train_n[:, 1].min() - 1, X_train_n[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = nn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

axes[0].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
axes[0].scatter(X_test_n[:, 0], X_test_n[:, 1], c=y_test,
                cmap='RdYlBu', edgecolors='black', s=30)
axes[0].set_title(f'NN Decision Boundary (test acc: {nn.accuracy(X_test_n, y_test):.2f})')

axes[1].plot(nn.loss_history)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Training Loss')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('02_nn_scratch_result.png', dpi=100, bbox_inches='tight')
plt.close()
print("OK Saved: 02_nn_scratch_result.png")


# ===========================================================
# BAGIAN 4: Vanishing/Exploding Gradients
# ===========================================================
print("\n=== Gradient Flow Analysis ===")
print("""
Masalah utama di deep networks:

1. VANISHING GRADIENTS
   - Sigmoid/Tanh: gradient max = 0.25 (sigmoid), 1.0 (tanh)
   - Di deep net: gradient = product of small numbers -> mendekati 0
   - Layer awal hampir tidak belajar!
   - Solusi: ReLU activation, BatchNorm, Residual connections

2. EXPLODING GRADIENTS
   - Gradient = product of large numbers -> infinity
   - Weights menjadi NaN
   - Solusi: Gradient clipping, proper initialization

3. DEAD NEURONS (khusus ReLU)
   - ReLU output = 0 jika input < 0
   - Gradient = 0 -> neuron "mati" dan tidak belajar
   - Solusi: Leaky ReLU, ELU, GELU

Dengan background Teknik Elektro, pikirkan ini sebagai:
- Vanishing = over-damped system (terlalu lambat)
- Exploding = under-damped system (tidak stabil)
- Good training = critically-damped (optimal convergence)
""")


# ===========================================================
# LATIHAN 11: Extend the Neural Network
# ===========================================================
"""
TARGET Learning Objectives:
   - Memperluas neural network dengan fitur modern
   - Memverifikasi backpropagation dengan gradient checking
   - Memahami efek berbagai weight initialization

PANDUAN LANGKAH-LANGKAH:

STEP 1: Tambahkan Support untuk Fitur Modern
---------------------------------------------
Modifikasi class NeuralNetwork untuk support:

   a) Leaky ReLU, ELU, GELU activation:
      - Leaky ReLU: max(alpha*x, x) dengan alpha=0.01
      - ELU: x jika x>0, alpha(exp(x)-1) jika x<=0
      - GELU: x * Phi(x) (Gaussian CDF)
      
   b) Dropout regularization:
      - During training: randomly set p% neurons to 0
      - During inference: scale outputs by (1-p)
      - Implementation: mask = (np.random.rand(shape) > p) / (1-p)
      
   c) Batch Normalization:
      - Normalize per batch: (x - batch_mean) / sqrt(batch_var + eps)
      - Learnable parameters: gamma (scale), beta (shift)
      - Running statistics untuk inference
      
   d) Multi-class output (softmax + cross-entropy):
      - Output layer: softmax instead of sigmoid
      - Loss: categorical cross-entropy
      - Gradient: probs - y_one_hot

   TIPS KENAPA fitur ini penting?
     - Leaky ReLU/ELU/GELU: menghindari dead neurons
     - Dropout: mencegah overfitting
     - BatchNorm: mempercepat training, memungkinkan higher lr
     - Softmax: generalisasi ke multi-class


STEP 2: Implementasi Gradient Checking
---------------------------------------
Verifikasi backpropagation dengan numerical gradient:

   a) Untuk setiap weight w:
      - Compute loss(w + epsilon)
      - Compute loss(w - epsilon)
      - Numerical gradient = (loss(w+epsilon) - loss(w-epsilon)) / (2*epsilon)
      
   b) Bandingkan dengan analytical gradient dari backprop
   c) Relative error harus < 1e-5
   
   Formula relative error:
   |num_grad - ana_grad| / (|num_grad| + |ana_grad| + eps)
   
   TIPS KENAPA gradient checking?
     - Backpropagation mudah salah implementasi
     - Gradient checking adalah unit test untuk backprop
     - Wajib dilakukan saat mengembangkan architecture baru


STEP 3: Eksperimen Weight Initialization
-----------------------------------------
   Bandingkan 5 strategi inisialisasi:
   
   a) Zero initialization:
      - weights = 0
      - Ekspektasi: semua neuron identical -> tidak belajar
      
   b) Random normal (0.01):
      - weights = np.random.randn() * 0.01
      - Ekspektasi: vanishing gradients di deep network
      
   c) Xavier initialization:
      - weights = np.random.randn() * sqrt(1/n_in)
      - Good untuk sigmoid/tanh
      
   d) He initialization:
      - weights = np.random.randn() * sqrt(2/n_in)
      - Good untuk ReLU (default di atas)
      
   e) Orthogonal initialization:
      - weights = orthogonal matrix * scale
      - Preserves norm through layers
      
   TEST:
   - Train network dengan masing-masing initialization
   - Plot: gradient magnitude per layer
   - Plot: training loss curve
   - Analisis: mana yang paling stabil?


STEP 4: Visualisasi Gradient Flow
---------------------------------
   Plot magnitude gradient di setiap layer selama training:
   
   a) Untuk setiap epoch, hitung mean absolute gradient per layer
   b) Plot: layer vs gradient magnitude (heatmap atau line plot)
   c) Identifikasi: ada layer dengan vanishing/exploding gradients?
   
   TIPS KENAPA visualisasi?
     - Membantu diagnose training issues
     - Layer dengan gradient ~0 = tidak belajar
     - Layer dengan gradient huge = unstable


TIPS HINTS:
   - Untuk GELU, gunakan approx: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x**3)))
   - Untuk gradient checking, epsilon = 1e-5
   - Untuk BatchNorm, simpan running_mean dan running_var
   - Orthogonal init: np.linalg.qr(np.random.randn(n, n))[0]

PERINGATAN COMMON MISTAKES:
   - Dropout diterapkan saat inference
   - BatchNorm menggunakan batch stats saat inference
   - Gradient checking dengan epsilon terlalu kecil -> numerical issues
   - Lupa scale dropout output saat inference

TARGET EXPECTED OUTPUT:
   - NeuralNetwork yang support modern features
   - Gradient checking passing (relative error < 1e-5)
   - He initialization paling stabil untuk ReLU
   - Visualisasi gradient flow yang informatif
"""


# ===========================================================
# CHALLENGE: Universal Function Approximator
# ===========================================================
"""
TARGET Learning Objectives:
   - Membuktikan teorema Universal Approximation
   - Memahami tradeoff depth vs width
   - Menganalisis representasi internal neural network

PANDUAN LANGKAH-LANGKAH:

STEP 1: Define Target Function
------------------------------
Buat fungsi target yang kompleks:
   f(x) = sin(x) * cos(2x) + 0.5 * sin(5x)
   
Generate data:
   x = np.linspace(-2*pi, 2*pi, 1000)
   y = f(x) + noise
   
TIPS KENAPA fungsi ini?
  - Non-linear (tidak bisa di-approximate dengan linear model)
  - Multiple frequency components
  - Memerlukan representasi hierarchical


STEP 2: Experiment with Architecture
------------------------------------
Coba berbagai konfigurasi:

   a) Depth experiment (width fixed = 32):
      - 1 hidden layer
      - 2 hidden layers
      - 3 hidden layers
      - 5 hidden layers
      - 10 hidden layers
      
   b) Width experiment (depth fixed = 2):
      - 4 neurons per layer
      - 8 neurons per layer
      - 16 neurons per layer
      - 32 neurons per layer
      - 64 neurons per layer
      
   TIPS KENAPA experiment ini?
     - Teorema UAT: 1 hidden layer cukup untuk approximate
       ANY continuous function (dengan width yang cukup)
     - Tapi dalam praktik: deep networks lebih efisien
     - Depth vs width tradeoff adalah active research area


STEP 3: Train and Evaluate
---------------------------
   Untuk setiap konfigurasi:
   a) Train sampai convergence
   b) Evaluate: MSE pada test set
   c) Plot: true function vs approximation
   d) Count: total number of parameters
   
   TIPS Analisis:
     - Mana yang lebih penting: depth atau width?
     - Berapa minimum parameters untuk error < 0.01?
     - Apakah deeper always better?


STEP 4: Analyze Internal Representations
------------------------------------------
   Untuk network terbaik, analisis hidden layers:
   
   a) Visualisasi activation patterns:
      - Forward pass data melalui network
      - Plot activation di setiap hidden layer
      - Apakah ada "feature detectors" yang muncul?
      
   b) Weight visualization:
      - Plot heatmap dari weight matrices
      - Apakah ada structure yang terbentuk?
      
   c) Feature evolution:
      - Input: simple sinusoid
      - Layer 1: ???
      - Layer 2: ???
      - Output: approximation
      - Apa yang di-learn di setiap layer?


STEP 5: Conclusion
------------------
   Buat kesimpulan:
   - Minimum depth & width untuk approximate fungsi ini dengan error < 0.01
   - Insight tentang representasi hierarchical
   - Recommendation untuk choosing architecture


TIPS HINTS:
   - Gunakan MSE loss untuk regression
   - Output layer: linear (tanpa activation)
   - Learning rate: 0.01-0.1
   - Early stopping jika loss stagnate
   - Smoothing dengan moving average untuk visualisasi

PERINGATAN COMMON MISTAKES:
   - Terlalu few parameters -> underfitting
   - Terlalu many parameters -> overfitting
   - Learning rate terlalu besar -> divergence
   - Tidak normalize input -> training instabil

TARGET EXPECTED OUTPUT:
   - Network dengan MSE < 0.01
   - Analysis: depth vs width tradeoff
   - Visualisasi internal representations
   - Kesimpulan praktis untuk memilih architecture

Ini fondasi untuk memahami kenapa deep learning bekerja!
"""

print("\n" + "="*50)
print("OK Modul selesai! Lanjut ke: 04-deep-learning/02_pytorch_fundamentals.py")
print("="*50)
