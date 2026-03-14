"""
=============================================================
FASE 4 — MODUL 1: NEURAL NETWORK FROM SCRATCH
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
# 📖 BAGIAN 1: Single Neuron (Perceptron)
# ===========================================================
# Satu neuron = Linear Regression/Logistic Regression!
# output = activation(w·x + b)

class Neuron:
    """Single neuron — building block dari neural network"""
    def __init__(self, n_inputs, activation='sigmoid'):
        self.w = np.random.randn(n_inputs) * 0.1
        self.b = 0.0
        self.activation = activation

    def forward(self, x):
        z = x @ self.w + self.b
        if self.activation == 'sigmoid':
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
print("📊 Saved: 01_activations.png")


# ===========================================================
# 📖 BAGIAN 2: Multi-Layer Neural Network from Scratch
# ===========================================================

class NeuralNetwork:
    """
    Multi-layer neural network from scratch.

    Architecture: Input → Hidden₁ → Hidden₂ → ... → Output
    Activation: ReLU untuk hidden layers, Sigmoid/Softmax untuk output
    Loss: Binary Cross-Entropy (binary) atau Cross-Entropy (multi-class)
    Optimizer: SGD with momentum
    """

    def __init__(self, layer_sizes, learning_rate=0.01, momentum=0.9):
        """
        layer_sizes: list, contoh [2, 16, 8, 1] = input(2), hidden(16,8), output(1)
        """
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
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
            self.velocity_w.append(np.zeros_like(w))
            self.velocity_b.append(np.zeros_like(b))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        """Forward pass — simpan semua intermediate values untuk backprop"""
        self.activations = [X]  # a[0] = input
        self.z_values = []       # pre-activation values

        current = X
        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            current = self.relu(z)  # Hidden layers: ReLU
            self.activations.append(current)

        # Output layer: Sigmoid (binary) atau linear
        z = current @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        output = self.sigmoid(z)
        self.activations.append(output)

        return output

    def backward(self, y):
        """
        Backpropagation — the CORE of neural network training.

        Chain rule: ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w

        Ini mirip dengan analisis cascaded systems di Teknik Elektro:
        transfer function total = product of individual transfer functions.
        Backprop = menghitung "gain" di setiap stage secara backward.
        """
        n = len(y)
        y = y.reshape(-1, 1) if y.ndim == 1 else y

        # Output layer gradient
        # ∂L/∂z_output = (a_output - y) untuk sigmoid + BCE
        delta = self.activations[-1] - y

        gradients_w = []
        gradients_b = []

        # Backpropagate through layers (dari output ke input)
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient untuk weights dan biases di layer i
            dw = (1/n) * (self.activations[i].T @ delta)
            db = (1/n) * np.sum(delta, axis=0)
            gradients_w.insert(0, dw)
            gradients_b.insert(0, db)

            if i > 0:
                # Propagate delta ke layer sebelumnya
                delta = (delta @ self.weights[i].T) * self.relu_derivative(self.z_values[i-1])

        # Update weights dengan momentum
        for i in range(len(self.weights)):
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.lr * gradients_w[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.lr * gradients_b[i]
            self.weights[i] += self.velocity_w[i]
            self.biases[i] += self.velocity_b[i]

    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        """Training loop dengan mini-batch"""
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
# 📖 BAGIAN 3: Test Neural Network
# ===========================================================

from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# Normalize
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
print("📊 Saved: 02_nn_scratch_result.png")


# ===========================================================
# 📖 BAGIAN 4: Vanishing/Exploding Gradients
# ===========================================================
print("\n=== Gradient Flow Analysis ===")
print("""
Masalah utama di deep networks:

1. VANISHING GRADIENTS
   - Sigmoid/Tanh: gradient max = 0.25 (sigmoid), 1.0 (tanh)
   - Di deep net: gradient = product of small numbers → mendekati 0
   - Layer awal hampir tidak belajar!
   - Solusi: ReLU activation, BatchNorm, Residual connections

2. EXPLODING GRADIENTS
   - Gradient = product of large numbers → infinity
   - Weights menjadi NaN
   - Solusi: Gradient clipping, proper initialization

3. DEAD NEURONS (khusus ReLU)
   - ReLU output = 0 jika input < 0
   - Gradient = 0 → neuron "mati" dan tidak belajar
   - Solusi: Leaky ReLU, ELU, GELU

Dengan background Teknik Elektro, pikirkan ini sebagai:
- Vanishing = over-damped system (terlalu lambat)
- Exploding = under-damped system (tidak stabil)
- Good training = critically-damped (optimal convergence)
""")


# ===========================================================
# 🏋️ EXERCISE 11: Extend the Neural Network
# ===========================================================
"""
1. Tambahkan support untuk:
   - Leaky ReLU, ELU, GELU activation
   - Dropout regularization
   - Batch Normalization
   - Multi-class output (softmax + cross-entropy)

2. Implementasi gradient checking:
   - Hitung numerical gradient: ∂L/∂w ≈ (L(w+ε) - L(w-ε)) / (2ε)
   - Bandingkan dengan analytical gradient dari backprop
   - Relative error harus < 1e-5

3. Eksperimen weight initialization:
   - Zero initialization (apa yang terjadi?)
   - Random normal (0.01)
   - Xavier initialization
   - He initialization
   - Bandingkan training speed dan final performance

4. Visualisasi gradient flow:
   - Plot magnitude gradient di setiap layer selama training
   - Identifikasi vanishing/exploding gradients
"""


# ===========================================================
# 🔥 CHALLENGE: Universal Function Approximator
# ===========================================================
"""
Buktikan bahwa neural network adalah Universal Function Approximator:

1. Buat target function yang kompleks:
   f(x) = sin(x) * cos(2x) + 0.5 * sin(5x)

2. Train neural network untuk approximate function ini:
   - Coba berbagai depth: 1, 2, 3, 5, 10 hidden layers
   - Coba berbagai width: 4, 8, 16, 32, 64 neurons per layer
   - Mana yang lebih penting: depth atau width?

3. Visualisasi:
   - True function vs NN approximation di setiap konfigurasi
   - Training loss curve per konfigurasi
   - "Representation" di setiap hidden layer (apa yang di-learn?)

4. Buat kesimpulan: minimum depth & width untuk approximate
   function ini dengan error < 0.01

Ini fondasi untuk memahami kenapa deep learning bekerja!
"""

print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 04-deep-learning/02_pytorch_fundamentals.py")
print("="*50)


# ===========================================================
# MILESTONE ASSESSMENT — 4.1 Neural Network from Scratch
# ===========================================================
# Referensi lengkap: ASSESSMENT.md (Fase 4, bagian 4.1)
#
# Level 1 — Bisa Dikerjakan (timer: 60 menit):
#   [ ] Forward propagation: input -> hidden (ReLU) -> output (sigmoid)
#   [ ] Backpropagation: hitung gradient untuk setiap layer
#   [ ] Training loop: init -> forward -> loss -> backward -> update
#
# Level 2 — Bisa Dijelaskan:
#   [ ] Chain rule dalam konteks backprop — gambar computation graph
#   [ ] Vanishing gradient: kenapa terjadi dengan sigmoid?
#   [ ] He vs Xavier initialization: kapan pakai masing-masing?
#
# Level 3 — Bisa Improvisasi (timer: 45 menit):
#   [ ] Batch normalization dari scratch
#   [ ] Dropout dari scratch (beda train vs eval)
#   [ ] Gradient checking untuk verifikasi backprop
#
# SKOR: ___/27
# TARGET PD: minimal 18/27 (rata-rata 2.0)
# ===========================================================
