"""
=============================================================
FASE 4 — MODUL 2: PYTORCH FUNDAMENTALS
=============================================================
Setelah belajar NN dari scratch, sekarang pakai PyTorch —
framework deep learning yang paling populer di research & production.

Kenapa PyTorch?
- Pythonic (feels like NumPy)
- Dynamic computation graph
- GPU acceleration dengan CUDA
- Ecosystem yang besar (torchvision, torchaudio, etc.)

Koneksi Teknik Elektro:
- Computational graph = signal flow graph
- Autograd = automatic sensitivity analysis
- GPU parallel = SIMD processing seperti DSP
- Optimizer = adaptive filter algorithms (LMS, RLS, etc.)

Durasi target: 4-5 jam
=============================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

print("=== PyTorch Version:", torch.__version__, "===")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))


# ===========================================================
# 📖 BAGIAN 1: Tensors — Dasar PyTorch
# ===========================================================
# Tensor = generalisasi scalar, vector, matrix ke N-dimensi
# Mirip NumPy tapi bisa di-GPU

# Create tensors
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.randn(3, 3)
z = torch.zeros(2, 3)
w = torch.ones(3, 2)

print("\n=== Tensor Basics ===")
print(f"x shape: {x.shape}, dtype: {x.dtype}")
print(f"y (random):\n{y}")

# Tensor ↔ NumPy (share memory!)
np_arr = np.array([[1, 2], [3, 4]])
tensor_from_np = torch.from_numpy(np_arr)
np_from_tensor = tensor_from_np.numpy()

# Tensor operations
print(f"\nTensor operations:")
print(f"  x + 10:\n{x + 10}")
print(f"  x @ x.T:\n{x @ x.T}")  # Matrix multiplication
print(f"  x.sum(): {x.sum().item()}")

# GPU acceleration (if available)
if torch.cuda.is_available():
    x_gpu = x.cuda()
    print(f"  x on GPU: {x_gpu.device}")
    x_cpu = x_gpu.cpu()
    print(f"  x back to CPU: {x_cpu.device}")


# ===========================================================
# 📖 BAGIAN 2: Autograd — Automatic Differentiation
# ===========================================================
# Core dari PyTorch: AUTOMATIC GRADIENT COMPUTATION
# Setiap tensor punya `.grad_fn` yang track operations
# `.backward()` menghitung gradient dengan chain rule

print("\n=== Autograd ===")
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3  # y = x³
y.backward()  # dy/dx = 3x² = 12
print(f"x = {x.item()}, y = x³ = {y.item()}")
print(f"dy/dx = {x.grad.item()} (expected: 12)")

# Multi-variable
x = torch.tensor([2.0, 3.0], requires_grad=True)
z = x[0] ** 2 + 3 * x[1]
z.backward(torch.tensor(1.0))
print(f"∂z/∂x = {x.grad}")  # [2x, 3] = [4, 3]


# ===========================================================
# 📖 BAGIAN 3: nn.Module — Building Neural Networks
# ===========================================================
# PyTorch menyarankan membuat model sebagai class yang
# mewarisi nn.Module

class SimpleNet(nn.Module):
    """
    Simple neural network with 2 hidden layers.
    
    Architecture:
      Input (n_features) → FC(64) → ReLU → FC(32) → ReLU → FC(1) → Sigmoid
      
    Parameters:
    -----------
    n_features : int
        Jumlah input features.
    n_classes : int, default 1
        Jumlah output classes (1 untuk binary).
        
    Notes:
    ------
    - nn.Linear = fully connected layer (y = xW + b)
    - ReLU activation setelah hidden layer
    - Sigmoid untuk binary classification
    - CrossEntropyLoss untuk multi-class (includes softmax)
    """
    
    def __init__(self, n_features, n_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, n_features)
            Input batch.
            
        Returns:
        --------
        torch.Tensor
            Output predictions.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# ===========================================================
# 📖 BAGIAN 4: Training Loop
# ===========================================================
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate data
X, y = make_classification(n_samples=1000, n_features=10,
                            n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

# DataLoader
# DataLoader mempermudah batching, shuffling, dll
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model, Loss, Optimizer
model = SimpleNet(n_features=10, n_classes=1)
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=0.01)
# Adam = Adaptive Moment Estimation
# Menggabungkan Momentum dengan adaptive learning rate per parameter

print("\n=== Training PyTorch Model ===")
loss_history = []
epochs = 100

for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        # Forward
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward (3 step penting!)
        optimizer.zero_grad()  # Reset gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    
    if (epoch + 1) % 20 == 0:
        # Evaluate
        with torch.no_grad():  # No gradient computation needed
            preds = model(X_test_t)
            preds_binary = (preds >= 0.5).float()
            acc = (preds_binary == y_test_t).float().mean()
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}")


# Visualisasi
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(loss_history)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True)

# Visualisasi weights
weights = model.fc1.weight.data.numpy().flatten()
axes[1].hist(weights, bins=50, edgecolor='black')
axes[1].set_xlabel('Weight Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of FC1 Weights')

plt.tight_layout()
plt.savefig('01_pytorch_training.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 01_pytorch_training.png")


# ===========================================================
# 📖 BAGIAN 5: Model Save & Load
# ===========================================================
print("\n=== Save & Load ===")
# Save
model_path = 'simple_net.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Load (create new instance, then load state)
model_loaded = SimpleNet(n_features=10, n_classes=1)
model_loaded.load_state_dict(torch.load(model_path))
print("Model loaded successfully")


# ===========================================================
# 📖 BAGIAN 6: Multi-Class dengan CrossEntropyLoss
# ===========================================================
class MultiClassNet(nn.Module):
    """
    Multi-class neural network.
    
    Beda dengan binary:
    - Output layer: n_classes neurons (tanpa activation)
    - Loss: CrossEntropyLoss (includes LogSoftmax)
    - Labels: LongTensor (class indices)
    """
    
    def __init__(self, n_features, n_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # NO activation! CrossEntropyLoss includes softmax
        return self.fc3(x)


# ===========================================================
# 🏋️ EXERCISE 12: PyTorch Mastery
# ===========================================================
"""
🎯 Learning Objectives:
   - Membangun CNN dari scratch dengan PyTorch
   - Mengimplementasikan custom loss function
   - Menggunakan learning rate scheduling
   - Melakukan hyperparameter tuning

📋 LANGKAH-LANGKAH:

STEP 1: Build Custom CNN
────────────────────────
Buat arsitektur CNN untuk MNIST:

   a) ConvBlock:
      - Conv2d(in, out, kernel=3, padding=1)
      - BatchNorm2d(out)
      - ReLU()
      - MaxPool2d(2)
      
   b) Architecture:
      Input (28x28x1) → ConvBlock(1→32) → ConvBlock(32→64)
      → Flatten → FC(7*7*64, 128) → Dropout(0.5)
      → FC(128, 10)
      
   💡 KENAPA arsitektur ini?
     - Conv layers: extract spatial features
     - BatchNorm: stabilize training
     - Dropout: prevent overfitting
     - Progressif: 28→14→7 (spatial) dan 1→32→64 (channels)


STEP 2: Implementasi Custom Loss Function
──────────────────────────────────────────
Buat loss function Focal Loss untuk class imbalance:

   Focal Loss = -α * (1 - p_t)^γ * log(p_t)
   
   dimana:
   - p_t = predicted probability untuk true class
   - γ (gamma) > 0: down-weight easy examples
   - α: class weighting
   
   💡 KENAPA Focal Loss?
     - Address class imbalance
     - Focus training pada hard examples
     - Sangat populer di object detection (RetinaNet)


STEP 3: Training dengan Learning Rate Scheduling
───────────────────────────────────────────────
Implementasi LR scheduling strategies:

   a) StepLR: decrease LR setiap N epochs
      scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
      
   b) ReduceLROnPlateau: decrease LR jika loss stagnate
      scheduler = ReduceLROnPlateau(optimizer, patience=5)
      
   c) Cosine Annealing:
      scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
      
   d) ExponentialLR:
      scheduler = ExponentialLR(optimizer, gamma=0.95)
      
   🧪 Experiment:
   - Train dengan masing-masing scheduler
   - Compare: final loss, convergence speed
   - Analisis: mana yang terbaik untuk dataset ini?


STEP 4: Hyperparameter Search
─────────────────────────────
Grid search beberapa hyperparameters:

   a) Learning rate: [0.1, 0.01, 0.001, 0.0001]
   b) Batch size: [16, 32, 64, 128]
   c) Optimizer: [SGD, Adam, RMSprop]
   d) Dropout rate: [0.2, 0.5, 0.8]
   
   🧪 Metrics:
   - Validation accuracy
   - Training time per epoch
   - Final loss
   
   💡 KENAPA grid search?
     - Hyperparameters sangat mempengaruhi performance
     - Tidak ada one-size-fits-all
     - Pattern: Adam + lr=0.001 + batch=32 sering bekerja


💡 HINTS:
   - nn.Conv2d documentation: torch docs essential
   - Focal Loss: implementasi manual dengan torch
   - CosineAnnealingLR: lr = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*t/T))
   - Grid search: nested loops atau itertools.product

⚠️ COMMON MISTAKES:
   - Lupa zero_grad() sebelum backward()
   - Simpan model state dict tanpa arsitektur
   - Learning rate terlalu besar → NaN loss
   - BatchNorm di test tanpa eval() mode
   - Memory leak: tensor yang tidak di-detach()

🎯 EXPECTED OUTPUT:
   - CNN dengan accuracy > 98% pada MNIST
   - Focal Loss implementation yang benar
   - Comparison table untuk setiap scheduler
   - Best hyperparameters combination
"""


# ===========================================================
# 🔥 CHALLENGE: Optimizer from Scratch
# ===========================================================
"""
🎯 Learning Objectives:
   - Mengimplementasikan SGD + Momentum dari scratch
   - Mengimplementasikan Adam optimizer dari scratch
   - Membandingkan custom optimizer dengan PyTorch built-in
   - Memahami matematika di balik adaptive learning rate

📋 LANGKAH-LANGKAH:

STEP 1: Implementasi SGD with Momentum
───────────────────────────────────────

   Algorithm:
   v_t = β * v_{t-1} - lr * gradient
   w_t = w_{t-1} + v_t
   
   Implementasi:
   a) Buat class SGDMomentum:
      - __init__(self, params, lr, momentum=0.9)
      - zero_grad(): reset gradients
      - step(): apply update rule
      
   b) State management:
      - Simpan velocity untuk setiap parameter
      - Initialize velocity = 0
      - Update velocity di setiap step
      
   💡 KENAPA momentum?
     - Accelerate convergence di valley-loss
     - Mencegah oscillation
     - Mirip dengan momentum di physics: inertia


STEP 2: Implementasi Adam Optimizer
────────────────────────────────────

   Adam = Adaptive Moment Estimation
   
   Algorithm:
   m_t = β1 * m_{t-1} + (1-β1) * g_t       (biased first moment)
   v_t = β2 * v_{t-1} + (1-β2) * g_t²      (biased second moment)
   m̂_t = m_t / (1-β1^t)                    (bias correction)
   v̂_t = v_t / (1-β2^t)                    (bias correction)
   w_t = w_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε)
   
   💡 KENAPA Adam?
     - Adaptive per-parameter learning rate
     - Combine momentum (first moment) dengan RMSprop (second moment)
     - Bias correction untuk iterasi awal
     - Default optimizer di banyak kasus


STEP 3: Verifikasi Correctness
──────────────────────────────
   a) Train model dengan custom optimizer
   b) Bandingkan dengan PyTorch optimizer:
      - Final loss harus sama (atau sangat dekat)
      - Loss curve harus mirip
      - Weights harus converge ke values yang sama
      
   💡 Verifikasi:
     - Gunakan seed yang sama
     - Bandingkan loss per epoch
     - Check: gradient computation sama


STEP 4: Analisis Per-Parameter Learning Rate
─────────────────────────────────────────────
   Visualisasi adaptive learning rate:
   
   a) Untuk setiap parameter, plot lr_effective = lr / sqrt(v_t)
   b) Identifikasi: parameter mana dengan lr tertinggi? terendah?
   c) Korelasikan dengan gradient magnitude
   
   💡 KENAPA analisis ini?
     - Memahami kenapa Adam bekerja
     - Sparse features mendapat lr yang lebih tinggi
     - Noisy gradients di-dampen oleh second moment


💡 HINTS:
   - Simpan state di dictionary: {param: {'m': ..., 'v': ...}}
   - Untuk Adam, bias correction sangat penting di epoch awal
   - ε = 1e-8 untuk numerical stability
   - Default PyTorch: β1=0.9, β2=0.999

⚠️ COMMON MISTAKES:
   - Lupa bias correction di Adam (khususnya epoch awal)
   - State tidak di-reset antar training run
   - Mengupdate parameter yang tidak memiliki grad
   - ε terlalu kecil → division by zero

🎯 EXPECTED OUTPUT:
   - Custom SGD+Momentum yang matching dengan PyTorch
   - Custom Adam yang matching dengan PyTorch
   - Visualisasi per-parameter adaptive learning rate
   - Kesimpulan: kapan menggunakan SGD vs Adam

Ini adalah implementasi fundamental yang membedakan
pemula dan engineer yang benar-benar mengerti!
"""

print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 04-deep-learning/03_cnn.py")
print("="*50)
