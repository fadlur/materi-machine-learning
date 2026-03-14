"""
=============================================================
FASE 4 — MODUL 2: PYTORCH FUNDAMENTALS
=============================================================
Sekarang kamu tahu cara kerja neural net dari dalam.
Saatnya pakai framework yang proper: PyTorch.

Kenapa PyTorch (bukan TensorFlow)?
- Lebih "Pythonic" dan intuitif
- Dynamic computation graph (easier debugging)
- Dominan di riset/akademik (relevan dengan background S2)
- Lebih mudah untuk custom architectures

Durasi target: 4-5 jam
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ===========================================================
# 📖 BAGIAN 1: Tensor Basics
# ===========================================================
# Tensor = NumPy array + GPU support + automatic differentiation

# Membuat tensor
x = torch.tensor([1.0, 2.0, 3.0])
print(f"\nTensor: {x}, dtype: {x.dtype}")

# Dari NumPy (zero-copy jika tipe sama!)
np_array = np.random.randn(3, 4)
t_from_np = torch.from_numpy(np_array)
print(f"From NumPy: shape={t_from_np.shape}")

# Random tensors
t_rand = torch.randn(3, 4)  # normal distribution
t_zeros = torch.zeros(2, 3)
t_ones = torch.ones(5)

# Operasi — PERSIS seperti NumPy
A = torch.randn(3, 4)
B = torch.randn(4, 2)
C = A @ B  # matrix multiplication
print(f"Matrix multiply: ({A.shape}) @ ({B.shape}) = {C.shape}")

# GPU (kalau ada)
if torch.cuda.is_available():
    x_gpu = x.to(device)
    print(f"On GPU: {x_gpu.device}")


# ===========================================================
# 📖 BAGIAN 2: Autograd — Automatic Differentiation
# ===========================================================
# INI yang membuat PyTorch powerful!
# Autograd = menghitung gradient OTOMATIS
# Kamu tidak perlu tulis backprop manual lagi!

# Simple example
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1  # y = x² + 2x + 1

y.backward()  # compute dy/dx automatically!
print(f"\ny = x² + 2x + 1")
print(f"x = {x.item()}")
print(f"y = {y.item()}")
print(f"dy/dx = {x.grad.item()} (expected: 2x + 2 = {2*x.item() + 2})")

# Multi-variable
w = torch.randn(3, requires_grad=True)
x = torch.randn(3)
y_true = torch.tensor(1.0)

# Forward pass
z = torch.dot(w, x)
y_pred = torch.sigmoid(z)
loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

# Backward pass
loss.backward()
print(f"\nAutograd gradients: {w.grad}")
print("→ Ini yang di fase 2 kamu hitung manual. Sekarang otomatis!")


# ===========================================================
# 📖 BAGIAN 3: Building Models with nn.Module
# ===========================================================

class SimpleNet(nn.Module):
    """Neural network pakai PyTorch"""
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))  # BatchNorm for stability
            layers.append(nn.Dropout(0.2))     # Dropout for regularization
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ===========================================================
# 📖 BAGIAN 4: Training Loop — The PyTorch Way
# ===========================================================

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Prepare data
X_np, y_np = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2)

# Normalize
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)

# DataLoader — handles batching and shuffling
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model, Loss, Optimizer
model = SimpleNet(input_size=2, hidden_sizes=[32, 16], output_size=1).to(device)
criterion = nn.BCEWithLogitsLoss()  # sigmoid + BCE combined (numerically stable)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
print("\n=== Training PyTorch Model ===")
train_losses = []
test_losses = []

for epoch in range(100):
    model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer.zero_grad()  # PENTING: reset gradients!
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t)
        test_pred = model(X_test_t)
        train_loss = criterion(train_pred, y_train_t).item()
        test_loss = criterion(test_pred, y_test_t).item()
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    if (epoch + 1) % 20 == 0:
        train_acc = ((torch.sigmoid(train_pred) > 0.5).float() == y_train_t).float().mean()
        test_acc = ((torch.sigmoid(test_pred) > 0.5).float() == y_test_t).float().mean()
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, "
              f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label='Train')
axes[0].plot(test_losses, label='Test')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Learning Curves')
axes[0].legend()
axes[0].grid(True)

# Decision boundary
model.eval()
with torch.no_grad():
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    Z = (torch.sigmoid(model(grid)) > 0.5).cpu().numpy().reshape(xx.shape)

axes[1].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu',
                edgecolors='black', s=20)
axes[1].set_title('Decision Boundary')

plt.tight_layout()
plt.savefig('03_pytorch_training.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 03_pytorch_training.png")


# ===========================================================
# 📖 BAGIAN 5: Model Saving & Loading
# ===========================================================

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
}, 'model_checkpoint.pth')
print("\n✅ Model saved to model_checkpoint.pth")

# Load model
checkpoint = torch.load('model_checkpoint.pth', weights_only=False)
model_loaded = SimpleNet(2, [32, 16], 1).to(device)
model_loaded.load_state_dict(checkpoint['model_state_dict'])
model_loaded.eval()
print("✅ Model loaded successfully")


# ===========================================================
# 📖 BAGIAN 6: Custom Dataset & DataLoader
# ===========================================================

from torch.utils.data import Dataset

class SensorDataset(Dataset):
    """Custom dataset untuk sensor data — pattern yang sering dipakai"""
    def __init__(self, features, labels, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


# ===========================================================
# 🏋️ EXERCISE 12: PyTorch Deep Dive
# ===========================================================
"""
1. Implementasi model yang sama dari Fase 4 Modul 1 (NN from scratch)
   tapi pakai PyTorch. Bandingkan:
   - Lines of code
   - Training speed
   - Final performance

2. Buat custom loss function:
   - Focal Loss: untuk imbalanced classification
     FL = -α(1-p)^γ * log(p)
   - Huber Loss: untuk robust regression

3. Implementasi learning rate scheduler:
   - CosineAnnealingLR
   - OneCycleLR
   - ReduceLROnPlateau
   Bandingkan training curves

4. Buat training pipeline yang REUSABLE:
   class Trainer:
       def __init__(self, model, criterion, optimizer, device)
       def train_epoch(self, train_loader)
       def evaluate(self, val_loader)
       def fit(self, train_loader, val_loader, epochs)
       def plot_history(self)
"""


# ===========================================================
# 🔥 CHALLENGE: Time Series Prediction
# ===========================================================
"""
Buat model untuk memprediksi sensor data (relevant EE!):

1. Generate synthetic sensor data:
   - Voltage sinusoidal dengan variasi amplitude
   - Trend (aging effect)
   - Seasonal pattern (daily/hourly)
   - Anomali sporadis

2. Buat sliding window dataset:
   - Input: 100 timesteps
   - Output: predict next 10 timesteps

3. Train simple feedforward network
   (nanti di modul RNN kita akan improve ini)

4. Evaluate:
   - MSE, MAE pada test set
   - Plot actual vs predicted
   - Apakah model bisa menangkap trend? Seasonal? Anomali?
"""

print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 04-deep-learning/03_cnn.py")
print("="*50)


# ===========================================================
# MILESTONE ASSESSMENT — 4.2 PyTorch Fundamentals
# ===========================================================
# Referensi lengkap: ASSESSMENT.md (Fase 4, bagian 4.2)
#
# Level 1 — Bisa Dikerjakan (timer: 30 menit):
#   [ ] Definisi model pakai nn.Module + nn.Sequential
#   [ ] Training loop: DataLoader, optimizer, loss, train()/eval()
#   [ ] Save dan load model (state_dict)
#
# Level 2 — Bisa Dijelaskan:
#   [ ] requires_grad dan cara kerja autograd
#   [ ] Kenapa model.eval() dan torch.no_grad() saat inference?
#   [ ] DataLoader: batch_size, shuffle, num_workers, pin_memory
#
# Level 3 — Bisa Improvisasi (timer: 45 menit):
#   [ ] Custom Dataset class
#   [ ] Custom loss function (Focal Loss)
#   [ ] LR scheduler: CosineAnnealing + warmup
#
# SKOR: ___/27
# TARGET PD: minimal 18/27 (rata-rata 2.0)
# ===========================================================
