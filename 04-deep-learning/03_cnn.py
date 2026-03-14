"""
=============================================================
FASE 4 — MODUL 3: CNN (Convolutional Neural Networks)
=============================================================
CNN = model yang PALING sukses untuk data grid (image, signal).

Koneksi LANGSUNG dengan Teknik Elektro:
- Convolution di CNN = PERSIS convolution di signal processing!
- Filter/kernel = impulse response dari sistem LTI
- Feature maps = output dari bank of filters
- Pooling = downsampling (decimation)

Kalau kamu paham konvolusi dari DSP, kamu sudah paham 80% CNN.

Durasi target: 4-5 jam
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ===========================================================
# 📖 BAGIAN 1: Konvolusi — DSP Meets Deep Learning
# ===========================================================

# 1D Convolution (signal processing perspective)
def conv1d_manual(signal, kernel):
    """Manual 1D convolution — persis seperti di DSP!"""
    k_len = len(kernel)
    output_len = len(signal) - k_len + 1
    output = np.zeros(output_len)
    for i in range(output_len):
        output[i] = np.sum(signal[i:i+k_len] * kernel)
    return output

# Demo: edge detection pada sinyal
t = np.linspace(0, 1, 200)
signal = np.zeros(200)
signal[50:100] = 1.0  # step signal
signal += 0.05 * np.random.randn(200)

# Different kernels (filters)
kernels = {
    'Moving Average': np.ones(5) / 5,       # low-pass filter
    'Edge Detection': np.array([-1, 0, 1]),  # derivative = high-pass
    'Laplacian': np.array([1, -2, 1]),       # 2nd derivative
}

fig, axes = plt.subplots(len(kernels) + 1, 1, figsize=(12, 8))
axes[0].plot(t, signal)
axes[0].set_title('Original Signal')

for ax, (name, kernel) in zip(axes[1:], kernels.items()):
    output = conv1d_manual(signal, kernel)
    ax.plot(output)
    ax.set_title(f'After {name} filter: {kernel}')

plt.tight_layout()
plt.savefig('01_convolution_demo.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 01_convolution_demo.png")
print("→ CNN LEARNS these kernels automatically from data!")


# ===========================================================
# 📖 BAGIAN 2: CNN Architecture
# ===========================================================

class CNN(nn.Module):
    """
    CNN untuk image classification.

    Architecture:
    Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC → Output

    Setiap Conv layer belajar filter yang berbeda:
    - Layer awal: edge, texture (low-level features)
    - Layer tengah: parts (mid-level)
    - Layer akhir: objects (high-level)
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (1,28,28) → (32,28,28)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),                              # (32,28,28) → (32,14,14)

            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (32,14,14) → (64,14,14)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                              # (64,14,14) → (64,7,7)

            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d(4),                      # → (128,4,4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ===========================================================
# 📖 BAGIAN 3: Train on MNIST
# ===========================================================

# Data augmentation & loading
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("\n=== Loading MNIST ===")
train_dataset = datasets.MNIST('./data', train=True, download=True,
                                transform=transform_train)
test_dataset = datasets.MNIST('./data', train=False,
                               transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Visualize samples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, (img, label) in enumerate(train_dataset):
    if i >= 10:
        break
    ax = axes[i // 5, i % 5]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')
plt.suptitle('MNIST Samples')
plt.tight_layout()
plt.savefig('02_mnist_samples.png', dpi=100, bbox_inches='tight')
plt.close()

# Train
model = CNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print("\n=== Training CNN on MNIST ===")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

train_losses = []
test_accuracies = []

for epoch in range(10):
    model.train()
    running_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    test_acc = correct / total
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    test_accuracies.append(test_acc)

    print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, test_acc={test_acc:.4f}")

print(f"\nFinal Test Accuracy: {test_accuracies[-1]:.4f}")


# ===========================================================
# 📖 BAGIAN 4: Visualize What CNN Learns
# ===========================================================

# Visualize first conv layer filters
filters = model.features[0].weight.data.cpu().numpy()
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < filters.shape[0]:
        ax.imshow(filters[i, 0], cmap='RdBu_r')
    ax.axis('off')
plt.suptitle('Learned Conv Filters (Layer 1)')
plt.tight_layout()
plt.savefig('03_learned_filters.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 03_learned_filters.png")
print("→ Notice: filters learned edge detectors, similar to Sobel/Prewitt!")


# ===========================================================
# 📖 BAGIAN 5: 1D CNN untuk Signal Classification
# ===========================================================
# Ini SANGAT relevan untuk Teknik Elektro!
# 1D CNN perfect untuk: vibration analysis, ECG, power quality, audio

class CNN1D(nn.Module):
    """1D CNN untuk signal/time series classification"""
    def __init__(self, n_channels, n_classes, signal_length):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),  # Global Average Pooling
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Quick demo with synthetic signal data
print("\n=== 1D CNN for Signal Classification ===")
n_samples = 1000
signal_length = 256
n_classes = 3

# Generate 3 classes of signals
X_signals = np.zeros((n_samples, 1, signal_length))
y_signals = np.zeros(n_samples, dtype=int)

t = np.linspace(0, 1, signal_length)
for i in range(n_samples):
    cls = i % n_classes
    y_signals[i] = cls
    if cls == 0:   # Normal: clean sinusoidal
        X_signals[i, 0] = np.sin(2*np.pi*10*t) + 0.1*np.random.randn(signal_length)
    elif cls == 1:  # Harmonic distortion
        X_signals[i, 0] = (np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*30*t)
                          + 0.1*np.random.randn(signal_length))
    else:           # Noisy / degraded
        X_signals[i, 0] = np.sin(2*np.pi*10*t) + 0.5*np.random.randn(signal_length)

# Split & convert
X_train_s = torch.FloatTensor(X_signals[:800]).to(device)
y_train_s = torch.LongTensor(y_signals[:800]).to(device)
X_test_s = torch.FloatTensor(X_signals[800:]).to(device)
y_test_s = torch.LongTensor(y_signals[800:]).to(device)

model_1d = CNN1D(n_channels=1, n_classes=3, signal_length=signal_length).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_1d.parameters(), lr=0.001)

for epoch in range(30):
    model_1d.train()
    outputs = model_1d(X_train_s)
    loss = criterion(outputs, y_train_s)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model_1d.eval()
with torch.no_grad():
    test_out = model_1d(X_test_s)
    _, predicted = torch.max(test_out, 1)
    test_acc = (predicted == y_test_s).float().mean()
print(f"Signal Classification Accuracy: {test_acc:.4f}")


# ===========================================================
# 🏋️ EXERCISE 13: CNN Experiments
# ===========================================================
"""
1. Modifikasi CNN architecture:
   - Coba berbagai kernel size (3, 5, 7)
   - Coba berbagai pooling (max, average, no pooling)
   - Tambahkan residual connections (skip connections)
   - Bandingkan performance pada MNIST

2. Implementasi CNN untuk CIFAR-10 (color images):
   - Input: 3x32x32 (RGB)
   - Target accuracy: >80%
   - Gunakan data augmentation (flip, crop, rotation)

3. Implementasi 1D CNN untuk:
   - ECG classification (generate synthetic ECG signals)
   - Vibration analysis (bearing fault detection)
   - Audio classification (different waveforms)
"""


# ===========================================================
# 🔥 CHALLENGE: Spectrogram CNN
# ===========================================================
"""
Gabungkan signal processing + CNN:

1. Generate audio/vibration signals (3+ kelas)
2. Compute spectrogram (STFT) dari setiap sinyal
3. Treat spectrogram sebagai "image" (2D)
4. Train 2D CNN pada spectrogram
5. Bandingkan dengan:
   - 1D CNN langsung pada raw signal
   - Classical ML pada hand-crafted features

Ini adalah pendekatan yang SANGAT populer di industri:
- Speech recognition: Mel-spectrogram + CNN
- Vibration analysis: spectrogram + CNN
- Power quality: spectrogram + CNN

Kamu menggabungkan keahlian EE (signal processing)
dengan ML (CNN) → ini competitive advantage kamu!
"""

print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 04-deep-learning/04_rnn_timeseries.py")
print("="*50)


# ===========================================================
# MILESTONE ASSESSMENT — 4.3 CNN
# ===========================================================
# Referensi lengkap: ASSESSMENT.md (Fase 4, bagian 4.3)
#
# Level 1 — Bisa Dikerjakan (timer: 45 menit):
#   [ ] Build CNN: Conv2d -> ReLU -> BatchNorm -> MaxPool -> FC
#   [ ] Train pada MNIST/CIFAR-10 dengan data augmentation
#   [ ] Plot loss curve dan confusion matrix
#
# Level 2 — Bisa Dijelaskan:
#   [ ] Conv layer = filter di signal processing — analogikan
#   [ ] Hitung output size: (W - K + 2P) / S + 1
#   [ ] Kenapa CNN > MLP untuk image/signal? (parameter sharing, invariance)
#   [ ] Stride, padding, dilation — efek masing-masing
#
# Level 3 — Bisa Improvisasi (timer: 60 menit):
#   [ ] 1D CNN untuk klasifikasi sinyal
#   [ ] Visualisasi learned filters dan feature maps
#   [ ] Tambahkan residual connections / skip connections
#
# SKOR: ___/30
# TARGET PD: minimal 20/30 (rata-rata 2.0)
# ===========================================================
