"""
=============================================================
FASE 4 — MODUL 3: CONVOLUTIONAL NEURAL NETWORKS (CNN)
=============================================================
CNN = model khusus untuk data yang punya structure spasial:
- Images (2D)
- Time series (1D)
- Video (3D)

Konsep kunci:
- Local receptive fields (filter/kernel)
- Weight sharing (satu filter di-aplikasikan ke semua lokasi)
- Hierarchical feature extraction

Koneksi Teknik Elektro:
- Convolution = operasi yang familiar dari DSP
- 1D CNN = FIR filter yang di-train
- Pooling = downsampling/decimation
- Feature maps = output dari filtering operation

Durasi target: 4-5 jam
============================================================="""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)


# ===========================================================
# 📖 BAGIAN 1: 2D Convolution dari Scratch
# ===========================================================

def conv2d_scratch(input, kernel, stride=1, padding=0):
    """
    Implementasi 2D convolution dari scratch.
    
    Parameters:
    -----------
    input : np.ndarray, shape (H, W)
        Input 2D array.
    kernel : np.ndarray, shape (kH, kW)
        Convolution kernel/filter.
    stride : int, default 1
        Step size untuk sliding window.
    padding : int, default 0
        Zero padding di sekitar input.
        
    Returns:
    --------
    np.ndarray
        Output feature map.
        
    Notes:
    ------
    Convolution formula:
    output[i, j] = sum(input[i*stride + m, j*stride + n] * kernel[m, n])
    
    Koneksi Teknik Elektro:
    - Ini adalah 2D FIR filter!
    - Kernel = impulse response
    - Stride = downsampling factor
    - Padding = zero-padding untuk maintain size
    """
    if padding > 0:
        input = np.pad(input, pad_width=padding, mode='constant')
    
    kH, kW = kernel.shape
    H, W = input.shape
    
    outH = (H - kH) // stride + 1
    outW = (W - kW) // stride + 1
    
    output = np.zeros((outH, outW))
    
    for i in range(outH):
        for j in range(outW):
            # Extract patch
            patch = input[i*stride:i*stride+kH, j*stride:j*stride+kW]
            # Element-wise multiply and sum (dot product)
            output[i, j] = np.sum(patch * kernel)
    
    return output


# Visualisasi convolution
def visualize_convolution():
    """Visualisasi proses convolution pada edge detection."""
    # Sobel edge detection kernel
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    
    # Create simple pattern
    img = np.zeros((10, 10))
    img[3:7, 2:8] = 1  # Square
    
    convolved = conv2d_scratch(img, sobel_x)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    axes[1].imshow(sobel_x, cmap='RdBu_r')
    axes[1].set_title('Kernel (Sobel X)')
    axes[1].axis('off')
    
    axes[2].imshow(convolved, cmap='RdBu_r')
    axes[2].set_title('Output (Edges)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('01_convolution_visualization.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("📊 Saved: 01_convolution_visualization.png")

visualize_convolution()


# ===========================================================
# 📖 BAGIAN 2: 1D Convolution untuk Time Series
# ===========================================================

def conv1d_scratch(signal, kernel, stride=1):
    """
    1D convolution — relevan untuk signal processing!
    
    Parameters:
    -----------
    signal : np.ndarray
        1D input signal.
    kernel : np.ndarray
        1D convolution kernel.
    stride : int, default 1
        
    Returns:
    --------
    np.ndarray
        Filtered signal.
        
    Notes:
    ------
    Ini identik dengan FIR filtering di DSP!
    - kernel = filter coefficients
    - stride = downsampling factor
    
    Koneksi Teknik Elektro:
    - Low-pass filter: kernel = moving average
    - High-pass filter: kernel = [1, -1]
    - Band-pass filter: kernel = windowed sinusoid
    """
    k = len(kernel)
    n = len(signal)
    out_len = (n - k) // stride + 1
    
    output = np.zeros(out_len)
    for i in range(out_len):
        patch = signal[i*stride:i*stride+k]
        output[i] = np.sum(patch * kernel)
    
    return output


# Demo: 1D convolution sebagai FIR filter
t = np.arange(1000)
signal = np.sin(2 * np.pi * 5 * t / 1000) + 0.5 * np.sin(2 * np.pi * 50 * t / 1000)

# Moving average = low-pass filter
kernel_lowpass = np.ones(20) / 20
filtered = conv1d_scratch(signal, kernel_lowpass)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(t, signal, alpha=0.7, label='Original')
axes[0].set_title('Original Signal (5Hz + 50Hz)')
axes[0].legend()
axes[0].set_xlabel('Sample')

axes[1].plot(filtered, label='Low-pass filtered')
axes[1].set_title('After Moving Average (20-pt)')
axes[1].legend()
axes[1].set_xlabel('Sample')

plt.tight_layout()
plt.savefig('02_1d_convolution.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 02_1d_convolution.png")


# ===========================================================
# 📖 BAGIAN 3: CNN untuk MNIST dengan PyTorch
# ===========================================================
class MNISTCNN(nn.Module):
    """
    CNN untuk klasifikasi MNIST.
    
    Architecture:
    Input(1x28x28) → Conv(32, 3x3) → ReLU → MaxPool(2)
                   → Conv(64, 3x3) → ReLU → MaxPool(2)
                   → Flatten(7x7x64) → FC(128) → Dropout(0.5) → FC(10)
    
    Parameters:
    -----------
    dropout : float, default 0.5
        Dropout probability untuk regularization.
        
    Notes:
    ------
    - Conv layers: extract hierarchical spatial features
    - MaxPool: reduce spatial dimensions, provide translation invariance
    - Dropout: prevent overfitting
    - Final FC: classification layer
    
    Koneksi Teknik Elektro:
    - Conv = matched filtering untuk pattern recognition
    - Pooling = decimation dengan max operation
    - Feature maps = spectrogram-like representation
    """
    
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Layer 1: 28x28 → 28x28 → 14x14
        x = self.pool(self.relu(self.conv1(x)))
        # Layer 2: 14x14 → 14x14 → 7x7
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Training setup
from torchvision import datasets, transforms

def train_mnist_cnn(epochs=5):
    """Train MNIST CNN dan visualisasi hasil."""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True,
                                    download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False,
                                   transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    model = MNISTCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n=== Training MNIST CNN ===")
    train_losses = []
    test_accs = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        acc = 100 * correct / total
        test_accs.append(acc)
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Test Acc={acc:.2f}%")
    
    # Visualisasi
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)
    
    axes[1].plot(test_accs)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Test Accuracy')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('03_mnist_cnn_training.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("📊 Saved: 03_mnist_cnn_training.png")
    
    return model

# Uncomment untuk training (memerlukan beberapa menit)
# model = train_mnist_cnn(epochs=3)


# ===========================================================
# 📖 BAGIAN 4: Visualisasi Feature Maps
# ===========================================================
def visualize_feature_maps(model, sample_image):
    """
    Visualisasi feature maps dari CNN.
    
    Parameters:
    -----------
    model : nn.Module
        Trained CNN model.
    sample_image : torch.Tensor
        Single image (1, 1, 28, 28).
        
    Notes:
    ------
    Feature maps menunjukkan "apa yang dilihat" CNN di setiap layer.
    Layer awal: edge detectors, corners
    Layer tengah: textures, patterns
    Layer akhir: object parts, semantic features
    
    Koneksi Teknik Elektro:
    - Mirip dengan spectrogram yang menunjukkan
      energy di setiap frequency band
    - Feature maps = spatial-frequency representation
    """
    model.eval()
    
    # Hook untuk capture intermediate outputs
    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    model.conv1.register_forward_hook(hook_fn('conv1'))
    model.conv2.register_forward_hook(hook_fn('conv2'))
    
    with torch.no_grad():
        model(sample_image)
    
    # Visualisasi
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    
    # Conv1 features (32 channels, tampilkan 8)
    feat1 = activations['conv1'][0]  # (32, 28, 28)
    for i in range(8):
        axes[0, i].imshow(feat1[i].cpu().numpy(), cmap='viridis')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Conv1 Ch{i}')
    
    # Conv2 features (64 channels, tampilkan 8)
    feat2 = activations['conv2'][0]  # (64, 14, 14)
    for i in range(8):
        axes[1, i].imshow(feat2[i].cpu().numpy(), cmap='viridis')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Conv2 Ch{i}')
    
    plt.tight_layout()
    plt.savefig('04_feature_maps.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("📊 Saved: 04_feature_maps.png")


# ===========================================================
# 📖 BAGIAN 5: 1D CNN untuk Signal Classification
# ===========================================================
class SignalCNN1D(nn.Module):
    """
    1D CNN untuk klasifikasi sinyal time series.
    
    Relevan untuk:
    - Fault detection di motor (vibration signals)
    - Power quality classification
    - ECG classification
    - Speech recognition
    
    Architecture:
    Input(1xN) → Conv1(32, kernel=7) → ReLU → MaxPool(4)
               → Conv1(64, kernel=5) → ReLU → MaxPool(4)
               → Conv1(128, kernel=3) → ReLU → GlobalAvgPool
               → FC(64) → FC(n_classes)
    
    Parameters:
    -----------
    n_classes : int
        Jumlah kelas output.
    input_length : int
        Panjang input signal.
        
    Notes:
    ------
    - Kernel sizes decreasing: large → medium → small
      (capture multi-scale patterns)
    - Global Average Pooling: reduce parameters, improve generalization
    - Dilated convolution: increase receptive field tanpa increasing params
    
    Koneksi Teknik Elektro:
    - Conv1D = FIR filter bank
    - Kernel size 7 ≈ analyze 7-sample window
    - Receptive field grows dengan depth (seperti multi-resolution analysis)
    """
    
    def __init__(self, n_classes=4, input_length=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(4)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ===========================================================
# 🏋️ EXERCISE 13: CNN Mastery
# ===========================================================
"""
🎯 Learning Objectives:
   - Memahami receptive field dan dilated convolution
   - Mengimplementasikan residual connections (ResNet)
   - Membangun CNN untuk domain-specific task

📋 LANGKAH-LANGKAH:

STEP 1: Receptive Field Analysis
────────────────────────────────
   a) Definisi: receptive field = ukuran area input yang
      mempengaruhi satu neuron di layer tersebut.
      
   b) Hitung receptive field untuk:
      - Conv(kernel=3, stride=1) → RF = 3
      - Conv(3) → MaxPool(2) → Conv(3) → RF = ?
      - Conv(5, dilation=2) → RF = ?
      
   c) Formula:
      RF_out = RF_in + (kernel_size - 1) * stride * dilation
      
   d) Buat fungsi calculate_receptive_field(layers) yang
      menghitung RF untuk sequence of layers.
      
   💡 KENAPA penting?
     - RF menentukan ukuran pattern yang bisa dideteksi
     - Small RF → small patterns (edges, textures)
     - Large RF → large patterns (objects, context)
     - Untuk time series: RF menentukan durasi event yang terdeteksi


STEP 2: Implementasi Dilated Convolution
────────────────────────────────────────
   a) Conv1d dengan dilation > 1:
      - Dilation = gap antara taps
      - Dilation=2: taps di [0, 2, 4] bukan [0, 1, 2]
      
   b) Buat class DilatedConvBlock:
      - Conv1d dengan increasing dilation
      - DilatedConv → BatchNorm → ReLU
      
   c) Bandingkan:
      - Dilated Conv: RF besar tanpa parameter banyak
      - Standard Conv: RF kecil atau parameter banyak
      - Visualisasi: receptive field coverage
      
   💡 KENAPA dilated conv?
     - Multi-scale analysis tanpa pooling
     - Preservasi resolution (penting untuk segmentation)
     - Efisien: parameter sama tapi RF lebih besar


STEP 3: Implementasi Residual Block (ResNet)
────────────────────────────────────────────
   a) BasicBlock:
      - Conv → BN → ReLU → Conv → BN
      - Skip connection: input + output
      - ReLU setelah addition
      
   b) BottleneckBlock:
      - Conv(1x1) → Conv(3x3) → Conv(1x1)
      - Reduce → Process → Restore channels
      - Lebih efisien untuk deep networks
      
   c) Visualisasi gradient flow:
      - Dengan skip connection: gradient bisa flow langsung
      - Tanpa skip: gradient harus melewati banyak layers
      - Plot: gradient magnitude per layer
      
   💡 KENAPA residual connections?
     - Memungkinkan training networks yang sangat deep (100+ layers)
     - Mitigasi vanishing gradient
     - Mirip dengan feedback control systems


STEP 4: Domain-Specific CNN
───────────────────────────
   Konteks: Bearing Fault Detection dari Vibration Signals
   
   a) Dataset: synthetic bearing vibration
      - Normal, Inner Race Fault, Outer Race Fault, Ball Fault
      - Sampling rate: 12 kHz, window: 2048 samples
      
   b) Architecture:
      - Input: 1x2048 (raw vibration)
      - Conv layers: extract bearing fault signatures
      - Pooling: reduce dimensionality
      - Classification: 4 classes
      
   c) Feature extraction analysis:
      - Apakah CNN mengekstrak BPFI, BPFO, BSF?
      - Visualisasi: FFT dari feature maps
      - Bandingkan dengan hand-crafted features (RMS, kurtosis)
      
   d) Evaluation:
      - Accuracy per fault type
      - Confusion matrix
      - Robustness ke noise


💡 HINTS:
   - Receptive field: RF = 1 + Σ((k_i - 1) * Π s_j)
   - Dilated conv: torch.nn.Conv1d(..., dilation=d)
   - Residual: F(x) + x (bisa pakai F.conv1d)
   - Untuk bearing: fault frequencies = (n/2) * fr * (1 ± d/D * cos(β))

⚠️ COMMON MISTAKES:
   - Output size tidak sesuai karena padding salah
   - Skip connection dimension mismatch
   - Dilation tidak compatible dengan kernel size
   - BatchNorm di test tanpa eval()
   - Terlalu aggressive pooling → information loss

🎯 EXPECTED OUTPUT:
   - Receptive field calculator
   - Dilated conv implementation
   - ResNet block implementation
   - Bearing fault CNN dengan accuracy > 90%
   - Feature map analysis yang menunjukkan fault frequency extraction
"""


# ===========================================================
# 🔥 CHALLENGE: CNN untuk Power Quality Classification
# ===========================================================
"""
🎯 Learning Objectives:
   - Mengaplikasikan CNN ke real-world power systems problem
   - Menggabungkan time domain dan frequency domain
   - Membangun production-ready signal classification pipeline

📋 LANGKAH-LANGKAH:

STEP 1: Generate Power Quality Dataset
───────────────────────────────────────
   Simulasi 4 kelas gangguan kualitas daya:
   
   a) Normal: sinusoidal murni 50Hz, THD < 3%
   b) Voltage Sag: tegangan turun 30-60% selama 0.1-0.5s
   c) Voltage Swell: tegangan naik 30-60% selama 0.1-0.5s
   d) Harmonic Distortion: THD > 10% (3rd, 5th, 7th harmonics)
   
   Parameter:
   - Sampling rate: 3.2 kHz (64 samples/cycle)
   - Window: 10 cycles = 640 samples
   - Dataset: 1000 windows per class
   
   💡 KENAPA kelas ini?
     - Very common di power systems
     - Setiap kelas punya signature yang distinct
     - Penting untuk protective relaying dan monitoring


STEP 2: Preprocessing
─────────────────────
   a) Normalisasi: scale ke [-1, 1]
   b) Optional: convert ke frequency domain (FFT)
   c) Data augmentation:
      - Add noise (SNR 30-40 dB)
      - Time shift (±10 samples)
      - Amplitude scaling (±5%)
      
   d) Train/val/test split: 70/15/15
   
   💡 KENAPA augmentation?
     - Simulasi variasi real-world
     - Mencegah overfitting
     - Memperkuat generalization


STEP 3: Architecture Design
───────────────────────────
   Desain CNN yang optimal untuk power signals:
   
   a) Raw signal path:
      - Conv1D layers dengan increasing filters
      - Kernel sizes: 7, 5, 3 (multi-scale)
      - Dilated conv untuk long-range dependencies
      
   b) Frequency domain path (optional):
      - FFT input → Conv1D pada spectrum
      - Focus pada harmonic content
      
   c) Fusion:
      - Concatenate features dari kedua path
      - Joint classification
      
   d) Regularization:
      - Dropout 0.3-0.5
      - BatchNorm setiap conv layer
      - Weight decay (L2 regularization)


STEP 4: Training & Evaluation
─────────────────────────────
   a) Training:
      - Loss: CrossEntropyLoss dengan class weights
      - Optimizer: AdamW (Adam + weight decay)
      - LR: 1e-3 dengan cosine annealing
      - Early stopping: patience 10 epochs
      
   b) Evaluation metrics:
      - Overall accuracy
      - Precision, recall, F1 per class
      - Confusion matrix
      - ROC curve per class
      
   c) Analysis:
      - Which class is hardest?
      - False positives analysis
      - Robustness ke noise level yang berbeda


STEP 5: Model Interpretability
──────────────────────────────
   a) Feature map visualization:
      - Plot feature maps dari setiap conv layer
      - Identifikasi: apakah model menangkap fault signatures?
      
   b) Class activation mapping (CAM):
      - Lihat region input yang paling penting
      - Validasi: apakah fault region di-highlight?
      
   c) Sensitivity analysis:
      - Perturbasi input: tambahkan noise ke region tertentu
      - Measure perubahan confidence
      - Identifikasi critical time regions


💡 HINTS:
   - Voltage sag/swell: abrupt changes di envelope
   - Harmonics: peaks di frequency domain
   - Untuk FFT path: window input dengan Hanning window
   - Class weights: inverse frequency untuk imbalance
   - AdamW: torch.optim.AdamW(params, lr, weight_decay)

⚠️ COMMON MISTAKES:
   - Training pada augmented data, test pada original
   - Window boundary effects (Gibbs phenomenon)
   - Class imbalance tanpa weighting
   - FFT phase information tidak relevant → use magnitude only
   - Model terlalu shallow untuk capture fault dynamics

🎯 EXPECTED OUTPUT:
   - Power quality classifier dengan accuracy > 95%
   - Confusion matrix yang clean (minimal misclassification)
   - Feature maps yang menunjukkan fault detection
   - Production-ready pipeline: preprocess → model → predict

Ini adalah aplikasi langsung dari deep learning ke power systems!
"""

print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 04-deep-learning/04_rnn_timeseries.py")
print("="*50)
