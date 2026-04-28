"""
=============================================================
FASE 4 — MODUL 4: RNN, LSTM, GRU, TRANSFORMER
=============================================================
Sequential data = data dengan order temporal:
- Time series (sensors, stocks, weather)
- Text (sentences, documents)
- Audio (speech, music)
- Video (frames)

Models untuk sequential data:
1. RNN (Recurrent Neural Network) — vanishing gradient
2. LSTM (Long Short-Term Memory) — solved vanishing
3. GRU (Gated Recurrent Unit) — simplified LSTM
4. Transformer (Attention) — state-of-the-art

Koneksi Teknik Elektro:
- RNN = Infinite Impulse Response (IIR) filter
- LSTM = state-space model dengan gating
- Transformer = matched filter bank (attention = correlation)
- Sequential prediction = system identification

Durasi target: 5-6 jam
============================================================="""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)


# ===========================================================
# 📖 BAGIAN 1: RNN Basics
# ===========================================================
class SimpleRNN(nn.Module):
    """
    Vanilla RNN implementation.
    
    Architecture:
    h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
    
    Parameters:
    -----------
    input_size : int
        Jumlah input features per timestep.
    hidden_size : int
        Jumlah hidden units.
    output_size : int
        Jumlah output classes.
        
    Notes:
    ------
    - h_t = hidden state yang di-pass antar timestep
    - Hidden state = "memory" dari sequence
    - Masalah utama: vanishing gradients di sequence panjang
    
    Koneksi Teknik Elektro:
    - RNN = IIR filter dengan nonlinear feedback
    - Hidden state = filter state/memory
    - Backprop through time (BPTT) = sensitivity analysis
      melalui time steps
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        """
        Parameters:
        -----------
        x : torch.Tensor, shape (batch, seq_len, input_size)
            Input sequence.
            
        Returns:
        --------
        torch.Tensor
            Output untuk setiap timestep.
        """
        batch_size, seq_len, _ = x.shape
        hidden = torch.zeros(batch_size, self.hidden_size)
        outputs = []
        
        for t in range(seq_len):
            # Combine input dan hidden state
            combined = torch.cat((x[:, t, :], hidden), dim=1)
            hidden = self.tanh(self.i2h(combined))
            output = self.h2o(hidden)
            outputs.append(output)
        
        return torch.stack(outputs, dim=1), hidden


# ===========================================================
# 📖 BAGIAN 2: LSTM — Long Short-Term Memory
# ===========================================================
class SimpleLSTM(nn.Module):
    """
    LSTM implementation — solves vanishing gradient problem.
    
    LSTM Gates:
    - Forget gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
    - Input gate: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
    - Candidate: c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
    - Cell state: c_t = f_t * c_{t-1} + i_t * c̃_t
    - Output gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
    - Hidden state: h_t = o_t * tanh(c_t)
    
    Parameters:
    -----------
    input_size : int
        Jumlah input features.
    hidden_size : int
        Jumlah hidden units.
        
    Notes:
    ------
    - Cell state = "conveyor belt" yang melewati time steps
    - Gates mengontrol apa yang di-forget, input, dan output
    - Gradient bisa flow langsung melalui cell state (tanpa
      melewati nonlinear activation)
    
    Koneksi Teknik Elektro:
    - LSTM = state-space model dengan controlled gates
    - Cell state = integrator (memory element)
    - Forget gate = reset mechanism
    - Input gate = enable/disable input
    - Output gate = output enable
    """
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # LSTM weights (combined untuk efisiensi)
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size)
        c = torch.zeros(batch_size, self.hidden_size)
        
        outputs = []
        for t in range(seq_len):
            combined = torch.cat((x[:, t, :], h), dim=1)
            gates = self.W(combined)
            
            # Split gates
            f, i, c_tilde, o = torch.split(gates, self.hidden_size, dim=1)
            
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            c_tilde = torch.tanh(c_tilde)
            o = torch.sigmoid(o)
            
            # Update cell dan hidden state
            c = f * c + i * c_tilde
            h = o * torch.tanh(c)
            
            outputs.append(h)
        
        return torch.stack(outputs, dim=1), (h, c)


# ===========================================================
# 📖 BAGIAN 3: Time Series Prediction
# ===========================================================
def generate_synthetic_series(n_samples=1000, noise=0.1):
    """
    Generate synthetic time series untuk training.
    
    Model: ARMA-like process
    y_t = 0.6*y_{t-1} - 0.3*y_{t-2} + 0.2*sin(t/10) + noise
    
    Koneksi Teknik Elektro:
    - ARMA = AutoRegressive Moving Average
    - Common model untuk time series di signal processing
    - Memprediksi y_t dari history = system identification
    """
    y = np.zeros(n_samples)
    for t in range(2, n_samples):
        y[t] = 0.6 * y[t-1] - 0.3 * y[t-2] + \
               0.2 * np.sin(t / 10) + np.random.randn() * noise
    return y


def create_sequences(data, seq_length):
    """
    Create input-output pairs untuk sequence prediction.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data.
    seq_length : int
        Panjang input sequence.
        
    Returns:
    --------
    X, y : np.ndarray
        X: shape (n_samples, seq_length, 1)
        y: shape (n_samples, 1)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


# Generate data
series = generate_synthetic_series(2000, noise=0.1)
seq_length = 50
X, y = create_sequences(series, seq_length)

# Split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)
y_test_t = torch.FloatTensor(y_test)

print(f"Train shape: {X_train_t.shape}, Test shape: {X_test_t.shape}")


# ===========================================================
# 📖 BAGIAN 4: Training LSTM untuk Time Series
# ===========================================================
class TimeSeriesLSTM(nn.Module):
    """
    LSTM untuk time series prediction.
    
    Architecture:
    Input(seq_len, 1) → LSTM(32) → Dropout → FC(16) → FC(1)
    
    Parameters:
    -----------
    input_size : int, default 1
    hidden_size : int, default 32
    num_layers : int, default 1
        Jumlah stacked LSTM layers.
        
    Notes:
    ------
    - num_layers > 1: stacked LSTM (LSTM output jadi input
      untuk LSTM berikutnya)
    - Dropout antara layers (jika num_layers > 1)
    - Final hidden state digunakan untuk prediction
    """
    
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        # LSTM output: (batch, seq, hidden), (hidden, cell)
        lstm_out, (hidden, cell) = self.lstm(x)
        # Use last hidden state
        last_hidden = hidden[-1]  # Take last layer
        return self.fc(last_hidden).squeeze()


# Training
model = TimeSeriesLSTM(input_size=1, hidden_size=32, num_layers=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\n=== Training LSTM ===")
losses = []
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    output = model(X_train_t)
    loss = criterion(output, y_train_t)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_loss = criterion(test_pred, y_test_t)
        print(f"  Epoch {epoch+1}: Train Loss={loss.item():.4f}, Test Loss={test_loss.item():.4f}")


# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(series)
axes[0].set_title('Full Time Series')
axes[0].set_xlabel('Time')

axes[1].plot(losses)
axes[1].set_title('Training Loss')
axes[1].set_xlabel('Epoch')
axes[1].grid(True)

model.eval()
with torch.no_grad():
    predictions = model(X_test_t).numpy()
    
axes[2].plot(y_test[:100], label='True', alpha=0.7)
axes[2].plot(predictions[:100], label='Predicted', alpha=0.7)
axes[2].set_title('Predictions vs True (first 100 test points)')
axes[2].legend()
axes[2].set_xlabel('Time')

plt.tight_layout()
plt.savefig('01_rnn_timeseries.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 01_rnn_timeseries.png")


# ===========================================================
# 📖 BAGIAN 5: Multi-Step Forecasting
# ===========================================================
def multi_step_forecast(model, initial_sequence, n_steps):
    """
    Multi-step forecasting menggunakan recursive prediction.
    
    Parameters:
    -----------
    model : nn.Module
        Trained model.
    initial_sequence : np.ndarray
        Sequence awal untuk memulai forecasting.
    n_steps : int
        Jumlah steps ke depan untuk diprediksi.
        
    Returns:
    --------
    np.ndarray
        Predicted values.
        
    Notes:
    ------
    Recursive forecasting:
    1. Prediksi step 1 dari sequence
    2. Append prediksi ke sequence
    3. Gunakan sequence yang di-update untuk prediksi step 2
    4. Ulangi sampai n_steps
    
    ⚠️ Error accumulates! Makin jauh forecast → makin tidak akurat.
    
    Alternatif: direct multi-step (prediksi semua langsung)
    atau seq2seq models.
    """
    model.eval()
    sequence = initial_sequence.copy()
    forecasts = []
    
    with torch.no_grad():
        for _ in range(n_steps):
            # Prepare input
            x = torch.FloatTensor(sequence[-seq_length:]).reshape(1, -1, 1)
            # Predict
            pred = model(x).item()
            forecasts.append(pred)
            # Append to sequence
            sequence = np.append(sequence, pred)
    
    return np.array(forecasts)


# ===========================================================
# 📖 BAGIAN 6: Transformer untuk Time Series
# ===========================================================
class TimeSeriesTransformer(nn.Module):
    """
    Transformer untuk time series forecasting.
    
    Architecture:
    Input → Linear Embedding → Positional Encoding
          → Transformer Encoder (multi-head attention)
          → FC → Output
    
    Parameters:
    -----------
    input_size : int, default 1
    d_model : int, default 64
        Embedding dimension.
    nhead : int, default 4
        Number of attention heads.
    num_layers : int, default 2
        Number of transformer encoder layers.
        
    Notes:
    ------
    - Transformer tidak punya inductive bias untuk sequential data
      (unlike RNN/LSTM)
    - Butuh positional encoding untuk inject order information
    - Multi-head attention = multiple "views" dari sequence
    - Bisa parallel processing (tidak sequential seperti RNN)
    
    Koneksi Teknik Elektro:
    - Attention = cross-correlation antara query dan key
    - Multi-head = multiple filter banks dengan frequencies berbeda
    - Positional encoding = carrier frequency untuk time index
    """
    
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: (batch, seq, 1)
        x = self.input_proj(x)  # (batch, seq, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # Use last position
        x = x[:, -1, :]
        return self.fc(x).squeeze()


class PositionalEncoding(nn.Module):
    """
    Positional encoding dengan sinusoidal functions.
    
    Formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Notes:
    ------
    - Memberikan informasi posisi ke transformer
    - Unik untuk setiap position (tidak ambiguous)
    - Dapat generalize ke sequence lengths yang tidak terlihat
      saat training
    
    Koneksi Teknik Elektro:
    - Ini adalah frequency-domain encoding!
    - pos = time index, i = frequency bin
    - Mirip dengan mel-frequency cepstral coefficients (MFCC)
    """
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ===========================================================
# 🏋️ EXERCISE 14: Sequential Models Mastery
# ===========================================================
"""
🎯 Learning Objectives:
   - Membangun seq2seq model untuk multi-step forecasting
   - Mengimplementasikan attention mechanism
   - Membandingkan RNN, LSTM, GRU, Transformer

📋 LANGKAH-LANGKAH:

STEP 1: Implementasi GRU
────────────────────────
GRU = Gated Recurrent Unit (simplified LSTM)

   a) Gates:
      - Reset gate: r_t = σ(W_r · [h_{t-1}, x_t])
      - Update gate: z_t = σ(W_z · [h_{t-1}, x_t])
      - Candidate: h̃_t = tanh(W · [r_t * h_{t-1}, x_t])
      - Hidden: h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
      
   b) Bandingkan dengan LSTM:
      - GRU punya 2 gates vs LSTM 3 gates
      - GRU lebih cepat (fewer parameters)
      - Performance serupa di banyak task
      
   c) Implementasi from scratch (PyTorch nn.Module):
      - GRUCell manual
      - Bandingkan dengan nn.GRU output
      
   💡 KENAPA GRU?
     - Lebih sederhana dari LSTM
     - Lebih cepat training
     - Cukup untuk banyak sequential tasks


STEP 2: Seq2Seq untuk Multi-Step Forecasting
────────────────────────────────────────────
   a) Architecture:
      - Encoder: LSTM/GRU yang process input sequence
      - Decoder: LSTM/GRU yang generate output sequence
      - Encoder final hidden state = initial decoder state
      
   b) Training:
      - Teacher forcing: gunakan true value sebagai input
        decoder (bukan prediksi sendiri)
      - Loss: MSE untuk setiap output timestep
      
   c) Inference:
      - Decoder input = prediksi sebelumnya (recursive)
      - Atau: predict all at once
      
   d) Evaluasi:
      - RMSE per forecast horizon
      - Plot: true vs predicted untuk setiap horizon
      
   💡 KENAPA seq2seq?
     - Directly predicts multiple steps
     - Tidak perlu recursive forecasting
     - Lebih stabil untuk long-term prediction


STEP 3: Attention Mechanism
───────────────────────────
   a) Implementasi scaled dot-product attention:
      Attention(Q, K, V) = softmax(QK^T / √d_k) V
      
   b) Self-attention untuk time series:
      - Q = K = V = hidden states dari encoder
      - Setiap timestep "melihat" semua timestep lain
      - Weight = importance dari setiap timestep
      
   c) Visualisasi attention weights:
      - Heatmap: time vs time
      - Identifikasi: timestep mana yang paling penting?
      - Apakah attention menangkap seasonal patterns?
      
   💡 KENAPA attention?
     - Model bisa fokus pada relevant timesteps
     - Interpretable: bisa lihat "what the model is looking at"
     - Menghindari bottleneck dari encoder final state


STEP 4: Comprehensive Comparison
─────────────────────────────────
   Bandingkan 4 arsitektur pada dataset yang sama:
   
   a) LSTM (baseline)
   b) GRU (simplified LSTM)
   c) Transformer (attention-based)
   d) LSTM + Attention (seq2seq with attention)
   
   Metrics:
   - RMSE pada test set
   - Training time per epoch
   - Inference time
   - Number of parameters
   - Memory usage
   
   💡 Analisis:
     - Mana yang terbaik untuk short-term vs long-term?
     - Tradeoff accuracy vs efficiency?
     - Kapan menggunakan masing-masing?


💡 HINTS:
   - GRU equations lebih simple dari LSTM
   - Seq2seq: encoder_output, (hidden, cell) = encoder(input)
   - Attention: torch.bmm(Q, K.transpose(1, 2)) untuk batch matrix multiply
   - Scaled attention: divide by sqrt(d_k) untuk stability
   - Teacher forcing ratio: decrease dari 1.0 ke 0.0 selama training

⚠️ COMMON MISTAKES:
   - Teacher forcing tanpa schedule → model tidak belajar
     inference (exposure bias)
   - Attention tanpa scaling → softmax terlalu sharp
   - Lupa positional encoding di transformer
   - Seq2seq tanpa EOS token → infinite generation
   - Gradient clipping tidak di-LSTM → exploding gradients

🎯 EXPECTED OUTPUT:
   - GRU implementation matching PyTorch nn.GRU
   - Seq2seq model dengan multi-step forecasting
   - Attention visualization heatmap
   - Comprehensive comparison table
"""


# ===========================================================
# 🔥 CHALLENGE: Transformer untuk Fault Prediction
# ===========================================================
"""
🎯 Learning Objectives:
   - Mengaplikasikan transformer ke real-world industrial problem
   - Menggabungkan multiple sensor streams
   - Membangun early warning system untuk predictive maintenance

📋 LANGKAH-LANGKAH:

STEP 1: Generate Multi-Sensor Dataset
─────────────────────────────────────
   Konteks: Motor bearing monitoring dengan 3 sensors:
   
   a) Sensors:
      - Accelerometer (vibration)
      - Thermocouple (temperature)
      - Current sensor (motor current)
      
   b) Normal operation (80%):
      - Vibration: white noise + 60Hz hum (minimal)
      - Temperature: 40°C ± 2°C (slow drift)
      - Current: sinusoidal 50Hz, 5A RMS
      
   c) Degrading (15%):
      - Vibration: increased BPFO frequency content
      - Temperature: gradual increase
      - Current: slight imbalance
      
   d) Failing (5%):
      - Vibration: high amplitude, impulsive
      - Temperature: > 70°C
      - Current: significant imbalance, harmonics
      
   e) Label: 0=normal, 1=degrading, 2=failing
   
   Dataset: 5000 windows, masing-masing 200 timesteps x 3 sensors


STEP 2: Preprocessing
─────────────────────
   a) Normalisasi per sensor:
      - Z-score: (x - mean) / std per sensor
      - Simpan statistics untuk inference
      
   b) Windowing:
      - Sequence length: 200 timesteps
      - Overlap: 50% (stride = 100)
      
   c) Data augmentation:
      - Add sensor noise (SNR 20-40 dB)
      - Time shift (±10 timesteps)
      - Random sensor dropout (simulate sensor failure)
      
   d) Class balancing:
      - Oversample failing class
      - Atau: weighted loss


STEP 3: Transformer Architecture
────────────────────────────────
   a) Input embedding:
      - Project 3 sensors ke d_model dimension
      - Add positional encoding
      - Add sensor type embedding (opsional)
      
   b) Transformer encoder:
      - Multi-head self-attention
      - Feed-forward network
      - Layer normalization
      - Dropout
      
   c) Classification head:
      - Global average pooling over time
      - FC layers
      - Softmax output
      
   d) Regularization:
      - Dropout 0.1-0.3
      - Label smoothing
      - Weight decay


STEP 4: Training Strategy
─────────────────────────
   a) Loss function:
      - Weighted cross-entropy (karena class imbalance)
      - Focal loss (fokus pada hard examples)
      
   b) Optimization:
      - AdamW dengan cosine annealing
      - Warmup epochs (5-10)
      - Gradient clipping (max_norm=1.0)
      
   c) Validation:
      - Early stopping (patience 15)
      - Save best model
      - Monitor per-class metrics


STEP 5: Evaluation & Deployment
───────────────────────────────
   a) Metrics:
      - Overall accuracy
      - Precision/recall/F1 per class
      - Confusion matrix
      - ROC-AUC per class
      
   b) Attention analysis:
      - Visualisasi attention weights
      - Identifikasi: sensor mana yang paling penting?
      - Identifikasi: timestep mana yang critical?
      
   c) Early warning system:
      - Confidence threshold untuk alert
      - False alarm rate vs detection rate
      - Lead time: berapa lama sebelum failure?
      
   d) Deployment considerations:
      - Model size (inference speed)
      - Input latency (real-time vs batch)
      - Sensor failure handling


💡 HINTS:
   - Bearing fault frequencies:
     BPFO = (n/2)*fr*(1-d/D*cos(β))
     BPFI = (n/2)*fr*(1+d/D*cos(β))
     BSF = (D/2d)*fr*(1-(d/D*cos(β))²)
   - Weighted loss: weights = 1/class_frequency
   - Focal loss: -α*(1-p)^γ * log(p)
   - Label smoothing: target = 0.9 (bukan 1.0)
   - Gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

⚠️ COMMON MISTAKES:
   - Data leakage: normalize sebelum split
   - Class imbalance tanpa weighting
   - Attention weights tidak di-visualisasi
   - Model terlalu besar untuk real-time inference
   - Tidak handle missing sensor data

🎯 EXPECTED OUTPUT:
   - Multi-sensor transformer classifier
   - Accuracy > 90% per class
   - Attention analysis yang informatif
   - Early warning system dengan lead time > 30 detik
   - Deployment-ready model (TorchScript atau ONNX)

Ini adalah aplikasi end-to-end deep learning untuk
predictive maintenance — skill yang sangat dicari di industri!
"""

print("\n" + "="*50)
print("🎉 FASE 4 SELESAI!")
print("="*50)
print("""
Kamu sekarang bisa:
✅ Neural network dari scratch (NumPy)
✅ PyTorch fundamentals (tensors, autograd, nn.Module)
✅ CNN untuk images dan signals
✅ RNN/LSTM/GRU/Transformer untuk time series

Sebelum lanjut:
1. Selesaikan Project 3: Computer Vision
2. Selesaikan semua exercise dan challenge

Lanjut ke: 05-advanced/01_transfer_learning.py
""")
