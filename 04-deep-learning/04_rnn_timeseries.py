"""
=============================================================
FASE 4 — MODUL 4: RNN & TIME SERIES
=============================================================
RNN (Recurrent Neural Network) = network dengan "memori".

Perfect untuk data sekuensial:
- Time series (sensor data, stock price)
- Text (NLP)
- Audio

Koneksi Teknik Elektro:
- RNN = IIR filter (output depends on previous output)
- LSTM gates = adaptive control mechanism
- Attention = frequency-selective filter

Durasi target: 4-5 jam
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===========================================================
# 📖 BAGIAN 1: RNN Basics
# ===========================================================
# RNN equation:
#   h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
#   y_t = W_hy * h_t
#
# Masalah: vanishing gradient untuk sequence panjang
# Solusi: LSTM (Long Short-Term Memory) atau GRU

# ===========================================================
# 📖 BAGIAN 2: LSTM untuk Time Series Prediction
# ===========================================================

class LSTMPredictor(nn.Module):
    """LSTM untuk time series forecasting"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Ambil output dari timestep terakhir
        out = self.fc(lstm_out[:, -1, :])
        return out


# ===========================================================
# 📖 BAGIAN 3: Synthetic Time Series Data
# ===========================================================

def generate_sensor_data(n_points=5000, noise_level=0.1):
    """
    Generate realistic sensor data:
    - Sinusoidal base (50Hz power)
    - Daily pattern (amplitude varies)
    - Slow degradation trend
    - Random noise
    """
    t = np.arange(n_points) / 100  # 100 Hz sampling

    # Multi-component signal
    base = np.sin(2 * np.pi * 0.5 * t)                   # slow oscillation
    daily = 0.3 * np.sin(2 * np.pi * t / 24)              # daily pattern
    trend = 0.001 * t                                       # degradation
    noise = noise_level * np.random.randn(n_points)

    signal = base + daily + trend + noise
    return signal, t


def create_sequences(data, seq_length, pred_length=1):
    """Create input sequences and targets for time series"""
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + pred_length])
    return np.array(X), np.array(y)


# Generate data
signal, t = generate_sensor_data(5000)

# Normalize
signal_mean = signal[:4000].mean()
signal_std = signal[:4000].std()
signal_norm = (signal - signal_mean) / signal_std

# Create sequences
SEQ_LENGTH = 50
PRED_LENGTH = 10
X, y = create_sequences(signal_norm, SEQ_LENGTH, PRED_LENGTH)

# Train/test split (TIME SERIES: NO RANDOM SPLIT!)
train_size = int(len(X) * 0.8)
X_train = torch.FloatTensor(X[:train_size]).unsqueeze(-1).to(device)
y_train = torch.FloatTensor(y[:train_size]).to(device)
X_test = torch.FloatTensor(X[train_size:]).unsqueeze(-1).to(device)
y_test = torch.FloatTensor(y[train_size:]).to(device)

print(f"=== Time Series Dataset ===")
print(f"Sequence length: {SEQ_LENGTH}")
print(f"Prediction length: {PRED_LENGTH}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ===========================================================
# 📖 BAGIAN 4: Training LSTM
# ===========================================================

model = LSTMPredictor(
    input_size=1,
    hidden_size=64,
    num_layers=2,
    output_size=PRED_LENGTH
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

print(f"\n=== Training LSTM ===")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

train_losses = []
test_losses = []

for epoch in range(50):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test).item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    test_losses.append(test_loss)
    scheduler.step(test_loss)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, "
              f"test_loss={test_loss:.6f}")


# ===========================================================
# 📖 BAGIAN 5: Evaluation & Visualization
# ===========================================================

model.eval()
with torch.no_grad():
    predictions = model(X_test).cpu().numpy()
    actuals = y_test.cpu().numpy()

# Denormalize
predictions = predictions * signal_std + signal_mean
actuals = actuals * signal_std + signal_mean

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training curves
axes[0, 0].plot(train_losses, label='Train')
axes[0, 0].plot(test_losses, label='Test')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MSE Loss')
axes[0, 0].set_title('Learning Curves')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Actual vs Predicted (first prediction step)
n_show = 200
axes[0, 1].plot(actuals[:n_show, 0], label='Actual', alpha=0.7)
axes[0, 1].plot(predictions[:n_show, 0], label='Predicted', alpha=0.7)
axes[0, 1].set_title('Actual vs Predicted (1-step ahead)')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Multi-step prediction example
sample_idx = 50
axes[1, 0].plot(range(PRED_LENGTH), actuals[sample_idx], 'bo-', label='Actual')
axes[1, 0].plot(range(PRED_LENGTH), predictions[sample_idx], 'rx-', label='Predicted')
axes[1, 0].set_title(f'Multi-step Forecast (sample {sample_idx})')
axes[1, 0].set_xlabel('Steps Ahead')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Error distribution
errors = predictions[:, 0] - actuals[:, 0]
axes[1, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].set_title(f'Prediction Error Distribution\nMean={errors.mean():.4f}, Std={errors.std():.4f}')
axes[1, 1].set_xlabel('Error')

plt.tight_layout()
plt.savefig('04_lstm_results.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 04_lstm_results.png")


# ===========================================================
# 📖 BAGIAN 6: GRU & Model Comparison
# ===========================================================

class GRUPredictor(nn.Module):
    """GRU — lebih simple dari LSTM, sering performance comparable"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])


class TransformerPredictor(nn.Module):
    """Transformer for time series — state-of-the-art"""
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, seq_length):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_length, d_model) * 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=d_model*4, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


print("\n=== Model Comparison ===")
models_config = {
    'LSTM': LSTMPredictor(1, 64, 2, PRED_LENGTH),
    'GRU': GRUPredictor(1, 64, 2, PRED_LENGTH),
    'Transformer': TransformerPredictor(1, 64, 4, 2, PRED_LENGTH, SEQ_LENGTH),
}

for name, mdl in models_config.items():
    mdl = mdl.to(device)
    n_params = sum(p.numel() for p in mdl.parameters())
    print(f"  {name}: {n_params:,} parameters")


# ===========================================================
# 🏋️ EXERCISE 14: RNN Experiments
# ===========================================================
"""
1. Train dan bandingkan LSTM, GRU, Transformer pada dataset yang sama
   - Metric: MSE, MAE, training time
   - Plot: actual vs predicted untuk ketiga model

2. Sequence-to-Sequence prediction:
   - Input: 100 timesteps
   - Output: 50 timesteps (bukan cuma 10)
   - Gunakan Teacher Forcing (input target di saat training)

3. Encoder-Decoder architecture:
   - Encoder LSTM membaca input sequence
   - Decoder LSTM generates output sequence
   - Ini basis dari banyak model NLP juga!

4. Attention mechanism (manual):
   - Implementasi Bahdanau attention
   - Visualisasi attention weights (timestep mana yang paling penting?)
"""


# ===========================================================
# 🔥 CHALLENGE: Predictive Maintenance System
# ===========================================================
"""
Buat sistem prediksi kegagalan mesin listrik:

1. Generate realistic sensor data (3000 timesteps per sample):
   - Normal operation → gradual degradation → failure
   - Multiple sensors: vibration, temperature, current, voltage
   - Remaining Useful Life (RUL) sebagai target

2. Feature extraction (time + frequency domain)

3. Model comparison:
   a. Classical: extract features → Random Forest
   b. 1D CNN: raw signal → CNN
   c. LSTM: sliding window → LSTM
   d. Hybrid: CNN features → LSTM

4. Evaluate:
   - RMSE dari RUL prediction
   - Confusion matrix untuk early warning (RUL < threshold)
   - Cost analysis: false alarm cost vs missed failure cost

5. Deploy consideration:
   - Real-time inference latency
   - Model size
   - Interpretability

Ini adalah use case NYATA yang sangat dicari di industri!
Simpan di projects/project_03_computer_vision/ (atau buat folder baru)
"""

print("\n" + "="*50)
print("🎉 FASE 4 SELESAI!")
print("="*50)
print("""
Kamu sekarang bisa:
✅ Bangun neural network dari nol (forward + backprop)
✅ Gunakan PyTorch untuk training models
✅ CNN untuk image dan signal classification
✅ RNN/LSTM untuk time series prediction

Sebelum lanjut:
1. Selesaikan Project 3 (Computer Vision atau Predictive Maintenance)
2. Pastikan semua exercise selesai
3. Bisa jelaskan backpropagation tanpa melihat kode

Lanjut ke: 05-advanced/01_transfer_learning.py
""")
