"""
=============================================================
FASE 5 — MODUL 2: NLP & TRANSFORMERS
=============================================================
Transformers = arsitektur yang MEREVOLUSI AI.
GPT, BERT, ChatGPT — semua berbasis Transformer.

Kenapa penting untuk ML engineer?
- Dominan di NLP (text processing)
- Sekarang juga dipakai di: vision, audio, time series
- Understanding transformers = understanding modern AI

Koneksi EE:
- Self-attention = adaptive matched filter
- Positional encoding = phase encoding (mirip OFDM!)
- Multi-head attention = bank of parallel filters

Durasi target: 5-6 jam (topik paling padat)
=============================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===========================================================
# 📖 BAGIAN 1: Self-Attention dari Nol
# ===========================================================
# Attention: "query" asks "what should I focus on?"
# Key-Value pairs provide the context
#
# Attention(Q, K, V) = softmax(QK^T / √d_k) * V

class SelfAttention(nn.Module):
    """Self-Attention mechanism from scratch"""
    def __init__(self, embed_size, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, embed_size)
        Q = self.query(x)  # (batch, seq_len, head_size)
        K = self.key(x)
        V = self.value(x)

        # Attention scores
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_size)
        # scores: (batch, seq_len, seq_len)

        # Softmax → attention weights
        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        output = attn_weights @ V
        return output, attn_weights


# Demo
embed_size = 64
seq_len = 10
batch_size = 1

x = torch.randn(batch_size, seq_len, embed_size)
attention = SelfAttention(embed_size, head_size=32)
output, weights = attention(x)

print("=== Self-Attention ===")
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Attention weights sum per query: {weights.sum(dim=-1)}")

# Visualize attention
plt.figure(figsize=(8, 6))
plt.imshow(weights[0].detach().numpy(), cmap='viridis')
plt.colorbar()
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.title('Self-Attention Weights')
plt.savefig('01_attention_weights.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 01_attention_weights.png")


# ===========================================================
# 📖 BAGIAN 2: Multi-Head Attention
# ===========================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention — multiple attention heads in parallel"""
    def __init__(self, embed_size, num_heads):
        super().__init__()
        assert embed_size % num_heads == 0
        self.head_size = embed_size // num_heads
        self.num_heads = num_heads

        self.heads = nn.ModuleList([
            SelfAttention(embed_size, self.head_size)
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        head_outputs = []
        all_weights = []
        for head in self.heads:
            out, weights = head(x)
            head_outputs.append(out)
            all_weights.append(weights)

        # Concatenate heads
        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.projection(concatenated)
        return output, all_weights


# ===========================================================
# 📖 BAGIAN 3: Transformer Block
# ===========================================================

class TransformerBlock(nn.Module):
    """Single Transformer encoder block"""
    def __init__(self, embed_size, num_heads, ff_size, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x)
        x = self.norm1(x + attn_out)  # Residual + LayerNorm

        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)  # Residual + LayerNorm

        return x


# ===========================================================
# 📖 BAGIAN 4: Positional Encoding
# ===========================================================
# Transformer tidak punya konsep "urutan" (semua posisi equal)
# → Positional encoding menambahkan informasi posisi
# → Menggunakan sin/cos (familiar dari signal processing!)

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding — Fourier-like!"""
    def __init__(self, embed_size, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float()
                             * (-math.log(10000.0) / embed_size))

        pe[:, 0::2] = torch.sin(position * div_term)  # sin untuk posisi genap
        pe[:, 1::2] = torch.cos(position * div_term)  # cos untuk posisi ganjil

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# Visualize positional encoding
pe = PositionalEncoding(64)
encoding = pe.pe[0, :50, :].numpy()

plt.figure(figsize=(12, 5))
plt.imshow(encoding.T, aspect='auto', cmap='RdBu_r')
plt.colorbar()
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Positional Encoding (like Fourier basis functions!)')
plt.savefig('02_positional_encoding.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 02_positional_encoding.png")
print("→ Each position has a unique 'frequency signature' — Fourier decomposition!")


# ===========================================================
# 📖 BAGIAN 5: Complete Transformer for Classification
# ===========================================================

class TransformerClassifier(nn.Module):
    """Complete Transformer for sequence classification"""
    def __init__(self, vocab_size, embed_size, num_heads, num_layers,
                 num_classes, max_len, ff_size=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, max_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_size, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len) of token indices
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        # Global average pooling over sequence
        x = x.mean(dim=1)

        return self.classifier(x)


# Quick test
model = TransformerClassifier(
    vocab_size=1000, embed_size=64, num_heads=4,
    num_layers=2, num_classes=5, max_len=100
)
test_input = torch.randint(0, 1000, (4, 50))  # batch=4, seq_len=50
output = model(test_input)
print(f"\n=== Transformer Classifier ===")
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ===========================================================
# 📖 BAGIAN 6: Menggunakan Hugging Face (Production Way)
# ===========================================================

print("""
\n=== Hugging Face Transformers ===
Untuk production, gunakan Hugging Face:

pip install transformers datasets

# Sentiment Analysis
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love machine learning!")

# Text Classification dengan fine-tuning
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Fine-tuning with Trainer API
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(output_dir="./results", num_train_epochs=3)
trainer = Trainer(model=model, args=training_args, train_dataset=train_data)
trainer.train()
""")


# ===========================================================
# 🏋️ EXERCISE 16: Transformer Experiments
# ===========================================================
"""
1. Text Classification:
   - Download movie review dataset (atau gunakan dari Hugging Face)
   - Train Transformer classifier dari scratch
   - Fine-tune BERT untuk task yang sama
   - Bandingkan: accuracy, training time, model size

2. Visualize Attention:
   - Untuk input kalimat, visualize attention weights
   - Identifikasi: kata mana yang "memperhatikan" kata mana?
   - Analisis: apakah attention weights masuk akal?

3. Transformer untuk Time Series:
   - Adapt Transformer classifier di atas untuk time series prediction
   - Replace embedding layer dengan linear projection
   - Bandingkan dengan LSTM dari fase sebelumnya
"""


# ===========================================================
# 🔥 CHALLENGE: Transformer untuk Signal Processing
# ===========================================================
"""
Gabungkan expertise EE + state-of-the-art ML:

1. Buat Transformer untuk signal classification:
   - Input: raw signal (1D time series)
   - Patch embedding (bukan word embedding): split signal into patches
   - Positional encoding
   - Transformer encoder
   - Classification head

2. Bandingkan dengan:
   a. 1D CNN (dari fase sebelumnya)
   b. LSTM
   c. CNN + Transformer (hybrid)

3. Analisis attention weights:
   - Apakah Transformer memperhatikan bagian sinyal yang "tepat"?
   - Misalnya: untuk fault detection, apakah attention fokus
     di bagian sinyal yang mengandung fault signature?

4. Ini disebut "Vision Transformer for 1D" — cutting-edge research!
"""

print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 05-advanced/03_generative_models.py")
print("="*50)
