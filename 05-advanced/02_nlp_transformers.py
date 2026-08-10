"""
=============================================================
FASE 5 - MODUL 2: NLP DAN TRANSFORMERS
=============================================================
Natural Language Processing (NLP) = memahami dan memproses bahasa manusia.

Transformer (2017, "Attention Is All You Need") merevolusi NLP:
- Self-attention: setiap token "melihat" semua token lain
- Parallel processing: lebih cepat dari RNN
- State-of-the-art di hampir semua NLP tasks

Koneksi Teknik Elektro:
- Self-attention = cross-correlation antara tokens
- Positional encoding = frequency-domain time representation
- Multi-head = parallel filter banks
- Transformer = matched filter untuk semantic patterns

Durasi target: 4-5 jam
============================================================="""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)


# ===========================================================
# BAGIAN 1: Self-Attention from Scratch
# ===========================================================
# Self-attention adalah mekanisme inti dari Transformer.
#
# KONSEP DASAR:
# - Setiap token dalam sequence menghasilkan tiga vektor: Query (Q), Key (K), Value (V).
# - Attention weight antara token i dan j dihitung sebagai similarity(Q_i, K_j).
# - Output untuk token i adalah weighted sum dari semua Value, dengan weights dari attention.
#
# KENAPA SCALING DENGAN sqrt(d_k)?
# - Ketika d_k besar, dot product QK^T bisa menjadi sangat besar.
# - Softmax dari nilai yang sangat besar menghasilkan distribusi yang sangat "sharp"
#   (hampir one-hot), yang membuat gradient kecil (vanishing gradients).
# - Scaling dengan sqrt(d_k) menjaga variance dari dot product tetap stabil.
#
# KONEKSI TEKNIK ELEKTRO:
# - QK^T = cross-correlation antara query dan key (seperti matched filter)
# - Softmax = normalization untuk probability distribution
# - Weighted sum V = output filter dengan adaptive weights
# - Attention map menunjukkan "konektivitas" antar token, mirip dengan
#   adjacency matrix di graph theory.

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention.
    
    Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Parameters:
    -----------
    Q : torch.Tensor, shape (batch, n_q, d_k)
        Query matrix.
    K : torch.Tensor, shape (batch, n_k, d_k)
        Key matrix.
    V : torch.Tensor, shape (batch, n_v, d_v)
        Value matrix. n_v = n_k.
    mask : torch.Tensor, optional
        Mask untuk mencegah attention ke future tokens (decoder).
        
    Returns:
    --------
    torch.Tensor
        Output attention.
    torch.Tensor
        Attention weights.
        
    Notes:
    ------
    - Q, K, V berasal dari input yang sama (self-attention)
    - Scaling dengan sqrt(d_k) mencegah softmax dari terlalu sharp
    - Mask untuk causal attention (decoder auto-regressive)
    - Complexity: O(n^2 * d) untuk sequence length n
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


# Demo attention
seq_len = 5
d_k = 4
Q = torch.randn(1, seq_len, d_k)
K = torch.randn(1, seq_len, d_k)
V = torch.randn(1, seq_len, d_k)

output, weights = scaled_dot_product_attention(Q, K, V)
print("=== Self-Attention Demo ===")
print(f"Input shape: {Q.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Attention weights:\n{weights[0]}")

# Visualisasi attention weights
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(weights[0].detach().numpy(), cmap='Blues')
ax.set_xticks(range(seq_len))
ax.set_yticks(range(seq_len))
ax.set_xlabel('Key Position')
ax.set_ylabel('Query Position')
ax.set_title('Self-Attention Weights')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('01_attention_weights.png', dpi=100, bbox_inches='tight')
plt.close()
print("PLOT Saved: 01_attention_weights.png")


# ===========================================================
# BAGIAN 2: Multi-Head Attention
# ===========================================================
# Multi-head attention = menjalankan multiple attention heads secara parallel.
#
# KENAPA MULTI-HEAD?
# - Satu attention head hanya bisa fokus pada satu jenis relasi.
# - Multiple heads memungkinkan model belajar berbagai jenis relasi
#   secara simultan (sintaksis, semantik, coreferensi, dll).
# - Ini mirip dengan multiple filter banks di signal processing,
#   di mana setiap filter menangkap frekuensi yang berbeda.
#
# IMPLEMENTASI:
# - Input diproyeksikan ke Q, K, V menggunakan linear layers.
# - Q, K, V di-split ke h heads, masing-masing dengan dimensi d_model/h.
# - Attention dihitung secara parallel untuk setiap head.
# - Hasil dari semua head di-concatenate dan diproyeksikan kembali.

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.
    
    Multiple attention heads berjalan secara parallel,
    masing-masing dengan representasi yang berbeda.
    
    Parameters:
    -----------
    d_model : int
        Model dimension.
    n_heads : int
        Jumlah attention heads.
        
    Notes:
    ------
    - d_k = d_v = d_model / n_heads
    - Setiap head = independent attention mechanism
    - Concatenate outputs dari semua heads
    - Linear projection untuk combine
    - n_heads harus membagi d_model agar split rata.
    
    Koneksi Teknik Elektro:
    - Multi-head = parallel filter banks
    - Setiap head = different "frequency response"
    - Concatenation = combining filter outputs
    """
    
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(attn_output)


# ===========================================================
# BAGIAN 3: Transformer Encoder Block
# ===========================================================
# Transformer encoder block = building block utama dari BERT dan model sejenis.
#
# ARSITEKTUR:
# Input -> Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm -> Output
#
# RESIDUAL CONNECTIONS (Add):
# - Menambahkan input ke output dari sub-layer: x + Sublayer(x).
# - Membantu gradient flow saat training deep networks.
# - Ini adalah teknik kritis yang memungkinkan Transformer bisa sangat dalam
#   (BERT-base punya 12 layer, BERT-large 24 layer).
#
# LAYER NORMALIZATION (Norm):
# - Normalisasi dilakukan PER SAMPLE, per layer.
# - Berbeda dengan BatchNorm yang normalisasi per batch.
# - LayerNorm lebih stabil untuk sequence data dengan variable length.
# - Formula: LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
#
# FEED-FORWARD NETWORK:
# - Dua linear layers dengan ReLU di tengah.
# - Hidden dimension biasanya 4x dari d_model (misal: d_model=512, d_ff=2048).
# - Ini menambahkan non-linearitas yang tidak bisa diberikan oleh attention alone.

class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block.
    
    Architecture:
    Input -> Multi-Head Attention -> Add & Norm
          -> Feed Forward -> Add & Norm -> Output
    
    Parameters:
    -----------
    d_model : int
        Model dimension.
    n_heads : int
        Jumlah attention heads.
    d_ff : int
        Feed-forward hidden dimension.
    dropout : float
        Dropout rate.
        
    Notes:
    ------
    - Add & Norm = residual connection + layer normalization
    - Feed Forward = 2 FC layers dengan ReLU
    - LayerNorm lebih stabil dari BatchNorm untuk sequence
    - Residual connections memungkinkan training deep networks
    - Dropout = regularization untuk mencegah overfitting
    
    Koneksi Teknik Elektro:
    - Residual = feedback path (seperti control systems)
    - LayerNorm = normalization per sample (unlike BatchNorm)
    - Feed Forward = nonlinear transformation (seperti amplifier dengan saturation)
    """
    
    def __init__(self, d_model=64, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-head attention with residual
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


# ===========================================================
# BAGIAN 4: Complete Transformer untuk Text Classification
# ===========================================================
# Text Transformer = menggabungkan embedding, positional encoding,
# multiple encoder blocks, dan classifier head.
#
# EMBEDDING LAYER:
# - Mengubah token IDs (integer) menjadi dense vectors.
# - Vocabulary size biasanya 10,000 - 50,000 untuk model kecil.
# - Embedding dimension = d_model (biasanya 128 - 1024).
#
# POSITIONAL ENCODING:
# - Attention bersifat permutation-invariant: urutan token tidak penting.
# - Tapi urutan sangat penting dalam bahasa!
# - Positional encoding menambahkan informasi posisi ke embedding.
# - Formula original Transformer menggunakan sinusoidal functions:
#   PE[pos, 2i] = sin(pos / 10000^(2i/d_model))
#   PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
# - Keuntungan sinusoidal: bisa extrapolate ke sequence yang lebih panjang.
# - Alternatif: learned positional embeddings (dipakai oleh BERT).
#
# GLOBAL AVERAGE POOLING:
# - Setelah encoder, kita punya sequence of vectors.
# - Untuk classification, perlu "meringkas" sequence jadi satu vektor.
# - Global average pooling = rata-rata semua token representations.
# - Alternatif: gunakan [CLS] token seperti BERT.

class TextTransformer(nn.Module):
    """
    Transformer untuk text classification.
    
    Architecture:
    Input -> Embedding + Positional Encoding
          -> Transformer Encoder (N layers)
          -> Global Average Pooling
          -> FC -> Output
    
    Parameters:
    -----------
    vocab_size : int
        Ukuran vocabulary.
    d_model : int
        Embedding dimension.
    n_heads : int
        Jumlah attention heads.
    n_layers : int
        Jumlah encoder layers.
    num_classes : int
        Jumlah output classes.
    max_len : int
        Maximum sequence length.
        
    Notes:
    ------
    - Text classification = sentiment analysis, topic classification, etc.
    - Global pooling = aggregate sequence information
    - Bisa juga menggunakan [CLS] token (seperti BERT)
    - Semakin banyak layers dan d_model, semakin powerful modelnya,
      tapi juga semakin banyak parameter dan training time.
    
    Koneksi Teknik Elektro:
    - Embedding = dictionary lookup (seperti lookup table di FPGA)
    - Positional encoding = time index carrier (seperti carrier signal di modulation)
    - Attention = adaptive filter yang fokus pada keywords
    """
    
    def __init__(self, vocab_size=10000, d_model=128, n_heads=4,
                 n_layers=2, num_classes=2, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(max_len, d_model)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x):
        # x: (batch, seq_len)
        seq_len = x.size(1)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        return self.classifier(x)


# ===========================================================
# BAGIAN 5: Using Pre-trained Transformers (HuggingFace)
# ===========================================================
"""
HuggingFace Transformers library menyediakan pre-trained models
yang bisa langsung digunakan atau di-fine-tune.

MODEL POPULER:
- BERT (Bidirectional Encoder Representations from Transformers):
  * Pre-trained dengan masked language modeling dan next sentence prediction.
  * Bidirectional: melihat context dari kiri dan kanan.
  * Great untuk: classification, NER, question answering.
  * Ukuran: BERT-base (110M params), BERT-large (340M params).

- GPT (Generative Pre-trained Transformer):
  * Autoregressive: hanya melihat tokens sebelumnya.
  * Great untuk: text generation, completion.
  * GPT-3: 175B params, GPT-4: estimated > 1T params.

- RoBERTa: Robustly optimized BERT approach.
  * Hanya pre-training dengan masked language modeling.
  * Biasanya lebih baik dari BERT original.

- DistilBERT: distilled version dari BERT.
  * 40% lebih kecil, 60% lebih cepat, 97% performance BERT.
  * Great untuk deployment dengan resource terbatas.

- T5 (Text-to-Text Transfer Transformer):
  * Semua NLP task diformat sebagai text-to-text.
  * Unified framework untuk translation, summarization, QA, dll.

KEUNTUNGAN MENGGUNAKAN PRE-TRAINED TRANSFORMERS:
- Sudah belajar representasi bahasa yang kaya dari corpus besar.
- Fine-tuning hanya membutuhkan data labeled yang sedikit.
- Implementasi dan weights sudah tersedia, tinggal download.

TRADEOFF:
- Model besar membutuhkan GPU dengan memory besar.
- Inference bisa lambat untuk real-time applications.
- Perlu handling untuk sequence length yang panjang.

Contoh penggunaan (perlu install transformers):

from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained tokenizer dan model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)

# Tokenize text
text = "This movie is amazing!"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Forward pass
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
"""


# ===========================================================
# LATIHAN 16: Transformer Implementation
# ===========================================================
"""
TARGET Learning Objectives:
   - Mengimplementasikan transformer encoder dari scratch
   - Memahami attention patterns
   - Mengaplikasikan ke text classification task

PANDUAN LANGKAH-LANGKAH:

STEP 1: Complete Self-Attention Implementation
----------------------------------------------
   a) Implementasi dari scratch (tanpa nn.MultiheadAttention):
      - Q, K, V projections
      - Scaled dot-product
      - Multi-head split dan concat
      - Output projection
      
   b) Verifikasi:
      - Bandingkan dengan PyTorch nn.MultiheadAttention
      - Output harus sama (dengan toleransi numerical)
      
   c) Analisis complexity:
      - Time: O(n^2 * d) untuk sequence length n
      - Space: O(n^2) untuk attention matrix
      - Bandingkan dengan RNN: O(n * d^2)
      - Transformer lebih cepat di GPU karena parallel,
        tapi memory usage lebih tinggi.


STEP 2: Causal (Masked) Self-Attention
--------------------------------------
   a) Implementasi untuk decoder (autoregressive):
      - Mask: upper-triangular matrix (prevent looking at future)
      - Query position i hanya bisa attend ke positions <= i
      
   b) Verifikasi:
      - Attention weights untuk position i:
        weights[i, j] = 0 untuk j > i
      - Sum weights[i, :i+1] ~ 1.0
      
   c) Analisis:
      - Kenapa causal attention penting untuk generation?
      - Bandingkan dengan bidirectional (encoder) attention


STEP 3: Positional Encoding Analysis
------------------------------------
   a) Visualisasi positional encoding matrix:
      - Plot PE[pos, dim] sebagai heatmap
      - Identifikasi: pattern sinusoidal
      
   b) Properties:
      - Uniqueness: setiap posisi punya encoding unik
      - Relative: PE[pos+k] bisa diekspresikan dari PE[pos]
      - Bounded: values dalam [-1, 1]
      
   c) Alternatives:
      - Learned positional embedding (dipelajari bersama model)
      - Rotary positional embedding (RoPE)
      - ALiBi (Attention with Linear Biases)
      - Bandingkan: mana yang lebih baik untuk sequence panjang?


STEP 4: Text Classification dengan Transformer
----------------------------------------------
   Dataset: IMDB sentiment analysis (atau synthetic)
   
   a) Preprocessing:
      - Tokenization (word-level atau subword)
      - Padding/truncation ke fixed length
      - Vocabulary building
      
   b) Model:
      - Embedding layer
      - Positional encoding
      - 2-4 transformer encoder layers
      - Global pooling
      - Classification head
      
   c) Training:
      - Cross-entropy loss
      - Adam optimizer
      - Learning rate warmup (sangat penting untuk Transformer!)
      - LR warmup: mulai dari 0, naik linear ke lr maksimum
        selama beberapa steps awal, lalu decay.
      
   d) Evaluation:
      - Accuracy, precision, recall
      - Attention visualization untuk interpretasi
      - Identifikasi: kata-kata mana yang paling di-attend?


TIPS:
   - Attention mask: torch.triu(torch.ones(n, n), diagonal=1) == 0
   - Positional encoding: PE[pos, 2i] = sin(pos/10000^(2i/d))
   - Tokenization: bisa pakai simple word splitting atau torchtext
   - Attention visualization: plt.imshow(attention_weights[0].detach())

PERINGATAN COMMON MISTAKES:
   - Attention tanpa scaling -> softmax terlalu sharp
   - Lupa positional encoding -> model tidak tahu urutan
   - Mask yang salah -> information leakage (causal)
   - Padding token di-attend -> harus mask padding positions
   - Learning rate terlalu besar -> training instabil

TARGET EXPECTED OUTPUT:
   - Self-attention implementation matching PyTorch
   - Causal attention dengan mask yang benar
   - Positional encoding visualization
   - Text classifier dengan accuracy > 80% pada IMDB
   - Attention heatmap yang interpretable
"""


# ===========================================================
# 🔥 CHALLENGE: Transformer untuk Fault Report Classification
# ===========================================================
"""
TARGET Learning Objectives:
   - Mengaplikasikan NLP ke industrial domain
   - Membangun information extraction pipeline
   - Menggabungkan structured dan unstructured data

PANDUAN LANGKAH-LANGKAH:

STEP 1: Dataset - Maintenance Log Classification
------------------------------------------------
   Konteks: Klasifikasi maintenance reports untuk predictive maintenance
   
   a) Data sources:
      - Maintenance logs: text descriptions of faults
      - Sensor readings: structured numerical data
      - Technician notes: free-form text
      
   b) Labels:
      - 0: No action needed
      - 1: Scheduled maintenance
      - 2: Urgent repair
      - 3: Critical failure
      
   c) Generate synthetic dataset (500 samples):
      Example: "Motor vibration increased by 30% over past week.
                Temperature reading 65 degC. Recommend inspection."
      -> Label: 2 (Urgent repair)


STEP 2: Preprocessing Pipeline
------------------------------
   a) Text preprocessing:
      - Lowercasing, punctuation removal
      - Tokenization (word or subword)
      - Stopword removal (opsional)
      - Stemming/Lemmatization (opsional)
      
   b) Numerical features:
      - Extract numbers from text (temperature, vibration, etc.)
      - Normalize ke [0, 1]
      
   c) Combined representation:
      - Text -> Transformer encoder
      - Numbers -> FC layer
      - Concatenate -> classifier


STEP 3: Model Architecture
--------------------------
   a) Text branch:
      - Pre-trained BERT (distilbert-base-uncased)
      - Fine-tune pada domain-specific data
      - Extract [CLS] token representation
      
   b) Numerical branch:
      - FC layers untuk sensor readings
      - BatchNorm untuk stabilitas
      
   c) Fusion:
      - Concatenate text dan numerical embeddings
      - Joint classification dengan attention
      
   d) Multi-task (opsional):
      - Task 1: Severity classification
      - Task 2: Component extraction (NER)
      - Task 3: Recommended action generation


STEP 4: Training & Evaluation
-----------------------------
   a) Training:
      - Transfer learning dari DistilBERT
      - Fine-tune dengan small learning rate (2e-5)
      - Class weights untuk imbalance
      
   b) Evaluation:
      - Classification metrics (accuracy, F1)
      - Confusion matrix per severity level
      - Error analysis: jenis report yang sering misclassified
      
   c) Interpretability:
      - Attention visualization: kata-kata penting
      - LIME/SHAP untuk local explanations
      - Confident vs uncertain predictions


STEP 5: Deployment Pipeline
---------------------------
   a) Real-time inference:
      - API endpoint untuk report classification
      - Response time < 500ms
      - Batch processing untuk historical data
      
   b) Alert system:
      - Critical failures -> immediate alert
      - Urgent repairs -> daily digest
      - Scheduled maintenance -> weekly report
      
   c) Feedback loop:
      - Collect technician feedback
      - Retrain model periodically
      - Track model drift


TIPS:
   - DistilBERT = 40% smaller, 60% faster, 97% of BERT performance
   - HuggingFace: AutoTokenizer, AutoModelForSequenceClassification
   - NER: BioBERT atau general BERT dengan custom labels
   - Fusion: weighted combination dari modalities
   - Response time: quantization (INT8) untuk inference cepat

PERINGATAN COMMON MISTAKES:
   - Fine-tune BERT dengan learning rate terlalu besar
   - Tidak handle out-of-vocabulary words
   - Numerical values sebagai text ("65 degC" -> tokenizer pecah)
   - Class imbalance tanpa weighting
   - Model terlalu lambat untuk real-time

TARGET EXPECTED OUTPUT:
   - Maintenance report classifier dengan F1 > 0.85
   - Attention visualization yang menunjukkan keywords
   - Fusion model yang menggabungkan text dan numerical
   - Deployment pipeline dengan response time < 500ms
   - Interpretable predictions dengan explanations

Ini adalah aplikasi NLP yang sangat valuable untuk
industrial operations dan predictive maintenance!
"""

print("\n" + "="*50)
print("OK Modul selesai! Lanjut ke: 05-advanced/03_generative_models.py")
print("="*50)
