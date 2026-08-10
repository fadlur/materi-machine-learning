"""
=============================================================
FASE 5 - MODUL 3: GENERATIVE MODELS
=============================================================
Generative models = model yang belajar distribusi data dan bisa
generate data baru yang mirip dengan training data.

Tiga kategori utama:
1. Autoencoders (AE) - compression & reconstruction
2. Variational Autoencoders (VAE) - probabilistic generation
3. Generative Adversarial Networks (GAN) - adversarial training

Aplikasi:
- Data augmentation
- Anomaly detection (reconstruction error)
- Image generation
- Dimensionality reduction
- Denoising

Koneksi Teknik Elektro:
- Autoencoder = encoder-decoder system (seperti codec)
- Latent space = compressed representation (seperti transform coding)
- GAN = two-player game (seperti game theory di control)
- VAE = probabilistic encoder dengan regularization

Durasi target: 4-5 jam
============================================================="""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)


# ===========================================================
# BAGIAN 1: Autoencoder (AE)
# ===========================================================
# Autoencoder = neural network yang belajar meng-copy input ke output
# melalui representasi terkompresi (latent space).
#
# KONSEP:
# - Encoder: memetakan input x ke latent representation z.
# - Decoder: memetakan z kembali ke reconstruction x_hat.
# - Loss: MSE(x, x_hat) - seberapa baik reconstruksi.
#
# KEGUNAAN AUTOENCODER:
# - Dimensionality reduction (seperti PCA tapi non-linear).
# - Denoising: train dengan noisy input dan clean target.
# - Anomaly detection: data anomaly akan punya reconstruction error tinggi.
# - Pre-training: weights encoder bisa diinisialisasi untuk downstream tasks.
#
# KELEMAHAN:
# - Latent space tidak terstruktur (tidak continuous atau interpretable).
# - Tidak bisa generate data baru secara langsung karena tidak ada
#   distribusi yang jelas di latent space.

class Autoencoder(nn.Module):
    """
    Autoencoder - compression dan reconstruction.
    
    Architecture:
    Input -> Encoder (compress) -> Latent Space -> Decoder (reconstruct) -> Output
    
    Parameters:
    -----------
    input_dim : int
        Input dimension.
    latent_dim : int
        Latent space dimension (compression ratio = input_dim/latent_dim).
        
    Notes:
    ------
    - Encoder: maps input ke latent representation
    - Decoder: maps latent representation ke reconstruction
    - Loss: MSE(input, reconstruction)
    - Latent space = compressed representation dari data
    - Aktivasi Sigmoid di decoder karena MNIST images normalized ke [0,1].
    
    Koneksi Teknik Elektro:
    - Encoder = analysis filter bank (decomposition)
    - Decoder = synthesis filter bank (reconstruction)
    - Latent space = transform coefficients
    - Reconstruction error = distortion measure
    """
    
    def __init__(self, input_dim=784, latent_dim=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


# ===========================================================
# BAGIAN 2: Variational Autoencoder (VAE)
# ===========================================================
# VAE = Autoencoder dengan latent space yang terstruktur secara probabilistik.
#
# BEDA UTAMA DENGAN AE:
# - Encoder output BUKAN satu vektor, tapi DUA vektor: mean (mu) dan log variance (logvar).
# - Latent vector z di-sample dari distribusi Gaussian: z ~ N(mu, sigma^2).
# - Loss = Reconstruction Loss + KL Divergence (regularization).
#
# REPARAMETERIZATION TRICK:
# - Sampling z dari N(mu, sigma^2) tidak bisa di-differentiate.
# - Solusi: z = mu + sigma * epsilon, di mana epsilon ~ N(0,1).
# - epsilon dianggap constant untuk backprop, sehingga gradient bisa mengalir
#   ke mu dan sigma.
#
# KL DIVERGENCE:
# - Mengukur seberapa jauh distribusi posterior q(z|x) dari prior p(z)=N(0,1).
# - Formula: KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2).
# - Ini "memaksa" latent space untuk terdistribusi normal di sekitar origin.
# - Hasilnya: latent space menjadi continuous dan bisa di-interpolate.
#
# KEGUNAAN VAE:
# - Generate data baru dengan sample z dari N(0,1) lalu decode.
# - Latent space interpolation: berjalan di latent space menghasilkan
#   transisi yang mulus antara dua data point.
# - Anomaly detection: data normal punya reconstruction error rendah DAN
#   KL divergence yang sesuai dengan prior.

class VAE(nn.Module):
    """
    Variational Autoencoder - generative model dengan probabilistic latent space.
    
    Beda dengan AE:
    - Encoder output: mean (mu) dan log variance (log sigma^2)
    - Latent: sample dari N(mu, sigma^2)
    - Loss: Reconstruction + KL Divergence
    
    Parameters:
    -----------
    input_dim : int
        Input dimension.
    latent_dim : int
        Latent space dimension.
        
    Notes:
    ------
    - Reparameterization trick: z = mu + sigma * epsilon, epsilon ~ N(0,1)
      (memungkinkan backpropagation melalui sampling)
    - KL Divergence: measures how far posterior dari prior N(0,1)
    - VAE bisa generate data baru dengan sample dari prior
    - Tradeoff antara reconstruction quality dan structured latent space
      dikontrol oleh bobot KL term (beta-VAE).
    
    Koneksi Teknik Elektro:
    - VAE = encoder dengan noisy channel
    - Latent = transmitted signal dengan additive noise
    - KL divergence = channel capacity constraint
    - Reparameterization = equivalent channel model
    """
    
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick.
        
        z = mu + sigma * epsilon, dimana epsilon ~ N(0, 1)
        
        Ini memungkinkan backpropagation melalui sampling
        dengan treating epsilon sebagai constant.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, reconstruction, x, mu, logvar):
        """
        VAE Loss = Reconstruction Loss + KL Divergence
        
        Reconstruction: MSE atau BCE
        KL: -0.5 * Sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


# ===========================================================
# BAGIAN 3: Training VAE pada MNIST
# ===========================================================
# Training VAE sedikit berbeda dari training model deterministik.
#
# TIPS TRAINING VAE:
# - Gunakan reconstruction loss yang sesuai dengan data:
#   * Binary Cross Entropy untuk data biner atau [0,1] (seperti MNIST).
#   * MSE untuk data continuous.
# - KL loss bisa didown-weight dengan beta untuk mengontrol tradeoff.
# - Monitoring: perhatikan BCE dan KLD secara terpisah.
#   * Jika KLD terlalu besar -> reconstruction jelek (terlalu ter-regularisasi).
#   * Jika KLD terlalu kecil -> latent space tidak terstruktur.

def train_vae(vae, train_loader, epochs=10, device='cpu'):
    """
    Training VAE.
    
    Parameters:
    -----------
    vae : VAE
        Model VAE.
    train_loader : DataLoader
        Training data loader.
    epochs : int, default 10
        Jumlah epochs.
    device : str, default 'cpu'
        Device untuk training.
        
    Returns:
    --------
    list
        Training losses per epoch.
        
    Notes:
    ------
    - Optimizer Adam dengan lr 1e-3 biasanya works well untuk VAE.
    - Loss per epoch bisa fluktuatif karena stochastic sampling.
    - Untuk MNIST, BCE reconstruction loss lebih umum daripada MSE.
    """
    vae = vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    
    losses = []
    for epoch in range(epochs):
        vae.train()
        train_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            optimizer.zero_grad()
            
            recon, mu, logvar = vae(data)
            loss = vae.loss_function(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader.dataset)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}")
    
    return losses


# ===========================================================
# BAGIAN 4: Generate New Samples dari VAE
# ===========================================================
# Setelah VAE trained, kita bisa generate data baru dengan:
# 1. Sample z dari prior N(0, I).
# 2. Pass z melalui decoder.
# 3. Output = synthetic data.
#
# KUALITAS GENERATION:
# - VAE cenderung menghasilkan output yang "blurry" karena MSE/BCE loss
#   menghukum per-pixel error secara independent.
# - GAN (Generative Adversarial Network) umumnya menghasilkan output
#   yang lebih tajam karena discriminator menilai keseluruhan image.
# - Tapi VAE lebih stabil untuk training dan punya latent space
#   yang terstruktur secara matematis.

def generate_samples(vae, n_samples=16, device='cpu'):
    """
    Generate new samples dengan sampling dari prior N(0,1).
    
    Parameters:
    -----------
    vae : VAE
        Trained VAE model.
    n_samples : int, default 16
        Jumlah samples untuk generate.
    device : str, default 'cpu'
        Device untuk generation.
        
    Returns:
    --------
    torch.Tensor
        Generated samples.
        
    Notes:
    ------
    - Sample z dari N(0, I)
    - Pass z melalui decoder
    - Output = new synthetic data
    - Kualitas bergantung pada seberapa baik VAE belajar distribusi.
    """
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, vae.latent_dim).to(device)
        samples = vae.decoder(z)
    return samples


# ===========================================================
# BAGIAN 5: Visualisasi Latent Space
# ===========================================================
# Salah satu keunggulan VAE adalah latent space yang terstruktur.
# Kita bisa memvisualisasikan ini untuk 2D latent space.
#
# INTERPRETASI:
# - Setiap titik = satu sample data di latent space.
# - Warna = label/true class.
# - Jika clustering bagus, berarti latent space terstruktur dengan baik.
# - Kita bisa melakukan "latent space walk": interpolate antara dua titik
#   dan decode setiap titik intermediate untuk melihat transisi mulus.

def visualize_latent_space(vae, test_loader, device='cpu'):
    """
    Visualisasi 2D latent space (untuk latent_dim=2).
    
    Parameters:
    -----------
    vae : VAE
        Trained VAE dengan latent_dim=2.
    test_loader : DataLoader
        Test data loader.
    device : str, default 'cpu'
        Device untuk computation.
        
    Notes:
    ------
    - Setiap point = latent representation dari satu digit
    - Warna = true label
    - Clustering menunjukkan quality dari latent space
    - Jika latent_dim > 2, gunakan t-SNE atau PCA untuk visualisasi 2D.
    """
    vae.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.view(-1, 784).to(device)
            mu, _ = vae.encode(data)
            latents.append(mu.cpu().numpy())
            labels.append(label.numpy())
    
    latents = np.concatenate(latents)
    labels = np.concatenate(labels)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1],
                         c=labels, cmap='tab10', alpha=0.5, s=5)
    plt.colorbar(scatter)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space (colored by digit)')
    plt.savefig('01_vae_latent_space.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("PLOT Saved: 01_vae_latent_space.png")


# ===========================================================
# BAGIAN 6: GAN - Generative Adversarial Network
# ===========================================================
# GAN = framework di mana dua neural network saling berkompetisi:
# - Generator (G): berusaha membuat data fake yang mirip real.
# - Discriminator (D): berusaha membedakan real vs fake.
#
# GAME THEORY PERSPECTIVE:
# - G dan D bermain minimax game.
# - D mencoba maximize log(D(x)) + log(1 - D(G(z))).
# - G mencoba minimize log(1 - D(G(z))) atau equivalently maximize log(D(G(z))).
# - Nash equilibrium tercapai ketika G menghasilkan distribusi yang sama
#   dengan data real, dan D tidak bisa membedakan (output = 0.5).
#
# KEUNTUNGAN GAN:
# - Generate data yang lebih tajam dan realistic dibanding VAE.
# - Tidak perlu explicit likelihood estimation.
#
# KEKURANGAN GAN:
# - Training notoriously unstable (mode collapse, vanishing gradients).
# - Sulit untuk evaluate secara objektif.
# - Tidak punya explicit latent space structure seperti VAE.

class Generator(nn.Module):
    """
    Generator GAN - generate fake data dari noise.
    
    Parameters:
    -----------
    latent_dim : int
        Dimension dari noise vector.
    output_dim : int
        Dimension dari output data.
        
    Notes:
    ------
    - Input: random noise z ~ N(0, I)
    - Output: synthetic data yang mirip real data
    - Goal: fool discriminator
    - LeakyReLU dipilih karena membantu gradient flow untuk negative values.
    - Tanh output untuk range [-1, 1]; Sigmoid untuk [0, 1].
    
    Koneksi Teknik Elektro:
    - Generator = signal synthesizer
    - Latent noise = random seed untuk synthesis
    - Upsampling = interpolation (seperti oversampling di DSP)
    """
    
    def __init__(self, latent_dim=100, output_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
        
    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    """
    Discriminator GAN - membedakan real vs fake data.
    
    Parameters:
    -----------
    input_dim : int
        Dimension dari input data.
        
    Notes:
    ------
    - Input: real data atau fake data dari generator
    - Output: probability bahwa input adalah real (1) atau fake (0)
    - Goal: correctly classify real vs fake
    - Dropout digunakan sebagai regularization.
    - LeakyReLU membantu menghindari dead neurons.
    
    Koneksi Teknik Elektro:
    - Discriminator = classifier/detector
    - Binary classification: signal present atau tidak
    - Adversarial training = jamming and anti-jamming scenario
    """
    
    def __init__(self, input_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)


# ===========================================================
# BAGIAN 7: GAN Training
# ===========================================================
# Training GAN memerlukan balancing act antara G dan D.
#
# BEST PRACTICES TRAINING GAN:
# - Label smoothing: real labels = 0.9 (bukan 1.0) untuk menghindari
#   discriminator yang terlalu confident.
# - Train D lebih sering dari G (misal: 5 D steps per 1 G step).
# - Gunakan learning rate kecil (misal: 0.0002).
# - Adam dengan beta1=0.5, beta2=0.999 lebih stabil.
# - Gradient clipping bisa membantu stabilitas.
#
# PROBLEM UMUM:
# - Mode collapse: G hanya generate variasi terbatas.
# - Vanishing gradients: D terlalu kuat, G tidak bisa belajar.
# - Oscillation: loss tidak converge.
#
# SOLUSI:
# - WGAN: mengganti BCE loss dengan Wasserstein distance.
# - Spectral Normalization: membatasi Lipschitz constant.
# - Progressive GAN: training dari resolusi rendah ke tinggi.

def train_gan(generator, discriminator, train_loader, epochs=50,
              latent_dim=100, device='cpu'):
    """
    Training GAN dengan adversarial training.
    
    Parameters:
    -----------
    generator : Generator
        Generator network.
    discriminator : Discriminator
        Discriminator network.
    train_loader : DataLoader
        Training data loader.
    epochs : int, default 50
        Jumlah epochs.
    latent_dim : int, default 100
        Latent dimension.
    device : str, default 'cpu'
        Device untuk training.
        
    Returns:
    --------
    tuple
        (generator_losses, discriminator_losses)
        
    Notes:
    ------
    - Alternating training:
      1. Train discriminator: classify real vs fake
      2. Train generator: fool discriminator
    - BCELoss untuk kedua networks
    - Label smoothing untuk stabilitas
    - detach() penting saat train D: mencegah gradient mengalir ke G.
    
    PERINGATAN: GAN training notoriously unstable!
    - Mode collapse: generator hanya produce limited variety
    - Vanishing gradients: discriminator terlalu kuat
    - Oscillation: tidak converge
    """
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    g_losses = []
    d_losses = []
    
    for epoch in range(epochs):
        g_epoch_loss = 0
        d_epoch_loss = 0
        
        for batch_idx, (real_data, _) in enumerate(train_loader):
            batch_size = real_data.size(0)
            real_data = real_data.view(-1, 784).to(device)
            
            # Labels
            real_labels = torch.ones(batch_size, 1).to(device) * 0.9  # Label smoothing
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real data
            real_output = discriminator(real_data)
            d_real_loss = criterion(real_output, real_labels)
            
            # Fake data
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_data = generator(z)
            fake_output = discriminator(fake_data.detach())
            d_fake_loss = criterion(fake_output, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            
            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, real_labels)  # Want discriminator to think fake is real
            
            g_loss.backward()
            g_optimizer.step()
            
            g_epoch_loss += g_loss.item()
            d_epoch_loss += d_loss.item()
        
        g_losses.append(g_epoch_loss / len(train_loader))
        d_losses.append(d_epoch_loss / len(train_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: D Loss={d_losses[-1]:.4f}, G Loss={g_losses[-1]:.4f}")
    
    return g_losses, d_losses


# ===========================================================
# LATIHAN 17: Generative Models
# ===========================================================
"""
TARGET Learning Objectives:
   - Membangun VAE dari scratch dengan benar
   - Mengimplementasikan GAN training loop
   - Menggunakan generative models untuk anomaly detection

PANDUAN LANGKAH-LANGKAH:

STEP 1: VAE Implementation & Analysis
-------------------------------------
   a) Implementasi complete VAE:
      - Encoder: input -> mu, log(sigma^2)
      - Reparameterization trick
      - Decoder: z -> reconstruction
      - Loss: BCE + beta * KL (beta-VAE)
      
   b) Eksperimen dengan beta:
      - beta = 0.1: emphasis pada reconstruction
      - beta = 1.0: balanced (standard VAE)
      - beta = 5.0: emphasis pada structured latent space
      - beta = 10.0: very structured (tapi reconstruction buruk)
      
   c) Visualisasi:
      - Latent space interpolation (walk through latent space)
      - Reconstruction quality per beta
      - KL divergence vs reconstruction tradeoff
      
   TIPS KENAPA beta?
     - beta mengontrol tradeoff antara reconstruction dan regularization
     - Higher beta = more structured latent space
     - Lower beta = better reconstruction


STEP 2: GAN Implementation & Training
-------------------------------------
   a) Implementasi Generator dan Discriminator:
      - Generator: noise -> data
      - Discriminator: data -> real/fake probability
      
   b) Training loop:
      - Alternating updates (1 D, 1 G)
      - Label smoothing untuk stabilitas
      - Gradient clipping (opsional)
      
   c) Monitoring:
      - Plot losses: D dan G
      - Visualisasi generated samples per epoch
      - Inception score (opsional)
      
   d) Troubleshooting:
      - Mode collapse: D loss -> 0, G loss naik
      - Vanishing gradients: D terlalu kuat
      - Oscillation: losses tidak stabil


STEP 3: Anomaly Detection dengan VAE
------------------------------------
   a) Train VAE pada normal data only
   b) Anomaly score = reconstruction error
   c) Threshold: > percentile 95 = anomaly
   d) Evaluate: precision, recall, F1
   
   TIPS KENAPA VAE untuk anomaly detection?
     - VAE belajar distribusi normal data
     - Anomaly = data yang tidak bisa direconstruct dengan baik
     - Probabilistic: bisa quantify uncertainty


STEP 4: Conditional Generation
------------------------------
   a) Conditional VAE (CVAE):
      - Input: data + label
      - Latent: conditioned on label
      - Generate: sample dari class tertentu
      
   b) Conditional GAN (CGAN):
      - Generator input: noise + label
      - Discriminator input: data + label
      - Generate: data dari class tertentu
      
   c) Aplikasi: generate synthetic data per class


TIPS:
   - VAE: reparameterization trick = z = mu + sigma * epsilon
   - GAN: label smoothing = real_labels = 0.9 (bukan 1.0)
   - GAN: train D lebih sering dari G (opsional)
   - Anomaly: reconstruction error = MSE(input, recon)
   - CVAE: concatenate label ke input encoder

PERINGATAN COMMON MISTAKES:
   - VAE tanpa reparameterization trick -> cannot backprop
   - KL loss terlalu besar -> dominates reconstruction
   - GAN: D terlalu kuat -> G tidak bisa belajar
   - GAN: tidak detach() saat train D -> G juga terupdate
   - Anomaly detection: train pada data dengan anomaly -> model
     belajar anomaly sebagai normal

TARGET EXPECTED OUTPUT:
   - VAE yang bisa generate realistic MNIST digits
   - GAN dengan stable training (minimal mode collapse)
   - Anomaly detector dengan F1 > 0.80
   - Conditional generation per digit class
"""


# ===========================================================
# 🔥 CHALLENGE: Synthetic Data Generation untuk Power Systems
# ===========================================================
"""
TARGET Learning Objectives:
   - Menggenerate synthetic power quality data
   - Menggunakan generative models untuk data augmentation
   - Membangun robust anomaly detection system

PANDUAN LANGKAH-LANGKAH:

STEP 1: Dataset - Power Quality Waveforms
-----------------------------------------
   a) Normal waveforms (base):
      - Pure 50Hz sinusoidal
      - THD < 3%
      - 1000 samples, 3.2 kHz sampling rate
      
   b) Anomaly types (minority classes):
      - Voltage sag: amplitude 70% selama 0.1s
      - Voltage swell: amplitude 130% selama 0.1s
      - Harmonic distortion: THD 15% (3rd, 5th, 7th)
      - Transient: high-frequency impulse
      
   c) Problem: dataset tidak balanced
      - Normal: 80%
      - Anomalies: 20% (distributed across types)


STEP 2: Build VAE untuk Normal Data
-----------------------------------
   a) Architecture:
      - Input: 1D waveform (1000 samples)
      - Encoder: Conv1D layers
      - Latent: 32 dimensions
      - Decoder: ConvTranspose1D layers
      
   b) Training:
      - Train HANYA pada normal data
      - Monitor: reconstruction quality
      - Save best model (lowest validation loss)


STEP 3: Anomaly Detection
-------------------------
   a) Reconstruction-based:
      - Test semua data (normal + anomaly)
      - Compute reconstruction error per sample
      - Threshold: mu + 3*sigma dari normal data errors
      
   b) Latent-space based:
      - Compute Mahalanobis distance di latent space
      - Anomaly = high distance dari normal cluster
      
   c) Evaluate:
      - ROC curve
      - AUC score
      - Precision/recall per anomaly type
      - Which anomaly type is hardest to detect?


STEP 4: Generate Synthetic Anomalies
------------------------------------
   a) Interpolate di latent space:
      - Ambil dua normal samples
      - Interpolate di latent space
      - Decode: hasil = intermediate waveform
      
   b) Add controlled noise:
      - Sample dari normal latent
      - Add noise ke latent
      - Decode: hasil = "perturbed" waveform
      
   c) Use untuk augmentation:
      - Generate synthetic anomalies
      - Balance dataset
      - Retrain classifier


STEP 5: Compare Classifiers
---------------------------
   a) Baseline: train pada original imbalanced data
   b) With VAE augmentation: train pada augmented data
   c) With SMOTE: traditional augmentation
   d) Evaluate: accuracy, F1 per class, robustness
   
   TIPS Analisis:
     - Mana augmentation method terbaik?
     - Apakah synthetic data improve generalization?
     - Quality assessment dari generated data


TIPS:
   - Conv1D encoder: kernel sizes 7, 5, 3
   - Latent space: 32 dimensions cukup untuk 1D signals
   - Reconstruction: MSE per sample
   - Threshold: adaptive (percentile-based)
   - Augmentation: generate 2-5x synthetic per class

PERINGATAN COMMON MISTAKES:
   - VAE train pada semua data (termasuk anomaly)
   - Threshold terlalu strict -> banyak false negatives
   - Synthetic data yang tidak realistic -> hurt performance
   - Tidak evaluate quality dari generated data
   - Classifier overfit ke synthetic data

TARGET EXPECTED OUTPUT:
   - VAE yang bisa reconstruct normal waveforms dengan <5% error
   - Anomaly detector dengan AUC > 0.90
   - Synthetic anomaly generation yang realistic
   - Improved classifier setelah augmentation
   - Clear analysis: when to use generative models

Ini adalah aplikasi cutting-edge dari generative models
ke power systems engineering!
"""

print("\n" + "="*50)
print("SELESAI FASE 5!")
print("="*50)
print("""
Kamu sekarang bisa:
OK Transfer learning dengan pre-trained models
OK NLP dengan transformers (self-attention, BERT)
OK Generative models (VAE, GAN)

Sebelum lanjut:
1. Selesaikan Project 4: NLP Pipeline
2. Review semua exercise dan challenge

Lanjut ke: 06-expert/01_expert_roadmap.py
""")
