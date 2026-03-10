"""
=============================================================
FASE 5 — MODUL 3: GENERATIVE MODELS
=============================================================
Generative models = model yang bisa MEMBUAT data baru.

Tiga arsitektur utama:
1. Variational Autoencoder (VAE) — probabilistic generation
2. Generative Adversarial Network (GAN) — adversarial training
3. Diffusion Models — iterative denoising (DALL-E, Stable Diffusion)

Koneksi EE:
- VAE encoder = compression (source coding)
- GAN = adversarial game theory (Nash equilibrium)
- Diffusion = iterative signal denoising

Durasi target: 4-5 jam
=============================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===========================================================
# 📖 BAGIAN 1: Autoencoder (Building Block)
# ===========================================================
# Autoencoder: Encoder → Latent Space → Decoder
# Goal: reconstruct input from compressed representation

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
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
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z


# ===========================================================
# 📖 BAGIAN 2: VAE (Variational Autoencoder)
# ===========================================================
# VAE menambahkan probabilistic structure ke latent space:
# - Encoder output: mean (μ) dan variance (σ²)
# - Sampling: z = μ + σ * ε, dimana ε ~ N(0,1) (reparameterization trick)
# - Loss = Reconstruction + KL Divergence

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        # Encoder
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)       # mean
        self.fc_logvar = nn.Linear(128, latent_dim)    # log-variance

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_shared(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + σ * ε"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def generate(self, n_samples):
        """Generate new samples from random latent vectors"""
        z = torch.randn(n_samples, self.fc_mu.out_features).to(
            next(self.parameters()).device)
        return self.decode(z)


def vae_loss(recon_x, x, mu, logvar):
    """VAE Loss = Reconstruction + KL Divergence"""
    # Reconstruction loss (binary cross-entropy)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence: D_KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


# ===========================================================
# 📖 BAGIAN 3: Train VAE on MNIST
# ===========================================================

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

LATENT_DIM = 2  # 2D for visualization
vae = VAE(784, LATENT_DIM).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

print("=== Training VAE on MNIST ===")
for epoch in range(20):
    vae.train()
    total_loss = 0
    for batch_X, _ in train_loader:
        batch_X = batch_X.view(-1, 784).to(device)
        recon, mu, logvar = vae(batch_X)
        loss = vae_loss(recon, batch_X, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataset)
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}: loss={avg_loss:.2f}")

# Visualize latent space
vae.eval()
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Encode test data to latent space
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10000)
test_X, test_y = next(iter(test_loader))
test_X = test_X.view(-1, 784).to(device)

with torch.no_grad():
    mu, _ = vae.encode(test_X)
    z = mu.cpu().numpy()

axes[0].scatter(z[:, 0], z[:, 1], c=test_y.numpy(), cmap='tab10', s=1, alpha=0.5)
axes[0].set_title('Latent Space (colored by digit)')
axes[0].colorbar = plt.colorbar(axes[0].collections[0], ax=axes[0])

# Generate new samples
with torch.no_grad():
    generated = vae.generate(64).cpu().numpy()

for i in range(8):
    for j in range(8):
        idx = i * 8 + j
        ax_img = axes[1].inset_axes([j/8, 1-(i+1)/8, 1/8, 1/8])
        ax_img.imshow(generated[idx].reshape(28, 28), cmap='gray')
        ax_img.axis('off')
axes[1].set_title('Generated Samples')
axes[1].axis('off')

# Interpolation in latent space
n_steps = 10
z1 = torch.randn(1, LATENT_DIM).to(device)
z2 = torch.randn(1, LATENT_DIM).to(device)
interpolations = []
for alpha in np.linspace(0, 1, n_steps):
    z_interp = (1 - alpha) * z1 + alpha * z2
    with torch.no_grad():
        img = vae.decode(z_interp).cpu().numpy()
    interpolations.append(img.reshape(28, 28))

for i, img in enumerate(interpolations):
    ax_img = axes[2].inset_axes([i/n_steps, 0.3, 1/n_steps, 0.4])
    ax_img.imshow(img, cmap='gray')
    ax_img.axis('off')
axes[2].set_title('Latent Space Interpolation')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('01_vae_results.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 01_vae_results.png")


# ===========================================================
# 📖 BAGIAN 4: GAN (Generative Adversarial Network)
# ===========================================================

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# Train GAN
LATENT_DIM_GAN = 64
generator = Generator(LATENT_DIM_GAN, 784).to(device)
discriminator = Discriminator(784).to(device)

opt_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

print("\n=== Training GAN on MNIST ===")
g_losses = []
d_losses = []

for epoch in range(30):
    for batch_X, _ in train_loader:
        batch_size = batch_X.size(0)
        real = batch_X.view(batch_size, -1).to(device)

        # Train Discriminator
        z = torch.randn(batch_size, LATENT_DIM_GAN).to(device)
        fake = generator(z).detach()

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        d_real = discriminator(real)
        d_fake = discriminator(fake)
        d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)

        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # Train Generator
        z = torch.randn(batch_size, LATENT_DIM_GAN).to(device)
        fake = generator(z)
        g_output = discriminator(fake)
        g_loss = criterion(g_output, real_labels)  # Generator wants D to output 1

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}: G_loss={g_loss.item():.4f}, D_loss={d_loss.item():.4f}")

# Generate samples
generator.eval()
with torch.no_grad():
    z = torch.randn(64, LATENT_DIM_GAN).to(device)
    generated_gan = generator(z).cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Generated samples
for i in range(8):
    for j in range(8):
        idx = i * 8 + j
        ax_img = axes[0].inset_axes([j/8, 1-(i+1)/8, 1/8, 1/8])
        ax_img.imshow(generated_gan[idx].reshape(28, 28), cmap='gray')
        ax_img.axis('off')
axes[0].set_title('GAN Generated Samples')
axes[0].axis('off')

# Loss curves
axes[1].plot(g_losses, label='Generator')
axes[1].plot(d_losses, label='Discriminator')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('GAN Training Losses')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('02_gan_results.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 02_gan_results.png")


# ===========================================================
# 🏋️ EXERCISE 17: Generative Model Experiments
# ===========================================================
"""
1. VAE improvements:
   - Convolutional VAE (conv encoder + conv decoder)
   - β-VAE (adjustable KL weight)
   - Conditional VAE (generate specific digits)

2. GAN improvements:
   - DCGAN (convolutional)
   - Wasserstein GAN (WGAN) — more stable training
   - Conditional GAN (cGAN)

3. Compare VAE vs GAN:
   - Quality of generated samples (FID score)
   - Training stability
   - Latent space properties
"""


# ===========================================================
# 🔥 CHALLENGE: Data Augmentation dengan Generative Models
# ===========================================================
"""
Skenario: kamu punya dataset kecil (100 samples) untuk fault detection.
Gunakan generative models untuk AUGMENT data!

1. Train VAE/GAN pada data yang ada
2. Generate synthetic samples untuk setiap class
3. Train classifier pada: original vs original + synthetic
4. Evaluate: apakah synthetic data membantu?

Ini SANGAT praktis: di industri, labeled fault data selalu sedikit.
Generative augmentation bisa menjadi solusi!
"""

print("\n" + "="*50)
print("🎉 FASE 5 SELESAI!")
print("="*50)
print("""
Kamu sekarang bisa:
✅ Transfer Learning (feature extraction & fine-tuning)
✅ Transformers (attention dari nol + Hugging Face)
✅ Generative Models (VAE & GAN)

Lanjut ke: 06-expert/ (Paper Implementation, MLOps, Production)
""")
