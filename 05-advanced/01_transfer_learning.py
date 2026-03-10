"""
=============================================================
FASE 5 — MODUL 1: TRANSFER LEARNING
=============================================================
Transfer Learning = menggunakan model yang sudah dilatih
pada dataset besar, lalu fine-tune untuk task kita.

Kenapa ini penting?
- Tidak perlu dataset besar
- Training jauh lebih cepat
- Performance biasanya lebih baik

Analogi EE: seperti menggunakan IC/modul yang sudah jadi
daripada membangun seluruh rangkaian dari transistor.

Durasi target: 3-4 jam
=============================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===========================================================
# 📖 BAGIAN 1: Konsep Transfer Learning
# ===========================================================

print("""
=== Transfer Learning Strategies ===

1. FEATURE EXTRACTION (Freeze backbone, train classifier only)
   - Gunakan pre-trained model sebagai feature extractor
   - Hanya train FC layers di akhir
   - Cocok untuk: dataset kecil, domain mirip

2. FINE-TUNING (Unfreeze some/all layers)
   - Mulai dari pre-trained weights
   - Train beberapa layer terakhir + classifier
   - Cocok untuk: dataset medium, domain agak berbeda

3. FULL FINE-TUNING (Unfreeze all)
   - Train semua layer dari pre-trained weights
   - Cocok untuk: dataset besar, domain berbeda

Layer awal CNN → generic features (edges, textures)
Layer akhir CNN → task-specific features

→ Semakin berbeda domain, semakin banyak layer yang perlu di-fine-tune.
""")


# ===========================================================
# 📖 BAGIAN 2: Transfer Learning dengan ResNet
# ===========================================================

class TransferModel(nn.Module):
    """ResNet-18 with custom classifier"""
    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()

        # Load pre-trained ResNet-18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_layers(self, n_layers=0):
        """Gradually unfreeze layers for fine-tuning"""
        layers = list(self.backbone.children())[:-1]  # exclude FC
        for layer in layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True


# ===========================================================
# 📖 BAGIAN 3: Training Pipeline
# ===========================================================

def train_transfer_model(model, train_loader, val_loader,
                          epochs=10, lr=0.001):
    """Training pipeline untuk transfer learning"""
    # Hanya optimize parameter yang requires_grad
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                val_loss += criterion(output, y).item()
                _, pred = torch.max(output, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step()

        if (epoch + 1) % 2 == 0:
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                  f"val_acc={val_acc:.4f}, trainable_params={n_trainable:,}")

    return history


# ===========================================================
# 📖 BAGIAN 4: Practical Example — menggunakan CIFAR-10
# ===========================================================

print("\n=== Transfer Learning on CIFAR-10 ===")
print("(Kalau belum download, ini akan download ~170MB)")

transform = transforms.Compose([
    transforms.Resize(224),  # ResNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Only download and demo a small subset for speed
# In practice, use the full dataset
print("\nTo run full training:")
print("  1. Uncomment the training code below")
print("  2. Expect ~90%+ accuracy with fine-tuning")
print("  3. Compare: train from scratch vs transfer learning")

"""
# Uncomment to run full training:
train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, num_workers=2)

# Strategy 1: Feature Extraction
print("\\n--- Feature Extraction ---")
model_fe = TransferModel(num_classes=10, freeze_backbone=True).to(device)
history_fe = train_transfer_model(model_fe, train_loader, val_loader, epochs=5, lr=0.001)

# Strategy 2: Fine-Tuning (unfreeze last 2 layers)
print("\\n--- Fine-Tuning (last 2 layers) ---")
model_ft = TransferModel(num_classes=10, freeze_backbone=True).to(device)
model_ft.unfreeze_layers(2)
history_ft = train_transfer_model(model_ft, train_loader, val_loader, epochs=10, lr=0.0001)
"""


# ===========================================================
# 🏋️ EXERCISE 15: Transfer Learning Experiments
# ===========================================================
"""
1. Bandingkan 3 strategi transfer learning pada CIFAR-10:
   a. Feature extraction (freeze all)
   b. Fine-tune last 2 conv blocks
   c. Fine-tune all layers
   - Plot accuracy curves untuk ketiga strategi
   - Analisis: mana yang terbaik dan kenapa?

2. Transfer learning untuk 1D signals:
   - Pre-train CNN pada sinyal sinusoidal besar
   - Fine-tune untuk klasifikasi sinyal baru yang lebih kecil
   - Bandingkan: transfer vs train from scratch

3. Gradual unfreezing:
   - Epoch 1-3: hanya train FC
   - Epoch 4-6: unfreeze layer4
   - Epoch 7-10: unfreeze layer3
   - Bandingkan dengan langsung fine-tune semua
"""


# ===========================================================
# 🔥 CHALLENGE: Domain Adaptation
# ===========================================================
"""
Skenario realistis:
- Kamu punya model yang dilatih pada data motor merek A
- Ingin deploy di motor merek B (domain berbeda!)

1. Generate 2 domain data:
   - Domain A: 1000 labeled samples (source)
   - Domain B: 100 labeled + 900 unlabeled samples (target)

2. Strategi:
   a. Direct transfer (train on A, test on B) — baseline
   b. Fine-tune on B's 100 labeled samples
   c. Domain adaptation: train on A + unlabeled B

3. Evaluasi: mana yang terbaik dengan limited target labels?
"""

print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 05-advanced/02_nlp_transformers.py")
print("="*50)
