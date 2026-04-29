"""
=============================================================
FASE 5 - MODUL 1: TRANSFER LEARNING
=============================================================
Transfer Learning = menggunakan knowledge dari model yang sudah
pre-trained pada dataset besar, untuk task yang berbeda (tapi related).

Mengapa transfer learning powerful?
- Pre-trained models (ImageNet) sudah belajar generic features:
  edges, textures, shapes, patterns
- Kita hanya perlu fine-tune untuk task-specific features
- Butuh MUCH less data dan training time

Dua pendekatan utama:
1. Feature Extraction: freeze backbone, train hanya classifier
2. Fine-tuning: unfreeze semua (atau sebagian), train dengan lr kecil

Koneksi Teknik Elektro:
- Pre-trained model = system yang sudah di-identify
- Fine-tuning = adaptasi parameter untuk new operating point
- Feature extraction = menggunakan system tanpa mengubah dynamics
- Layer freezing = fixed controller parameters

Durasi target: 3-4 jam
============================================================="""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)


# ===========================================================
# BAGIAN 1: Pre-trained Models
# ===========================================================
# PyTorch menyediakan banyak pre-trained models:
# ResNet, VGG, EfficientNet, MobileNet, etc.
#
# KENAPA PRE-TRAINED MODELS PENTING?
# - Training dari awal pada dataset besar (seperti ImageNet dengan 1.2M gambar)
#   membutuhkan waktu berminggu-minggu dan resource GPU yang besar.
# - Pre-trained models sudah "belajar" representasi visual dasar yang universal.
# - Konsep ini mirip dengan bagaimana manusia belajar: kita tidak belajar
#   mengenali objek dari nol setiap kali, tapi memanfaatkan pengetahuan visual
#   yang sudah kita miliki.
#
# BEST PRACTICES MEMILIH PRE-TRAINED MODEL:
# - ResNet: Good balance antara accuracy dan speed, cocok untuk general purpose.
# - EfficientNet: Scaling yang lebih efisien (compound scaling depth, width, resolution).
# - MobileNet: Optimized untuk mobile dan edge devices dengan inverted residuals.
# - Vision Transformer (ViT): State-of-the-art untuk banyak task, tapi lebih berat.
#
# TRADEOFF:
# - Model besar = accuracy lebih tinggi, tapi inference lebih lambat dan
#   membutuhkan lebih banyak memory.
# - Model kecil = lebih cepat, tapi mungkin kurang akurat untuk task kompleks.

print("=== Available Pre-trained Models ===")
# Load ResNet-18 (pre-trained on ImageNet)
resnet = models.resnet18(pretrained=True)
print(f"ResNet-18 architecture:\n{resnet}")

# Layer terakhir (fc) adalah classifier untuk 1000 ImageNet classes
# Kita bisa ganti dengan layer baru untuk task kita
# PERINGATAN: pretrained=True akan download weights ~45MB saat pertama kali.

print("\n" + "="*60)
print("LAYERS IN RESNET-18:")
print("="*60)
for name, module in resnet.named_children():
    print(f"  {name}: {module.__class__.__name__}")


# ===========================================================
# BAGIAN 2: Feature Extraction
# ===========================================================
# Pendekatan 1: Freeze semua layers kecuali classifier
# Backbone = feature extractor yang fixed
#
# KAPAN MENGGUNAKAN FEATURE EXTRACTION?
# - Dataset sangat kecil (< 1000 images per class): fine-tuning semua layer
#   bisa menyebabkan overfitting karena terlalu banyak parameter yang di-train.
# - Dataset baru sangat mirip dengan dataset pre-training (misal: natural images).
# - Resource terbatas dan butuh training cepat.
#
# KEUNTUNGAN:
# - Training sangat cepat karena hanya classifier yang di-update.
# - Tidak perlu learning rate yang sangat kecil.
# - Overfitting lebih rendah karena backbone tetap robust.
#
# KEKURANGAN:
# - Akurasi mungkin tidak maksimal jika task sangat berbeda dari pre-training.
# - Tidak bisa adaptasi pada level feature yang lebih rendah.

class FeatureExtractor(nn.Module):
    """
    Feature extraction menggunakan pre-trained ResNet.
    
    Parameters:
    -----------
    num_classes : int
        Jumlah kelas untuk task baru.
    pretrained : bool, default True
        Gunakan weights pre-trained dari ImageNet.
        
    Notes:
    ------
    - Semua layers di-freeze (requires_grad=False)
    - Hanya classifier layer yang di-train
    - Cocok untuk dataset kecil (100-1000 images)
    - Training sangat cepat (hanya update classifier weights)
    - Kita mengganti fc layer terakhir karena ImageNet punya 1000 classes,
      sedangkan task kita mungkin hanya punya 10 classes.
    
    Koneksi Teknik Elektro:
    - Pre-trained backbone = black box system yang sudah di-kalibrasi
    - Classifier = observer/estimator yang di-tune untuk measurand baru
    - Frozen layers = fixed plant dynamics yang tidak boleh diubah
    """
    
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Freeze all layers
        # requires_grad=False berarti PyTorch tidak akan menghitung gradient
        # untuk parameter ini, sehingga tidak di-update saat backward pass.
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Replace classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


# ===========================================================
# BAGIAN 3: Fine-tuning
# ===========================================================
# Fine-tuning = mengupdate SEMUA atau SEBAGIAN parameter pre-trained model
# untuk menyesuaikan dengan task baru.
#
# KAPAN MENGGUNAKAN FINE-TUNING?
# - Dataset cukup besar (> 10,000 images): model punya cukup data untuk
#   belajar adaptasi tanpa overfitting.
# - Dataset baru cukup berbeda dari dataset pre-training.
# - Butuh akurasi maksimal dan willing untuk trade-off dengan training time.
#
# STRATEGI GRADUAL UNFREEZING:
# - Layer awal (layer1, layer2) belajar features yang lebih generic
#   (edges, textures). Sebaiknya di-freeze lebih lama atau pakai lr lebih kecil.
# - Layer akhir (layer3, layer4) belajar features yang lebih task-specific
#   (shapes, object parts). Bisa di-unfreeze lebih awal dengan lr lebih besar.
# - Classifier (fc) selalu di-train dari awal karena completely new.
#
# DISCRIMINATIVE LEARNING RATE:
# - Learning rate lebih kecil untuk layers awal (misal: 1e-5)
# - Learning rate lebih besar untuk layers akhir (misal: 1e-3)
# - Ini mencegah "catastrophic forgetting" di early layers.

class FineTunedModel(nn.Module):
    """
    Fine-tuning dengan gradual unfreezing.
    
    Strategy:
    1. Awal: freeze semua kecuali classifier (seperti feature extraction)
    2. Setelah classifier converge: unfreeze layer4
    3. Setelah layer4 converge: unfreeze layer3
    4. Continue sampai semua layers unfrozen
    
    Parameters:
    -----------
    num_classes : int
        Jumlah kelas untuk task baru.
    pretrained : bool, default True
        Gunakan weights pre-trained dari ImageNet.
        
    Notes:
    ------
    - Learning rate lebih kecil untuk layers awal
    - Learning rate lebih besar untuk layers akhir
    - Layer yang di-freeze lebih awal = features lebih generic
    - Layer yang di-unfreeze lebih akhir = features lebih task-specific
    - Gradual unfreezing mirip dengan adaptive control yang bertahap.
    
    Koneksi Teknik Elektro:
    - Gradual unfreezing = adaptive control dengan varying bandwidth
    - Lower lr untuk early layers = slow adaptation untuk
      stable dynamics (seperti integrator dengan time constant besar)
    - Higher lr untuk late layers = fast adaptation untuk
      output adjustment (seperti proportional gain yang tinggi)
    """
    
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def freeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Always unfreeze classifier
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
            
    def unfreeze_layers(self, layer_names):
        for name, param in self.backbone.named_parameters():
            if any(ln in name for ln in layer_names):
                param.requires_grad = True
                
    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


# ===========================================================
# BAGIAN 4: Learning Rate Scheduling per Layer
# ===========================================================
# DISCRIMINATIVE LEARNING RATE adalah teknik kunci dalam transfer learning.
# Setiap layer group mendapatkan learning rate yang berbeda.
#
# KENAPA PERLU DISKRIMINATIF LR?
# - Early layers sudah belajar features yang sangat generic dan robust.
#   Mengubahnya dengan lr besar bisa merusak representasi yang sudah bagus.
# - Late layers lebih task-specific, jadi boleh di-update lebih agresif.
# - Classifier baru sama sekali belum belajar apa-apa, jadi perlu lr paling besar.
#
# BEST PRACTICES:
# - Classifier: 10x base_lr (karena dimulai dari random initialization)
# - Layer4 (layer paling dalam): 1x base_lr
# - Layer3: 0.1x base_lr
# - Layer2: 0.01x base_lr
# - Layer1 (layer paling awal): 0.001x base_lr
#
# TOOLS YANG MENDUKUNG:
# - PyTorch: optim.Adam dengan parameter groups
# - fastai: fit_one_cycle dengan discriminative learning rate bawaan
# - transformers: Trainer dengan learning_rate sebagai dict

def get_optimizer_with_layer_lr(model, base_lr=0.001):
    """
    Different learning rates untuk different layer groups.
    
    Strategy:
    - Classifier: 10x base_lr (fast adaptation)
    - Layer4: 1x base_lr
    - Layer3: 0.1x base_lr
    - Layer2: 0.01x base_lr
    - Layer1: 0.001x base_lr (very slow adaptation)
    
    Parameters:
    -----------
    model : nn.Module
        Model dengan multiple layer groups.
    base_lr : float, default 0.001
        Base learning rate.
        
    Returns:
    --------
    torch.optim.Optimizer
        Optimizer dengan parameter groups.
        
    Notes:
    ------
    - Discriminative fine-tuning = key untuk transfer learning
    - Early layers punya features yang lebih generic
    - Late layers punya features yang lebih task-specific
    - Mengubah early layers dengan lr besar bisa merusak
      pre-trained features (catastrophic forgetting)
    - Adam optimizer dipilih karena adaptive momentum membantu
      stabilitas dengan lr yang berbeda-beda.
    """
    param_groups = [
        {'params': model.backbone.layer1.parameters(), 'lr': base_lr * 0.001},
        {'params': model.backbone.layer2.parameters(), 'lr': base_lr * 0.01},
        {'params': model.backbone.layer3.parameters(), 'lr': base_lr * 0.1},
        {'params': model.backbone.layer4.parameters(), 'lr': base_lr},
        {'params': model.backbone.fc.parameters(), 'lr': base_lr * 10},
    ]
    return optim.Adam(param_groups)


# ===========================================================
# BAGIAN 5: Training Pipeline
# ===========================================================
# Training pipeline untuk transfer learning biasanya dibagi dalam 2 phase:
#
# PHASE 1: FEATURE EXTRACTION
# - Freeze backbone, hanya train classifier.
# - Tujuannya: mendapatkan initial weights untuk classifier yang masuk akal.
# - Biasanya 5-10 epochs saja dengan lr standar (misal: 1e-3).
#
# PHASE 2: FINE-TUNING
# - Unfreeze sebagian atau semua layers.
# - Gunakan discriminative learning rate.
# - Training lebih lambat dan perlu lebih banyak epochs.
# - Early stopping sangat direkomendasikan untuk mencegah overfitting.
#
# MONITORING:
# - Selalu monitor validation accuracy, bukan hanya training loss.
# - Jika validation accuracy turun sementara training accuracy naik,
#   itu tanda overfitting -> stop training atau turunkan lr.

def train_transfer_learning(model, train_loader, val_loader,
                            epochs=10, device='cpu'):
    """
    Training pipeline untuk transfer learning.
    
    Parameters:
    -----------
    model : nn.Module
        Model (FeatureExtractor atau FineTunedModel).
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    epochs : int, default 10
        Jumlah training epochs.
    device : str, default 'cpu'
        Device untuk training ('cpu' atau 'cuda').
        
    Returns:
    --------
    dict
        Training history (loss dan accuracy).
        
    Notes:
    ------
    - Phase 1: Train classifier only (frozen backbone)
    - Phase 2: Fine-tune dengan discriminative lr
    - Early stopping berdasarkan validation accuracy
    - Learning rate harus lebih kecil saat fine-tuning untuk
      mencegah catastrophic forgetting.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Phase 1: Feature extraction
    print("\n=== Phase 1: Feature Extraction ===")
    if hasattr(model, 'freeze_all'):
        model.freeze_all()
    
    optimizer = optim.Adam(model.get_trainable_params(), lr=0.001)
    
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs // 2):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"  Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    # Phase 2: Fine-tuning
    print("\n=== Phase 2: Fine-tuning ===")
    if hasattr(model, 'unfreeze_layers'):
        model.unfreeze_layers(['layer4', 'layer3'])
    
    optimizer = get_optimizer_with_layer_lr(model, base_lr=0.0001)
    
    for epoch in range(epochs // 2, epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"  Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    return history


# ===========================================================
# BAGIAN 6: Data Augmentation untuk Transfer Learning
# ===========================================================
# Data augmentation adalah teknik CRITICAL untuk transfer learning,
# terutama saat dataset kecil.
#
# JENIS AUGMENTATION UNTUK IMAGES:
# - Geometric: RandomCrop, RandomHorizontalFlip, Rotation
# - Photometric: ColorJitter (brightness, contrast, saturation, hue)
# - Advanced: AutoAugment, RandAugment (policy-based)
#
# NORMALIZATION:
# - HARUS menggunakan ImageNet mean dan std jika menggunakan pre-trained weights.
# - Jika tidak, distribusi input tidak sesuai dengan yang model ekspektasi,
#   dan performance akan jatuh drastis.
# - Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
#
# TRADEOFF AUGMENTATION:
# - Terlalu banyak augmentation = model sulit belajar (signal-to-noise rendah).
# - Terlalu sedikit augmentation = overfitting pada dataset kecil.
# - General rule: lebih agresif untuk dataset yang lebih kecil.

def get_transforms(is_training=True):
    """
    Data augmentation transforms.
    
    Training transforms:
    - RandomResizedCrop: random crop dengan scale [0.8, 1.0]
    - RandomHorizontalFlip: flip horizontal (50% probability)
    - ColorJitter: random brightness/contrast
    - Normalize: ImageNet mean & std
    
    Validation transforms:
    - Resize: resize ke 256
    - CenterCrop: crop 224x224 dari center
    - Normalize: sama dengan training
    
    Parameters:
    -----------
    is_training : bool, default True
        Jika True, return training transforms.
        Jika False, return validation transforms.
        
    Returns:
    --------
    torchvision.transforms.Compose
        Compose of transforms.
        
    Notes:
    ------
    - Augmentation = regularization yang efektif
    - Transfer learning lebih robust dengan augmentation
    - Normalization HARUS menggunakan ImageNet statistics
      (karena model pre-trained dengan stats tersebut)
    - RandomResizedCrop(224, scale=(0.8, 1.0)) artinya:
      ambil random crop dengan area 80%-100% dari gambar asli,
      lalu resize ke 224x224.
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])


# ===========================================================
# LATIHAN 15: Transfer Learning Mastery
# ===========================================================
"""
TARGET Learning Objectives:
   - Membandingkan feature extraction vs fine-tuning
   - Mengimplementasikan gradual unfreezing
   - Mengevaluasi impact dari pre-trained weights

PANDUAN LANGKAH-LANGKAH:

STEP 1: Baseline - Training from Scratch
-----------------------------------------
   a) Buat ResNet-18 tanpa pre-trained weights (pretrained=False)
   b) Train pada dataset kecil (misal: CIFAR-10 subset 1000 images)
   c) Record:
      - Final accuracy
      - Training time
      - Convergence speed
      
   TIPS KENAPA baseline?
     - Untuk membuktikan benefit dari transfer learning
     - Expectation: from scratch akan underfit dengan data kecil


STEP 2: Feature Extraction
--------------------------
   a) Load ResNet-18 dengan pre-trained weights (pretrained=True)
   b) Freeze semua layers kecuali classifier
   c) Train classifier pada dataset yang sama
   d) Record metrics yang sama
   
   TIPS KENAPA feature extraction?
     - Paling cepat (hanya train classifier)
     - Paling sedikit overfitting
     - Cocok untuk dataset sangat kecil (<1000 images)
     - Backbone = robust feature extractor


STEP 3: Fine-tuning dengan Unfreezing Strategy
----------------------------------------------
   a) Strategy 1: Unfreeze all at once
      - Unfreeze semua layers
      - Gunakan lr yang sangat kecil (1e-5)
      - Train semua layers bersama
      
   b) Strategy 2: Gradual unfreezing
      - Epoch 1-5: classifier only
      - Epoch 6-10: + layer4
      - Epoch 11-15: + layer3
      - Epoch 16-20: all layers
      
   c) Strategy 3: Discriminative fine-tuning
      - Set different lr untuk setiap layer group
      - Classifier: lr=1e-3
      - Layer4: lr=1e-4
      - Layer3: lr=1e-5
      - Layer2: lr=1e-6
      - Layer1: lr=1e-7
      
   d) Record metrics untuk setiap strategy


STEP 4: Comparison & Analysis
-----------------------------
   Buat comparison table:
   
   | Method | Val Acc | Training Time | Params Updated | Overfitting |
   |--------|---------|---------------|----------------|-------------|
   | Scratch|   ?     |      ?        |      All       |     ?       |
   | Feature|   ?     |      ?        |    Classifier  |     ?       |
   | Full FT|   ?     |      ?        |      All       |     ?       |
   | Gradual|   ?     |      ?        |   Progressive  |     ?       |
   | Discrim|   ?     |      ?        |      All       |     ?       |
   
   Analisis:
   - Mana yang terbaik untuk accuracy?
   - Mana yang terbaik untuk efficiency?
   - Kapan menggunakan masing-masing?
   - Impact dari dataset size?


TIPS:
   - Gunakan CIFAR-10 atau buat synthetic dataset
   - Untuk small dataset, feature extraction biasanya cukup
   - Untuk medium dataset, gradual unfreezing optimal
   - Discriminative lr memerlukan careful tuning
   - Plot learning curves untuk setiap method

PERINGATAN COMMON MISTAKES:
   - Menggunakan lr besar saat fine-tuning -> merusak pre-trained weights
   - Tidak freeze BatchNorm layers -> domain shift
   - Normalization stats yang salah (harus ImageNet)
   - Overfitting karena terlalu banyak parameters di-train
   - Tidak menggunakan augmentation untuk small dataset

TARGET EXPECTED OUTPUT:
   - Clear comparison: transfer learning >> from scratch
   - Optimal strategy untuk dataset size tertentu
   - Learning curves yang menunjukkan convergence
   - Recommendation guide untuk transfer learning
"""


# ===========================================================
# CHALLENGE: Transfer Learning untuk Industrial Inspection
# ===========================================================
"""
TARGET Learning Objectives:
   - Mengaplikasikan transfer learning ke real industrial problem
   - Mengimplementasikan multi-task learning
   - Membangun domain adaptation pipeline

PANDUAN LANGKAH-LANGKAH:

STEP 1: Dataset - Industrial Defect Detection
---------------------------------------------
   Konteks: PCB (Printed Circuit Board) defect detection
   
   a) Dataset:
      - Normal PCBs: 500 images
      - Defect types:
        * Missing component (200 images)
        * Wrong orientation (200 images)
        * Solder bridge (150 images)
        * Component damage (150 images)
      - Total: 1200 images, 5 classes
      
   b) Data augmentation (kritis untuk dataset kecil):
      - Geometric: rotation (+/-15 deg), scale (0.9-1.1)
      - Photometric: brightness, contrast, blur
      - Noise: Gaussian, salt-and-pepper
      - Cutout: random occlusion (simulate inspection angles)


STEP 2: Model Selection
-----------------------
   a) Backbone options:
      - ResNet-18 (fast, good for small data)
      - ResNet-50 (more capacity, slower)
      - EfficientNet-B0 (good accuracy/efficiency tradeoff)
      - MobileNet-V2 (for edge deployment)
      
   b) Evaluation criteria:
      - Accuracy
      - Inference time (ms per image)
      - Model size (MB)
      - FLOPs
      
   c) Pilih backbone yang optimal untuk use case


STEP 3: Training dengan Multiple Strategies
-------------------------------------------
   a) Strategy A: Feature Extraction
      - Freeze backbone, train classifier
      - Quick baseline
      
   b) Strategy B: Gradual Unfreezing
      - Unfreeze dari layer terakhir ke pertama
      - Monitor validation loss untuk setiap stage
      
   c) Strategy C: Full Fine-tuning dengan Regularization
      - Unfreeze all dengan very small lr
      - Heavy augmentation
      - Label smoothing, mixup, cutmix
      
   d) Strategy D: Multi-Task Learning
      - Primary task: defect classification
      - Auxiliary task: defect localization (heatmap)
      - Shared backbone, separate heads
      - Loss = alpha*classification_loss + beta*localization_loss


STEP 4: Domain Adaptation (opsional tapi powerful)
--------------------------------------------------
   Jika ada domain shift (synthetic -> real):
   
   a) Domain Adversarial Neural Network (DANN):
      - Domain classifier yang berusaha membedakan
        source vs target domain
      - Feature extractor berusaha "fool" domain classifier
      - Result: domain-invariant features
      
   b) Maximum Mean Discrepancy (MMD):
      - Minimize MMD antara source dan target distributions
      - Add MMD loss ke training objective


STEP 5: Evaluation & Deployment
-------------------------------
   a) Metrics:
      - Per-class precision/recall (imbalanced classes)
      - Mean Average Precision (mAP)
      - Inference time pada target hardware
      
   b) Error analysis:
      - Confusion matrix
      - Failure cases (false positives/negatives)
      - Visualisasi attention maps
      
   c) Deployment:
      - Export ke TorchScript atau ONNX
      - Quantization untuk edge deployment
      - Benchmark pada target device


TIPS:
   - EfficientNet = scaling depth, width, resolution secara compound
   - Mixup: lambda*x1 + (1-lambda)*x2, label = lambda*y1 + (1-lambda)*y2
   - CutMix: replace region dari image dengan region dari image lain
   - Label smoothing: y = 0.9 (bukan 1.0), others = 0.025
   - Domain adaptation: sangat powerful untuk synthetic-to-real

PERINGATAN COMMON MISTAKES:
   - Augmentation yang terlalu aggressive -> domain shift
   - Fine-tuning dengan lr besar -> catastrophic forgetting
   - Tidak menangani class imbalance
   - Model terlalu besar untuk deployment target
   - Tidak benchmark inference speed

TARGET EXPECTED OUTPUT:
   - Defect detection model dengan mAP > 0.85
   - Comparison: from scratch vs transfer learning
   - Optimized model untuk deployment
   - Deployment-ready artifacts (ONNX/TorchScript)

Ini adalah skill yang sangat dicari di manufacturing dan
industrial automation!
"""

print("\n" + "="*50)
print("OK Modul selesai! Lanjut ke: 05-advanced/02_nlp_transformers.py")
print("="*50)
