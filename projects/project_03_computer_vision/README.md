# Project 3: Computer Vision / Signal Deep Learning

## 🎯 Tujuan
Menerapkan deep learning (CNN/RNN/Transformer) untuk klasifikasi sinyal atau image.
Tunjukkan kemampuan PyTorch + deployment.

## 🚀 Production-Ready Requirements
> **New:** Deploy dengan Docker dan buat interactive demo.

## Deliverables
1. **Model Architecture** — custom CNN/RNN/Transformer
2. **Training Pipeline** — proper training with PyTorch
3. **Evaluation** — comprehensive metrics & visualizations
4. **Deployment** — FastAPI + Docker (NEW)
5. **Demo App** — Streamlit/Gradio (NEW)

## Dataset
Pilih salah satu:
- **MNIST / Fashion-MNIST** (untuk quick iteration)
- **CIFAR-10** (medium complexity)
- **Custom:** Spectrograms dari sinyal EE (⭐ recommended!)
- **Industrial Defect Detection** ( MVTec AD — challenging)

## Checklist
### Data Preparation
- [ ] Dataset preparation (images atau sinyal)
- [ ] Data augmentation (rotation, flip, noise)
- [ ] Train/val/test split yang proper
- [ ] DataLoader dengan batching & shuffling

### Model Development
- [ ] Implement CNN (2D untuk images ATAU 1D untuk sinyal)
- [ ] Training loop dengan proper validation
- [ ] Learning curves (train vs val) → save as PNG
- [ ] Early stopping & checkpoint saving
- [ ] Confusion matrix
- [ ] Visualize learned filters / feature maps
- [ ] Bandingkan: from scratch vs transfer learning

### Production (NEW)
- [ ] **Model export** — ONNX atau TorchScript
- [ ] **FastAPI serving** — `/predict` endpoint untuk images
- [ ] **Docker containerization** — multi-stage build
- [ ] **Streamlit demo** — upload image → prediction + confidence
- [ ] **Model versioning** — save dengan timestamp & config

### Advanced (Opsional)
- [ ] Grad-CAM untuk explainability
- [ ] Model quantization (INT8)
- [ ] Batch inference optimization

## Hasil
*(Tulis hasil di sini setelah selesai)*

### Training Results
| Model | Val Accuracy | Val F1 | Epochs | Training Time |
|-------|-------------|--------|--------|---------------|
| Custom CNN | | | | |
| ResNet18 (Transfer) | | | | |
| EfficientNet (Transfer) | | | | |

### Deployment Info
- API endpoint: `http://localhost:8000/predict`
- Docker image: `docker run -p 8000:8000 cv-project`
- Streamlit demo: `streamlit run demo.py`
