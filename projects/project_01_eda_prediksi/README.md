# Project 1: Exploratory Data Analysis & Prediction

## 🎯 Tujuan
Menerapkan semua konsep dari Fase 1 & 2 dalam satu pipeline ML end-to-end.
Ini adalah portfolio piece pertama — tunjukkan kemampuan data storytelling.

## 🚀 Production-Ready Requirements
> **New:** Project ini harus bisa di-demo dan dijelaskan dalam 5 menit.

## Deliverables
1. **EDA Notebook** — eksplorasi data lengkap dengan visualisasi
2. **Model Pipeline** — dari raw data sampai prediction
3. **Prediction API** — FastAPI endpoint (NEW — leverage backend exp!)
4. **Report** — insight dan kesimpulan (tulis di file ini!)

## Dataset
Pilih salah satu:
- UCI Power Consumption Dataset (⭐ recommended untuk EE)
- California Housing (sklearn)
- Custom dataset dari domain Teknik Elektro

## Checklist
### Data Exploration
- [ ] Load & inspect data
- [ ] Handle missing values & outliers
- [ ] Visualisasi (minimal 6 plot berbeda)
- [ ] Feature engineering (minimal 3 fitur baru)
- [ ] Statistical summary & correlation analysis

### Modeling
- [ ] Train linear regression FROM SCRATCH
- [ ] Bandingkan dengan sklearn
- [ ] Proper evaluation (train/test split, cross-validation)
- [ ] Residual analysis
- [ ] Hyperparameter tuning (untuk sklearn version)

### Production Elements (NEW)
- [ ] **FastAPI endpoint** `/predict` dengan input validation
- [ ] **Dockerfile** untuk containerization
- [ ] **Unit tests** untuk data pipeline
- [ ] **README** dengan setup instructions
- [ ] **Git repository** dengan clean commit history

### Documentation
- [ ] Tulis insight per langkah (bukan cuma kode!)
- [ ] Business interpretation dari findings
- [ ] Recommendation untuk stakeholder

## Hasil
*(Tulis hasil di sini setelah selesai)*

### Model Performance
| Model | RMSE | MAE | R² | Notes |
|-------|------|-----|-----|-------|
| Scratch LR | | | | |
| Sklearn LR | | | | |
| Sklearn Ridge | | | | |

### Key Insights
1. 
2. 
3. 

### API Usage
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.0, "feature2": 2.0}'
```
