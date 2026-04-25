# Project 2: Klasifikasi Sinyal / Sensor

## 🎯 Tujuan
Menerapkan classical ML untuk klasifikasi sinyal/sensor data.
Ini menunjukkan domain expertise Teknik Elektro + ML skills.

## 🚀 Production-Ready Requirements
> **New:** Tunjukkan kemampuan MLOps dengan experiment tracking.

## Deliverables
1. **Signal Processing Pipeline** — preprocessing sinyal
2. **Feature Engineering** — time-domain & frequency-domain features
3. **Model Comparison** — minimal 4 algoritma
4. **MLflow Tracking** — experiments & model registry (NEW)
5. **Report** — performance & domain insights

## Dataset
Pilih salah satu:
- **EEG Motor Imagery** (BCI competition)
- **Power Quality Disturbances** (synthetic atau real)
- ** Bearing Fault Diagnosis** (NASA IMS dataset)
- **Custom:** Data dari riset/lab S2 kamu

## Checklist
### Signal Processing
- [ ] Load & visualize raw signals
- [ ] Preprocessing: filtering, denoising, normalization
- [ ] Time-domain features: RMS, crest factor, skewness, kurtosis
- [ ] Frequency-domain features: FFT, PSD, spectral centroid
- [ ] Time-frequency: STFT atau wavelet (opsional)

### Feature Engineering
- [ ] Extract statistical features per window
- [ ] Feature selection (correlation, mutual information)
- [ ] Feature scaling & normalization
- [ ] Dimensionality reduction (PCA untuk visualization)

### Modeling
- [ ] Logistic Regression
- [ ] Random Forest / XGBoost
- [ ] SVM dengan kernel RBF
- [ ] k-NN
- [ ] Proper cross-validation (stratified, time-series aware)
- [ ] Hyperparameter tuning (GridSearchCV)

### MLOps (NEW)
- [ ] **MLflow tracking** — log semua experiments
- [ ] **Model registry** — version dan stage (staging/production)
- [ ] **Hydra config** — hyperparameter di config file
- [ ] **Reproducibility** — seed, environment, dependencies

### Documentation
- [ ] Domain knowledge explanation (EE context)
- [ ] Feature importance analysis
- [ ] Confusion matrix per class
- [ ] Error analysis: which classes are confused?

## Hasil
*(Tulis hasil di sini setelah selesai)*

### Model Comparison (MLflow)
| Model | Accuracy | F1-Score | Precision | Recall | MLflow Run ID |
|-------|----------|----------|-----------|--------|---------------|
| | | | | | |

### Key Insights
1. 
2. 
3. 
