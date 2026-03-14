# Self-Assessment: Siap Apply Kerjaan Machine Learning?

## Filosofi Assessment Ini

Assessment ini bukan ujian akademik. Ini adalah **reality check** yang mensimulasikan
apa yang benar-benar ditanyakan dan diharapkan di dunia kerja ML:

1. **Technical Interview** -- bisa jelaskan konsep tanpa buka catatan
2. **Coding Challenge** -- bisa implementasi dari nol dalam waktu terbatas
3. **Take-Home Assignment** -- bisa deliver end-to-end solution
4. **System Design** -- bisa desain ML pipeline untuk problem nyata

### Cara Menggunakan

- Kerjakan assessment **TANPA buka materi, Google, atau ChatGPT**
- Set timer untuk setiap soal (simulasi interview)
- Scoring: **Jujur ke diri sendiri.** Kalau ragu, berarti belum PD.
- Setiap fase punya 3 level:
  - **Level 1 (Bisa Dikerjakan)** = minimum requirement
  - **Level 2 (Bisa Dijelaskan)** = siap interview
  - **Level 3 (Bisa Improvisasi)** = standout candidate

### Scoring Guide

Untuk setiap soal, beri nilai diri sendiri:

- 0 = Blank, tidak tahu harus mulai dari mana
- 1 = Tahu konsepnya tapi tidak bisa implementasi
- 2 = Bisa implementasi tapi masih perlu googling beberapa hal
- 3 = Bisa implementasi sendiri dari nol + jelaskan ke orang lain

### Target "PD" untuk Apply:

| Level Posisi         | Minimum Score per Fase | Total Minimum |
| -------------------- | ---------------------- | ------------- |
| Junior ML Engineer   | Avg 2.0 di Fase 1-3    | 60% overall   |
| ML Engineer          | Avg 2.5 di Fase 1-4    | 70% overall   |
| Senior ML Engineer   | Avg 2.5 di Fase 1-5    | 75% overall   |
| ML Lead / Specialist | Avg 2.5 di Fase 1-6    | 80% overall   |

---

## FASE 0: Setup & Tools

### Milestone: "Saya bisa setup ML project dari nol"

**Checklist (Pass/Fail):**

- [ ] Bisa setup Python environment (venv/conda) tanpa tutorial
- [ ] Bisa install semua dependency dari requirements.txt
- [ ] Bisa setup git repo, buat .gitignore yang proper untuk ML project
- [ ] Bisa jalankan Jupyter notebook dan Python script interaktif

> **Kapan lanjut:** Semua checklist harus centang. Ini fondasi, tidak ada kompromi.

---

## FASE 1: Fondasi Data

### Milestone: "Saya bisa manipulasi data apapun dengan percaya diri"

### 1.1 NumPy Essentials

**Level 1 -- Bisa Dikerjakan (timer: 30 menit):**

- [ ] Buat array 2D (5x3), hitung mean per kolom, normalisasi z-score -- TANPA loop
- [ ] Implementasi matrix multiplication manual, bandingkan dengan np.dot
- [ ] Buat boolean mask: filter semua elemen > mean + 1 std

**Level 2 -- Bisa Dijelaskan:**

- [ ] Jelaskan broadcasting rules NumPy: kapan 2 array bisa di-broadcast?
- [ ] Kenapa vectorized operation lebih cepat dari loop Python? (hint: C backend, memory layout)
- [ ] Apa bedanya .copy() vs view? Kapan ini jadi masalah?

**Level 3 -- Bisa Improvisasi (timer: 45 menit):**

- [ ] Implementasi softmax yang numerically stable (tanpa overflow)
- [ ] Implementasi cosine similarity matrix untuk batch of vectors
- [ ] Implementasi one-hot encoding pakai NumPy saja

**Skor: \_\_\_/27**

### 1.2 Pandas Essentials

**Level 1 -- Bisa Dikerjakan (timer: 30 menit):**

- [ ] Load CSV, cek missing values, handle dengan strategy yang tepat
- [ ] Groupby + aggregasi: hitung mean, std, count per kategori
- [ ] Buat 3 feature baru: rolling mean, lag feature, rate of change

**Level 2 -- Bisa Dijelaskan:**

- [ ] Kapan pakai fillna(mean) vs interpolate vs drop? Trade-off masing-masing?
- [ ] Jelaskan bedanya loc vs iloc vs at vs iat
- [ ] Kenapa concat bisa lebih baik dari append dalam loop?

**Level 3 -- Bisa Improvisasi (timer: 45 menit):**

- [ ] Tulis function: input = raw DataFrame, output = ML-ready (X, y) dengan semua preprocessing
- [ ] Handle mixed types: numeric scaling + categorical encoding dalam satu pipeline
- [ ] Detect dan handle outlier menggunakan IQR method

**Skor: \_\_\_/27**

### 1.3 Visualisasi

**Level 1 -- Bisa Dikerjakan (timer: 20 menit):**

- [ ] Buat histogram + boxplot untuk distribusi data
- [ ] Buat correlation heatmap
- [ ] Buat scatter plot dengan warna berdasarkan target class

**Level 2 -- Bisa Dijelaskan:**

- [ ] Kapan pakai histogram vs KDE vs boxplot?
- [ ] Bagaimana membaca correlation heatmap -- apa yang dicari?
- [ ] Kenapa visualisasi penting sebelum modeling? Beri 3 contoh insight yang hanya bisa dilihat dari plot

**Level 3 -- Bisa Improvisasi (timer: 30 menit):**

- [ ] Plot decision boundary untuk arbitrary 2D classifier
- [ ] Buat multi-panel figure (2x3 subplot) untuk model evaluation report
- [ ] Plot frequency spectrum dari time-domain signal (FFT)

**Skor: \_\_\_/27**

### FASE 1 Total: \_\_\_/81

---

## FASE 2: ML dari Nol

### Milestone: "Saya mengerti apa yang terjadi di balik sklearn.fit()"

### 2.1 Linear Regression from Scratch

**Level 1 -- Bisa Dikerjakan (timer: 45 menit):**

- [ ] Implementasi linear regression dengan Normal Equation: (X^T X)^-1 X^T y
- [ ] Implementasi linear regression dengan Gradient Descent
- [ ] Hitung R-squared, MSE, MAE secara manual

**Level 2 -- Bisa Dijelaskan:**

- [ ] Kapan Normal Equation lebih baik dari GD? Kapan sebaliknya?
- [ ] Apa yang terjadi kalau learning rate terlalu besar? Terlalu kecil?
- [ ] Jelaskan overfitting pada polynomial regression -- kenapa degree tinggi berbahaya?
- [ ] Apa bedanya Ridge (L2) vs Lasso (L1)? Kapan pakai yang mana?

**Level 3 -- Bisa Improvisasi (timer: 60 menit):**

- [ ] Tambahkan L2 regularization ke GD implementation
- [ ] Implementasi mini-batch GD dengan learning rate decay
- [ ] Implementasi k-fold cross-validation secara manual

**Skor: \_\_\_/30**

### 2.2 Logistic Regression from Scratch

**Level 1 -- Bisa Dikerjakan (timer: 45 menit):**

- [ ] Implementasi sigmoid function (numerically stable)
- [ ] Implementasi binary cross-entropy loss
- [ ] Implementasi logistic regression: fit dengan GD, predict_proba, predict

**Level 2 -- Bisa Dijelaskan:**

- [ ] Kenapa MSE tidak cocok untuk classification? (hint: non-convex loss surface)
- [ ] Jelaskan precision vs recall trade-off. Beri contoh skenario dimana masing-masing lebih penting
- [ ] Kenapa accuracy bisa misleading pada imbalanced data?

**Level 3 -- Bisa Improvisasi (timer: 60 menit):**

- [ ] Extend ke multiclass: implementasi softmax regression
- [ ] Implementasi ROC curve dan AUC dari scratch
- [ ] Handle imbalanced data: class weighting di loss function

**Skor: \_\_\_/27**

### 2.3 Gradient Descent Deep Dive

**Level 1 -- Bisa Dikerjakan (timer: 30 menit):**

- [ ] Implementasi vanilla GD, SGD, dan mini-batch GD
- [ ] Plot loss curve untuk membandingkan ketiganya
- [ ] Implementasi momentum

**Level 2 -- Bisa Dijelaskan:**

- [ ] Gambar/jelaskan kenapa SGD lebih "zig-zag" tapi bisa escape local minima
- [ ] Apa peran momentum? Analogikan dengan fisika/kontrol
- [ ] Jelaskan Adam: apa yang di-track? Kenapa butuh bias correction?

**Level 3 -- Bisa Improvisasi (timer: 45 menit):**

- [ ] Implementasi Adam optimizer dari scratch
- [ ] Implementasi learning rate scheduler: cosine annealing
- [ ] Visualisasi trajectory optimizer di 2D loss surface (contour plot + path)

**Skor: \_\_\_/27**

### 2.4 Model Evaluation

**Level 1 -- Bisa Dikerjakan (timer: 30 menit):**

- [ ] Hitung confusion matrix, precision, recall, F1 secara manual
- [ ] Implementasi k-fold cross-validation
- [ ] Buat proper train/test split (dan jelaskan kenapa tidak boleh random untuk time series)

**Level 2 -- Bisa Dijelaskan:**

- [ ] Sebutkan 5 sumber data leakage dan cara mencegahnya
- [ ] Kapan pakai stratified k-fold vs regular k-fold vs time series split?
- [ ] Jelaskan bias-variance tradeoff: apa yang berubah kalau model makin complex?

**Level 3 -- Bisa Improvisasi (timer: 45 menit):**

- [ ] Implementasi nested cross-validation (inner loop tuning, outer loop evaluation)
- [ ] Implementasi grid search dengan proper preprocessing di dalam CV loop
- [ ] Jelaskan kenapa normalisasi harus dilakukan SETELAH split (dan demonstrasikan dampaknya)

**Skor: \_\_\_/27**

### FASE 2 Total: \_\_\_/111

---

## FASE 3: Classical ML

### Milestone: "Saya bisa pilih, tune, dan evaluate model yang tepat untuk problem apapun"

### 3.1 Supervised Learning

**Level 1 -- Bisa Dikerjakan (timer: 45 menit):**

- [ ] Train dan evaluate: Decision Tree, Random Forest, SVM, Gradient Boosting menggunakan sklearn
- [ ] Plot feature importance dari Random Forest
- [ ] Jalankan GridSearchCV untuk tuning hyperparameter

**Level 2 -- Bisa Dijelaskan:**

- [ ] Jelaskan bedanya bagging (Random Forest) vs boosting (XGBoost). Gambar diagram.
- [ ] Kapan SVM lebih baik dari tree-based? Kapan sebaliknya?
- [ ] Apa arti "max_depth" di decision tree? Hubungannya dengan overfitting?
- [ ] Kenapa Random Forest lebih robust dari single Decision Tree?

**Level 3 -- Bisa Improvisasi (timer: 60 menit):**

- [ ] Desain experiment: bandingkan 5+ model dengan proper CV, hasilkan tabel + plot
- [ ] Analisis: model mana yang cocok untuk data kecil? Data besar? Banyak fitur noisy?
- [ ] Implementasi simple Bayesian Optimization untuk hyperparameter tuning

**Skor: \_\_\_/30**

### 3.2 Unsupervised Learning

**Level 1 -- Bisa Dikerjakan (timer: 30 menit):**

- [ ] Run K-Means, tentukan k optimal (elbow + silhouette)
- [ ] Run PCA, plot explained variance, pilih jumlah komponen
- [ ] Run Isolation Forest untuk anomaly detection

**Level 2 -- Bisa Dijelaskan:**

- [ ] Kapan K-Means gagal? (hint: non-convex clusters) Apa alternatifnya?
- [ ] PCA: apa yang direpresentasikan oleh eigenvalue dan eigenvector?
- [ ] Bedanya PCA vs t-SNE: kapan pakai yang mana?

**Level 3 -- Bisa Improvisasi (timer: 45 menit):**

- [ ] Implementasi K-Means dari scratch (termasuk K-means++ initialization)
- [ ] Kombinasikan PCA + clustering: reduce dimensi lalu cluster
- [ ] Anomaly detection pipeline: preprocess, model, evaluate, visualize

**Skor: \_\_\_/27**

### 3.3 Feature Engineering

**Level 1 -- Bisa Dikerjakan (timer: 30 menit):**

- [ ] Apply: StandardScaler, MinMaxScaler, log transform pada data skewed
- [ ] Buat time-domain features: mean, std, min, max, skewness, kurtosis
- [ ] Buat frequency-domain features: dominant frequency, spectral energy

**Level 2 -- Bisa Dijelaskan:**

- [ ] Kapan pakai StandardScaler vs MinMaxScaler vs RobustScaler?
- [ ] Kenapa feature engineering sering lebih impactful dari model selection?
- [ ] Jelaskan feature selection methods: filter vs wrapper vs embedded

**Level 3 -- Bisa Improvisasi (timer: 60 menit):**

- [ ] Desain domain-specific features untuk electrical engineering data
      (power factor, THD, crest factor, I^2t thermal, vibration severity)
- [ ] Build feature pipeline: raw data -> features -> selection -> ready for ML
- [ ] Jalankan dan bandingkan 3 feature selection methods, analisis hasilnya

**Skor: \_\_\_/27**

### FASE 3 Total: \_\_\_/84

---

## FASE 4: Deep Learning

### Milestone: "Saya bisa build, train, dan debug neural network dari zero"

### 4.1 Neural Network from Scratch

**Level 1 -- Bisa Dikerjakan (timer: 60 menit):**

- [ ] Implementasi forward propagation: input -> hidden (ReLU) -> output (sigmoid)
- [ ] Implementasi backpropagation: hitung gradient untuk setiap layer
- [ ] Training loop: full pipeline dari init weights -> forward -> loss -> backward -> update

**Level 2 -- Bisa Dijelaskan:**

- [ ] Jelaskan chain rule dalam konteks backprop. Gambar computation graph.
- [ ] Apa itu vanishing gradient? Kenapa terjadi dengan sigmoid di hidden layer?
- [ ] He vs Xavier initialization: kapan pakai masing-masing?

**Level 3 -- Bisa Improvisasi (timer: 45 menit):**

- [ ] Tambahkan batch normalization ke MLP dari scratch
- [ ] Tambahkan dropout dari scratch (beda behavior train vs eval)
- [ ] Implementasi gradient checking untuk verifikasi backprop

**Skor: \_\_\_/27**

### 4.2 PyTorch Fundamentals

**Level 1 -- Bisa Dikerjakan (timer: 30 menit):**

- [ ] Definisi model pakai nn.Module + nn.Sequential
- [ ] Tulis training loop lengkap: DataLoader, optimizer, loss, model.train(), model.eval()
- [ ] Save dan load model (state_dict)

**Level 2 -- Bisa Dijelaskan:**

- [ ] Apa fungsi requires_grad? Bagaimana autograd bekerja?
- [ ] Kenapa perlu model.eval() dan torch.no_grad() saat inference?
- [ ] Jelaskan DataLoader: batch_size, shuffle, num_workers, pin_memory

**Level 3 -- Bisa Improvisasi (timer: 45 menit):**

- [ ] Tulis custom Dataset class untuk data sendiri
- [ ] Implementasi custom loss function (e.g., Focal Loss)
- [ ] Implementasi LR scheduler: CosineAnnealing + warmup

**Skor: \_\_\_/27**

### 4.3 CNN

**Level 1 -- Bisa Dikerjakan (timer: 45 menit):**

- [ ] Build CNN: Conv2d -> ReLU -> BatchNorm -> MaxPool -> FC
- [ ] Train pada MNIST atau CIFAR-10 dengan data augmentation
- [ ] Plot training/validation loss curve dan confusion matrix

**Level 2 -- Bisa Dijelaskan:**

- [ ] Apa yang dilakukan convolutional layer? Analogikan dengan filter di signal processing.
- [ ] Bagaimana menghitung output size: (W - K + 2P) / S + 1
- [ ] Kenapa CNN lebih baik dari MLP untuk image/signal? (hint: parameter sharing, translation invariance)
- [ ] Jelaskan stride, padding, dilation -- efek masing-masing

**Level 3 -- Bisa Improvisasi (timer: 60 menit):**

- [ ] Build 1D CNN untuk klasifikasi sinyal (bukan gambar)
- [ ] Visualisasi learned filters dan feature maps
- [ ] Tambahkan residual connections (skip connections)

**Skor: \_\_\_/30**

### 4.4 RNN & Time Series

**Level 1 -- Bisa Dikerjakan (timer: 45 menit):**

- [ ] Build LSTM model dengan PyTorch untuk time series prediction
- [ ] Buat sliding window dataset (sequence -> next value)
- [ ] Train dan evaluate: plot actual vs predicted

**Level 2 -- Bisa Dijelaskan:**

- [ ] Gambar arsitektur LSTM cell: forget gate, input gate, output gate
- [ ] Kenapa LSTM mengatasi vanishing gradient? Apa peran cell state?
- [ ] Apa bedanya LSTM vs GRU? Kapan pilih yang mana?

**Level 3 -- Bisa Improvisasi (timer: 60 menit):**

- [ ] Build Transformer model untuk time series (bukan RNN)
- [ ] Multi-step forecasting: predict banyak step ke depan
- [ ] Bandingkan LSTM vs GRU vs Transformer: akurasi dan speed

**Skor: \_\_\_/27**

### FASE 4 Total: \_\_\_/111

---

## FASE 5: Advanced Topics

### Milestone: "Saya bisa leverage state-of-the-art dan adaptasi ke problem baru"

### 5.1 Transfer Learning

**Level 1 -- Bisa Dikerjakan (timer: 30 menit):**

- [ ] Load pretrained ResNet, ganti final layer, train pada custom dataset
- [ ] Freeze/unfreeze layer: feature extraction vs fine-tuning
- [ ] Plot learning curves, bandingkan akurasi strategi yang berbeda

**Level 2 -- Bisa Dijelaskan:**

- [ ] Kapan pakai feature extraction vs fine-tuning? Buat decision tree.
- [ ] Kenapa model pretrained di ImageNet berguna untuk task lain?
- [ ] Apa risiko fine-tuning pada dataset kecil? Bagaimana mitigasinya?

**Level 3 -- Bisa Improvisasi (timer: 45 menit):**

- [ ] Implementasi gradual unfreezing strategy
- [ ] Transfer learning untuk domain non-image (sinyal 1D, time series)
- [ ] Domain adaptation: source domain berbeda dari target domain

**Skor: \_\_\_/27**

### 5.2 NLP & Transformers

**Level 1 -- Bisa Dikerjakan (timer: 45 menit):**

- [ ] Implementasi self-attention dari scratch (Q, K, V, scaled dot-product)
- [ ] Fine-tune BERT via Hugging Face untuk text classification
- [ ] Visualisasi attention weights

**Level 2 -- Bisa Dijelaskan:**

- [ ] Jelaskan self-attention langkah per langkah. Kenapa "scaled"?
- [ ] Apa bedanya encoder-only (BERT) vs decoder-only (GPT) vs encoder-decoder (T5)?
- [ ] Kenapa Transformer menggantikan RNN? Apa kelemahannya?
- [ ] Jelaskan positional encoding: kenapa perlu? Bagaimana cara kerjanya?

**Level 3 -- Bisa Improvisasi (timer: 60 menit):**

- [ ] Build Transformer classifier dari scratch (bukan pakai library)
- [ ] Adaptasi Transformer untuk non-text: sinyal, time series
- [ ] Implementasi Transformer dengan patch embedding untuk 1D signal

**Skor: \_\_\_/30**

### 5.3 Generative Models

**Level 1 -- Bisa Dikerjakan (timer: 45 menit):**

- [ ] Build dan train VAE pada MNIST
- [ ] Build dan train GAN sederhana pada MNIST
- [ ] Generate samples, plot hasil

**Level 2 -- Bisa Dijelaskan:**

- [ ] VAE: jelaskan ELBO, reparameterization trick, KL divergence
- [ ] GAN: jelaskan adversarial training. Kenapa training GAN sulit?
- [ ] Bedanya VAE vs GAN: kelebihan dan kekurangan masing-masing

**Level 3 -- Bisa Improvisasi (timer: 45 menit):**

- [ ] Gunakan generative model untuk data augmentation (generate synthetic data)
- [ ] Latent space interpolation: generate smooth transition antara 2 sample
- [ ] Conditional generation: generate sample dari class tertentu

**Skor: \_\_\_/27**

### FASE 5 Total: \_\_\_/84

---

## FASE 6: Expert & Production

### Milestone: "Saya bisa deliver ML solution ke production"

### 6.1 Paper Implementation

**Level 1 -- Bisa Dikerjakan:**

- [ ] Baca dan pahami 1 ML paper lengkap (e.g., "Attention Is All You Need")
- [ ] Identifikasi: problem, method, architecture, experiment setup, results
- [ ] Tulis ringkasan 1 halaman dengan bahasa sendiri

**Level 2 -- Bisa Dijelaskan:**

- [ ] Jelaskan paper yang sudah dibaca seolah-olah presentasi 10 menit
- [ ] Identifikasi limitation dan potensi improvement
- [ ] Bandingkan dengan approach lain untuk problem yang sama

**Level 3 -- Bisa Improvisasi:**

- [ ] Reimplementasi paper dari scratch (minimal simplified version)
- [ ] Reproduce key results (meski dengan dataset yang lebih kecil)
- [ ] Tulis blog post / laporan tentang pengalaman implementasi

**Skor: \_\_\_/27**

### 6.2 MLOps

**Level 1 -- Bisa Dikerjakan (timer: 60 menit):**

- [ ] Setup experiment tracking (MLflow atau W&B): log metrics, parameters, models
- [ ] Tulis unit test untuk data preprocessing dan model pipeline
- [ ] Buat configuration management (Hydra atau simple YAML)

**Level 2 -- Bisa Dijelaskan:**

- [ ] Kenapa experiment tracking penting? Masalah apa yang dipecahkan?
- [ ] Apa itu data versioning? Kenapa git saja tidak cukup untuk data?
- [ ] Jelaskan CI/CD untuk ML: bedanya dengan CI/CD software biasa

**Level 3 -- Bisa Improvisasi (timer: 60 menit):**

- [ ] Setup DVC untuk version data dan model
- [ ] Buat GitHub Actions pipeline: test -> train -> evaluate -> deploy
- [ ] Implementasi model registry: versioning model + metadata

**Skor: \_\_\_/27**

### 6.3 Production & Deployment

**Level 1 -- Bisa Dikerjakan (timer: 60 menit):**

- [ ] Serve model via FastAPI: endpoint predict yang menerima JSON, return prediksi
- [ ] Buat Dockerfile: containerize ML app
- [ ] Buat Streamlit/Gradio demo untuk model

**Level 2 -- Bisa Dijelaskan:**

- [ ] Apa itu model optimization? Jelaskan quantization dan pruning
- [ ] Kenapa monitoring penting di production? Apa itu data drift?
- [ ] Jelaskan perbedaan batch inference vs real-time inference

**Level 3 -- Bisa Improvisasi (timer: 90 menit):**

- [ ] Implementasi A/B testing framework sederhana
- [ ] Setup monitoring: detect data drift, alert kalau performance turun
- [ ] End-to-end: data -> preprocess -> train -> API -> Docker -> monitoring

**Skor: \_\_\_/27**

### FASE 6 Total: \_\_\_/81

---

## GRAND ASSESSMENT: Interview Simulation

Ini adalah assessment final yang mensimulasikan real interview.

### Simulasi 1: Coding Challenge (timer: 90 menit)

Tanpa library (hanya NumPy), implementasi:

- [ ] Logistic Regression dengan mini-batch GD dan L2 regularization
- [ ] K-fold cross-validation
- [ ] Classification report (precision, recall, F1 per class)
- [ ] Plot: learning curve, confusion matrix, ROC curve

### Simulasi 2: System Design (timer: 60 menit)

Desain ML system untuk: **"Predictive maintenance untuk 1000 mesin industri"**

- [ ] Data: sensor apa yang dipakai? Sampling rate? Storage?
- [ ] Feature engineering: fitur apa yang diekstrak?
- [ ] Model: arsitektur apa? Kenapa? Training strategy?
- [ ] Deployment: bagaimana serving? Monitoring? Retraining?
- [ ] Buat diagram arsitektur end-to-end

### Simulasi 3: Debugging Challenge (timer: 45 menit)

Diberikan ML pipeline yang "broken":

- [ ] Model overfit (99% train, 60% test) -- diagnosa dan fix
- [ ] Ada data leakage -- identifikasi sumbernya
- [ ] Learning rate terlalu besar -- kenali dari symptoms dan fix
- [ ] Imbalanced data tidak di-handle -- perbaiki pipeline

### Simulasi 4: Paper Discussion (timer: 30 menit)

Pilih 1 paper yang pernah dibaca:

- [ ] Jelaskan problem, method, dan key contribution dalam 5 menit
- [ ] Jawab pertanyaan "Apa limitation utama approach ini?"
- [ ] Jawab pertanyaan "Bagaimana kamu akan improve paper ini?"

---

## SCORING SUMMARY

Isi setelah selesai assessment:

| Fase                  | Skor      | Max     | Persentase  |
| --------------------- | --------- | ------- | ----------- |
| Fase 0: Setup         | Pass/Fail | -       | -           |
| Fase 1: Fondasi Data  | \_\_\_    | 81      | \_\_\_%     |
| Fase 2: ML dari Nol   | \_\_\_    | 111     | \_\_\_%     |
| Fase 3: Classical ML  | \_\_\_    | 84      | \_\_\_%     |
| Fase 4: Deep Learning | \_\_\_    | 111     | \_\_\_%     |
| Fase 5: Advanced      | \_\_\_    | 84      | \_\_\_%     |
| Fase 6: Expert        | \_\_\_    | 81      | \_\_\_%     |
| **TOTAL**             | \_\_\_    | **552** | **\_\_\_**% |

### Interpretasi:

| Score Range | Verdict               | Action                                           |
| ----------- | --------------------- | ------------------------------------------------ |
| < 50%       | Belum siap            | Fokus review materi, ulangi exercise             |
| 50-60%      | Bisa apply Junior pos | Perkuat weakness area, mulai bikin portfolio     |
| 60-70%      | Ready untuk interview | Latih soft skill: jelaskan konsep, design system |
| 70-80%      | Competitive candidate | Fokus ke project portfolio & networking          |
| > 80%       | Strong candidate      | Apply dengan PD, negosiasi gaji dengan data      |

### Area yang Perlu Diperkuat:

Setelah scoring, identifikasi 3 area terlemah:

1. ***
2. ***
3. ***

Action plan:

- Area 1: ******\_\_\_******
- Area 2: ******\_\_\_******
- Area 3: ******\_\_\_******

---

## Tips Interview ML yang Sering Muncul

### Pertanyaan yang Hampir Selalu Ditanya:

1. "Jelaskan bias-variance tradeoff"
2. "Apa bedanya L1 vs L2 regularization?"
3. "Bagaimana kamu handle imbalanced data?"
4. "Jelaskan proses kamu dari data mentah sampai model deployed"
5. "Ceritakan project ML yang pernah kamu kerjakan, apa challengenya?"

### Yang Bikin Interviewer Impressed:

- Bisa jelaskan KENAPA, bukan hanya APA
- Punya intuisi: "untuk data seperti ini, saya akan coba X dulu karena..."
- Jujur kalau tidak tahu, tapi bisa reason through
- Punya portfolio project yang end-to-end (bukan cuma notebook)
- Bisa diskusi trade-off (akurasi vs latency, complexity vs interpretability)

### Red Flags di Interview:

- Tidak bisa jelaskan apa yang dilakukan model yang dipakai
- Hanya bisa pakai sklearn tanpa tahu apa yang terjadi di dalamnya
- Tidak pernah deploy model atau buat API
- Tidak familiar dengan version control atau experiment tracking
- Jawaban template tanpa konteks spesifik
