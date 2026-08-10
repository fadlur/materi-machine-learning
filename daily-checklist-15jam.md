# 📆 Daily Checklist — 1.5 Jam/Hari (Dari NumPy → Job-Ready)

> **Untuk:** S2 Teknik Elektro yang sedang transisi ke ML/AI Engineer, dengan komitmen **1.5 jam per hari**.
> **Posisi sekarang:** Sudah sampai materi **NumPy** (asumsi: `00-setup/setup_environment.py` sudah jalan & semua package OK).
> **Filosofi tetap sama:** _Build First, Library Later_ · _Deliberate Practice_ · _Project-Driven_.

---

## ⚠️ Jujur soal Timeline

`90-day-action-plan.md` di-reka dengan asumsi **6–8 jam/hari**. Karena kamu komitmen **1.5 jam/hari**, timeline realistisnya jadi:

| Rencana lama | Rencana baru (1.5 jam/hari) |
| ------------ | --------------------------- |
| 90 hari      | **±180 hari (±6 bulan)**    |
| ~7 jam/hari  | 90 menit/hari (konsisten)   |

**Ini normal dan sehat.** Kualitas > kecepatan. Kuncinya: **konsistensi harian**, bukan durasi. Semua tanggal sasaran di bawah adalah perkiraan — sesuaikan dengan ritmemu.

> **Hubungan dengan file lain:**
>
> - `90-day-action-plan.md` → sasaran besar & milestone (tetap jadi kompas).
> - `daily-checklist.md` → template **rutinitas harian** (sosmed/digital hygiene dll). Sesuaikan bloknya jadi 90 menit.
> - **File ini** → **jadwal konten per hari** (apa yang dikerjakan tiap hari).

---

## 🎯 Cara Pakai 90 Menit (Setiap Hari Belajar)

```
⏱️ 00:00–00:10  REVIEW  — baca ulang catatan/commit kemarin (2 menit cek git log)
⏱️ 00:10–00:55  BELAJAR — baca teori + JALANKAN semua contoh kode di bagian target
⏱️ 00:55–01:20  PRAKTEK — kerjakan exercise/challenge, coba modifikasi kode sendiri
⏱️ 01:20–01:30  RECAP   — tulis 3 takeaways, tandai yang masih bingung, git commit
```

### Aturan Main (wajib dibaca sekali)

- **Rule 1 — Jalankan semua kode.** Jangan cuma baca. Setiap contoh harus di-run sendiri.
- **Rule 2 — Exercise dijawab dulu.** Jalankan `python <modul>.py --exercise` → target semua **PASS**.
- **Rule 3 — Stuck >20 menit?** Buka `solutions/` untuk arah (pahami, jangan copy-paste buta).
- **Rule 4 — 1 commit per hari belajar** dengan pesan deskriptif, misal `fase3: implementasi SVM`.
- **Rule 5 — Hari Rest = istirahat beneran.** Kalau tertinggal, baru pakai hari rest untuk catch-up.
- **Rule 6 — Catatan dalam Bahasa Indonesia**, istilah kode tetap Inggris.
- **Rule 7 — Tiap akhir fase: review + push ke GitHub** (fase harus "hijau" dulu sebelum lanjut).

---

## 📊 Ringkasan Timeline

| Fase                                | Rentang Hari | Estimasi    | Milestone                                    |
| ----------------------------------- | ------------ | ----------- | -------------------------------------------- |
| 1. Fondasi Data (NumPy→Visualisasi) | Day 1–24     | ~3.5 minggu | Portfolio #1: EDA Notebook                   |
| 2. ML dari Nol                      | Day 25–48    | ~3.5 minggu | Portfolio #2: Prediction API                 |
| 3. Classical ML                     | Day 49–72    | ~3.5 minggu | Portfolio #3: Signal Classification + MLflow |
| 4. Deep Learning                    | Day 73–102   | ~4.5 minggu | Portfolio #4: CV + Docker                    |
| 5. Advanced (TL, NLP, Gen)          | Day 103–126  | ~3.5 minggu | Portfolio #5: NLP + Demo                     |
| 6. Expert Roadmap                   | Day 127–131  | ~1 minggu   | Paham paper + MLOps + LLM                    |
| 7. Production ML                    | Day 132–146  | ~2 minggu   | Feature store, monitoring, RAG               |
| 8. Flagship + Career Prep           | Day 147–180  | ~5 minggu   | FLAGSHIP project + job-ready                 |

---

# 🗓️ FASE 1 — Fondasi Data (Day 1–24)

## 📦 Blok 1: NumPy (`01-fondasi-data/01_numpy_essentials.py`)

| Day | Topik                                             | Deliverable hari ini                                                |
| --- | ------------------------------------------------- | ------------------------------------------------------------------- |
| 1   | BAGIAN 1 — Array Creation & Basic Operations      | [ ] Semua contoh jalan; paham `shape`, `dtype`, `nbytes`, `ndim`    |
| 2   | BAGIAN 2 — Broadcasting & Vectorization           | [ ] Bisa jelaskan aturan broadcasting; demo vectorized vs loop      |
| 3   | BAGIAN 3 — Linear Algebra                         | [ ] Matmul, inverse, eig, SVD, `solve`, `pinv` — semua contoh jalan |
| 4   | BAGIAN 4 — Indexing & Slicing                     | [ ] Paham **view vs copy**, boolean indexing, fancy indexing        |
| 5   | BAGIAN 5 — Practical ML Ops                       | [ ] `softmax`, `pairwise_distance`, `binary_cross_entropy` dipahami |
| 6   | 🏋️ EXERCISE 1 (3 fungsi)                          | [ ] `python 01_numpy_essentials.py --exercise` → semua **PASS**     |
| 7   | 🛌 Rest / Review (opsional: 🔥 CHALLENGE numerik) | [ ] (Opsional) bandingkan matmul manual vs NumPy, hitung speedup    |

## 🐼 Blok 2: Pandas (`01-fondasi-data/02_pandas_essentials.py`)

| Day | Topik                                       | Deliverable hari ini                                  |
| --- | ------------------------------------------- | ----------------------------------------------------- |
| 8   | BAGIAN 1 — Membuat & Membaca Data           | [ ] Load CSV/DataFrame lancar; paham `dtypes`         |
| 9   | BAGIAN 2 — Inspection & Cleaning            | [ ] Handle missing value & duplicate dengan benar     |
| 10  | BAGIAN 3 — Filtering, Grouping, Aggregation | [ ] `groupby`, `merge`, `pivot` dikuasai              |
| 11  | BAGIAN 4 — Feature Engineering              | [ ] Bikin fitur baru dari kolom existing              |
| 12  | BAGIAN 5 — Persiapan Data untuk ML          | [ ] Train/test split & encoding paham                 |
| 13  | 🏋️ Exercise + review                        | [ ] `--exercise` PASS; catat pola yang sering dipakai |
| 14  | 🛌 Rest / Review                            | [ ] Review NumPy + Pandas (20 soal cepat dari kepala) |

## 📊 Blok 3: Visualisasi (`01-fondasi-data/03_visualisasi.py`)

| Day | Topik                                | Deliverable hari ini                               |
| --- | ------------------------------------ | -------------------------------------------------- |
| 15  | BAGIAN 1 — Distribusi Data           | [ ] Histogram/KDE; `plt.savefig()` konsisten       |
| 16  | BAGIAN 2 — Relasi Antar Fitur        | [ ] Scatter, pairplot, heatmap korelasi            |
| 17  | BAGIAN 3 — Visualisasi untuk ML      | [ ] Plot data train vs test dengan benar           |
| 18  | BAGIAN 4 — Model Performance         | [ ] Confusion matrix, ROC, precision-recall curve  |
| 19  | BAGIAN 5 — Signal Processing (EE!)   | [ ] Plot sinyal waktu & frekuensi — modal EE kamu! |
| 20  | BAGIAN 6 — Feature Importance + wrap | [ ] Semua `.png` tersimpan; modul tuntas           |
| 21  | 🛌 Rest / Review                     | [ ] Review 3 modul fase 1                          |

## 📈 Blok 4: Project 1a — EDA (`projects/project_01_eda_prediksi/`)

| Day | Topik                 | Deliverable hari ini                                                         |
| --- | --------------------- | ---------------------------------------------------------------------------- |
| 22  | Setup + EDA Notebook  | [ ] Pilih dataset nyata; notebook berisi distribusi, missing value, korelasi |
| 23  | Insight + Dokumentasi | [ ] Tulis 3–5 insight di README; **push ke GitHub**                          |
| 24  | 🛌 Rest / Review      | [ ] 🎯 **Milestone Fase 1: Portfolio #1 EDA selesai**                        |

---

# 🧮 FASE 2 — ML dari Nol (NumPy Only, Day 25–48)

## 📐 Blok 5: Linear Regression (`02-ml-dari-nol/01_linear_regression_scratch.py`)

| Day | Topik                                     | Deliverable hari ini                           |
| --- | ----------------------------------------- | ---------------------------------------------- |
| 25  | BAGIAN 1 — Closed-Form (Normal Equation)  | [ ] Implementasi + verifikasi vs `sklearn`     |
| 26  | BAGIAN 2 — Gradient Descent               | [ ] Implementasi GD dari nol; loss curve turun |
| 27  | BAGIAN 3 — Polynomial & Overfitting       | [ ] Paham bias-variance tradeoff secara visual |
| 28  | BAGIAN 4 — Regularization (Ridge & Lasso) | [ ] 🏋️ Exercise PASS; paham efek lambda        |
| 29  | 🛌 Rest / Review                          | [ ] Review LR dari nol                         |

## 🎯 Blok 6: Logistic Regression (`02-ml-dari-nol/02_logistic_regression_scratch.py`)

| Day | Topik                                      | Deliverable hari ini                                  |
| --- | ------------------------------------------ | ----------------------------------------------------- |
| 30  | BAGIAN 1 — Sigmoid & Binary Classification | [ ] Paham probabilitas vs logit                       |
| 31  | BAGIAN 2 — Cross-Entropy Loss              | [ ] Turunkan & implementasikan loss                   |
| 32  | BAGIAN 3 — Full Implementation             | [ ] Class `LogisticRegression` dari nol (fit/predict) |
| 33  | BAGIAN 4 — Test pada Synthetic Data        | [ ] Decision boundary plot tersimpan                  |
| 34  | 🛌 Rest / Review                           | [ ] Review LogReg                                     |

## ⛰️ Blok 7: Gradient Descent Deep (`02-ml-dari-nol/03_gradient_descent_deep.py`)

| Day | Topik                                   | Deliverable hari ini                              |
| --- | --------------------------------------- | ------------------------------------------------- |
| 35  | BAGIAN 1 — Visualisasi Loss Landscape   | [ ] Plot contour/3D loss landscape                |
| 36  | BAGIAN 2 — Vanilla vs SGD vs Mini-batch | [ ] Bandingkan 3 mode; catat trade-off            |
| 37  | BAGIAN 3 — Momentum                     | [ ] Paham kenapa momentum mempercepat konvergensi |
| 38  | BAGIAN 4 — Adam Optimizer               | [ ] 🏋️ Exercise PASS; implementasi Adam dari nol  |
| 39  | 🛌 Rest / Review                        | [ ] Review optimizer                              |

## 📏 Blok 8: Evaluasi Model (`02-ml-dari-nol/04_evaluasi_model.py`)

| Day | Topik                              | Deliverable hari ini                                             |
| --- | ---------------------------------- | ---------------------------------------------------------------- |
| 40  | BAGIAN 1 — Regression Metrics      | [ ] MAE, MSE, RMSE, R² dari nol                                  |
| 41  | BAGIAN 2 — Classification Metrics  | [ ] Precision, Recall, F1, AUC dari nol                          |
| 42  | BAGIAN 3 — Proper Cross-Validation | [ ] Implementasi K-Fold manual + stratified                      |
| 43  | BAGIAN 4 — Data Leakage (⚠️)       | [ ] Bisa sebut 5+ contoh leakage                                 |
| 44  | 🛌 Rest / Review                   | [ ] **Self-test: implementasi 1 algoritma dari nol tanpa lihat** |

## 🚀 Blok 9: Project 1b — Prediction Pipeline + Deploy

| Day | Topik                   | Deliverable hari ini                                             |
| --- | ----------------------- | ---------------------------------------------------------------- |
| 45  | Pipeline Prediksi       | [ ] Scratch vs sklearn dibanding (metric table)                  |
| 46  | FastAPI `/predict`      | [ ] Endpoint jalan; test via `curl`/browser                      |
| 47  | Docker + Tests + README | [ ] `Dockerfile` + tests + README; **push GitHub**               |
| 48  | 🛌 Rest / Review        | [ ] 🎯 **Milestone Fase 2: Portfolio #2 Prediction API selesai** |

---

# 🌳 FASE 3 — Classical ML (Day 49–72)

## 🌲 Blok 10: Supervised Learning (`03-classical-ml/01_supervised_learning.py`)

| Day | Topik                                        | Deliverable hari ini                        |
| --- | -------------------------------------------- | ------------------------------------------- |
| 49  | BAGIAN 1 — Decision Trees                    | [ ] Paham splitting criteria (Gini/entropy) |
| 50  | BAGIAN 2 — Random Forest                     | [ ] Paham bagging + feature randomness      |
| 51  | BAGIAN 3 — SVM                               | [ ] Paham margin & kernel intuition         |
| 52  | BAGIAN 4 — Gradient Boosting                 | [ ] Paham boosting (XGBoost/LightGBM dasar) |
| 53  | BAGIAN 5 & 6 — Grand Comparison + GridSearch | [ ] 🏋️ Exercise PASS; tuning paham          |
| 54  | 🛌 Rest / Review                             | [ ] Review supervised                       |

## 🧩 Blok 11: Unsupervised Learning (`03-classical-ml/02_unsupervised_learning.py`)

| Day | Topik                        | Deliverable hari ini                        |
| --- | ---------------------------- | ------------------------------------------- |
| 55  | BAGIAN 1 — K-Means           | [ ] Implementasi manual + elbow method      |
| 56  | BAGIAN 2 — DBSCAN            | [ ] Paham eps & min_samples; noise handling |
| 57  | BAGIAN 3 — PCA               | [ ] Paham eigen vs SVD; variance explained  |
| 58  | BAGIAN 4 — t-SNE             | [ ] Paham kapan t-SNE cocok (visualisasi)   |
| 59  | BAGIAN 5 — Anomaly Detection | [ ] Isolation Forest / one-class paham      |
| 60  | 🛌 Rest / Review             | [ ] Review unsupervised                     |

## 🛠️ Blok 12: Feature Engineering (`03-classical-ml/03_feature_engineering.py`)

| Day | Topik                                            | Deliverable hari ini                         |
| --- | ------------------------------------------------ | -------------------------------------------- |
| 61  | BAGIAN 1 — Numerical Transformations             | [ ] Scaling, log, power transform            |
| 62  | BAGIAN 2 — Domain-Specific Features (EE!)        | [ ] Fitur domain sinyal — nilai jual kamu    |
| 63  | BAGIAN 3 — Time Series Features                  | [ ] Lag, rolling, statistical features       |
| 64  | BAGIAN 4 — Frequency Domain                      | [ ] FFT-based features (koneksi DSP!)        |
| 65  | BAGIAN 5 & 6 — Feature Selection + Dim Reduction | [ ] 🏋️ Exercise PASS; filter/embedding paham |
| 66  | 🛌 Rest / Review                                 | [ ] Review feature engineering               |

## 📡 Blok 13: Project 2 — Klasifikasi Sinyal (`projects/project_02_klasifikasi_sinyal/`)

| Day | Topik                  | Deliverable hari ini                              |
| --- | ---------------------- | ------------------------------------------------- |
| 67  | Setup + Baseline       | [ ] Pilih dataset sinyal; baseline model jalan    |
| 68  | Feature Engineering EE | [ ] Fitur time+frequency domain diekstrak         |
| 69  | Training + Tuning      | [ ] Eksperimen model; metric tercatat             |
| 70  | MLflow + Hydra         | [ ] Tracking eksperimen + config-driven           |
| 71  | README + Push          | [ ] Dokumentasi lengkap; **push GitHub**          |
| 72  | 🛌 Rest / Review       | [ ] 🎯 **Milestone Fase 3: Portfolio #3 selesai** |

---

# 🧠 FASE 4 — Deep Learning (Day 73–102)

## 🔢 Blok 14: Neural Net dari Scratch (`04-deep-learning/01_neural_net_scratch.py`)

| Day | Topik                                 | Deliverable hari ini                      |
| --- | ------------------------------------- | ----------------------------------------- |
| 73  | BAGIAN 1 — Single Neuron (Perceptron) | [ ] Forward pass manual paham             |
| 74  | BAGIAN 2 — Multi-Layer NN (Forward)   | [ ] Implementasi forward pass berlapis    |
| 75  | BAGIAN 2 lanjut — Backpropagation     | [ ] Backprop manual (gradien diturunkan)  |
| 76  | BAGIAN 3 — Test NN                    | [ ] NN klasifikasi jalan; loss turun      |
| 77  | 🏋️ Exercise + self-test               | [ ] Implementasi NN dari nol tanpa lihat  |
| 78  | 🛌 Rest / Review                      | [ ] Review NN scratch (modul terpenting!) |

## 🔥 Blok 15: PyTorch Fundamentals (`04-deep-learning/02_pytorch_fundamentals.py`)

| Day | Topik                    | Deliverable hari ini                      |
| --- | ------------------------ | ----------------------------------------- |
| 79  | BAGIAN 1 — Tensors       | [ ] Bandingkan konsep dengan NumPy        |
| 80  | BAGIAN 2 — Autograd      | [ ] Paham `requires_grad` & backward      |
| 81  | BAGIAN 3 — nn.Module     | [ ] Bangun model dengan `nn.Module`       |
| 82  | BAGIAN 4 — Training Loop | [ ] Loop training valid; validation paham |
| 83  | 🛌 Rest / Review         | [ ] Review PyTorch                        |

## 🖼️ Blok 16: CNN (`04-deep-learning/03_cnn.py`)

| Day | Topik                                   | Deliverable hari ini                        |
| --- | --------------------------------------- | ------------------------------------------- |
| 84  | BAGIAN 1 — 2D Convolution dari Scratch  | [ ] Konvolusi manual = korelasi DSP (EE!)   |
| 85  | BAGIAN 2 — 1D Convolution (Time Series) | [ ] Conv1D untuk sinyal paham               |
| 86  | BAGIAN 3 — CNN MNIST (PyTorch)          | [ ] Akurasi bagus; training pipeline bersih |
| 87  | BAGIAN 4 — Visualisasi Feature Maps     | [ ] Feature map tersimpan sebagai `.png`    |
| 88  | 🏋️ Exercise + review                    | [ ] Exercise PASS; arsitektur CNN paham     |
| 89  | 🛌 Rest / Review                        | [ ] Review CNN                              |

## 🔁 Blok 17: RNN / Time Series (`04-deep-learning/04_rnn_timeseries.py`)

| Day | Topik                             | Deliverable hari ini                        |
| --- | --------------------------------- | ------------------------------------------- |
| 90  | BAGIAN 1 — RNN Basics             | [ ] Paham hidden state & vanishing gradient |
| 91  | BAGIAN 2 — LSTM                   | [ ] Paham gates (input/forget/output)       |
| 92  | BAGIAN 3 — Time Series Prediction | [ ] Data windowing paham                    |
| 93  | BAGIAN 4 — Training LSTM          | [ ] Model time series train sukses          |
| 94  | BAGIAN 5 — Multi-Step Forecasting | [ ] Forecast multi-step; plot tersimpan     |
| 95  | 🛌 Rest / Review                  | [ ] Review RNN/LSTM                         |

## 🤖 Blok 18: Project 3 — Computer Vision (`projects/project_03_computer_vision/`)

| Day | Topik                          | Deliverable hari ini                              |
| --- | ------------------------------ | ------------------------------------------------- |
| 96  | Setup Data + Baseline CNN      | [ ] Dataset & baseline jalan                      |
| 97  | Custom CNN + Training          | [ ] Model custom train sampai akurasi baik        |
| 98  | Transfer Learning (pretrained) | [ ] Fine-tune model pretrained; bandingkan        |
| 99  | ONNX Export                    | [ ] Model diexport ke ONNX & diuji                |
| 100 | Docker + Streamlit Demo        | [ ] Container jalan; demo interaktif              |
| 101 | README + Push                  | [ ] Dokumentasi; **push GitHub**                  |
| 102 | 🛌 Rest / Review               | [ ] 🎯 **Milestone Fase 4: Portfolio #4 selesai** |

---

# 🚀 FASE 5 — Advanced (Day 103–126)

## 🧬 Blok 19: Transfer Learning (`05-advanced/01_transfer_learning.py`)

| Day | Topik                                           | Deliverable hari ini                               |
| --- | ----------------------------------------------- | -------------------------------------------------- |
| 103 | BAGIAN 1 & 2 — Pre-trained + Feature Extraction | [ ] Pakai model pretrained sbg feature extractor   |
| 104 | BAGIAN 3 — Fine-tuning                          | [ ] Fine-tune sebagian layer paham                 |
| 105 | BAGIAN 4 & 5 — LR Scheduling + Pipeline         | [ ] Scheduler per layer; pipeline rapi             |
| 106 | BAGIAN 6 — Data Augmentation                    | [ ] 🏋️ Exercise PASS; augmentasi menaikkan akurasi |
| 107 | 🛌 Rest / Review                                | [ ] Review transfer learning                       |

## 📚 Blok 20: NLP & Transformers (`05-advanced/02_nlp_transformers.py`)

| Day | Topik                                          | Deliverable hari ini                 |
| --- | ---------------------------------------------- | ------------------------------------ |
| 108 | BAGIAN 1 — Self-Attention dari Scratch         | [ ] Implementasi `Q·Kᵀ/√d` dari nol  |
| 109 | BAGIAN 2 — Multi-Head Attention                | [ ] Paham heads & concat             |
| 110 | BAGIAN 3 — Transformer Encoder Block           | [ ] Encoder block (attn + FFN + LN)  |
| 111 | BAGIAN 4 — Transformer utk Text Classification | [ ] Training klasifikasi teks sukses |
| 112 | BAGIAN 5 — HuggingFace Pre-trained             | [ ] Load & fine-tune BERT dasar      |
| 113 | 🛌 Rest / Review                               | [ ] Review transformer               |

## 🎨 Blok 21: Generative Models (`05-advanced/03_generative_models.py`)

| Day | Topik                                  | Deliverable hari ini                 |
| --- | -------------------------------------- | ------------------------------------ |
| 114 | BAGIAN 1 — Autoencoder                 | [ ] Rekonstruksi input paham         |
| 115 | BAGIAN 2 — VAE                         | [ ] Paham reparameterization trick   |
| 116 | BAGIAN 3 — Training VAE (MNIST)        | [ ] Training sukses; loss turun      |
| 117 | BAGIAN 4 & 5 — Generate + Latent Space | [ ] Sampel baru + visualisasi latent |
| 118 | BAGIAN 6 — GAN dasar                   | [ ] Paham generator vs discriminator |
| 119 | 🛌 Rest / Review                       | [ ] Review generative                |

## 📝 Blok 22: Project 4 — NLP Pipeline (`projects/project_04_nlp_pipeline/`)

| Day | Topik                 | Deliverable hari ini                              |
| --- | --------------------- | ------------------------------------------------- |
| 120 | Setup + Preprocessing | [ ] Dataset + tokenization pipeline               |
| 121 | Fine-tune BERT        | [ ] Fine-tune BERT sukses                         |
| 122 | Evaluasi + Eksperimen | [ ] Metric & perbandingan tercatat                |
| 123 | FastAPI Endpoint      | [ ] Endpoint klasifikasi teks jalan               |
| 124 | Gradio Demo + HF Hub  | [ ] Demo interaktif + push ke HF Hub              |
| 125 | README + Push         | [ ] Dokumentasi; **push GitHub**                  |
| 126 | 🛌 Rest / Review      | [ ] 🎯 **Milestone Fase 5: Portfolio #5 selesai** |

---

# 🧭 FASE 6 — Expert Roadmap (Day 127–131)

## 📖 Blok 23: `06-expert/01_expert_roadmap.py`

| Day | Topik                                                | Deliverable hari ini                                   |
| --- | ---------------------------------------------------- | ------------------------------------------------------ |
| 127 | BAGIAN 1 — Paper Reading Guide                       | [ ] Pilih 1 paper (mulai yg mudah: ResNet / Attention) |
| 128 | BAGIAN 2 — Paper Implementation Exercise             | [ ] Implementasi inti paper (simplified)               |
| 129 | BAGIAN 3 — MLOps Roadmap                             | [ ] Peta MLOps di kepala (tracking → deploy → monitor) |
| 130 | BAGIAN 4 & 5 — LLM Engineering + Production Patterns | [ ] Paham pola deploy (serving, batch, edge)           |
| 131 | 🛌 Rest / Review                                     | [ ] Review fase 6; siap masuk production ML            |

---

# 🏭 FASE 7 — Production ML (Day 132–146)

## 🗃️ Blok 24: Feature Stores (`07-production-ml/01_feature_stores.py`)

| Day | Topik                             | Deliverable hari ini                |
| --- | --------------------------------- | ----------------------------------- |
| 132 | BAGIAN 1 — Konsep Feature Store   | [ ] Paham online vs offline feature |
| 133 | BAGIAN 2 — Implementasi Sederhana | [ ] Feature store sederhana jalan   |
| 134 | BAGIAN 3 — Demo                   | [ ] Demo terhubung ke pipeline      |
| 135 | 🛌 Rest / Review                  | [ ] Review feature store            |

## 📡 Blok 25: Model Monitoring (`07-production-ml/02_model_monitoring.py`)

| Day | Topik                                   | Deliverable hari ini                  |
| --- | --------------------------------------- | ------------------------------------- |
| 136 | BAGIAN 1 — Drift Detection Methods      | [ ] Paham data drift vs concept drift |
| 137 | BAGIAN 2 — Implementasi Drift Detection | [ ] Deteksi drift (PSI/KS) jalan      |
| 138 | BAGIAN 3 — Performance Monitor          | [ ] Monitoring performa model         |
| 139 | BAGIAN 4 — Demo                         | [ ] Demo monitoring (Evidently)       |
| 140 | 🛌 Rest / Review                        | [ ] Review monitoring                 |

## 🤖 Blok 26: LLM Engineering (`07-production-ml/03_llm_engineering.py`)

| Day | Topik                                  | Deliverable hari ini                        |
| --- | -------------------------------------- | ------------------------------------------- |
| 141 | BAGIAN 1 — LLM Landscape               | [ ] Peta model & API paham                  |
| 142 | BAGIAN 2 — Prompt Engineering Patterns | [ ] 5+ pola prompt dikuasai                 |
| 143 | BAGIAN 3 — RAG Implementation          | [ ] RAG pipeline (chunk → embed → retrieve) |
| 144 | BAGIAN 4 — Fine-tuning & PEFT          | [ ] Paham LoRA/QLoRA                        |
| 145 | Mini RAG App                           | [ ] Document QA sederhana jalan             |
| 146 | 🛌 Rest / Review                       | [ ] Review fase 7                           |

---

# 🏆 FASE 8 — Flagship Project + Career Prep (Day 147–180)

## 🚢 Blok 27: Project 5 — Flagship Part 1 (`projects/project_05_end_to_end/`)

| Day | Topik                            | Deliverable hari ini                        |
| --- | -------------------------------- | ------------------------------------------- |
| 147 | Definisi Masalah + Arsitektur    | [ ] Problem statement & diagram end-to-end  |
| 148 | Data Pipeline + Feature Store    | [ ] Pipeline data versioning (DVC optional) |
| 149 | Training + Perbandingan Model    | [ ] 3+ model dibandingkan                   |
| 150 | MLflow Tracking + Model Registry | [ ] Eksperimen & model registry rapi        |
| 151 | API FastAPI                      | [ ] Endpoint prediksi + schema pydantic     |
| 152 | Docker + Monitoring (Evidently)  | [ ] Container + monitoring drift            |
| 153 | 🛌 Rest / Review                 | [ ] Review progress flagship                |

## 🏁 Blok 28: Project 5 — Flagship Part 2

| Day | Topik                          | Deliverable hari ini                            |
| --- | ------------------------------ | ----------------------------------------------- |
| 154 | CI/CD (GitHub Actions) + Tests | [ ] Automated test + build pipeline             |
| 155 | Frontend Streamlit             | [ ] UI demo interaktif                          |
| 156 | Dokumentasi + README           | [ ] README lengkap (approach & hasil)           |
| 157 | Polish + Demo                  | [ ] **Push GitHub**; siap di-showcase           |
| 158 | 🛌 Rest / Review               | [ ] 🎯 **Milestone: FLAGSHIP project selesai!** |

## 🎤 Blok 29: Career Prep Intensif (`08-career-prep/`)

| Day | Topik                                        | Deliverable hari ini                               |
| --- | -------------------------------------------- | -------------------------------------------------- |
| 159 | `01_ml_interview_prep.py` — Coding Patterns  | [ ] 5+ pola coding ML dihafal                      |
| 160 | `01_ml_interview_prep.py` — ML Theory Q&A    | [ ] Bisa jawab 15+ pertanyaan teori                |
| 161 | `01_ml_interview_prep.py` — Math for ML      | [ ] Linear algebra, prob, optimisasi siap          |
| 162 | `02_ml_system_design.py` — RADIO-M Framework | [ ] Framework hafal & bisa dipakai                 |
| 163 | Case Study — Recommendation System           | [ ] Desain lengkap (dari requirement → scale)      |
| 164 | 🛌 Rest / Review                             | [ ] Review system design                           |
| 165 | Case Study — Fraud Detection                 | [ ] Desain kedua selesai                           |
| 166 | `03_resume_portfolio_guide.py` — Resume      | [ ] Resume tailored ML Engineer                    |
| 167 | `03_resume_portfolio_guide.py` — Portfolio   | [ ] 5 proyek rapi di GitHub                        |
| 168 | `03_resume_portfolio_guide.py` — LinkedIn    | [ ] Headline, About, Featured diisi                |
| 169 | LeetCode / Python Practice                   | [ ] 5 soal easy/medium                             |
| 170 | Mock Interview #1 (Technical)                | [ ] Rekam & evaluasi jawaban                       |
| 171 | 🛌 Rest / Review                             | [ ] Perbaiki kelemahan dari mock #1                |
| 172 | Mock Interview #2 (System Design)            | [ ] Desain 1 kasus baru di bawah tekanan           |
| 173 | Mock Interview #3 (Full Simulation)          | [ ] Simulasi lengkap (intro → teknis → behavioral) |
| 174 | Portfolio Website                            | [ ] GitHub Pages/Streamlit portfolio live          |
| 175 | Apply Batch 1                                | [ ] 10 aplikasi (tailored)                         |
| 176 | Apply Batch 2                                | [ ] 10 aplikasi + referral request                 |
| 177 | Review & Iterate                             | [ ] Update portfolio berdasar feedback             |
| 178 | Apply Batch 3 + Networking                   | [ ] 10 aplikasi + LinkedIn engagement              |
| 179 | Apply Batch 4 + Follow-up                    | [ ] Total 40+ aplikasi; follow-up aktif            |
| 180 | 🎓 Graduation Day                            | [ ] Cek checklist kelulusan di bawah               |

---

## 🎓 Day 180 — Graduation Checklist

- [ ] 5+ portfolio projects di GitHub dengan README lengkap
- [ ] 1 **FLAGSHIP** project: End-to-End ML System (API + Docker + Monitoring + CI/CD)
- [ ] Bisa implementasi Linear Regression, Neural Net, CNN, Transformer **dari nol**
- [ ] Bisa deploy model dengan FastAPI + Docker + MLflow
- [ ] Bisa jawab 50+ ML interview questions
- [ ] Bisa design 6+ ML systems (recommendation, fraud detection, dll)
- [ ] Resume & LinkedIn optimized
- [ ] Portfolio website live
- [ ] Minimal 3 mock interviews completed
- [ ] 40+ job applications sent

---

## 📊 Weekly Check-in (tiap akhir minggu)

1. **Apa yang saya build minggu ini?** (deliverables)
2. **Apa yang saya pelajari?** (key insights)
3. **Apa yang membuat saya stuck?** (blockers)
4. **Apa yang perlu di-adjust?** (plan changes)
5. **Berapa jam efektif minggu ini?** (target: 10.5 jam = 7 × 1.5 jam)

---

## 🚨 Recovery Plan — Kalau Ketinggalan

**Tertinggal 1–2 hari:**

- Lewati hari Rest → langsung lanjut ke konten berikutnya.

**Tertinggal 3–5 hari:**

- Fokus deliverable utama; skip challenge/opsional. **Jangan skip project** — itu portfolio kamu.

**Tertinggal 1 minggu+:**

- Re-evaluasi timeline (mungkin jadi ±7 bulan). Itu OK — kualitas > kecepatan.

> 💪 **Mindset:** Kamu bukan "beginner yang baru belajar ML". Kamu **backend engineer yang sedang expand ke ML infrastructure** — kombinasi langka yang dicari perusahaan. Konsisten 90 menit sehari > sprint 8 jam seminggu sekali.

---

**Siap mulai? Mulai dari Day 1 besok pagi.** 🚀
