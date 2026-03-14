# 🧠 Machine Learning: From Zero to Production

## Filosofi Anti-Tutorial Hell

**Masalah tutorial hell:** kamu mengikuti langkah demi langkah, copy-paste kode,
merasa paham, tapi begitu dihadapkan problem baru — blank.

**Solusi kurikulum ini:**

1. **Build First, Library Later** — bangun algoritma dari nol dengan NumPy sebelum pakai sklearn
2. **Deliberate Practice** — setiap modul punya tantangan yang TIDAK ada jawabannya di tutorial
3. **Project-Driven** — setiap fase berakhir dengan proyek open-ended, bukan step-by-step
4. **Spaced Repetition** — konsep yang sama muncul di konteks berbeda di fase selanjutnya
5. **Debug > Run** — sengaja ada kode yang salah untuk difix, karena debugging = belajar

**Keunggulan latar belakang Teknik Elektro:**

- Linear algebra, kalkulus, probabilitas → sudah kuat, tinggal mapping ke ML context
- Signal processing → langsung relevan dengan time series, CNN (konvolusi!), feature engineering
- Control theory → ada paralel dengan reinforcement learning & optimization
- Pengalaman lab/riset → mindset eksperimental sudah terbentuk

---

## Roadmap Overview

```
FASE 0: Setup & Tools                    [~1 hari]
  └── Environment, git, workflow

FASE 1: Fondasi Data                     [~1 minggu]
  └── NumPy, Pandas, Visualisasi
  └── 🏗️ Mini Project: EDA dataset nyata

FASE 2: ML dari Nol (NumPy Only!)       [~2 minggu]
  └── Linear Regression from scratch
  └── Logistic Regression from scratch
  └── Gradient Descent deep-dive
  └── Evaluasi model dari nol
  └── 🏗️ Project 1: Prediksi + Full Pipeline

FASE 3: Classical ML (sklearn)           [~2 minggu]
  └── Supervised: SVM, Tree, Ensemble
  └── Unsupervised: Clustering, PCA
  └── Feature Engineering
  └── 🏗️ Project 2: Klasifikasi Sinyal

FASE 4: Deep Learning                    [~3 minggu]
  └── Neural Net from scratch
  └── PyTorch fundamentals
  └── CNN (koneksi ke signal processing!)
  └── RNN & Time Series
  └── 🏗️ Project 3: Computer Vision

FASE 5: Advanced Topics                  [~3 minggu]
  └── Transfer Learning
  └── NLP & Transformers
  └── Generative Models (VAE, GAN)
  └── 🏗️ Project 4: NLP Pipeline

FASE 6: Expert & Production              [~4 minggu]
  └── Paper Implementation
  └── MLOps (experiment tracking, CI/CD)
  └── Model Serving & Deployment
  └── 🏗️ Project 5: End-to-End ML System
```

---

## Aturan Main

### ❌ Yang TIDAK boleh dilakukan:

- Copy-paste kode tanpa modifikasi
- Skip exercise/tantangan karena "sudah paham teorinya"
- Langsung pakai library sebelum membangun versi scratch-nya
- Menghabiskan lebih dari 3 hari di satu modul tanpa maju

### ✅ Yang HARUS dilakukan:

- Tulis kode sendiri, bahkan kalau mirip contoh
- Eksperimen: ubah hyperparameter, ubah data, lihat apa yang terjadi
- Dokumentasikan insight di setiap notebook (bukan cuma kode, tapi PEMAHAMAN)
- Setiap project HARUS punya README yang menjelaskan approach & hasil
- Commit ke git setiap selesai satu modul

### 🎯 Graduation Criteria per Fase:

Kamu boleh lanjut ke fase berikutnya HANYA jika:

1. Semua exercise selesai
2. Project fase tersebut sudah complete dengan README
3. Bisa menjelaskan konsep ke orang lain TANPA melihat kode

---

## Cara Menggunakan

```bash
# Setup environment
cd 00-setup
python setup_environment.py

# Mulai dari fase 1
cd ../01-fondasi-data
python 01_numpy_essentials.py

# Atau jalankan interaktif di VS Code / Jupyter
# Setiap file .py bisa dijalankan langsung atau di interactive mode
```

Setiap file Python berisi:

- 📖 **Penjelasan** — teori ringkas yang relevan
- 💻 **Kode contoh** — implementasi yang bisa dirun
- 🏋️ **Exercise** — latihan dengan petunjuk, TANPA jawaban
- 🔥 **Challenge** — soal open-ended untuk eksplorasi mandiri

---

## Estimasi Total: ~3-4 bulan (part-time, 1-2 jam/hari)

Dengan background S2 Teknik Elektro, beberapa bagian bisa lebih cepat:

- Fase 1 bisa selesai lebih cepat (math foundation sudah ada)
- Fase 2 gradient descent / optimization → sudah familiar
- Fase 4 CNN convolution → direct mapping dari signal processing

**Kunci sukses: KONSISTENSI > INTENSITAS**

Lebih baik 1 jam setiap hari daripada 8 jam sekali seminggu.

---

## Self-Assessment: Siap Apply Kerjaan ML?

Lihat **[ASSESSMENT.md](ASSESSMENT.md)** untuk self-assessment lengkap.

Setiap chapter file (.py) juga sudah dilengkapi **Milestone Assessment** di bagian akhir,
berisi checklist 3 level yang bisa dikerjakan setelah menyelesaikan modul:

- **Level 1** (Bisa Dikerjakan) = minimum requirement
- **Level 2** (Bisa Dijelaskan) = siap interview
- **Level 3** (Bisa Improvisasi) = standout candidate

Target "PD" untuk apply berdasarkan posisi:

| Level Posisi         | Minimum Score per Fase | Total Minimum |
| -------------------- | ---------------------- | ------------- |
| Junior ML Engineer   | Avg 2.0 di Fase 1-3    | 60% overall   |
| ML Engineer          | Avg 2.5 di Fase 1-4    | 70% overall   |
| Senior ML Engineer   | Avg 2.5 di Fase 1-5    | 75% overall   |
| ML Lead / Specialist | Avg 2.5 di Fase 1-6    | 80% overall   |
