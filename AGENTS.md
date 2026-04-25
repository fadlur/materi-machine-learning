# AGENTS.md — Panduan untuk AI Coding Agent

> File ini ditujukan untuk AI coding agent yang bekerja pada proyek ini.  
> Pembaca diasumsikan tidak tahu apa-apa tentang proyek ini sebelumnya.

---

## 1. Ikhtisar Proyek

Proyek ini adalah **kurikulum pembelajaran mandiri Machine Learning (ML)** yang disusun secara bertahap dari nol hingga level produksi. Target pengguna adalah seseorang dengan latar belakang S2 Teknik Elektro yang sedang transisi ke bidang ML/AI.

**Versi ini di-optimasi untuk ML/AI Engineer track** dengan fokus pada:
- Job-readiness dalam 90 hari
- MLOps & production skills (leverage backend experience)
- LLM Engineering (tren 2025-2026)
- Interview preparation (coding + system design + behavioral)
- Portfolio production-ready

**Filosofi utama:**
- *Build First, Library Later* — bangun algoritma dari nol dengan NumPy sebelum menggunakan library seperti scikit-learn atau PyTorch.
- *Deliberate Practice* — setiap modul berisi latihan yang TIDAK ada jawabannya di tutorial.
- *Project-Driven* — setiap fase diakhiri dengan proyek mandiri (open-ended).
- *Debug > Run* — sengaja ada kode yang perlu diperbaiki, karena debugging = belajar.

**Estimasi waktu:** ~90 hari (sprint intensif, 6-8 jam/hari).

---

## 2. Struktur Direktori

```
.
├── README.md                      ← Roadmap & aturan main (Updated for ML Engineer track)
├── AGENTS.md                      ← File ini
├── 90-day-action-plan.md          ← Jadwal harian 90 hari ← NEW
├── 00-setup/                      ← Setup environment & cek dependensi
│   └── setup_environment.py       ← (Updated with MLOps & LLM packages)
├── 01-fondasi-data/               ← Fase 1: NumPy, Pandas, Visualisasi
│   ├── 01_numpy_essentials.py
│   ├── 02_pandas_essentials.py
│   └── 03_visualisasi.py
├── 02-ml-dari-nol/                ← Fase 2: ML dari nol (NumPy only)
│   ├── 01_linear_regression_scratch.py
│   ├── 02_logistic_regression_scratch.py
│   ├── 03_gradient_descent_deep.py
│   └── 04_evaluasi_model.py
├── 03-classical-ml/               ← Fase 3: sklearn & model selection
│   ├── 01_supervised_learning.py
│   ├── 02_unsupervised_learning.py
│   └── 03_feature_engineering.py
├── 04-deep-learning/              ← Fase 4: Neural nets & PyTorch
│   ├── 01_neural_net_scratch.py
│   ├── 02_pytorch_fundamentals.py
│   ├── 03_cnn.py
│   └── 04_rnn_timeseries.py
├── 05-advanced/                   ← Fase 5: Transfer learning, NLP, Generative
│   ├── 01_transfer_learning.py
│   ├── 02_nlp_transformers.py
│   └── 03_generative_models.py
├── 06-expert/                     ← Fase 6: Paper impl, MLOps, Production, LLM ← UPDATED
│   └── 01_expert_roadmap.py
├── 07-career-prep/                ← Fase 7: Interview prep & system design ← NEW
│   ├── 01_ml_interview_prep.py
│   ├── 02_ml_system_design.py
│   └── 03_resume_portfolio_guide.py
├── 08-production-ml/              ← Fase 8: Feature stores, monitoring, LLM ops ← NEW
│   ├── 01_feature_stores.py
│   ├── 02_model_monitoring.py
│   └── 03_llm_engineering.py
└── projects/                      ← Proyek mandiri (portfolio)
    ├── project_01_eda_prediksi/   ← (Updated with API requirements)
    ├── project_02_klasifikasi_sinyal/  ← (Updated with MLOps requirements)
    ├── project_03_computer_vision/     ← (Updated with Docker requirements)
    ├── project_04_nlp_pipeline/        ← (Updated with deployment requirements)
    └── project_05_end_to_end/          ← (MAJOR UPDATE — FLAGSHIP project)
```

### Pola Penamaan File
- File modul: `{nomor_urut}_{nama_modul_snake_case}.py`
- File proyek: `README.md` di dalam folder `projects/project_{NN}_{nama_proyek}/`
- Output visualisasi: disimpan sebagai `.png` dengan nama deskriptif (contoh: `01_loss_curve.png`, `02_overfitting_demo.png`).

---

## 3. Technology Stack (Updated)

| Kategori | Library / Tool |
|----------|----------------|
| Bahasa | Python 3.10+ |
| Data & Array | NumPy, Pandas |
| Visualisasi | Matplotlib, Seaborn, Plotly |
| Classical ML | scikit-learn, XGBoost, LightGBM |
| Deep Learning | PyTorch, torchvision, torchaudio |
| NLP / LLM | Hugging Face `transformers`, `datasets`, LangChain, OpenAI API |
| MLOps | MLflow, Weights & Biases, DVC, Hydra |
| Deployment | FastAPI, Docker, GitHub Actions |
| Monitoring | Evidently AI, Prometheus (basic) |
| Cloud (opsional) | AWS SageMaker / GCP Vertex AI |
| Environment | VS Code + Jupyter Interactive |

**Tidak ada file konfigurasi build** seperti `pyproject.toml`, `setup.py`, `requirements.txt`, atau `Makefile` di root level.  
Dependensi di-install secara manual via `pip` (lihat `00-setup/setup_environment.py` untuk daftar lengkap).

---

## 4. Cara Menjalankan

### 4.1 Cek Environment
```bash
cd 00-setup
python setup_environment.py
```
Script ini memeriksa versi Python dan keberadaan package utama. Jika ada yang kurang, akan dicetak perintah install.

### 4.2 Menjalankan Modul
Setiap file `.py` bisa dijalankan langsung:
```bash
cd 01-fondasi-data
python 01_numpy_essentials.py
```
Atau dijalankan secara interaktif di VS Code / Jupyter.

### 4.3 Menjalankan Proyek
Setiap folder di `projects/` berisi `README.md` dengan checklist deliverables.  
Proyek dijalankan sebagai notebook/script mandiri, bukan sebagai package yang di-import.

---

## 5. Gaya Kode & Konvensi

### 5.1 Bahasa
- **Semua komentar, docstring, dan dokumentasi ditulis dalam Bahasa Indonesia.**
- Nama variabel dan fungsi menggunakan Bahasa Inggris (konvensi Python umum).
- Gunakan istilah Teknik Elektro untuk menjelaskan konsep ML (misal: "gradient descent = steepest descent optimization", "CNN convolution = konvolusi di DSP").

### 5.2 Struktur Setiap File Modul
Setiap file `.py` mengikuti pola yang konsisten:

```python
"""
=============================================================
FASE {N} — MODUL {M}: {JUDUL MODUL}
=============================================================
...penjelasan tujuan modul & koneksi ke Teknik Elektro...
...durasi target...
=============================================================
"""

# ===========================================================
# 📖 BAGIAN X: {Judul Teori}
# ===========================================================
# ...penjelasan konsep dengan komentar panjang...
# ...kode contoh yang bisa di-run...

# ===========================================================
# 🏋️ EXERCISE {N}: {Deskripsi Tantangan}
# ===========================================================
"""
...instruksi latihan dalam docstring multi-line...
...fungsi sering diberikan signature tapi body-nya `pass` atau di-comment...
"""

# ===========================================================
# 🔥 CHALLENGE: {Deskripsi Open-Ended}
# ===========================================================
"""
...tantangan yang lebih kompleks & terbuka...
"""

print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: {path_file_berikutnya}")
print("="*50)
```

### 5.3 Aturan Penulisan Kode
- Gunakan **class-based implementation** untuk model ML (contoh: `LinearRegressionClosedForm`, `NeuralNetwork`).
- Setiap class harus memiliki method `fit`, `predict`, dan (jika relevan) `score`.
- Sertakan **visualisasi** untuk setiap hasil utama (loss curve, decision boundary, confusion matrix, dll).
- Simpan plot ke file `.png` dengan `plt.savefig()`; jangan hanya `plt.show()`.
- Gunakan `np.random.seed(42)` untuk reproducibility.
- Komentar penjelasan teori diletakkan di **atas** blok kode, bukan di samping.

### 5.4 Penandaan Visual dalam Kode
- `📖` — Bagian teori & contoh
- `🏋️` — Exercise (latihan wajib)
- `🔥` — Challenge (tantangan open-ended)
- `✅` — Penanda modul selesai
- `⚠️` — Peringatan penting (misal: data leakage)
- `📊` — Penanda file gambar tersimpan
- `🎯` — Interview prep questions

---

## 6. Organisasi Konten per Fase

| Fase | Topik | File Utama |
|------|-------|------------|
| 0 | Setup environment | `00-setup/setup_environment.py` |
| 1 | NumPy, Pandas, Matplotlib/Seaborn | `01-fondasi-data/*.py` |
| 2 | Linear/Logistic Regression, GD, Evaluasi | `02-ml-dari-nol/*.py` |
| 3 | Supervised (sklearn), Unsupervised, Feature Engineering | `03-classical-ml/*.py` |
| 4 | NN from scratch, PyTorch, CNN, RNN/LSTM | `04-deep-learning/*.py` |
| 5 | Transfer Learning, Transformers, VAE/GAN | `05-advanced/*.py` |
| 6 | Paper implementation, MLOps, Production, LLM | `06-expert/01_expert_roadmap.py` |
| 7 | ML Coding Interview, System Design, Resume | `07-career-prep/*.py` |
| 8 | Feature Stores, Monitoring, LLM Engineering | `08-production-ml/*.py` |

---

## 7. Strategi Pengujian

Proyek ini **tidak menggunakan framework unit testing formal** (tidak ada `pytest`, `unittest`, atau CI/CD). Pengujian dilakukan secara manual dengan pola berikut:

1. **Inline assertions** di bagian bawah exercise:
   ```python
   assert np.allclose(X_norm.mean(axis=0), 0, atol=1e-10), "Mean harus ~0"
   ```
2. **Visual inspection** — plot disimpan sebagai `.png` dan diperiksa secara visual.
3. **Metric comparison** — bandingkan hasil model scratch dengan library (contoh: LinearRegression dari nol vs `sklearn.linear_model.LinearRegression`).
4. **Reproducibility check** — jalankan ulang dan pastikan output sama (seed = 42).

Jika kamu menambahkan fitur baru, pastikan:
- Kode bisa di-run tanpa error.
- Output numerik masuk akal (tidak `NaN` atau `inf` kecuali sengaja).
- Plot tersimpan dengan benar.

---

## 8. Proyek & Deliverables

Setiap fase penting diakhiri dengan proyek di folder `projects/`:

| Proyek | Fase | Fokus | Production Elements |
|--------|------|-------|---------------------|
| `project_01_eda_prediksi` | 1–2 | EDA + prediksi end-to-end | FastAPI, Dockerfile, tests |
| `project_02_klasifikasi_sinyal` | 3 | Klasifikasi sensor dengan domain knowledge EE | MLflow tracking, Hydra config |
| `project_03_computer_vision` | 4 | CNN/RNN untuk image atau sinyal | Docker, ONNX export, Streamlit demo |
| `project_04_nlp_pipeline` | 5 | NLP dengan Transformer/BERT | FastAPI, Gradio demo, HF Hub |
| `project_05_end_to_end` | 6–8 | Sistem ML production-ready (FLAGSHIP) | API + Docker + monitoring + CI/CD |

Setiap proyek memiliki `README.md` dengan checklist.  
**Aturan:** setiap proyek HARUS memiliki README yang menjelaskan approach & hasil.

---

## 9. Pertimbangan Keamanan

- Proyek ini bersifat **educational/lokal**. Tidak ada API key, credential, atau data sensitif yang disimpan di repository.
- Jika menambahkan MLOps tools (MLflow, W&B) di fase 6, pastikan tidak meng-commit API key ke git.
- Model checkpoint (`.pth`, `.pkl`) bisa menjadi besar — pertimbangkan `.gitignore` jika menambahkannya.
- Tidak ada mekanisme sandboxing; script dijalankan langsung di environment user.

---

## 10. Tips untuk Agent

- **Jangan mengubah filosofi "from scratch"** — jika memperbaiki bug di fase 2, pastikan implementasi tetap menggunakan NumPy murni, bukan langsung mengganti dengan `sklearn`.
- **Pertahankan bahasa Indonesia** untuk semua komentar, docstring, dan pesan error.
- **Pertahankan struktur file**: header `===`, penanda `📖`/`🏋️`/`🔥`, dan pesan akhir modul.
- **Jika menambahkan modul baru**, ikuti konvensi penamaan `{nomor}_{nama_snake_case}.py`.
- **Jika menambahkan exercise**, berikan assertion atau test case yang jelas.
- **Jika memodifikasi kode visualisasi**, pastikan `plt.savefig()` tetap ada dan nama file konsisten.
- Repository ini hanya memiliki **1 commit** (`first commit`) — jangan melakukan `git commit` atau `git push` kecuali secara eksplisit diminta oleh user.
- **Modul baru (07-career-prep, 08-production-ml)** menggunakan format yang sama dengan modul existing — pertahankan konsistensi.
- **File `90-day-action-plan.md`** adalah dokumen panduan jadwal — gunakan bahasa Indonesia yang conversational dan motivational.

---

*File ini di-generate berdasarkan eksplorasi menyeluruh terhadap seluruh isi repository. Versi ini di-update untuk ML Engineer track.*
