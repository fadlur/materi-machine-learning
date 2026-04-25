# Project 5: End-to-End ML System — FLAGSHIP PORTFOLIO

> **Ini adalah portfolio piece TERPENTING.**
> Target: Production-ready system yang bisa di-demo ke interviewer.
> Timeline: 10-14 hari (Minggu 9-10 dari 90-day plan)

---

## 🎯 Tujuan

Membangun sistem ML end-to-end yang production-ready.
Bukan sekadar notebook — ini adalah **software product** dengan:
- Clean architecture
- API endpoints
- Docker deployment
- Monitoring & experiment tracking

---

## 📋 Pilihan Topik

### OPSI A: Predictive Maintenance Dashboard ⭐ (Recommended untuk EE background)
- **Domain:** Manufacturing / Power Systems
- **Data:** Sensor readings (vibration, temperature, current)
- **Models:** Anomaly detection + RUL prediction
- **Why:** Langsung leverage S2 EE knowledge!

### OPSI B: Smart Power Quality Monitor ⭐ (Recommended untuk EE background)
- **Domain:** Power Systems
- **Data:** Voltage/current waveforms
- **Models:** Event detection (sag, swell, harmonic) + classification
- **Why:** Direct mapping dari signal processing coursework!

### OPSI C: LLM-Powered Engineering Assistant 🆕 (Recommended untuk AI Engineer track)
- **Domain:** Technical documentation
- **Data:** Manual, datasheet, paper EE
- **Models:** RAG + Fine-tuned LLM
- **Why:** Tren terbesar 2025-2026, unique combination EE + LLM

### OPSI D: Custom dari Riset S2
- **Domain:** Topik thesis/riset kamu
- **Models:** Tergantung domain
- **Why:** Deep expertise, bisa jadi differentiator

---

## 🏗️ Architecture Requirements

Setiap opsi HARUS mengikuti architecture pattern ini:

```
┌─────────────────────────────────────────────────────────┐
│                   CLIENT LAYER                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │ Streamlit   │ or │ React/Vue   │ or │ Mobile App  │ │
│  │ Dashboard   │    │ (opsional)  │    │ (opsional)  │ │
│  └──────┬──────┘    └─────────────┘    └─────────────┘ │
└─────────┼───────────────────────────────────────────────┘
          │ HTTP/WebSocket
┌─────────▼───────────────────────────────────────────────┐
│                   API LAYER                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │  FastAPI / Flask                                │   │
│  │  - /predict (synchronous)                       │   │
│  │  - /predict/batch (asynchronous)                │   │
│  │  - /health (health check)                       │   │
│  │  - /metrics (Prometheus metrics)                │   │
│  │  - /docs (auto-generated Swagger)               │   │
│  └─────────────────────────────────────────────────┘   │
└─────────┬───────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────┐
│              MODEL SERVING LAYER                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │ Model A     │    │ Model B     │    │ Model C     │ │
│  │ (classical) │    │ (deep learning)│   │ (ensemble)  │ │
│  └─────────────┘    └─────────────┘    └─────────────┘ │
└─────────┬───────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────┐
│              FEATURE PIPELINE                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │ Raw Data    │───▶│ Feature Eng │───▶│ Feature     │ │
│  │ Ingestion   │    │ Pipeline    │    │ Store       │ │
│  └─────────────┘    └─────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────┐
│              MONITORING LAYER                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │ MLflow      │    │ Evidently   │    │ Prometheus  │ │
│  │ (experiments)│   │ (drift)     │    │ (system)    │ │
│  └─────────────┘    └─────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## ✅ Deliverables (Wajib)

### 1. Code Repository
```
project-05-end-to-end/
├── README.md                 # Dokumentasi lengkap
├── requirements.txt          # Dependencies
├── Dockerfile                # Containerization
├── docker-compose.yml        # Multi-service setup
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI/CD
├── config/
│   ├── model.yaml            # Hydra config
│   └── serving.yaml          # API config
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py      # Data loading
│   │   └── preprocessing.py  # Data cleaning
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py    # Feature computation
│   │   └── store.py          # Feature store logic
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py          # Training script
│   │   ├── evaluate.py       # Evaluation script
│   │   └── registry.py       # Model versioning
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── api.py            # FastAPI app
│   │   ├── middleware.py     # Logging, auth
│   │   └── schemas.py        # Pydantic models
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── drift.py          # Drift detection
│   │   └── metrics.py        # Performance tracking
│   └── utils/
│       ├── __init__.py
│       └── logger.py         # Logging config
├── notebooks/
│   ├── 01_eda.ipynb          # EDA & insights
│   ├── 02_experiments.ipynb  # Model experiments
│   └── 03_analysis.ipynb     # Error analysis
├── tests/
│   ├── __init__.py
│   ├── test_data.py          # Data pipeline tests
│   ├── test_features.py      # Feature tests
│   ├── test_model.py         # Model tests
│   └── test_api.py           # API tests
├── artifacts/                # (gitignored)
│   ├── models/               # Saved models
│   └── data/                 # Processed data
└── docs/
    ├── architecture.md       # System design doc
    └── api.md                # API documentation
```

### 2. API Endpoints
- [ ] `POST /predict` — Single prediction
- [ ] `POST /predict/batch` — Batch prediction
- [ ] `GET /health` — Health check
- [ ] `GET /metrics` — Prometheus metrics
- [ ] `GET /model/info` — Model metadata
- [ ] `POST /feedback` — Feedback loop (optional tapi recommended)

### 3. Monitoring
- [ ] MLflow experiment tracking (minimal 10 experiments)
- [ ] Model performance metrics (accuracy, latency, throughput)
- [ ] Data drift detection (Evidently AI atau custom)
- [ ] Alert system (email/Slack untuk degradation)

### 4. Testing
- [ ] Unit tests untuk data pipeline (>80% coverage)
- [ ] Unit tests untuk feature engineering
- [ ] Integration tests untuk API
- [ ] Load tests (Locust atau k6)

### 5. Documentation
- [ ] README dengan setup instructions
- [ ] API documentation (Swagger UI otomatis)
- [ ] Architecture decision records (ADR)
- [ ] Demo video (2-3 menit, unlisted YouTube)

---

## 📊 Checklist Detail

### Data Pipeline
- [ ] Data acquisition (API, file, atau synthetic generator)
- [ ] Data validation (schema, range, missing values)
- [ ] Feature engineering (minimal 10 features)
- [ ] Feature store implementation (simplified OK)
- [ ] Data versioning dengan DVC

### Model Training
- [ ] Minimal 3 model architectures dibandingkan
- [ ] Hyperparameter tuning (grid search atau bayesian)
- [ ] Cross-validation dengan proper splitting
- [ ] Error analysis (confusion matrix, error cases)
- [ ] Model comparison report

### Serving
- [ ] FastAPI application dengan async endpoints
- [ ] Input validation dengan Pydantic
- [ ] Error handling & logging
- [ ] Rate limiting
- [ ] Request/response logging

### Frontend
- [ ] Streamlit dashboard dengan:
  - Upload data / input form
  - Prediction results
  - Model confidence / explanation
  - Historical predictions
  - Model performance charts

### Deployment
- [ ] Dockerfile (multi-stage build recommended)
- [ ] Docker Compose (app + monitoring)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Cloud deployment (opsional: AWS/GCP/Heroku)

---

## 🎤 Demo Script (Untuk Interview)

Siapkan 3-menit demo dengan script ini:

```
[0:00] "Ini adalah [Nama Project], sistem ML untuk [problem]."

[0:30] "Arsitekturnya: data pipeline → feature store → 
        model serving → API → dashboard."

[1:00] Demo: Upload sample data → prediction → results

[1:30] "Kita track semua experiment dengan MLflow. 
        Ini comparison 3 model yang kita coba."

[2:00] "Untuk monitoring, kita deteksi data drift 
        dengan statistical tests."

[2:30] "Tech stack: PyTorch, FastAPI, Docker, MLflow."

[2:45] "Source code ada di GitHub: [link]"
```

---

## 🚀 Quick Start (Untuk Reviewer)

```bash
# Clone repo
git clone https://github.com/username/project-05-end-to-end.git
cd project-05-end-to-end

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Train model
python src/models/train.py --config config/model.yaml

# Start API
uvicorn src.serving.api:app --reload

# Start dashboard
streamlit run dashboard.py

# Docker
docker-compose up --build
```

---

## 🏆 Success Criteria

Project ini SUKSES jika:
- [ ] Bisa dijalankan dengan `docker-compose up` tanpa error
- [ ] API response time <200ms untuk single prediction
- [ ] Streamlit dashboard bisa demo live
- [ ] Semua tests passing
- [ ] MLflow menunjukkan minimal 10 experiments
- [ ] README cukup jelas untuk orang lain menjalankan
- [ ] Kamu bisa jelaskan setiap komponen dalam interview

---

## 💡 Tips dari Background Backend Kamu

**Leverage kekuatanmu:**
- API design → FastAPI akan sangat natural
- Docker → Containerization sudah familiar
- Testing → Unit tests untuk ML pipeline = transfer skill
- Database → Feature store design = database schema design
- CI/CD → GitHub Actions untuk ML = mirip dengan backend CI/CD

**Jangan over-engineer:**
- Focus pada end-to-end flow yang works
- Lebih baik simple tapi complete, daripada complex tapi broken
- Portfolio piece harus bisa demo dalam 3 menit

---

*Ini adalah portfolio FLAGSHIP kamu. Habiskan waktu untuk polish.*
