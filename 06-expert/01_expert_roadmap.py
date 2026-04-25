"""
=============================================================
FASE 6 — EXPERT LEVEL + LLM ENGINEERING
=============================================================
Di fase ini, kamu bukan lagi belajar tools — kamu belajar
MINDSET dan WORKFLOW seorang ML engineer profesional.

Tiga pilar:
1. Paper Implementation — bisa membaca dan implementasi paper riset
2. MLOps — experiment tracking, reproducibility, CI/CD
3. Production — serving, monitoring, scaling
4. LLM Engineering — RAG, fine-tuning, agents (NEW — tren 2025-2026)

Durasi target: 3-4 minggu
=============================================================
"""

print("""
╔══════════════════════════════════════════════════════════╗
║              FASE 6: EXPERT LEVEL                       ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  MODUL 1: PAPER IMPLEMENTATION                           ║
║  ────────────────────────────────────                    ║
║  Skill membaca dan implementasi research paper.          ║
║                                                          ║
║  Cara baca paper ML:                                     ║
║  1. Abstract + Conclusion dulu (5 min)                   ║
║  2. Figures & Tables (10 min)                            ║
║  3. Method section (30 min)                              ║
║  4. Full read (1 hour)                                   ║
║  5. Implementation (2-4 hours)                           ║
║                                                          ║
║  Papers yang WAJIB dibaca (in order):                    ║
║                                                          ║
║  ① "Attention Is All You Need" (Vaswani et al., 2017)   ║
║     → Transformer architecture                           ║
║     → Kamu sudah build dari scratch di Fase 5!          ║
║                                                          ║
║  ② "Deep Residual Learning" (He et al., 2015)           ║
║     → ResNet, skip connections                           ║
║     → Implementasi ResNet block dari nol                 ║
║                                                          ║
║  ③ "BERT" (Devlin et al., 2018)                         ║
║     → Pre-training + fine-tuning paradigm                ║
║     → Fine-tune BERT untuk text classification           ║
║                                                          ║
║  ④ "An Image is Worth 16x16 Words" (Dosovitskiy, 2020) ║
║     → Vision Transformer (ViT)                           ║
║     → Implementasi patch embedding + transformer         ║
║                                                          ║
║  ⑤ "LoRA: Low-Rank Adaptation" (Hu et al., 2021)       ║
║     → Efficient fine-tuning untuk LLM                    ║
║     → Hanya train 0.1% parameters!                       ║
║                                                          ║
║  ⑥ Paper dari domain EE:                                ║
║     → "Deep Learning for Fault Diagnosis" (review)       ║
║     → "Temporal Fusion Transformers" (time series)       ║
║     → Pilih 1 paper terbaru dari domain risetmu          ║
║                                                          ║
║  Exercise:                                               ║
║  - Implementasi salah satu paper di atas                 ║
║  - Reproduce hasil (minimal mendekati)                   ║
║  - Tulis README yang menjelaskan:                        ║
║    * Apa masalah yang dipecahkan?                        ║
║    * Apa ide utamanya?                                   ║
║    * Apa hasilmu vs paper?                               ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  MODUL 2: MLOps                                          ║
║  ────────────────                                        ║
║  "Machine learning is 10% model, 90% engineering."       ║
║                                                          ║
║  Tools yang perlu dikuasai:                              ║
║                                                          ║
║  1. Experiment Tracking: MLflow / Weights & Biases       ║
║     pip install mlflow wandb                             ║
║                                                          ║
║     import mlflow                                        ║
║     with mlflow.start_run():                             ║
║         mlflow.log_param("lr", 0.001)                    ║
║         mlflow.log_metric("accuracy", 0.95)              ║
║         mlflow.log_artifact("model.pth")                 ║
║                                                          ║
║  2. Data Version Control: DVC                            ║
║     pip install dvc                                      ║
║     dvc init                                             ║
║     dvc add data/training_data.csv                       ║
║     git add data/training_data.csv.dvc                   ║
║                                                          ║
║  3. Configuration Management: Hydra                      ║
║     pip install hydra-core                               ║
║     → Config files instead of hardcoded hyperparameters  ║
║                                                          ║
║  4. Testing ML Code:                                     ║
║     - Unit tests untuk data pipeline                     ║
║     - Integration tests untuk model training             ║
║     - Data validation (Great Expectations, Pandera)      ║
║                                                          ║
║  5. CI/CD for ML:                                        ║
║     - GitHub Actions untuk automated testing             ║
║     - Automated model training on new data               ║
║     - Model registry & versioning                        ║
║                                                          ║
║  Exercise:                                               ║
║  - Setup MLflow untuk salah satu project sebelumnya      ║
║  - Track minimal 10 experiments dengan hyperparameter    ║
║    berbeda                                               ║
║  - Buat Hydra config untuk project tersebut              ║
║  - Tulis unit tests untuk data pipeline                  ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  MODUL 3: PRODUCTION & DEPLOYMENT                        ║
║  ────────────────────────────                            ║
║                                                          ║
║  1. Model Serving:                                       ║
║     - FastAPI untuk REST API                             ║
║     - TorchServe untuk PyTorch models                    ║
║     - Gradio / Streamlit untuk demo apps                 ║
║                                                          ║
║     from fastapi import FastAPI                          ║
║     app = FastAPI()                                      ║
║                                                          ║
║     @app.post("/predict")                                ║
║     async def predict(data: InputData):                  ║
║         result = model.predict(data.features)            ║
║         return {"prediction": result}                    ║
║                                                          ║
║  2. Model Optimization:                                  ║
║     - Quantization (float32 → int8)                      ║
║     - Pruning (remove unimportant weights)               ║
║     - Knowledge Distillation (large → small model)       ║
║     - ONNX export for cross-platform deployment          ║
║                                                          ║
║  3. Docker Containerization:                             ║
║     FROM python:3.11-slim                                ║
║     COPY requirements.txt .                              ║
║     RUN pip install -r requirements.txt                  ║
║     COPY . /app                                          ║
║     CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]    ║
║                                                          ║
║  4. Monitoring in Production:                            ║
║     - Data drift detection                               ║
║     - Model performance monitoring                       ║
║     - Alerting on degradation                            ║
║     - A/B testing framework                              ║
║                                                          ║
║  Exercise:                                               ║
║  - Deploy salah satu model dengan FastAPI                ║
║  - Buat Dockerfile                                       ║
║  - Buat Streamlit demo untuk model                       ║
║  - Implementasi simple monitoring (log predictions)      ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  MODUL 4: LLM ENGINEERING (NEW — Wajib 2025-2026)       ║
║  ─────────────────────────────────                     ║
║                                                          ║
║  LLM (Large Language Models) adalah skill #1 yang        ║
║  dicari di AI Engineer roles. Background backend kamu    ║
║  sangat cocok untuk LLM deployment & infrastructure.     ║
║                                                          ║
║  1. Prompt Engineering:                                  ║
║     - Zero-shot, few-shot, chain-of-thought              ║
║     - Prompt templates & versioning                      ║
║     - Structured output (JSON mode, function calling)    ║
║                                                          ║
║  2. RAG (Retrieval-Augmented Generation):                ║
║     - Document ingestion & chunking                      ║
║     - Vector database (ChromaDB, Pinecone, Weaviate)     ║
║     - Embedding models (OpenAI, Sentence-Transformers)   ║
║     - Retrieval strategies & reranking                   ║
║                                                          ║
║  3. Fine-tuning LLM:                                     ║
║     - LoRA / QLoRA (efficient fine-tuning)               ║
║     - Dataset preparation untuk instruction tuning       ║
║     - Evaluation metrics untuk LLM                       ║
║                                                          ║
║  4. LLM Agents:                                          ║
║     - ReAct pattern (Reasoning + Acting)                 ║
║     - Tool use / Function calling                        ║
║     - Multi-agent systems                                ║
║                                                          ║
║  5. LLM Deployment:                                      ║
║     - vLLM / TGI untuk high-throughput serving           ║
║     - Streaming responses                                ║
║     - Cost optimization (caching, batching)              ║
║                                                          ║
║  Exercise:                                               ║
║  - Buat RAG system untuk dokumen domain EE (misal:      ║
║    paper, datasheet, manual)                             ║
║  - Deploy sebagai FastAPI dengan streaming               ║
║  - Bandingkan: vanilla LLM vs RAG                        ║
║  - (Opsional) Fine-tune model kecil dengan LoRA         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ===========================================================
# 🔥 FINAL PROJECT: End-to-End ML System
# ===========================================================

print("""
╔══════════════════════════════════════════════════════════╗
║           PROJECT 5: END-TO-END ML SYSTEM                ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Buat sistem ML end-to-end yang production-ready.        ║
║  Pilih salah satu:                                       ║
║                                                          ║
║  OPSI A: Predictive Maintenance Dashboard                ║
║  ─────────────────────────────────────                   ║
║  - Real-time sensor data processing                      ║
║  - Multi-model ensemble (classical + deep learning)      ║
║  - Anomaly detection + RUL prediction                    ║
║  - Streamlit dashboard                                   ║
║  - REST API untuk integrasi                              ║
║  - MLflow experiment tracking                            ║
║  - Docker deployment                                     ║
║                                                          ║
║  OPSI B: Smart Power Quality Monitor                     ║
║  ─────────────────────────────────────                   ║
║  - Power signal analysis (voltage, current)              ║
║  - Automatic event detection (sag, swell, harmonic)      ║
║  - CNN/Transformer pada spectrogram                      ║
║  - Classification + severity scoring                     ║
║  - Web dashboard dengan real-time visualization          ║
║  - Alert system                                          ║
║                                                          ║
║  OPSI C: LLM-Powered Engineering Assistant               ║
║  ─────────────────────────────────────                   ║
║  - RAG atas manual, datasheet, paper EE                  ║
║  - Conversational interface (chat)                       ║
║  - Source citation untuk setiap jawaban                  ║
║  - FastAPI backend + Streamlit frontend                  ║
║  - Document upload & management                          ║
║                                                          ║
║  OPSI D: Custom Project dari Riset S2                    ║
║  ─────────────────────────────────────                   ║
║  - Ambil topik dari riset/thesis S2 kamu                 ║
║  - Apply ML untuk solve problem di domain tsb            ║
║  - Full pipeline: data → model → deploy → monitor        ║
║                                                          ║
║  Requirements:                                           ║
║  ✓ Clean, documented code                                ║
║  ✓ Proper ML pipeline (no data leakage!)                 ║
║  ✓ Experiment tracking (MLflow/W&B)                      ║
║  ✓ Model comparison (minimal 3 models)                   ║
║  ✓ API endpoint (FastAPI)                                ║
║  ✓ Simple frontend (Streamlit/Gradio)                    ║
║  ✓ Docker deployment                                     ║
║  ✓ README with full documentation                        ║
║  ✓ Git version controlled                                ║
║  ✓ (Untuk LLM) RAG atau Fine-tuning evidence             ║
║                                                          ║
║  Ini akan menjadi PORTFOLIO PIECE terbaikmu.             ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

print("="*60)
print("🎓 CONGRATULATIONS!")
print("="*60)
print("""
Jika kamu sudah sampai di sini dan menyelesaikan semua
exercise + projects, kamu sudah di level ADVANCED ML Engineer.

Skill yang sudah kamu miliki:
✅ Fundamental ML (dari nol sampai sklearn)
✅ Deep Learning (CNN, RNN, Transformer)
✅ Advanced topics (transfer learning, generative models)
✅ LLM Engineering (RAG, fine-tuning, agents) ← HOT SKILL!
✅ MLOps (experiment tracking, testing, CI/CD)
✅ Production (API, Docker, monitoring)

Apa selanjutnya?
1. Terus baca paper terbaru (arxiv.org/list/cs.LG)
2. Kontribusi ke open source ML projects
3. Ikut kompetisi Kaggle (sekarang dengan fondasi yang KUAT)
4. Spesialisasi di intersection EE + ML:
   - Signal Processing + Deep Learning
   - Power Systems + ML
   - IoT + Edge ML
   - Robotics + Reinforcement Learning
5. Build LLM-powered products — tren terbesar 2025-2026!

Remember: The best ML engineer is one who NEVER STOPS LEARNING.
""")
