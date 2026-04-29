"""
=============================================================
FASE 6 - MODUL 1: EXPERT ROADMAP
=============================================================
Fase ini membuka pandangan ke level expert:
- Paper reading dan implementation
- MLOps dan production deployment
- LLM engineering (tren 2025-2026)
- Research dan innovation

Ini bukan tutorial step-by-step, tapi roadmap yang
memandu kamu ke arah expert-level skills.

Koneksi Teknik Elektro:
- MLOps = DevOps untuk systems dengan ML components
- Production ML = control systems dengan adaptive controllers
- LLM = large-scale signal processing dengan attention
- Research = system identification untuk unknown systems

Durasi target: 5-7 jam (self-directed)
============================================================="""

import numpy as np

np.random.seed(42)


# ===========================================================
# BAGIAN 1: Paper Reading Guide
# ===========================================================
print("="*60)
print("BAGIAN 1: CARA MEMBACA PAPER ML")
print("="*60)

paper_reading_guide = """
Reading ML papers adalah SKILL yang harus dilatih.

TARGET STRATEGI PEMBACAAN (3-pass method):

Pass 1 (5-10 menit):
  OK Baca: title, abstract, introduction, conclusion
  OK Baca: section dan sub-section headings
  OK Baca: mathematical content (cek apakah familiar)
  OK Goal: categorize paper, assess relevance
  OK Decision: continue atau skip?
  
  DETAIL:
  - Abstract memberikan overview singkat dari masalah, method, dan hasil.
  - Introduction menjelaskan motivasi dan kontribusi utama.
  - Conclusion merangkum apa yang dicapai dan limitation.
  - Dalam 5-10 menit, kamu harus bisa menjawab:
    * Tipe paper apa ini? (theory, algorithm, application, survey)
    * Apakah relevan dengan yang sedang saya pelajari?
    * Apakah ada asumsi yang tidak realistis?

Pass 2 (30-60 menit):
  OK Baca seluruh paper tanpa detil proofs
  OK Focus: main contributions, methods, results
  OK Highlight: key equations dan figures
  OK Goal: understand paper tanpa implementasi
  OK Decision: implement atau tidak?
  
  DETAIL:
  - Baca method section dengan cermat tapi tidak perlu derive semua.
  - Lihat experiments: dataset, metrics, baseline comparison.
  - Catat notations dan definitions yang tidak familiar.
  - Tandai bagian yang memerlukan review lebih lanjut.

Pass 3 (berjam-jam):
  OK Reproduce: implement dari paper
  OK Reproduce: run experiments
  OK Analyze: compare dengan baseline
  OK Goal: fully understand setiap detail
  
  DETAIL:
  - Ini adalah ultimate test of understanding.
  - Mulai dari simple case (1D atau toy dataset).
  - Implementasi dari scratch, bukan copy-paste.
  - Bandingkan dengan hasil yang dilaporkan di paper.

PANDUAN REPOSITORY PAPER PENTING:

Computer Vision:
  - AlexNet (2012) - deep learning breakthrough
  - ResNet (2015) - residual connections
  - EfficientNet (2019) - compound scaling
  - Vision Transformer (2020) - attention untuk images

NLP / Transformers:
  - Attention Is All You Need (2017) - Transformer
  - BERT (2018) - bidirectional pre-training
  - GPT-3 (2020) - large language models
  - LLaMA (2023) - efficient LLM

Generative Models:
  - VAE (2013) - variational autoencoder
  - GAN (2014) - generative adversarial networks
  - Diffusion Models (2020) - stable diffusion
  - Flow Matching (2022) - continuous normalizing flows

Optimization:
  - Adam (2014) - adaptive moment estimation
  - BatchNorm (2015) - normalize activations
  - LayerNorm (2016) - per-sample normalization
  - LoRA (2021) - low-rank adaptation

TIPS MEMBACA PAPER:
  - Mulai dari survey/review papers
  - Follow citation chain (backward dan forward)
  - Join reading groups (diskusi membantu pemahaman)
  - Implementasi = ultimate test of understanding
  - Blog posts (Distill, Lilian Weng) = great intuition

TOOLS:
  - arXiv.org - preprint repository
  - PapersWithCode - paper + implementation
  - Connected Papers - visualisasi citation network
  - Zotero - reference management
"""
print(paper_reading_guide)


# ===========================================================
# BAGIAN 2: Paper Implementation Exercise
# ===========================================================
print("="*60)
print("BAGIAN 2: IMPLEMENTASI PAPER")
print("="*60)

implementation_guide = """
TARGET STRATEGI IMPLEMENTASI PAPER:

STEP 1: Pahami Paper (Pass 1 & 2)
---------------------------------
  - Identifikasi: main contribution
  - Identifikasi: key equations
  - Identifikasi: experimental setup
  - Identifikasi: datasets dan metrics
  - Catat: hyperparameters yang digunakan
  - Pahami: asumsi dan limitation

STEP 2: Setup Environment
-------------------------
  - Buat repository baru (jangan campur dengan project lain)
  - Install dependencies (catat versi yang spesifik)
  - Download datasets (atau buat synthetic jika dataset besar)
  - Setup logging dan experiment tracking
  - Buat struktur folder yang rapi

STEP 3: Implementasi Bertahap
-----------------------------
  - Mulai dari simple case (1D atau toy dataset)
  - Verifikasi: setiap component berfunksi
  - Unit test: key functions
  - Bandingkan: dengan paper's reported results
  - Debug: gunakan gradient checking untuk layers baru

STEP 4: Eksperimen dan Analisis
-------------------------------
  - Reproduce: main results dari paper
  - Extend: ablation studies (hilangkan satu komponen, lihat efeknya)
  - Extend: hyperparameter sensitivity
  - Extend: apply ke dataset lain
  - Catat: semua observation di logbook

PANDUAN CHECKLIST IMPLEMENTASI:
  [ ] Pahami matematika di balik paper
  [ ] Implementasi dari scratch (bukan copy-paste)
  [ ] Verifikasi dengan unit tests
  [ ] Bandingkan dengan baseline
  [ ] Reproduce main results
  [ ] Dokumentasi dengan docstrings
  [ ] README dengan instructions
  [ ] Requirements.txt / environment.yml

TIPS STARTER PAPERS UNTUK IMPLEMENTASI:
  - "Attention Is All You Need" (Transformer)
  - "Deep Residual Learning for Image Recognition" (ResNet)
  - "Auto-Encoding Variational Bayes" (VAE)
  - "Adam: A Method for Stochastic Optimization" (Adam)
  - Pilih paper yang ada kode referensi di PapersWithCode
"""
print(implementation_guide)


# ===========================================================
# BAGIAN 3: MLOps Roadmap
# ===========================================================
print("="*60)
print("BAGIAN 3: MLOPS ROADMAP")
print("="*60)

mlops_roadmap = """
MLOps = Machine Learning Operations
Goal: deploy dan maintain ML models di production dengan reliability.

MLOps adalah evolusi dari DevOps yang disesuaikan untuk lifecycle ML.
Bedanya dengan DevOps biasa:
- ML punya komponen data yang perlu versioning
- Model bisa "degrade" over time (bukan hanya code yang rusak)
- Experimentation adalah bagian integral dari development
- Governance dan compliance lebih kompleks

MLOPS STACK:

1. EXPERIMENT TRACKING
   Tools: MLflow, Weights & Biases, Neptune
   Purpose: track experiments, hyperparameters, metrics
   Kenapa penting:
   - Reproducibility: bisa mengulang experiment persis sama
   - Collaboration: tim bisa melihat experiment satu sama lain
   - Decision making: bandingkan banyak runs untuk pilih yang terbaik
   EE Analogy: data logger untuk system identification

2. MODEL REGISTRY
   Tools: MLflow Model Registry, DVC
   Purpose: version control untuk models
   Kenapa penting:
   - Model versioning seperti code versioning
   - Staging: development -> staging -> production
   - Rollback: bisa kembali ke model sebelumnya jika ada masalah
   EE Analogy: revision control untuk controller designs

3. FEATURE STORE
   Tools: Feast, Tecton, SageMaker Feature Store
   Purpose: central storage untuk features
   Kenapa penting:
   - Consistency: training dan serving menggunakan features yang sama
   - Reusability: features bisa dipakai banyak model
   - Governance: tracking dan auditing
   EE Analogy: shared signal conditioning unit

4. MODEL MONITORING
   Tools: Evidently AI, Arize, WhyLabs
   Purpose: detect drift, track performance
   Kenapa penting:
   - Model performance bisa menurun tanpa kita sadari
   - Data drift dan concept drift perlu dideteksi dini
   - Alerting untuk action timely
   EE Analogy: fault detection system untuk controllers

5. ORCHESTRATION
   Tools: Airflow, Prefect, Kubeflow
   Purpose: schedule dan manage pipelines
   Kenapa penting:
   - Training pipeline perlu dijalankan secara berkala
   - Dependencies antar tasks perlu dikelola
   - Failure handling dan retry logic
   EE Analogy: PLC scheduling system

6. DEPLOYMENT
   Tools: Docker, Kubernetes, AWS SageMaker, GCP Vertex AI
   Purpose: deploy models sebagai services
   Kenapa penting:
   - Containerization memastikan consistency antar environment
   - Auto-scaling untuk handle traffic variance
   - Blue-green deployment untuk zero-downtime updates
   EE Analogy: embedded system deployment

PANDUAN MLOps CHECKLIST:
  [ ] Version control untuk code (Git)
  [ ] Version control untuk data (DVC)
  [ ] Experiment tracking (MLflow/W&B)
  [ ] Automated testing (unit, integration)
  [ ] CI/CD pipeline (GitHub Actions/GitLab CI)
  [ ] Model registry dan versioning
  [ ] Monitoring dan alerting
  [ ] Documentation dan runbooks

TARGET LEARNING PATH MLOps:
  Week 1: Setup MLflow tracking
  Week 2: Docker containerization
  Week 3: CI/CD dengan GitHub Actions
  Week 4: Model monitoring dengan Evidently
  Week 5: Feature store dengan Feast
  Week 6: End-to-end pipeline dengan Airflow

TIPS:
  - Mulai dari tools yang paling simple (MLflow, Docker)
  - Jangan over-engineer dari awal
  - Dokumentasi adalah bagian dari MLOps, bukan afterthought
  - Background backend/DevOps kamu adalah keunggulan besar di sini!
"""
print(mlops_roadmap)


# ===========================================================
# BAGIAN 4: LLM Engineering
# ===========================================================
print("="*60)
print("BAGIAN 4: LLM ENGINEERING")
print("="*60)

llm_engineering = """
LLM (Large Language Models) = tren terbesar di AI 2023-2026.
Sebagai ML Engineer, kamu perlu memahami:

LLM LANDSCAPE:

Open Source:
  - LLaMA 2/3 (Meta) - base untuk banyak derivative models
  - Mistral - efficient dan powerful
  - Falcon - open-source SOTA
  - Qwen (Alibaba) - multilingual support
  - Gemma (Google) - 2B, 7B untuk research

API-based:
  - GPT-4/4o (OpenAI)
  - Claude 3 (Anthropic)
  - Gemini (Google)

Specialized:
  - Code: CodeLlama, StarCoder, DeepSeek-Coder
  - Medical: Med-PaLM
  - Vision: LLaVA, GPT-4V

TRADEOFF OPEN SOURCE VS API:
- Open Source: full control, no per-request cost, data privacy,
  tapi butuh infrastructure dan expertise.
- API: instant access, no maintenance, latest models,
  tapi ada cost per token, data privacy concern, rate limits.

LLM ENGINEERING SKILLS:

1. PROMPT ENGINEERING
   - Zero-shot, few-shot, chain-of-thought
   - System prompts dan user prompts
   - Structured output (JSON, XML)
   - Prompt templates dan version control
   Kenapa penting: prompt adalah "interface" ke LLM. Prompt yang baik
   bisa meningkatkan performance secara dramatis tanpa fine-tuning.

2. RETRIEVAL-AUGMENTED GENERATION (RAG)
   - Vector databases (Chroma, Pinecone, Weaviate)
   - Document chunking dan embedding
   - Retrieval strategies (semantic, keyword, hybrid)
   - Re-ranking dan context injection
   Kenapa penting: RAG memungkinkan LLM mengakses knowledge yang
   tidak ada di training data, tanpa perlu fine-tuning.

3. FINE-TUNING
   - LoRA (Low-Rank Adaptation) - parameter-efficient
   - QLoRA - quantized LoRA (memory efficient)
   - Full fine-tuning (jika resources cukup)
   - Instruction tuning dan RLHF
   Kenapa penting: fine-tuning membuat model lebih specialized
   untuk domain tertentu dengan data yang relatif sedikit.

4. DEPLOYMENT
   - vLLM - high-throughput inference
   - Text Generation Inference (HuggingFace)
   - Quantization (INT8, INT4) untuk edge
   - Batch processing dan streaming
   Kenapa penting: LLM deployment memerlukan optimasi khusus
   karena model yang sangat besar.

5. EVALUATION
   - BLEU, ROUGE, BERTScore
   - Human evaluation
   - LLM-as-a-judge (GPT-4 evaluate outputs)
   - Custom metrics untuk domain-specific tasks
   Kenapa penting: evaluation LLM lebih kompleks dari classification
   karena output adalah free-form text.

PANDUAN LLM PROJECT CHECKLIST:
  [ ] Define use case dan success metrics
  [ ] Choose: API vs self-hosted vs fine-tuned
  [ ] Design prompt template
  [ ] Implement RAG jika perlu knowledge base
  [ ] Setup monitoring (cost, latency, quality)
  [ ] Implement guardrails (safety, bias)
  [ ] Evaluate dengan comprehensive test set

TIPS KONEKSI TEKNIK ELEKTRO:
  - LLM = large-scale nonlinear system dengan attention
  - RAG = adaptive filter dengan external memory
  - Fine-tuning = system adaptation untuk new operating conditions
  - Quantization = reduced-precision signal processing
"""
print(llm_engineering)


# ===========================================================
# BAGIAN 5: Production Deployment Patterns
# ===========================================================
print("="*60)
print("BAGIAN 5: PRODUCTION DEPLOYMENT PATTERNS")
print("="*60)

deployment_patterns = """
TARGET DEPLOYMENT ARCHITECTURE PATTERNS:

1. BATCH PREDICTION
   - Model runs secara periodic (hourly, daily)
   - Input: large dataset
   - Output: predictions untuk seluruh dataset
   - Use case: forecasting, reporting, recommendation refresh
   - EE Analogy: batch processing di SCADA
   
   Tools: Airflow, Spark, AWS Batch
   Pros: simple, cost-effective untuk large volume
   Cons: tidak real-time, latency tinggi

2. REAL-TIME API
   - Model sebagai REST/gRPC service
   - Input: single request
   - Output: immediate prediction
   - Use case: recommendation, fraud detection, chatbot
   - EE Analogy: real-time controller
   
   Tools: FastAPI, Flask, gRPC, AWS SageMaker Endpoints
   Pros: low latency, immediate response
   Cons: butuh auto-scaling, lebih kompleks

3. STREAMING
   - Model process continuous stream
   - Input: Kafka/Kinesis stream
   - Output: stream of predictions
   - Use case: IoT, sensor monitoring, log analysis
   - EE Analogy: digital signal processing pipeline
   
   Tools: Kafka, Flink, Spark Streaming, AWS Kinesis
   Pros: handle continuous data, low latency
   Cons: complex state management, harder to debug

4. EDGE DEPLOYMENT
   - Model runs on edge device
   - Input: local sensor data
   - Output: local decision
   - Use case: autonomous systems, mobile, industrial sensors
   - EE Analogy: embedded controller
   
   Tools: TensorRT, ONNX Runtime, TensorFlow Lite
   Pros: no network dependency, ultra-low latency
   Cons: limited compute, model size constraints

PANDUAN PRODUCTION CHECKLIST:
  [ ] Model latency < requirement (e.g., <100ms p99)
  [ ] Throughput > requirement (e.g., >1000 QPS)
  [ ] Error handling dan fallback
  [ ] Logging dan observability
  [ ] A/B testing framework
  [ ] Rollback strategy
  [ ] Cost monitoring
  [ ] Security (input validation, rate limiting)

MODEL LIFECYCLE:
  1. Development -> 2. Staging -> 3. Production
  4. Monitor -> 5. Retrain -> 6. Deploy
  (repeat 4-6)

PERINGATAN ANTI-PATTERNS:
  - Training-serving skew: preprocessing berbeda antara training dan serving
  - Data leakage di pipeline: label terexpose ke features
  - Model tanpa monitoring: deploy dan lupakan
  - Manual deployment process: error-prone dan tidak reproducible
  - No rollback plan: jika model rusak di production, tidak bisa kembali
"""
print(deployment_patterns)


# ===========================================================
# LATIHAN 18: Expert Skills Practice
# ===========================================================
"""
TARGET Learning Objectives:
   - Membaca dan memahami paper ML
   - Mengimplementasikan paper secara mandiri
   - Membangun MLOps pipeline sederhana
   - Mendesain LLM-based application

PANDUAN LANGKAH-LANGKAH:

STEP 1: Paper Reading Practice
------------------------------
   a) Pilih satu paper dari list di atas
   b) Lakukan 3-pass reading
   c) Buat summary:
      - Problem yang di-solve
      - Method / key contribution
      - Results dan comparison
      - Strengths dan limitations
      - Relevance ke bidangmu
      
   d) Implementasi:
      - Implement key algorithm dari paper
      - Test pada toy dataset
      - Bandingkan dengan baseline


STEP 2: MLOps Pipeline
----------------------
   a) Setup experiment tracking:
      - Install MLflow atau W&B
      - Log hyperparameters, metrics, artifacts
      - Compare multiple runs
      
   b) Containerization:
      - Buat Dockerfile untuk model
      - Build dan test image
      - Push ke registry (Docker Hub atau private)
      
   c) CI/CD:
      - Setup GitHub Actions
      - Automated testing
      - Automated deployment ke staging


STEP 3: LLM Application Design
------------------------------
   a) Pilih use case:
      - Customer support chatbot
      - Code review assistant
      - Document summarization
      - Domain-specific Q&A
      
   b) Design architecture:
      - Model selection (API vs self-hosted)
      - RAG design (jika perlu)
      - Prompt engineering
      - Evaluation strategy
      
   c) Prototype:
      - Build minimal working version
      - Test dengan real data
      - Measure quality dan cost


TIPS:
   - Paper: mulai dari yang lebih simple (Adam, BatchNorm)
   - MLOps: MLflow sangat mudah untuk setup
   - LLM: OpenAI API untuk prototyping cepat
   - Focus pada reproducibility dan documentation

PERINGATAN COMMON MISTAKES:
   - Implementasi paper tanpa paham matematika
   - MLOps over-engineering untuk small project
   - LLM tanpa proper evaluation
   - Tidak dokumentasi experiments
   - Production tanpa monitoring

TARGET EXPECTED OUTPUT:
   - Paper summary dengan critical analysis
   - Working implementation dari key algorithm
   - MLOps pipeline dengan experiment tracking
   - LLM prototype dengan evaluation metrics

Ini adalah fondasi untuk menjadi ML Engineer expert!
"""


# ===========================================================
# CHALLENGE: End-to-End Production System
# ===========================================================
"""
TARGET Learning Objectives:
   - Membangun production ML system dari awal sampai deploy
   - Mengintegrasikan semua skills dari fase 1-6
   - Mendemonstrasikan capabilities sebagai ML Engineer

PANDUAN LANGKAH-LANGKAH:

STEP 1: Define Problem
----------------------
   Pilih satu problem domain:
   
   a) Predictive Maintenance:
      - Input: sensor data dari industrial equipment
      - Output: failure prediction + recommended action
      - Value: reduce downtime, optimize maintenance schedule
      
   b) Quality Control:
      - Input: images dari production line
      - Output: defect detection + classification
      - Value: reduce defect escape rate
      
   c) Energy Optimization:
      - Input: historical consumption + weather + occupancy
      - Output: optimal scheduling + anomaly alerts
      - Value: reduce energy cost, detect waste
      
   d) Custom (dari domainmu sendiri)


STEP 2: System Design
---------------------
   a) Data pipeline:
      - Sources (sensors, databases, APIs)
      - Ingestion (batch vs streaming)
      - Storage (data lake, feature store)
      - Processing (ETL, feature engineering)
      
   b) Model pipeline:
      - Training (experiment tracking, hyperparameter tuning)
      - Evaluation (validation, testing)
      - Registration (model versioning)
      - Deployment (serving infrastructure)
      
   c) Monitoring:
      - Data drift detection
      - Model performance tracking
      - System health (latency, errors, cost)
      - Alerting dan escalation


STEP 3: Implementation
----------------------
   a) Data layer:
      - Ingestion pipeline
      - Feature engineering
      - Data validation (Great Expectations)
      
   b) Model layer:
      - Training pipeline (MLflow)
      - Model evaluation
      - Model registry
      
   c) Serving layer:
      - API (FastAPI/Flask)
      - Containerization (Docker)
      - Orchestration (Kubernetes)
      
   d) Monitoring layer:
      - Evidently AI untuk drift detection
      - Prometheus + Grafana untuk metrics
      - Logging dengan structured format


STEP 4: Documentation & Presentation
------------------------------------
   a) Architecture diagram:
      - System architecture
      - Data flow
      - Deployment topology
      
   b) Documentation:
      - README dengan setup instructions
      - API documentation (OpenAPI/Swagger)
      - Runbook untuk operations
      - On-call playbook
      
   c) Presentation:
      - Problem statement
      - Solution approach
      - Results dan metrics
      - Lessons learned
      - Future improvements


TIPS:
   - Start simple, iterate
   - Focus pada one metric yang matters
   - Document assumptions dan decisions
   - Test failover dan rollback
   - Measure business value (bukan hanya ML metrics)

PERINGATAN COMMON MISTAKES:
   - Over-engineering dari awal
   - Tidak define success metrics
   - Ignore data quality issues
   - Deploy tanpa monitoring
   - Tidak plan untuk model updates

TARGET EXPECTED OUTPUT:
   - Production-ready ML system
   - Clean architecture dan documentation
   - Monitoring dan alerting
   - Demonstrable business value
   - Portfolio-worthy project

Ini adalah CAPSTONE dari kurikulum ini -
membuktikan bahwa kamu bisa deliver production ML systems!
"""

print("\n" + "="*50)
print("OK Modul selesai! Lanjut ke: 07-career-prep/01_ml_interview_prep.py")
print("="*50)
