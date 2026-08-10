"""
=============================================================
FASE 8 - MODUL 2: ML SYSTEM DESIGN
=============================================================
System design interview = mendesain end-to-end ML system
untuk problem yang realistis.

Framework: RADIO-M
- R: Requirements (functional dan non-functional)
- A: Architecture (high-level design)
- D: Data (sources, pipeline, storage)
- I: Integration (serving, monitoring)
- O: Optimization (scale, performance)
- M: Maintenance (retraining, drift)

Koneksi Teknik Elektro:
- System design = system architecture design
- Data pipeline = signal acquisition dan processing
- Model serving = real-time control system
- Monitoring = fault detection system
- Scalability = handling load seperti power grid

Durasi target: 4-5 jam
============================================================="""

import numpy as np

np.random.seed(42)


# ===========================================================
# BAGIAN 1: RADIO-M Framework
# ===========================================================
print("="*60)
print("BAGIAN 1: RADIO-M FRAMEWORK")
print("="*60)

radio_m_framework = """
TARGET R - REQUIREMENTS:

Functional Requirements:
  - What does the system need to do?
  - Input: what data? Format? Frequency?
  - Output: what predictions? Format? Latency?
  - Features: real-time? Batch? Both?
  
  DETAIL:
  Functional requirements menjawab "WHAT" system harus lakukan.
  Contoh: "System harus bisa merekomendasikan 10 produk per user."
  Pisahkan antara must-have (P0) dan nice-to-have (P1).

Non-Functional Requirements:
  - Latency: p99 < X ms?
  - Throughput: X QPS?
  - Availability: 99.9%? 99.99%?
  - Scalability: handle X users?
  - Cost: budget constraints?
  
  DETAIL:
  Non-functional requirements menjawab "HOW WELL" system harus bekerja.
  SLI (Service Level Indicator) = metric yang diukur (e.g., latency).
  SLO (Service Level Objective) = target untuk SLI (e.g., p99 < 100ms).
  SLA (Service Level Agreement) = kontrak dengan consequences jika SLO tidak tercapai.

Example (Recommendation System):
  Functional:
    - Recommend 10 items per user
    - Update recommendations daily
    - Support multiple content types
  Non-Functional:
    - Latency: p99 < 200ms
    - Throughput: 100K QPS
    - Availability: 99.99%

TARGET A - ARCHITECTURE:

High-Level Components:
  - Data ingestion layer
  - Feature engineering pipeline
  - Model training pipeline
  - Model serving layer
  - Monitoring dan logging
  - User interface (if applicable)

Design Patterns:
  - Lambda architecture (batch + speed layer)
  - Kappa architecture (streaming only)
  - Microservices (independent deployable units)
  - Event-driven (async processing)
  
  DETAIL:
  - Lambda: batch layer untuk accuracy, speed layer untuk latency.
    Complex karena dua codebase.
  - Kappa: streaming-only, simpler. Semua data di-stream.
  - Microservices: scaling independent, fault isolation.
    Tapi overhead operational lebih tinggi.
  - Event-driven: loose coupling, async processing.
    Cocok untuk high-throughput systems.

TARGET D - DATA:

Data Sources:
  - Structured (SQL databases, data warehouses)
  - Unstructured (logs, images, text)
  - Streaming (Kafka, Kinesis)
  - External APIs

Data Pipeline:
  - Ingestion (batch vs streaming)
  - Processing (ETL/ELT)
  - Storage (raw, processed, features)
  - Validation (schema, quality)

Feature Engineering:
  - Online features (real-time computation)
  - Offline features (batch pre-computation)
  - Feature store (centralized storage)
  
  DETAIL:
  - ETL: Extract-Transform-Load. Transform sebelum load.
  - ELT: Extract-Load-Transform. Transform di data warehouse.
    Lebih fleksibel untuk analytics.
  - Feature store: Feast, Tecton, SageMaker Feature Store.
    Centralized untuk konsistensi training-serving.

TARGET I - INTEGRATION:

Model Serving:
  - REST API (synchronous)
  - gRPC (high-performance)
  - Message queue (asynchronous)
  - Batch inference (periodic)

Caching:
  - Redis/Memcached untuk hot predictions
  - CDN untuk static content
  - Cache invalidation strategy
  
  DETAIL:
  - REST API: simple, universal. Cocok untuk most use cases.
  - gRPC: binary protocol, lebih cepat. Cocok untuk internal services.
  - Message queue: Kafka, RabbitMQ. Untuk async processing.
  - Caching strategy: TTL (time-to-live), LRU (least recently used).

TARGET O - OPTIMIZATION:

Performance:
  - Model quantization (INT8, INT4)
  - Model pruning (remove unnecessary weights)
  - Knowledge distillation (smaller student model)
  - Batch inference (amortize overhead)
  
  DETAIL:
  - Quantization: mengurangi precision weights. INT8 = 4x smaller.
  - Pruning: remove weights dengan magnitude kecil. Sparse model.
  - Distillation: train small model untuk mimic large model.
    Student learns from teacher's soft labels.
  - Batch inference: process multiple requests sekaligus.

Scalability:
  - Horizontal scaling (add more instances)
  - Vertical scaling (bigger machines)
  - Load balancing (distribute traffic)
  - Auto-scaling (dynamic resource allocation)

TARGET M - MAINTENANCE:

Model Retraining:
  - Schedule: daily, weekly, triggered
  - Data freshness: how recent should training data be?
  - Pipeline automation: Airflow, Prefect
  
  DETAIL:
  - Scheduled retraining: fixed interval (e.g., weekly).
  - Triggered retraining: ketika drift terdeteksi.
  - Progressive retraining: update model dengan data baru
    tanpa training dari awal (continual learning).

Monitoring:
  - Data drift (input distribution changes)
  - Concept drift (relationship changes)
  - Model performance (accuracy, latency)
  - System health (errors, throughput)

Alerting:
  - Threshold-based alerts
  - Anomaly detection on metrics
  - Escalation policies
  
  DETAIL:
  - Escalation: level 1 (email) -> level 2 (Slack) -> level 3 (PagerDuty).
  - Runbook: dokumentasi cara handle setiap alert.

Rollback:
  - Previous model version
  - Feature flags (disable model)
  - Fallback to heuristic
"""
print(radio_m_framework)


# ===========================================================
# BAGIAN 2: Case Study - Recommendation System
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 2: CASE STUDY - RECOMMENDATION SYSTEM")
print("="*60)

recommendation_system = """
TARGET PROBLEM: Design a recommendation system for an e-commerce platform.

R - REQUIREMENTS:
  Functional:
    - Recommend 10 products per user per page
    - Support: homepage, product page, cart page
    - Include: similar items, frequently bought together
  Non-Functional:
    - Latency: p99 < 100ms
    - Throughput: 500K QPS
    - Availability: 99.99%
    - Freshness: update daily

A - ARCHITECTURE:
  +-------------+    +--------------+    +---------------+
  |   Client    |--->|  API Gateway |--->|  Load Balancer|
  +-------------+    +--------------+    +-------+-------+
                                                 |
                          +----------------------+----------------------+
                          |                      |                      |
                    +-----v-----+        +-------v-------+        +-----v-----+
                    | Candidate |        |  Ranking      |        |  Re-rank  |
                    | Generation|        |   Model       |        |  (Business|
                    |           |        |               |        |   Rules)  |
                    +-----------+        +---------------+        +-----------+
                          |                      |                      |
                    +-----v-----+        +-------v-------+        +-----v-----+
                    |   Redis   |        |   Feature     |        |   A/B Test|
                    |   Cache   |        |    Store      |        |   Engine  |
                    +-----------+        +---------------+        +-----------+

  DETAIL:
  - Candidate Generation: mengurangi dari jutaan items ke ratusan/ratusan.
    Harus sangat cepat (< 10ms).
  - Ranking Model: deep learning model yang accurate tapi lebih lambat.
    Mengurangi dari ratusan ke puluhan.
  - Re-ranking: business rules, diversity, freshness.
    Final filter ke 10 items.

D - DATA:
  User Features:
    - Demographics (age, location, device)
    - Behavior (clicks, purchases, views)
    - Context (time, season, trending)
  
  Item Features:
    - Metadata (category, brand, price)
    - Content (images, description, reviews)
    - Statistics (popularity, conversion rate)
  
  Interaction Data:
    - User-Item matrix (sparse)
    - Clickstream logs
    - Purchase history

I - INTEGRATION:
  Candidate Generation:
    - Collaborative Filtering (fast, broad)
    - Content-Based (diverse)
    - Trending/Popular (exploration)
    - Total: 1000 candidates per request
  
  Ranking:
    - Deep learning model (accurate, slow)
    - Features: user, item, interaction
    - Output: score per candidate
    - Top 100 from ranking
  
  Re-ranking:
    - Business rules (diversity, freshness)
    - Filters (out of stock, age-restricted)
    - Final: top 10

O - OPTIMIZATION:
  - Cache popular recommendations (Redis)
  - Pre-compute offline recommendations (daily batch)
  - Real-time untuk user-specific (online)
  - Model quantization untuk inference cepat
  - CDN untuk images

M - MAINTENANCE:
  - Daily retraining untuk ranking model
  - Weekly retraining untuk candidate generation
  - Monitor: CTR, conversion rate, diversity
  - Alert: drop in metrics > 10%
  - A/B test: new models vs baseline

PANDUAN METRICS:
  Online:
    - Click-Through Rate (CTR)
    - Conversion Rate
    - Revenue per User
    - Diversity (intra-list similarity)
  
  Offline:
    - Precision@K, Recall@K
    - NDCG (Normalized Discounted Cumulative Gain)
    - MAP (Mean Average Precision)
"""
print(recommendation_system)


# ===========================================================
# BAGIAN 3: Case Study - Fraud Detection
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 3: CASE STUDY - FRAUD DETECTION")
print("="*60)

fraud_detection = """
TARGET PROBLEM: Design a real-time fraud detection system.

R - REQUIREMENTS:
  Functional:
    - Score setiap transaction (0-1 fraud probability)
    - Block high-risk transactions (>0.9)
    - Review medium-risk (0.7-0.9)
    - Support: credit card, wire transfer, digital wallet
  Non-Functional:
    - Latency: p99 < 50ms (synchronous blocking)
    - Throughput: 10K TPS
    - False positive rate: < 0.1% (jangan block legitimate!)
    - Explainability: provide reason for block

A - ARCHITECTURE:
  Transaction -> API -> Feature Store -> Model -> Decision -> Action
                    |
              Monitoring -> Alerting -> Human Review

  DETAIL:
  - API harus synchronous blocking karena decision menentukan
    apakah transaction di-approve atau di-block.
  - Feature store menyediakan hot features dengan < 10ms latency.
  - Decision service menggabungkan model score dengan business rules.

D - DATA:
  Real-Time Features:
    - Transaction amount, time, location
    - Device fingerprint
    - Velocity: transactions per hour/day
  
  Historical Features:
    - User history (avg amount, typical merchants)
    - Merchant history (fraud rate)
    - Network: connected fraudulent accounts
  
  External:
    - IP reputation
    - Device reputation
    - Geographic risk scores

I - INTEGRATION:
  Real-Time Path:
    - Kafka for event streaming
    - Flink/Spark Streaming untuk feature computation
    - Model inference: < 50ms
    - Decision: auto-block atau review queue
  
  Async Path:
    - Deep analysis untuk flagged transactions
    - Graph analysis untuk network fraud
    - Human review untuk edge cases

O - OPTIMIZATION:
  - Feature caching (hot features in Redis)
  - Model ensemble: fast model (50ms) + slow model (200ms)
    -> fast untuk block/allow, slow untuk review
  - Batch scoring untuk non-blocking analysis
  - Geographic distribution (edge deployment)

M - MAINTENANCE:
  - Adversarial environment: fraudsters adapt!
  - Continuous retraining (daily/weekly)
  - Label latency: fraud confirmed after days/weeks
  - Feedback loop: confirmed fraud -> retraining
  - Monitor: fraud rate, false positive rate, latency

PANDUAN METRICS:
  - Precision: dari flagged, berapa yang benar fraud?
  - Recall: dari actual fraud, berapa yang tertangkap?
  - False Positive Rate: legitimate yang di-block
  - Detection latency: waktu dari fraud terjadi ke terdeteksi
  - Financial impact: $ saved dari prevented fraud
"""
print(fraud_detection)


# ===========================================================
# BAGIAN 4: Back-of-the-Envelope Calculations
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 4: BACK-OF-THE-ENVELOPE CALCULATIONS")
print("="*60)

def calculate_storage_requirements():
    """
    Contoh perhitungan storage requirements.
    
    Problem: Simpan 1 tahun data sensor (1M devices, 1 reading/minute)
    
    Notes:
    ------
    - Back-of-the-envelope calculations adalah skill penting di system design.
    - Gunakan powers of 10 untuk simplifikasi.
    - Selalu pertimbangkan compression.
    """
    devices = 1_000_000
    readings_per_day = 24 * 60  # 1 per minute
    days_per_year = 365
    bytes_per_reading = 100  # timestamp + value + metadata
    
    total_readings = devices * readings_per_day * days_per_year
    total_bytes = total_readings * bytes_per_reading
    total_gb = total_bytes / (1024**3)
    total_tb = total_gb / 1024
    
    print("Storage Calculation:")
    print(f"  Devices: {devices:,}")
    print(f"  Readings/day/device: {readings_per_day:,}")
    print(f"  Total readings/year: {total_readings:,.0f}")
    print(f"  Total storage: {total_gb:,.0f} GB = {total_tb:,.0f} TB")
    
    # With compression (10:1 ratio)
    compressed_tb = total_tb / 10
    print(f"  With compression: {compressed_tb:,.0f} TB")
    
    return total_tb


def calculate_training_time():
    """
    Contoh perhitungan training time.
    
    Problem: Train model dengan 10M samples, 100 epochs
    
    Notes:
    ------
    - Training time = (samples / batch_size) * epochs * time_per_batch
    - Distributed training bisa mempercepat tapi tidak linear
      (overhead communication).
    - Time per batch bergantung pada model complexity dan hardware.
    """
    samples = 10_000_000
    epochs = 100
    batch_size = 256
    time_per_batch_ms = 500  # milliseconds
    
    batches_per_epoch = samples / batch_size
    total_batches = batches_per_epoch * epochs
    total_time_ms = total_batches * time_per_batch_ms
    total_time_hours = total_time_ms / (1000 * 3600)
    
    print("\nTraining Time Calculation:")
    print(f"  Samples: {samples:,}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {batches_per_epoch:,.0f}")
    print(f"  Total batches: {total_batches:,.0f}")
    print(f"  Training time: {total_time_hours:,.1f} hours")
    
    # With distributed training (8 GPUs)
    distributed_time = total_time_hours / 8
    print(f"  With 8 GPUs: {distributed_time:,.1f} hours")
    
    return total_time_hours


# Run calculations
calculate_storage_requirements()
calculate_training_time()


# ===========================================================
# LATIHAN 20: System Design Practice
# ===========================================================
"""
TARGET Learning Objectives:
   - Mengaplikasikan RADIO-M framework
   - Melakukan back-of-the-envelope calculations
   - Mendesain ML systems untuk berbagai use cases

PANDUAN LANGKAH-LANGKAH:

STEP 1: Practice Case Studies
-----------------------------
   Desain sistem untuk scenarios berikut:
   
   a) Search Ranking:
      - Input: query + documents
      - Output: ranked list
      - Scale: 1B documents, 1M queries/day
      
   b) Content Moderation:
      - Input: images, text, video
      - Output: safe/unsafe + reason
      - Scale: 1M uploads/hour
      
   c) Demand Forecasting:
      - Input: historical sales + external factors
      - Output: future demand per SKU
      - Scale: 100K SKUs, daily forecasts
      
   d) Predictive Maintenance:
      - Input: sensor data dari industrial equipment
      - Output: failure probability + recommended action
      - Scale: 10K machines, real-time monitoring
      
   e) Personalization:
      - Input: user profile + content
      - Output: personalized feed
      - Scale: 100M users, 1B content items

   Untuk setiap case:
   - Apply RADIO-M framework
   - Buat architecture diagram
   - Lakukan back-of-the-envelope calculations
   - Identify key tradeoffs
   - Discuss failure modes


STEP 2: Back-of-the-Envelope Drills
-----------------------------------
   Latihan perhitungan cepat:
   
   a) Storage:
      - How much storage for 1 year of user logs?
      - How much for 1B images at 1MB each?
      - Compression ratio untuk time series data?
      
   b) Compute:
      - Training time untuk 100M samples dengan 8 GPUs?
      - Inference cost untuk 1M requests/day?
      - Feature computation latency untuk 100 features?
      
   c) Network:
      - Bandwidth untuk streaming 1M sensors?
      - Latency untuk cross-region data transfer?
      - CDN cost untuk serving 1M images/day?
      
   d) Memory:
      - RAM untuk loading model dengan 1B parameters?
      - Redis memory untuk caching 1M predictions?
      - GPU memory untuk batch size 128 dengan ResNet-50?


STEP 3: Tradeoff Analysis
-------------------------
   Untuk setiap system, analyze tradeoffs:
   
   a) Latency vs Accuracy:
      - Complex model = better accuracy, slower inference
      - Simple model = faster inference, lower accuracy
      - Solution: cascade (fast -> slow)
      
   b) Freshness vs Cost:
      - Real-time updates = expensive
      - Batch updates = cheaper, less fresh
      - Solution: hybrid (batch + real-time delta)
      
   c) Precision vs Recall:
      - High precision = fewer false positives
      - High recall = fewer false negatives
      - Solution: depends on business cost
      
   d) Scalability vs Complexity:
      - Distributed system = scalable, complex
      - Monolith = simple, not scalable
      - Solution: start simple, evolve


STEP 4: Mock System Design Interviews
-------------------------------------
   a) Timed practice (45 menit)
   b) Whiteboard/diagram tool
   c) Think out loud
   d) Handle follow-up questions
   e) Iterate based on feedback
   
   TIPS Interview structure:
     - Clarify requirements (5 menit)
     - High-level design (10 menit)
     - Deep dive (20 menit)
     - Tradeoffs dan extensions (10 menit)


TIPS:
   - Start dengan requirements, jangan langsung solution
   - Draw diagram (visualisasi membantu)
   - Quantify dengan numbers (storage, latency, throughput)
   - Identify bottlenecks dan propose solutions
   - Discuss failure modes dan mitigation
   - Mention monitoring dan maintenance

PERINGATAN COMMON MISTAKES:
   - Langsung dive ke teknis tanpa understand requirements
   - Tidak quantify ("fast", "big" -> berapa?)
   - Ignore non-functional requirements
   - Single point of failure
   - Tidak discuss monitoring
   - Over-engineering dari awal

TARGET EXPECTED OUTPUT:
   - 5+ system designs dengan RADIO-M framework
   - Back-of-the-envelope calculation skills
   - Architecture diagrams untuk setiap system
   - Tradeoff analysis documents
   - Confident untuk system design interviews

Practice makes perfect!
"""


# ===========================================================
# 🔥 CHALLENGE: Design Production ML System
# ===========================================================
"""
TARGET Learning Objectives:
   - Mendesain end-to-end ML system untuk real use case
   - Mengintegrasikan semua aspects: data, model, serving, monitoring
   - Membuat presentasi yang convincing

PANDUAN LANGKAH-LANGKAH:

STEP 1: Choose Real Use Case
----------------------------
   Pilih satu problem dari domain yang kamu kenal:
   
   a) Smart Grid:
      - Load forecasting untuk grid optimization
      - Anomaly detection untuk power quality
      - Renewable energy prediction
      
   b) Manufacturing:
      - Visual inspection untuk defect detection
      - Predictive maintenance untuk critical equipment
      - Process optimization untuk yield improvement
      
   c) Telecommunications:
      - Network traffic prediction
      - Anomaly detection untuk network failures
      - Customer churn prediction
      
   d) Transportation:
      - Traffic flow prediction
      - Predictive maintenance untuk vehicles
      - Route optimization


STEP 2: Complete System Design
------------------------------
   Apply RADIO-M framework secara lengkap:
   
   Requirements:
   - Interview stakeholders (simulated)
   - Define success metrics
   - Identify constraints (budget, timeline, regulations)
   
   Architecture:
   - High-level diagram
   - Component interactions
   - Technology stack
   
   Data:
   - Data sources dan collection
   - Pipeline design
   - Storage strategy
   - Feature engineering
   
   Integration:
   - Model serving strategy
   - API design
   - Caching strategy
   
   Optimization:
   - Performance optimization
   - Scalability plan
   - Cost optimization
   
   Maintenance:
   - Retraining strategy
   - Monitoring plan
   - Alerting thresholds
   - Rollback plan


STEP 3: Create Deliverables
---------------------------
   a) Architecture Diagram:
      - High-level system diagram
      - Data flow diagram
      - Deployment diagram
      
   b) Technical Document:
      - Problem statement
      - Solution approach
      - Technology choices (dengan justification)
      - Tradeoff analysis
      - Risk assessment
      
   c) Implementation Plan:
      - Phased approach (MVP -> full system)
      - Timeline dan milestones
      - Resource requirements
      - Success criteria


STEP 4: Present and Defend
--------------------------
   a) Prepare presentation (15-20 slides)
   b) Practice delivery (30-45 menit)
   c) Anticipate questions:
      - Kenapa pilih technology X?
      - Bagaimana handle scale 10x?
      - Apa single points of failure?
      - Bagaimana measure success?
      - Apa plan untuk 6 bulan ke depan?
   d) Iterate berdasarkan feedback


TIPS:
   - Start dengan problem yang familiar
   - Gunakan real numbers (bukan abstrak)
   - Reference existing systems ("seperti Netflix recommendation")
   - Show awareness dari tradeoffs
   - Demonstrate operational thinking (monitoring, maintenance)
   - Be pragmatic (tidak perlu perfect dari awal)

PERINGATAN COMMON MISTAKES:
   - Over-engineering untuk MVP
   - Ignore operational aspects
   - Tidak quantify requirements
   - Technology choices tanpa justification
   - Tidak consider failure modes
   - Ignore data privacy/security

TARGET EXPECTED OUTPUT:
   - Complete system design document
   - Professional architecture diagrams
   - Implementation roadmap
   - Presentation yang convincing
   - Ready untuk defend di interview

Ini adalah kunci untuk memenangkan system design interview!
"""

print("\n" + "="*50)
print("OK Modul selesai! Lanjut ke: 08-career-prep/03_resume_portfolio_guide.py")
print("="*50)
