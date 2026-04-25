"""
=============================================================
FASE 7 — MODUL 2: ML SYSTEM DESIGN INTERVIEW
=============================================================
System Design untuk ML Engineer berbeda dengan Software Engineer:
- Bukan hanya "design API dan database"
- Tapi juga "design data pipeline, model pipeline, dan serving"

Target: Bisa menjawab dengan framework yang terstruktur dalam 45 menit.

Durasi target: 1 minggu (6 case studies)
=============================================================
"""

# ===========================================================
# 📖 BAGIAN 1: Framework ML System Design
# ===========================================================

print("""
╔══════════════════════════════════════════════════════════╗
║     ML SYSTEM DESIGN FRAMEWORK: RADIO-M                ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Gunakan framework ini untuk setiap case study:          ║
║                                                          ║
║  R — Requirements (functional & non-functional)          ║
║     * Fitur apa yang harus dibuild?                      ║
║     * Scale: QPS, latency, throughput?                   ║
║     * Accuracy vs latency tradeoff?                      ║
║                                                          ║
║  A — Architecture (high-level design)                    ║
║     * Data pipeline (batch vs streaming)                 ║
║     * Feature store                                      ║
║     * Model training pipeline                            ║
║     * Model serving (real-time vs batch)                 ║
║     * Monitoring & feedback loop                         ║
║                                                          ║
║  D — Data (features, labels, storage)                    ║
║     * What features? How to compute?                     ║
║     * Feature engineering pipeline                       ║
║     * Label collection (explicit vs implicit)            ║
║     * Data freshness requirements                        ║
║                                                          ║
║  I — Implementation (model & algorithm)                  ║
║     * Model selection (simple → complex)                 ║
║     * Training strategy (online vs batch)                ║
║     * Evaluation metrics (offline vs online)             ║
║                                                          ║
║  O — Operations (deployment & monitoring)                ║
║     * Serving infrastructure                             ║
║     * A/B testing framework                              ║
║     * Model versioning & rollback                        ║
║     * Monitoring: data drift, model degradation          ║
║                                                          ║
║  M — Metrics & Scale                                     ║
║     * Back-of-envelope calculation                       ║
║     * Storage estimation                                 ║
║     * Compute requirements                               ║
║     * Cost estimation                                    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ===========================================================
# 📖 BAGIAN 2: System Design Case Studies
# ===========================================================

print("""
╔══════════════════════════════════════════════════════════╗
║     CASE STUDY 1: RECOMMENDATION SYSTEM                  ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  "Design a recommendation system for YouTube/Netflix"    ║
║                                                          ║
║  R — Requirements:                                       ║
║  - Recommend videos to 2B users                          ║
║  - Latency: <200ms untuk homepage load                   ║
║  - Freshness: new videos discoverable dalam 1 jam        ║
║  - Metrics: CTR, watch time, diversity                   ║
║                                                          ║
║  A — Architecture:                                       ║
║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ ║
║  │ Event Stream│───▶│Feature Store│───▶│Candidate    │ ║
║  │ (Kafka)     │    │(Redis/Feast)│    │Generation   │ ║
║  └─────────────┘    └─────────────┘    │(ANN index)  │ ║
║                                         └──────┬──────┘ ║
║                                                │         ║
║  ┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐ ║
║  │ Model Store │◀───│Training     │◀───│Feature Eng  │ ║
║  │(S3/MLflow)  │    │Pipeline     │    │Pipeline     │ ║
║  └──────┬──────┘    │(Spark/Beam) │    └─────────────┘ ║
║         │           └─────────────┘                     ║
║         │                                               ║
║  ┌──────▼──────┐    ┌─────────────┐                    ║
║  │Ranker Model │───▶│Serving API  │───▶ User          ║
║  │(TF/PyTorch) │    │(FastAPI/gRPC)│                    ║
║  └─────────────┘    └─────────────┘                    ║
║                                                          ║
║  D — Data:                                               ║
║  - User features: demographics, watch history, liked     ║
║  - Item features: video embedding, category, duration    ║
║  - Context features: time, device, location              ║
║  - Label: watch > 30 seconds (implicit)                  ║
║                                                          ║
║  I — Model:                                              ║
║  - Candidate generation: Two-tower neural network        ║
║  - Ranking: Deep neural network dengan Wide&Deep         ║
║  - Re-ranking: Diversity filter, business rules          ║
║                                                          ║
║  O — Operations:                                         ║
║  - A/B testing: 1% traffic untuk model baru              ║
║  - Monitoring: CTR drop > 5% → auto rollback             ║
║  - Retraining: daily batch training                      ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ===========================================================
# 🏋️ EXERCISE 1: Design Your Own
# ===========================================================
"""
Gunakan framework RADIO-M untuk design sistem berikut:

Case 2: "Design a fraud detection system for e-commerce"
- Scale: 10M transactions/hari
- Latency: <100ms untuk real-time decision
- False positive rate: <1% (jangan decline legitimate!)

Case 3: "Design a search ranking system for e-commerce"
- Scale: 100M products, 500M queries/hari
- Latency: <50ms
- Metrics: relevance, revenue, diversity

Case 4: "Design a predictive maintenance system for factory"
- Scale: 10K sensors, data every 1 second
- Latency: can be batch (hourly)
- Metrics: precision (false alarm costly!)

Case 5: "Design a content moderation system"
- Scale: 1M posts/hour
- Latency: <500ms
- Multi-modal: text, image, video

Case 6: "Design an LLM-powered customer support chatbot"
- Scale: 100K conversations/hari
- Latency: <2 seconds per response
- Requirements: accurate, safe, on-brand
"""


# ===========================================================
# 📖 BAGIAN 3: Key Design Decisions
# ===========================================================

print("""
╔══════════════════════════════════════════════════════════╗
║     KEY DESIGN DECISIONS IN ML SYSTEMS                 ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  1. REAL-TIME vs BATCH PREDICTION                      ║
║  ┌─────────────────┬─────────────────┐                  ║
║  │ Real-time       │ Batch           │                  ║
║  ├─────────────────┼─────────────────┤                  ║
║  │ Fraud detection │ Recommendation  │                  ║
║  │ Search ranking  │ Email campaign  │                  ║
║  │ Autocomplete    │ Report generation│                 ║
║  │ <100ms latency  │ Hours OK        │                  ║
║  │ Complex infra   │ Simple scheduler│                  ║
║  └─────────────────┴─────────────────┘                  ║
║                                                          ║
║  2. ONLINE vs BATCH LEARNING                           ║
║  ┌─────────────────┬─────────────────┐                  ║
║  │ Online          │ Batch           │                  ║
║  ├─────────────────┼─────────────────┤                  ║
║  │ Data drifts fast│ Data stable     │                  ║
║  │ Compute scarce  │ Compute available│                 ║
║  │ Example: ads    │ Example: images │                  ║
║  └─────────────────┴─────────────────┘                  ║
║                                                          ║
║  3. FEATURE STORE ARCHITECTURE                         ║
║  ┌─────────────────────────────────────┐                ║
║  │ Online Store (Redis)                │                ║
║  │ - Low latency (<5ms)                │                ║
║  │ - Pre-computed features             │                ║
║  │ - User profile, item embedding      │                ║
║  ├─────────────────────────────────────┤                ║
║  │ Offline Store (Data Warehouse)      │                ║
║  │ - Batch feature computation         │                ║
║  │ - Historical data                   │                ║
║  │ - Training data generation          │                ║
║  └─────────────────────────────────────┘                ║
║                                                          ║
║  4. MODEL SERVING PATTERNS                             ║
║  ┌─────────────────────────────────────┐                ║
║  │ Pattern          │ Use Case         │                ║
║  ├──────────────────┼──────────────────┤                ║
║  │ Single model     │ Simple API       │                ║
║  │ Ensemble         │ High accuracy    │                ║
║  │ Cascade          │ Cost optimization│                ║
║  │ Multi-model      │ Multi-task       │                ║
║  └─────────────────────────────────────┘                ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ===========================================================
# 🏋️ EXERCISE 2: Back-of-Envelope Calculations
# ===========================================================
"""
Latihan estimasi untuk setiap case:

Case 1 (Recommendation):
- 2B users, 100 videos/user/session
- Each video feature vector: 128 floats (512 bytes)
- Feature store size?
- QPS untuk serving?
- Daily training data size?

Case 2 (Fraud Detection):
- 10M transactions/day
- Each transaction: 50 features (200 bytes)
- Model size: 10MB
- Serving infrastructure needed?
- Training time estimation?

Case 3 (Search):
- 500M queries/day
- 100M products, each 1KB embedding
- ANN index size?
- Memory requirements?
"""


def estimate_recommendation_storage():
    """
    Hitung estimasi storage untuk feature store recommendation.
    
    Asumsi:
    - 2B users
    - 1000 videos per user (history)
    - Each interaction: user_id (8B) + video_id (8B) + features (512B)
    """
    # TODO: Hitung total storage dalam TB
    pass


def estimate_fraud_qps():
    """
    Hitung QPS untuk fraud detection system.
    
    Asumsi:
    - 10M transactions/day
    - Peak = 3x average
    """
    # TODO: Hitung average QPS dan peak QPS
    pass


# ===========================================================
# 🔥 CHALLENGE: Mock System Design Interview
# ===========================================================
"""
Lakukan mock interview dengan setup:

- Timer: 45 menit
- Interviewer: mentor atau AI (ChatGPT dengan prompt system design)
- Format:
  * 2 min: Clarify requirements
  * 10 min: High-level design
  * 20 min: Deep dive (data, model, serving)
  * 10 min: Scale & tradeoffs
  * 3 min: Summary

Tips:
1. SELALU mulai dengan requirements clarification
2. Jangan langsung ke model — design data pipeline dulu
3. Mention tradeoffs secara eksplisit
4. Gunakan numbers untuk estimasi
5. Akhiri dengan "what would I do with more time"

Target: Selesaikan 6 case studies sebelum apply.
"""


print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 07-career-prep/03_resume_portfolio_guide.py")
print("="*50)
