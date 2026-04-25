"""
=============================================================
FASE 8 — MODUL 1: FEATURE STORES & DATA VERSIONING
=============================================================
Feature store adalah komponen kritis di production ML systems.
Ini menjembatani gap antara data engineering dan ML engineering.

Background backend kamu sangat relevan di sini:
- Database design → Feature store schema design
- API design → Feature serving APIs
- Caching → Online feature store
- Data pipelines → Feature computation pipelines

Durasi target: 3-4 hari
=============================================================
"""

import numpy as np
import pandas as pd

# ===========================================================
# 📖 BAGIAN 1: Kenapa Feature Store?
# ===========================================================

print("""
╔══════════════════════════════════════════════════════════╗
║     KENAPA FEATURE STORE?                                ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  TANPA FEATURE STORE:                                    ║
║  ┌─────────┐    ┌─────────┐    ┌─────────┐             ║
║  │ Data Scientist A │    │ Data Scientist B │            ║
║  │ Feature: avg_click_7d │    │ Feature: click_avg_week │ ║
║  │ Logic berbeda!        │    │ Logic berbeda!         │ ║
║  └─────────┘    └─────────┘    └─────────┘             ║
║  → Training-serving skew, inconsistency, duplication    ║
║                                                          ║
║  DENGAN FEATURE STORE:                                   ║
║  ┌─────────┐    ┌─────────┐    ┌─────────┐             ║
║  │ Training│◀───│Feature  │───▶│ Serving │             ║
║  │ Pipeline│    │Store    │    │ API     │             ║
║  └─────────┘    └─────────┘    └─────────┘             ║
║       ▲                              │                  ║
║       └────── SAME FEATURE LOGIC ────┘                  ║
║                                                          ║
║  BENEFITS:                                               ║
║  1. Feature reuse antar team & project                   ║
║  2. Training-serving consistency                         ║
║  3. Feature versioning & lineage                         ║
║  4. Point-in-time correctness (no data leakage!)         ║
║  5. Feature sharing & discovery                          ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ===========================================================
# 📖 BAGIAN 2: Arsitektur Feature Store
# ===========================================================

print("""
╔══════════════════════════════════════════════════════════╗
║     FEATURE STORE ARCHITECTURE                           ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  ┌─────────────────────────────────────────────┐         ║
║  │            DATA SOURCES                     │         ║
║  │  (Database, Stream, Files, API)             │         ║
║  └──────────────────┬──────────────────────────┘         ║
║                     │                                    ║
║  ┌──────────────────▼──────────────────────────┐         ║
║  │        FEATURE COMPUTATION                  │         ║
║  │  - Batch (Spark, Pandas)                    │         ║
║  │  - Streaming (Flink, Kafka Streams)         │         ║
║  │  - On-demand (Python function)              │         ║
║  └──────────────────┬──────────────────────────┘         ║
║                     │                                    ║
║        ┌────────────┴────────────┐                      ║
║        ▼                         ▼                      ║
║  ┌─────────────┐          ┌─────────────┐              ║
║  │ OFFLINE     │          │ ONLINE      │              ║
║  │ STORE       │          │ STORE       │              ║
║  │ (Data Lake/ │          │ (Redis/     │              ║
║  │  Warehouse) │          │  DynamoDB)  │              ║
║  │             │          │             │              ║
║  │ - Historical│          │ - Low latency│             ║
║  │ - Training  │          │ - Real-time │              ║
║  │ - Batch pred│          │ - Pre-computed│            ║
║  └──────┬──────┘          └──────┬──────┘              ║
║         │                        │                     ║
║         ▼                        ▼                     ║
║  ┌─────────────────────────────────────────┐            ║
║  │         FEATURE SERVING                 │            ║
║  │  - GetFeatures() for training           │            ║
║  │  - GetOnlineFeatures() for serving      │            ║
║  └─────────────────────────────────────────┘            ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ===========================================================
# 📖 BAGIAN 3: Feature Store Sederhana (Dari Nol)
# ===========================================================
# Kita akan bangun feature store sederhana untuk memahami konsep.
# Di production, gunakan Feast, Tecton, atau SageMaker Feature Store.

class SimpleFeatureStore:
    """
    Feature store sederhana untuk memahami konsep.
    
    Konsep:
    - FeatureGroup: kelompok feature yang related (user, item, interaction)
    - OfflineStore: penyimpanan untuk training (Parquet/CSV)
    - OnlineStore: penyimpanan untuk serving (in-memory dict)
    """
    
    def __init__(self):
        self.offline_store = {}  # {feature_group: pd.DataFrame}
        self.online_store = {}   # {feature_group: {entity_id: features}}
        self.feature_definitions = {}  # Metadata
    
    def register_feature_group(self, name, features, entities):
        """
        Register feature group dengan definisi.
        
        Parameters:
        -----------
        name : str
            Nama feature group, e.g., "user_features"
        features : list
            List feature names, e.g., ["age", "avg_purchase_30d"]
        entities : list
            List entity keys, e.g., ["user_id"]
        """
        self.feature_definitions[name] = {
            'features': features,
            'entities': entities
        }
        print(f"✅ Registered feature group: {name}")
        print(f"   Features: {features}")
        print(f"   Entities: {entities}")
    
    def ingest_batch(self, feature_group, df):
        """
        Ingest data ke offline store.
        
        Di production, ini akan: 
        - Compute features dengan Spark
        - Validate data quality (Great Expectations)
        - Write ke data lake (S3 + Parquet)
        """
        self.offline_store[feature_group] = df.copy()
        
        # Build online store (pre-compute untuk serving)
        entities = self.feature_definitions[feature_group]['entities']
        self.online_store[feature_group] = {}
        
        for _, row in df.iterrows():
            entity_key = tuple(row[e] for e in entities)
            features = {f: row[f] for f in self.feature_definitions[feature_group]['features']}
            self.online_store[feature_group][entity_key] = features
        
        print(f"📊 Ingested {len(df)} rows to '{feature_group}'")
    
    def get_online_features(self, feature_group, entity_keys):
        """
        Get features untuk serving (low latency!).
        
        Di production:
        - Query Redis/DynamoDB
        - Latency target: <5ms
        - Fallback ke on-demand computation kalau missing
        """
        results = []
        for key in entity_keys:
            key_tuple = tuple(key) if isinstance(key, list) else (key,)
            features = self.online_store.get(feature_group, {}).get(key_tuple, {})
            results.append(features)
        return pd.DataFrame(results)
    
    def get_historical_features(self, feature_group, entity_keys, timestamps):
        """
        Get features untuk training dengan point-in-time correctness.
        
        ⚠️ INI PENTING: Kita harus ambil feature values
        SEBELUM timestamp tertentu, bukan data terbaru!
        Ini mencegah data leakage.
        """
        df = self.offline_store.get(feature_group, pd.DataFrame())
        results = []
        
        for key, ts in zip(entity_keys, timestamps):
            # Filter: entity match AND timestamp <= event time
            entity_col = self.feature_definitions[feature_group]['entities'][0]
            mask = (df[entity_col] == key) & (df['timestamp'] <= ts)
            
            # Ambil record terbaru sebelum timestamp
            matched = df[mask].sort_values('timestamp').iloc[-1:]
            results.append(matched)
        
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ===========================================================
# 💻 CONTOH: Feature Store untuk E-Commerce
# ===========================================================

print("\n" + "="*50)
print("CONTOH: Feature Store untuk E-Commerce")
print("="*50)

# Inisialisasi feature store
store = SimpleFeatureStore()

# Register feature groups
store.register_feature_group(
    name="user_features",
    features=["age", "total_purchases", "avg_order_value", "days_since_last_purchase"],
    entities=["user_id"]
)

store.register_feature_group(
    name="item_features", 
    features=["price", "category", "avg_rating", "popularity_score"],
    entities=["item_id"]
)

# Generate sample data
np.random.seed(42)
n_users = 1000

user_df = pd.DataFrame({
    'user_id': range(n_users),
    'timestamp': pd.date_range('2024-01-01', periods=n_users, freq='H'),
    'age': np.random.randint(18, 65, n_users),
    'total_purchases': np.random.poisson(10, n_users),
    'avg_order_value': np.random.exponential(50, n_users).round(2),
    'days_since_last_purchase': np.random.exponential(30, n_users).round(0)
})

item_df = pd.DataFrame({
    'item_id': range(100),
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
    'price': np.random.exponential(100, 100).round(2),
    'category': np.random.choice(['electronics', 'clothing', 'food', 'books'], 100),
    'avg_rating': np.random.uniform(3.0, 5.0, 100).round(1),
    'popularity_score': np.random.exponential(1000, 100).round(0)
})

# Ingest ke feature store
store.ingest_batch("user_features", user_df)
store.ingest_batch("item_features", item_df)

# Simulasi: serving request (real-time prediction)
print("\n🚀 Online Feature Serving (for prediction):")
user_features = store.get_online_features("user_features", entity_keys=[42, 100, 555])
print(user_features)

# Simulasi: training data generation (point-in-time)
print("\n📚 Historical Features (for training — NO LEAKAGE!):")
train_keys = [42, 100]
train_timestamps = pd.to_datetime(['2024-01-10', '2024-01-15'])
# Note: Contoh ini sederhana, perlu adjust timestamp logic


# ===========================================================
# 📖 BAGIAN 4: Data Versioning dengan DVC
# ===========================================================

print("""
╔══════════════════════════════════════════════════════════╗
║     DATA VERSIONING: DVC (Data Version Control)          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Masalah: Dataset berubah, tapi tidak ada versioning!   ║
║  Solusi: DVC = Git untuk data                            ║
║                                                          ║
║  WORKFLOW:                                               ║
║  ┌─────────┐    ┌─────────┐    ┌─────────┐             ║
║  │ Raw Data│───▶│ Process │───▶│ Features│             ║
║  │ (S3)    │    │ Pipeline│    │ (DVC)   │             ║
║  └─────────┘    └─────────┘    └────┬────┘             ║
║                                     │                    ║
║  ┌─────────┐    ┌─────────┐    ┌────▼────┐             ║
║  │ Git     │◀───│ DVC     │◀───│ .dvc    │             ║
║  │ (code)  │    │ (data)  │    │  files  │             ║
║  └─────────┘    └─────────┘    └─────────┘             ║
║                                                          ║
║  COMMANDS:                                               ║
║  $ dvc init                    # Initialize DVC          ║
║  $ dvc remote add -d myremote s3://bucket/path          ║
║  $ dvc add data/features.csv   # Track data              ║
║  $ git add data/features.csv.dvc                         ║
║  $ git commit -m "Add features v1"                       ║
║  $ dvc push                    # Upload ke remote        ║
║                                                          ║
║  REPRODUCIBILITY:                                        ║
║  $ dvc repro                     # Re-run pipeline       ║
║  $ dvc metrics show              # Show metrics          ║
║  $ dvc params diff               # Compare params        ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ===========================================================
# 🏋️ EXERCISE 1: Build Your Feature Store
# ===========================================================
"""
Bangun feature store untuk domain EE / Signal Processing:

Feature Groups:
1. sensor_features
   - Entities: sensor_id
   - Features: avg_voltage_1h, max_current_1h, thd_1h, 
               temperature, vibration_rms
               
2. equipment_features
   - Entities: equipment_id
   - Features: age_days, last_maintenance_days, 
               failure_count_1y, manufacturer
               
3. operational_features
   - Entities: [sensor_id, timestamp]
   - Features: load_factor, ambient_temp, humidity

Tugas:
1. Register semua feature groups
2. Generate synthetic data (1000 sensors, 30 days)
3. Implementasi get_online_features untuk real-time prediction
4. Implementasi get_historical_features dengan point-in-time
5. Validasi: pastikan tidak ada data leakage!

Bonus:
- Tambahkan data quality checks (missing values, range checks)
- Simulasi latency test untuk online serving
"""


# ===========================================================
# 🏋️ EXERCISE 2: DVC Pipeline
# ===========================================================
"""
Buat DVC pipeline untuk project kamu:

1. Buat dvc.yaml:
   stages:
     prepare:
       cmd: python src/prepare.py
       deps:
         - data/raw.csv
         - src/prepare.py
       outs:
         - data/processed.csv
     
     feature_engineering:
       cmd: python src/features.py
       deps:
         - data/processed.csv
         - src/features.py
       outs:
         - data/features.csv
     
     train:
       cmd: python src/train.py
       deps:
         - data/features.csv
         - src/train.py
       params:
         - config/model.yaml:
             - learning_rate
             - epochs
       metrics:
         - metrics.json:
             - accuracy
             - f1_score
       outs:
         - models/model.pkl

2. Jalankan: dvc repro
3. Ganti hyperparameter, jalankan lagi
4. Bandingkan: dvc metrics diff
5. Push ke remote: dvc push
"""


# ===========================================================
# 🔥 CHALLENGE: Production-Ready Feature Pipeline
# ===========================================================
"""
Bangun feature pipeline production-ready:

Requirements:
1. Batch feature computation (daily schedule)
2. Streaming feature computation (real-time events)
3. Feature validation (Great Expectations / Pandera)
4. Feature serving API (FastAPI)
5. Feature monitoring (drift detection)

Arsitektur:
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Kafka   │───▶│ Streaming│───▶│  Online  │
│  Events  │    │ Features │    │  Store   │
└──────────┘    └──────────┘    └──────────┘
                                     ▲
┌──────────┐    ┌──────────┐    ┌────┴───┐
│ Data Lake│───▶│  Batch   │───▶│ Offline│
│ (Parquet)│    │ Features │    │ Store  │
└──────────┘    └──────────┘    └────────┘

Deliverable:
- Code repository dengan README
- Feature definitions dan metadata
- API documentation
- Monitoring dashboard (Streamlit)
"""


print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 08-production-ml/02_model_monitoring.py")
print("="*50)
