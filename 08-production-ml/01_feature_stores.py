"""
=============================================================
FASE 8 - MODUL 1: FEATURE STORES
=============================================================
Feature Store = centralized storage untuk features yang
digunakan oleh ML models.

Mengapa feature store penting?
- Consistency: training dan serving menggunakan features yang sama
- Reusability: features bisa di-share antar teams dan models
- Governance: tracking dan versioning features
- Efficiency: avoid redundant computation

Koneksi Teknik Elektro:
- Feature store = shared signal conditioning unit
- Online store = real-time buffer untuk immediate access
- Offline store = historical data logger untuk analysis
- Feature transformation = DSP pipeline (filtering, scaling)

Durasi target: 3-4 jam
============================================================="""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

np.random.seed(42)


# ===========================================================
# BAGIAN 1: Feature Store Concepts
# ===========================================================
print("="*60)
print("BAGIAN 1: KONSEP FEATURE STORE")
print("="*60)

feature_store_concepts = """
TARGET ARSITEKTUR FEATURE STORE:

+---------------------------------------------------------+
|                    FEATURE STORE                        |
+---------------------------------------------------------+
|                                                         |
|  +--------------+         +--------------+             |
|  | ONLINE STORE |         | OFFLINE STORE|             |
|  |  (Low Latency|         |  (Batch/     |             |
|  |   Real-time) |         |   Historical)|             |
|  +------+-------+         +------+-------+             |
|         |                        |                      |
|         v                        v                      |
|  +--------------+         +--------------+             |
|  |  Redis /     |         |  Data Lake / |             |
|  |  DynamoDB    |         |  Warehouse   |             |
|  |  (ms access) |         |  (batch)     |             |
|  +--------------+         +--------------+             |
|                                                         |
|  +-------------------------------------------------+   |
|  |         FEATURE TRANSFORMATION ENGINE            |   |
|  |  (compute features dari raw data)               |   |
|  +-------------------------------------------------+   |
|                                                         |
+---------------------------------------------------------+

TARGET ONLINE STORE:
- Purpose: real-time serving (API calls)
- Latency: < 10ms
- Storage: key-value (Redis, DynamoDB)
- Data: latest feature values
- Use case: recommendation, fraud detection

DETAIL:
- Online store harus sangat cepat karena dipanggil setiap request.
- Redis adalah pilihan populer karena in-memory dan mendukung
  data structures seperti hashes dan sorted sets.
- DynamoDB adalah managed alternative di AWS.
- Data di-update secara real-time atau near real-time.

TARGET OFFLINE STORE:
- Purpose: training data generation
- Latency: minutes to hours (batch)
- Storage: data warehouse (BigQuery, Snowflake)
- Data: historical feature values
- Use case: model training, backtesting

DETAIL:
- Offline store menyimpan historical data dalam format
  yang cocok untuk batch processing.
- Data warehouse seperti BigQuery atau Snowflake bisa
  handle queries analytical yang kompleks.
- Data di-update secara batch (misal: nightly ETL).

TARGET POINT-IN-TIME CORRECTNESS:
- Training: features harus dari waktu sebelum label
- Serving: features dari current time
- Challenge: avoid data leakage!

DETAIL:
- Data leakage adalah masalah SERIOUS di ML.
- Contoh: menggunakan future information untuk predict past.
- Point-in-time correctness memastikan kita tidak menggunakan
  information yang belum tersedia pada waktu prediction.
- Ini memerlukan timestamp tracking di setiap feature.

TARGET FEATURE TYPES:
1. Raw Features: langsung dari data source
2. Transformed Features: hasil transformation
3. Aggregated Features: statistics over time windows
4. Derived Features: combinations dari features lain

Koneksi Teknik Elektro:
- Feature transformation = signal conditioning
- Online store = sample-and-hold circuit
- Offline store = data recorder
- Point-in-time = causality constraint
"""
print(feature_store_concepts)


# ===========================================================
# BAGIAN 2: Simple Feature Store Implementation
# ===========================================================
# Feature store production seperti Feast, Tecton, atau SageMaker
# Feature Store punya kompleksitas tinggi.
# Implementasi ini adalah educational version untuk memahami konsep dasar.
#
# KOMPONEN UTAMA FEATURE STORE:
# 1. Feature Registry: metadata tentang features (nama, tipe, owner, dll)
# 2. Online Store: key-value storage untuk real-time serving
# 3. Offline Store: historical storage untuk training
# 4. Transformation Engine: compute features dari raw data

class SimpleFeatureStore:
    """
    Simplified feature store untuk educational purposes.
    
    Parameters:
    -----------
    online_store : dict
        In-memory key-value store untuk online serving.
    offline_store : pd.DataFrame
        DataFrame dengan timestamped features.
        
    Notes:
    ------
    - Production feature stores: Feast, Tecton, SageMaker Feature Store
    - This implementation untuk memahami konsep dasar
    - Feature store memisahkan feature computation dari model training
    
    Koneksi Teknik Elektro:
    - Online store = real-time register (current value)
    - Offline store = historical log (time-series data)
    - Feature transformation = signal processing pipeline
    """
    
    def __init__(self):
        self.online_store = {}  # entity_id -> {feature_name: value}
        self.offline_store = []  # List of records dengan timestamp
        self.feature_definitions = {}  # feature_name -> metadata
        
    def register_feature(self, name: str, dtype: str,
                         description: str, transformation: Optional[Any] = None):
        """
        Register feature definition.
        
        Parameters:
        -----------
        name : str
            Feature name.
        dtype : str
            Data type ('numeric', 'categorical', 'boolean').
        description : str
            Human-readable description.
        transformation : callable, optional
            Function untuk transform raw value.
            
        Notes:
        ------
        - Feature registry adalah metadata catalog.
        - Setiap feature harus punya owner dan documentation.
        - Versioning: feature_name:v1, feature_name:v2.
        """
        self.feature_definitions[name] = {
            'dtype': dtype,
            'description': description,
            'transformation': transformation,
            'created_at': datetime.now()
        }
        print(f"Registered feature: {name} ({dtype})")
        
    def ingest_online(self, entity_id: str, features: Dict[str, Any]):
        """
        Ingest features ke online store.
        
        Parameters:
        -----------
        entity_id : str
            Unique identifier (e.g., user_id, device_id).
        features : dict
            Feature values.
            
        Notes:
        ------
        - Online store di-update secara real-time.
        - Setiap update meng-overwrite value sebelumnya.
        - TTL (time-to-live) bisa diatur untuk auto-expire.
        """
        if entity_id not in self.online_store:
            self.online_store[entity_id] = {}
        
        # Apply transformations jika ada
        for name, value in features.items():
            if name in self.feature_definitions:
                transform = self.feature_definitions[name]['transformation']
                if transform:
                    value = transform(value)
            self.online_store[entity_id][name] = value
            
    def ingest_offline(self, entity_id: str, features: Dict[str, Any],
                       timestamp: datetime):
        """
        Ingest features ke offline store dengan timestamp.
        
        Parameters:
        -----------
        entity_id : str
            Unique identifier.
        features : dict
            Feature values.
        timestamp : datetime
            Timestamp untuk point-in-time correctness.
            
        Notes:
        ------
        - Offline store menyimpan historical data.
        - Timestamp critical untuk point-in-time correctness.
        - Data di-append, tidak di-overwrite.
        """
        record = {
            'entity_id': entity_id,
            'timestamp': timestamp,
            **features
        }
        self.offline_store.append(record)
        
    def get_online_features(self, entity_id: str,
                            feature_names: List[str]) -> Dict[str, Any]:
        """
        Retrieve features dari online store.
        
        Parameters:
        -----------
        entity_id : str
            Entity identifier.
        feature_names : list
            List of feature names to retrieve.
            
        Returns:
        --------
        dict
            Feature values.
            
        Notes:
        ------
        - Latency critical: harus < 10ms
        - Missing features -> handle dengan default values
        - Fallback mechanism untuk partial failures
        """
        if entity_id not in self.online_store:
            return {name: None for name in feature_names}
        
        entity_features = self.online_store[entity_id]
        return {
            name: entity_features.get(name, None)
            for name in feature_names
        }
        
    def get_offline_features(self, entity_ids: List[str],
                             feature_names: List[str],
                             timestamps: List[datetime]) -> pd.DataFrame:
        """
        Retrieve historical features dengan point-in-time correctness.
        
        Parameters:
        -----------
        entity_ids : list
            List of entity identifiers.
        feature_names : list
            List of feature names.
        timestamps : list
            Point-in-time untuk setiap entity.
            
        Returns:
        --------
        pd.DataFrame
            Features pada waktu yang diminta.
            
        Notes:
        ------
        - Point-in-time: ambil features SEBELUM timestamp
        - Ini mencegah data leakage di training!
        - Complexity: O(n) dengan n = offline store size
        - Untuk production, gunakan time-series database
          atau data warehouse dengan partitioning.
        """
        df = pd.DataFrame(self.offline_store)
        results = []
        
        for entity_id, timestamp in zip(entity_ids, timestamps):
            # Filter: entity match dan timestamp <= requested time
            mask = (df['entity_id'] == entity_id) & (df['timestamp'] <= timestamp)
            entity_data = df[mask]
            
            if len(entity_data) == 0:
                results.append({'entity_id': entity_id})
                continue
            
            # Ambil record terbaru sebelum timestamp
            latest = entity_data.loc[entity_data['timestamp'].idxmax()]
            result = {'entity_id': entity_id}
            for name in feature_names:
                result[name] = latest.get(name, None)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_feature_statistics(self, feature_name: str) -> Dict[str, float]:
        """
        Compute statistics untuk feature.
        
        Parameters:
        -----------
        feature_name : str
            Feature name.
            
        Returns:
        --------
        dict
            Statistics (mean, std, min, max, missing %).
            
        Notes:
        ------
        - Feature statistics berguna untuk monitoring dan data quality.
        - Track drift dengan membandingkan statistics over time.
        - Missing value rate menunjukkan data quality issues.
        """
        df = pd.DataFrame(self.offline_store)
        if feature_name not in df.columns:
            return {}
        
        values = df[feature_name].dropna()
        return {
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            'missing_pct': df[feature_name].isna().mean() * 100,
            'count': len(values)
        }


# ===========================================================
# BAGIAN 3: Demo Feature Store
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 3: DEMO FEATURE STORE")
print("="*60)

# Initialize
store = SimpleFeatureStore()

# Register features
store.register_feature('voltage_rms', 'numeric',
                       'RMS voltage reading')
store.register_feature('current_rms', 'numeric',
                       'RMS current reading')
store.register_feature('power_factor', 'numeric',
                       'Power factor')
store.register_feature('temperature', 'numeric',
                       'Equipment temperature in Celsius')
store.register_feature('equipment_type', 'categorical',
                       'Type of equipment')

# Ingest online (real-time)
store.ingest_online('motor_001', {
    'voltage_rms': 220.5,
    'current_rms': 5.2,
    'power_factor': 0.92,
    'temperature': 45.0
})

# Ingest offline (historical)
now = datetime.now()
for i in range(10):
    store.ingest_offline('motor_001', {
        'voltage_rms': 220 + np.random.randn(),
        'current_rms': 5 + np.random.randn(),
        'power_factor': 0.9 + np.random.randn() * 0.05,
        'temperature': 40 + i * 2 + np.random.randn() * 2
    }, now - timedelta(hours=i))

# Retrieve online
online_features = store.get_online_features('motor_001',
    ['voltage_rms', 'current_rms', 'temperature'])
print(f"\nOnline features: {online_features}")

# Retrieve offline dengan point-in-time
offline_features = store.get_offline_features(
    ['motor_001'] * 3,
    ['voltage_rms', 'temperature'],
    [now - timedelta(hours=i) for i in [1, 3, 5]]
)
print(f"\nOffline features (point-in-time):\n{offline_features}")

# Statistics
stats = store.get_feature_statistics('temperature')
print(f"\nTemperature statistics: {stats}")


# ===========================================================
# BAGIAN 4: Feature Engineering Patterns
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 4: FEATURE ENGINEERING PATTERNS")
print("="*60)

feature_patterns = """
TARGET TIME-BASED FEATURES:

Window Aggregations:
  - Rolling mean: mean(X) over last N time units
  - Rolling std: standard deviation over window
  - Exponential moving average: weighted average dengan decay
  
  EE Connection: moving average = low-pass filter

Lag Features:
  - X(t-1), X(t-2), ..., X(t-k)
  - Capture temporal dependencies
  
  EE Connection: delay elements di digital filters

Time Since Events:
  - Time since last maintenance
  - Time since last failure
  - Time of day, day of week, season

TARGET DOMAIN-SPECIFIC FEATURES (EE):

Power Systems:
  - RMS, peak, crest factor
  - THD (Total Harmonic Distortion)
  - Power factor, real/reactive power
  - Voltage unbalance
  
  DETAIL:
  - THD = sqrt(sum(V_h^2)) / V_1, h = 2,3,...
  - Crest factor = peak / RMS
  - Power factor = P / |S| = cos(phi)

Signal Processing:
  - FFT coefficients
  - Spectral energy per band
  - Zero crossing rate
  - Signal entropy
  
  DETAIL:
  - FFT coefficients bisa jadi features untuk classification
  - Spectral energy = integral dari power spectral density
  - Zero crossing rate berguna untuk voice activity detection

Control Systems:
  - Settling time, rise time, overshoot
  - Steady-state error
  - Control effort

TARGET FEATURE TRANSFORMATION:

Scaling:
  - StandardScaler: z-score normalization
    z = (x - mu) / sigma
  - MinMaxScaler: scale ke [0, 1]
    x_scaled = (x - min) / (max - min)
  - RobustScaler: median dan IQR
    x_scaled = (x - median) / IQR
    Lebih robust terhadap outlier.

Encoding:
  - One-hot: categorical dengan few categories
  - Target encoding: categorical dengan many categories
    replace category dengan mean target value
  - Embedding: learnable representations
    Digunakan di deep learning untuk high-cardinality categories.

Interaction:
  - X1 * X2 (multiplicative)
  - X1 / X2 (ratio)
  - X1^2, sqrt(X1) (non-linear)

TARGET FEATURE SELECTION:

Filter Methods:
  - Correlation: remove highly correlated features
  - Mutual Information: relevance to target
  - Statistical tests: chi-square, ANOVA
  
  DETAIL:
  - Filter methods cepat dan scalable
  - Tapi tidak mempertimbangkan interaksi antar features

Wrapper Methods:
  - Forward selection: add features one by one
  - Backward elimination: remove features one by one
  - Recursive Feature Elimination (RFE)
  
  DETAIL:
  - Wrapper methods lebih akurat tapi computationally expensive
  - Bergantung pada model performance

Embedded Methods:
  - Lasso: L1 regularization (sparse)
  - Tree importance: feature importance dari random forest
  - Permutation importance: shuffle dan measure impact
  
  DETAIL:
  - Permutation importance adalah gold standard
    untuk interpretasi feature importance.
  - Shuffle satu feature, measure performance drop.
"""
print(feature_patterns)


# ===========================================================
# LATIHAN 22: Build Feature Store
# ===========================================================
"""
TARGET Learning Objectives:
   - Membangun feature store untuk use case tertentu
   - Mengimplementasikan point-in-time correctness
   - Membuat feature transformation pipeline

PANDUAN LANGKAH-LANGKAH:

STEP 1: Design Feature Store untuk Use Case
-------------------------------------------
   Pilih use case:
   
   a) Predictive Maintenance:
      - Entities: equipment_id
      - Features: sensor readings, operational hours, maintenance history
      - Online: current status untuk real-time monitoring
      - Offline: historical untuk training failure prediction
      
   b) Recommendation System:
      - Entities: user_id, item_id
      - Features: user behavior, item metadata, interaction history
      - Online: real-time personalization
      - Offline: batch model training
      
   c) Fraud Detection:
      - Entities: transaction_id, user_id, device_id
      - Features: transaction amount, velocity, device fingerprint
      - Online: real-time scoring
      - Offline: historical analysis


STEP 2: Implement Core Functionality
------------------------------------
   a) Feature registration dan versioning
   b) Online ingestion dan serving
   c) Offline ingestion dengan timestamps
   d) Point-in-time retrieval
   e) Feature statistics dan monitoring
   
   TIPS KENAPA versioning?
     - Features bisa berubah over time
     - Model trained dengan v1, serving dengan v1
     - New model trained dengan v2
     - Backward compatibility


STEP 3: Feature Transformation Pipeline
---------------------------------------
   a) Define transformations:
      - Scaling: StandardScaler, MinMaxScaler
      - Encoding: OneHotEncoder, TargetEncoder
      - Aggregation: rolling mean, cumulative sum
      - Domain-specific: THD, crest factor, power factor
      
   b) Pipeline execution:
      - Raw data -> Transformation -> Store
      - Reproducible: same raw data -> same features
      - Versioned: transformations punya version
      
   c) Testing:
      - Unit tests untuk setiap transformation
      - Integration tests untuk pipeline
      - Data quality checks


STEP 4: Integration dengan Model Training
-----------------------------------------
   a) Generate training dataset:
      - Specify: entity_ids, timestamps, feature_names
      - Retrieve: point-in-time correct features
      - Join dengan labels
      
   b) Point-in-time correctness validation:
      - Features dari waktu sebelum label
      - No data leakage
      - Reproducible
      
   c) Training pipeline:
      - Feature retrieval -> Model training -> Model evaluation
      - Automated dengan Airflow/Prefect
      - Track lineage: model -> features -> raw data


TIPS:
   - Point-in-time: feature_timestamp <= label_timestamp
   - Versioning: feature_name:v1, feature_name:v2
   - Backfill: compute historical features untuk new features
   - Monitoring: track feature distributions dan drift
   - Metadata: document setiap feature dengan owner, description

PERINGATAN COMMON MISTAKES:
   - Data leakage: features dari waktu setelah label
   - Tidak version features
   - Missing data tanpa proper handling
   - Feature computation yang tidak reproducible
   - No monitoring untuk feature drift

TARGET EXPECTED OUTPUT:
   - Feature store dengan online dan offline stores
   - Point-in-time correct feature retrieval
   - Feature transformation pipeline
   - Integration dengan model training
   - Feature monitoring dan statistics
"""


# ===========================================================
# CHALLENGE: Production Feature Store
# ===========================================================
"""
TARGET Learning Objectives:
   - Membangun production-ready feature store
   - Mengintegrasikan dengan existing data infrastructure
   - Implementasi monitoring dan governance

PANDUAN LANGKAH-LANGKAH:

STEP 1: Architecture Design
---------------------------
   Design feature store untuk manufacturing company:
   
   Data Sources:
   - SCADA systems (real-time sensors)
   - ERP systems (maintenance records, equipment specs)
   - Quality systems (inspection results)
   - External (weather, market data)
   
   Users:
   - Data Scientists (training data)
   - ML Engineers (feature serving)
   - Analysts (feature exploration)
   - Operations (monitoring)


STEP 2: Implementation
----------------------
   a) Online Store (Redis):
      - Key: equipment_id
      - Value: JSON dengan latest features
      - TTL: 24 hours
      - Latency: < 5ms
      
   b) Offline Store (BigQuery/Snowflake):
      - Table: features dengan partition by date
      - Schema: entity_id, timestamp, feature_1, ..., feature_n
      - Backfill: 2 years historical data
      
   c) Feature Computation (Spark/Flink):
      - Batch: nightly computation untuk historical
      - Streaming: real-time computation untuk online
      - Transformations: defined sebagai code
      
   d) API Layer (FastAPI):
      - /features/online: real-time retrieval
      - /features/offline: batch retrieval
      - /features/statistics: feature monitoring
      - /features/register: new feature registration


STEP 3: Monitoring & Governance
-------------------------------
   a) Feature Monitoring:
      - Distribution tracking (mean, std, percentiles)
      - Drift detection (PSI, KS test)
      - Missing value rate
      - Latency monitoring
      
   b) Data Quality:
      - Schema validation
      - Range checks
      - Freshness checks
      - Anomaly detection
      
   c) Governance:
      - Feature ownership
      - Access control
      - Lineage tracking
      - Documentation


STEP 4: Integration
-------------------
   a) Model Training:
      - Automated dataset generation
      - Point-in-time correctness
      - Feature versioning
      
   b) Model Serving:
      - Real-time feature lookup
      - Feature caching
      - Fallback untuk missing features
      
   c) Monitoring:
      - Feature drift alerts
      - Performance degradation alerts
      - Data quality alerts


TIPS:
   - Redis: hashes untuk features per entity
   - BigQuery: partitioned tables untuk query performance
   - Spark: DataFrame API untuk transformations
   - FastAPI: async endpoints untuk concurrent requests
   - Monitoring: Prometheus + Grafana

PERINGATAN COMMON MISTAKES:
   - Online dan offline inconsistency
   - No point-in-time correctness
   - Feature drift tanpa detection
   - No fallback untuk missing features
   - Scalability issues (single point of failure)
   - No data lineage tracking

TARGET EXPECTED OUTPUT:
   - Production feature store architecture
   - Working implementation dengan Redis + BigQuery
   - Feature monitoring dashboard
   - Data quality checks
   - Integration documentation

Ini adalah fondasi untuk MLOps di production!
"""

print("\n" + "="*50)
print("OK Modul selesai! Lanjut ke: 08-production-ml/02_model_monitoring.py")
print("="*50)
