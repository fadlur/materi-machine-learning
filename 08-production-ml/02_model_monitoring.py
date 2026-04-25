"""
=============================================================
FASE 8 — MODUL 2: MODEL MONITORING & DRIFT DETECTION
=============================================================
Model di production akan degradasi — itu pasti.
Tugas ML Engineer: deteksi DULU sebelum user complain.

Background backend kamu relevan:
- Monitoring tools (Prometheus, Grafana) → ML monitoring
- Alerting systems → Model degradation alerts
- Logging → Prediction logging & analysis

Durasi target: 3-4 hari
=============================================================
"""

import numpy as np
import pandas as pd
from scipy import stats

# ===========================================================
# 📖 BAGIAN 1: Kenapa Model Degradasi?
# ===========================================================

print("""
╔══════════════════════════════════════════════════════════╗
║     KENAPA MODEL DEGRADASI DI PRODUCTION?                ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  1. DATA DRIFT                                           ║
║     ┌─────────────┐    ┌─────────────┐                 ║
║     │ Training    │    │ Production  │                 ║
║     │ Age: 20-40  │───▶│ Age: 40-60  │                 ║
║     │ Income: 5K  │    │ Income: 8K  │                 ║
║     └─────────────┘    └─────────────┘                 ║
║     → Model tidak pernah lihat data seperti ini!        ║
║                                                          ║
║  2. CONCEPT DRIFT                                        ║
║     ┌─────────────┐    ┌─────────────┐                 ║
║     │ 2020: Work  │    │ 2024: Work  │                 ║
║     │ From Home   │───▶│ From Office │                 ║
║     │ = anomaly   │    │ = normal    │                 ║
║     └─────────────┘    └─────────────┘                 ║
║     → Relationship feature-target berubah!              ║
║                                                          ║
║  3. UPSTREAM DATA CHANGES                                ║
║     - Feature computation pipeline berubah              ║
║     - Schema drift (kolom baru/hilang)                  ║
║     - Data source berganti                              ║
║                                                          ║
║  4. ADVERSARIAL BEHAVIOR                                 ║
║     - Users adapt ke model (contoh: fraudsters)         ║
║     - Model predictions memengaruhi behavior            ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ===========================================================
# 📖 BAGIAN 2: Monitoring Metrics
# ===========================================================

print("""
╔══════════════════════════════════════════════════════════╗
║     WHAT TO MONITOR?                                     ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  1. DATA QUALITY                                         ║
║     - Missing value rate per feature                     ║
║     - Feature range violations                           ║
║     - Schema changes (column additions/removals)         ║
║     - Data freshness (last update timestamp)             ║
║                                                          ║
║  2. DATA DRIFT                                           ║
║     - Statistical tests: KS test, Chi-square, PSI        ║
║     - Distribution comparison (histograms)               ║
║     - Feature correlation changes                        ║
║                                                          ║
║  3. MODEL PERFORMANCE                                    ║
║     - Business metrics: conversion, revenue, CTR         ║
║     - ML metrics: accuracy, precision, recall, AUC       ║
║     - Latency: p50, p95, p99                             ║
║     - Throughput: predictions/second                     ║
║                                                          ║
║  4. PREDICTION DISTRIBUTION                              ║
║     - Output distribution shift                          ║
║     - Confidence score trends                            ║
║     - Class imbalance in predictions                     ║
║                                                          ║
║  5. SYSTEM HEALTH                                        ║
║     - API error rates                                    ║
║     - Resource usage (CPU, memory, GPU)                  ║
║     - Queue depth (untuk async processing)               ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ===========================================================
# 💻 CONTOH: Drift Detection Implementation
# ===========================================================

class DriftDetector:
    """
    Sederhana tapi powerful: deteksi drift dengan statistical tests.
    
    Di production, gunakan Evidently AI, WhyLabs, atau custom monitoring.
    """
    
    def __init__(self, reference_data, psi_threshold=0.2, ks_threshold=0.05):
        """
        Parameters:
        -----------
        reference_data : pd.DataFrame
            Data training / baseline
        psi_threshold : float
            Population Stability Index threshold
        ks_threshold : float
            Kolmogorov-Smirnov test p-value threshold
        """
        self.reference = reference_data
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.reference_distributions = {}
        
        # Pre-compute reference distributions
        for col in reference_data.select_dtypes(include=[np.number]).columns:
            self.reference_distributions[col] = {
                'mean': reference_data[col].mean(),
                'std': reference_data[col].std(),
                'hist': np.histogram(reference_data[col], bins=10, density=True)
            }
    
    def calculate_psi(self, expected, actual, buckets=10):
        """
        Population Stability Index (PSI).
        
        PSI < 0.1: No change
        0.1 ≤ PSI < 0.25: Slight change
        PSI ≥ 0.25: Significant change
        """
        # Bin data
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints[-1] += 1e-8  # Include max value
        
        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        
        # Avoid division by zero
        expected_percents = np.clip(expected_percents, 1e-10, 1)
        actual_percents = np.clip(actual_percents, 1e-10, 1)
        
        psi = np.sum((actual_percents - expected_percents) * 
                     np.log(actual_percents / expected_percents))
        return psi
    
    def detect_drift(self, current_data):
        """
        Detect drift antara reference dan current data.
        
        Returns:
        --------
        dict dengan drift report per feature
        """
        report = {}
        
        for col in self.reference_distributions:
            if col not in current_data.columns:
                report[col] = {'status': 'MISSING', 'alert': True}
                continue
            
            ref_values = self.reference[col].dropna()
            curr_values = current_data[col].dropna()
            
            # PSI test
            psi = self.calculate_psi(ref_values, curr_values)
            
            # KS test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_values, curr_values)
            
            # Determine status
            if psi >= self.psi_threshold:
                status = 'DRIFT'
            elif psi >= 0.1:
                status = 'WARNING'
            else:
                status = 'OK'
            
            report[col] = {
                'status': status,
                'psi': psi,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'alert': status in ['DRIFT', 'WARNING']
            }
        
        return report
    
    def print_report(self, report):
        """Print drift report dalam format yang readable."""
        print("\n" + "="*60)
        print("📊 DRIFT DETECTION REPORT")
        print("="*60)
        
        for feature, metrics in report.items():
            status_icon = {
                'OK': '✅',
                'WARNING': '⚠️',
                'DRIFT': '🔴',
                'MISSING': '❌'
            }.get(metrics['status'], '?')
            
            print(f"{status_icon} {feature:20s} | Status: {metrics['status']:10s} | "
                  f"PSI: {metrics.get('psi', 0):.4f}")
        
        alerts = sum(1 for m in report.values() if m.get('alert', False))
        print(f"\n🚨 Total alerts: {alerts}/{len(report)}")


# ===========================================================
# 💻 DEMO: Drift Detection
# ===========================================================

print("\n" + "="*50)
print("DEMO: Drift Detection")
print("="*50)

# Generate reference data (training distribution)
np.random.seed(42)
reference = pd.DataFrame({
    'feature_a': np.random.normal(100, 15, 1000),
    'feature_b': np.random.exponential(2, 1000),
    'feature_c': np.random.uniform(0, 1, 1000)
})

# Initialize detector
detector = DriftDetector(reference)

# Test 1: No drift (same distribution)
current_normal = pd.DataFrame({
    'feature_a': np.random.normal(100, 15, 500),
    'feature_b': np.random.exponential(2, 500),
    'feature_c': np.random.uniform(0, 1, 500)
})

print("\nTest 1: No Drift (same distribution)")
report = detector.detect_drift(current_normal)
detector.print_report(report)

# Test 2: Drift (shifted distribution)
current_drift = pd.DataFrame({
    'feature_a': np.random.normal(130, 20, 500),  # Shifted mean!
    'feature_b': np.random.exponential(5, 500),   # Different lambda!
    'feature_c': np.random.uniform(0, 1, 500)
})

print("\nTest 2: With Drift (shifted distribution)")
report = detector.detect_drift(current_drift)
detector.print_report(report)


# ===========================================================
# 📖 BAGIAN 3: Monitoring Infrastructure
# ===========================================================

print("""
╔══════════════════════════════════════════════════════════╗
║     MONITORING INFRASTRUCTURE SETUP                      ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  PATTERN 1: Log & Analyze                                ║
║  ┌─────────┐    ┌─────────┐    ┌─────────┐             ║
║  │ Model   │───▶│ Logs    │───▶│ Analysis│             ║
║  │ Serving │    │ (JSON)  │    │ (Daily) │             ║
║  └─────────┘    └─────────┘    └─────────┘             ║
║  Simple, tapi reactive (laggy)                           ║
║                                                          ║
║  PATTERN 2: Real-time Metrics                            ║
║  ┌─────────┐    ┌─────────┐    ┌─────────┐             ║
║  │ Model   │───▶│Prometheus│───▶│Grafana  │             ║
║  │ Serving │    │Metrics  │    │Dashboard│             ║
║  └─────────┘    └─────────┘    └─────────┘             ║
║  Real-time, butuh infrastructure                         ║
║                                                          ║
║  PATTERN 3: Dedicated ML Monitoring (Recommended)        ║
║  ┌─────────┐    ┌─────────┐    ┌─────────┐             ║
║  │ Model   │───▶│Evidently│───▶│Dashboard│             ║
║  │ Serving │    │ /WhyLabs│    │ /Alerts │             ║
║  └─────────┘    └─────────┘    └─────────┘             ║
║  Built for ML, drift detection included                  ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ===========================================================
# 💻 CONTOH: Prediction Logging & Monitoring
# ===========================================================

class ModelMonitor:
    """
    Simple model monitoring system.
    
    Di production, gunakan ini sebagai starting point,
    lalu migrate ke Evidently AI atau custom monitoring.
    """
    
    def __init__(self, model_name, version):
        self.model_name = model_name
        self.version = version
        self.predictions_log = []
        self.metrics_history = []
    
    def log_prediction(self, features, prediction, probability=None, 
                       latency_ms=None, timestamp=None):
        """Log setiap prediction untuk analysis."""
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        log_entry = {
            'timestamp': timestamp,
            'model_name': self.model_name,
            'model_version': self.version,
            'prediction': prediction,
            'probability': probability,
            'latency_ms': latency_ms
        }
        
        # Add feature snapshots (select key features)
        if isinstance(features, dict):
            for key, value in features.items():
                log_entry[f'feature_{key}'] = value
        
        self.predictions_log.append(log_entry)
    
    def compute_metrics(self, window='1H'):
        """Compute metrics dalam time window."""
        if not self.predictions_log:
            return {}
        
        df = pd.DataFrame(self.predictions_log)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        metrics = {}
        
        # Prediction distribution
        metrics['prediction_distribution'] = df['prediction'].value_counts().to_dict()
        
        # Latency stats
        if 'latency_ms' in df.columns:
            metrics['latency_p50'] = df['latency_ms'].median()
            metrics['latency_p95'] = df['latency_ms'].quantile(0.95)
            metrics['latency_p99'] = df['latency_ms'].quantile(0.99)
        
        # Throughput
        metrics['predictions_per_hour'] = len(df) / ((df.index.max() - df.index.min()).total_seconds() / 3600 + 0.001)
        
        return metrics
    
    def generate_alert(self, metrics):
        """Generate alert kalau metrics di luar threshold."""
        alerts = []
        
        if metrics.get('latency_p95', 0) > 200:
            alerts.append("🔴 Latency p95 > 200ms")
        
        if metrics.get('latency_p99', 0) > 500:
            alerts.append("🔴 Latency p99 > 500ms")
        
        pred_dist = metrics.get('prediction_distribution', {})
        if pred_dist:
            total = sum(pred_dist.values())
            max_class_ratio = max(pred_dist.values()) / total
            if max_class_ratio > 0.95:
                alerts.append("⚠️ Prediction distribution skewed (>95% one class)")
        
        return alerts


# ===========================================================
# 🏋️ EXERCISE 1: Build Monitoring Dashboard
# ===========================================================
"""
Buat monitoring dashboard dengan Streamlit:

Requirements:
1. Upload reference data dan current data
2. Tampilkan drift detection report (PSI, KS test)
3. Visualisasi distribution comparison (histogram side-by-side)
4. Alert panel untuk features yang drift
5. Export report ke JSON/CSV

Streamlit app structure:
- Sidebar: upload files, configure thresholds
- Main: tabs for Overview, Feature Analysis, Alerts
- Use plotly for interactive charts
"""


# ===========================================================
# 🏋️ EXERCISE 2: A/B Testing Framework
# ===========================================================
"""
Implementasi sederhana A/B testing untuk model:

Requirements:
1. Route traffic: 50% model A, 50% model B
2. Log predictions dari kedua model
3. Compute metrics per model
4. Statistical significance test (t-test untuk continuous, chi-square untuk binary)
5. Auto-promote model B kalau significantly better

Design:
class ABTestFramework:
    def __init__(self, model_a, model_b, traffic_split=0.5)
    def predict(self, features) -> route ke model A atau B
    def evaluate(self) -> comparison report
    def should_promote(self) -> bool
"""


# ===========================================================
# 🔥 CHALLENGE: End-to-End Monitoring System
# ===========================================================
"""
Bangun monitoring system untuk project kamu:

Components:
1. Prediction logger (middleware di FastAPI)
2. Drift detector (schedule: daily batch)
3. Metrics dashboard (Streamlit atau Grafana)
4. Alert system (email/Slack kalau drift detected)
5. Model performance tracker (compare dengan baseline)

Deliverables:
- Code repository
- Docker Compose setup (app + monitoring)
- Documentation: how to add new metrics, how to configure alerts
- Demo: simulasi drift dan lihat alert
"""


print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 08-production-ml/03_llm_engineering.py")
print("="*50)
