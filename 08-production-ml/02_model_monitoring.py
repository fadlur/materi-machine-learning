"""
=============================================================
FASE 8 - MODUL 2: MODEL MONITORING
=============================================================
Model monitoring = memastikan model bekerja dengan baik
setelah deployment.

Mengapa monitoring penting?
- Data drift: input distribution berubah
- Concept drift: relationship antara input dan output berubah
- Performance degradation: accuracy menurun
- System issues: latency, errors, resource usage

Koneksi Teknik Elektro:
- Model monitoring = fault detection system
- Drift detection = change detection di signals
- Alerting = alarm system untuk abnormal conditions
- Performance tracking = efficiency monitoring

Durasi target: 3-4 jam
============================================================="""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

np.random.seed(42)


# ===========================================================
# BAGIAN 1: Drift Detection Methods
# ===========================================================
print("="*60)
print("BAGIAN 1: DRIFT DETECTION METHODS")
print("="*60)

drift_methods = """
TARGET JENIS DRIFT:

1. DATA DRIFT (Covariate Shift)
   - P(X) berubah, tapi P(Y|X) tetap
   - Contoh: sensor calibration drift, seasonal changes
   - Detection: compare feature distributions
   
   DETAIL:
   - Data drift terjadi ketika distribusi input berubah.
   - Model mungkin masih valid (P(Y|X) sama), tapi data yang
     diterima berbeda dari training data.
   - Contoh: sensor yang kalibrasinya bergeser sedikit.

2. CONCEPT DRIFT
   - P(Y|X) berubah, tapi P(X) tetap
   - Contoh: policy changes, economic shifts
   - Detection: compare model performance
   
   DETAIL:
   - Concept drift terjadi ketika relationship antara input
     dan output berubah.
   - Ini lebih serious karena model menjadi invalid.
   - Contoh: fraud pattern berubah karena fraudsters adapt.

3. LABEL DRIFT
   - P(Y) berubah
   - Contoh: fraud rate naik karena new attack pattern
   - Detection: monitor label distribution
   
   DETAIL:
   - Label drift terjadi ketika class balance berubah.
   - Bisa mempengaruhi model performance meski model valid.
   - Contoh: economic downturn meningkatkan default rate.

TARGET STATISTICAL TESTS UNTUK DRIFT:

1. KOLMOGOROV-SMIRNOV TEST (KS Test)
   - Compare two distributions
   - Null hypothesis: distributions are same
   - Statistic: maximum difference antara CDFs
   - Good untuk: continuous features
   
   DETAIL:
   - KS statistic = sup_x |F1(x) - F2(x)|
   - P-value < alpha -> reject null -> drift detected
   - Non-parametric: tidak memerlukan asumsi distribusi

2. POPULATION STABILITY INDEX (PSI)
   - Measure stability antara dua populations
   - Formula: Sum(%Actual - %Expected) * ln(%Actual / %Expected)
   - Threshold: PSI < 0.1 (stable), 0.1-0.25 (warning), > 0.25 (drift)
   - Good untuk: risk scoring, credit models
   
   DETAIL:
   - PSI sangat populer di financial industry.
   - Mengukur perubahan distribusi dalam bins.
   - Tapi sensitive ke binning strategy.

3. CHI-SQUARE TEST
   - Compare categorical distributions
   - Good untuk: categorical features
   
   DETAIL:
   - Chi-square = Sum((O - E)^2 / E)
   - O = observed frequency, E = expected frequency

4. WASSERSTEIN DISTANCE
   - Earth mover's distance
   - Intuitive: minimum "work" untuk transform satu distribution ke lainnya
   - Good untuk: multimodal distributions
   
   DETAIL:
   - Wasserstein distance lebih smooth dari KS statistic.
   - Bisa digunakan sebagai loss function (WGAN).
   - Lebih informatif untuk continuous distributions.

5. MAXIMUM MEAN DISCREPANCY (MMD)
   - Compare distributions di RKHS (Reproducing Kernel Hilbert Space)
   - Good untuk: high-dimensional data
   
   DETAIL:
   - MMD menggunakan kernel trick untuk compare distributions.
   - Bisa detect drift di embedding space.
   - Computationally more expensive.

TARGET MONITORING METRICS:

Model Performance:
  - Accuracy, Precision, Recall, F1
  - AUC-ROC, AUC-PR
  - Log loss, Brier score
  - Business metrics (revenue, cost savings)
  
  DETAIL:
  - Business metrics lebih penting dari ML metrics.
  - Contoh: "cost savings" lebih meaningful daripada "F1 score".
  - Always tie ML metrics ke business impact.

Data Quality:
  - Missing value rate
  - Feature range violations
  - Schema violations
  - Data freshness
  
  DETAIL:
  - Data quality issues bisa menyebabkan model failures.
  - Schema violations = pipeline broken.
  - Freshness = data tidak stale.

System Health:
  - Latency (p50, p95, p99)
  - Throughput (QPS)
  - Error rate
  - Resource usage (CPU, memory, GPU)
  
  DETAIL:
  - P99 latency lebih penting daripada average latency.
  - Resource usage menunjukkan capacity planning needs.

Koneksi Teknik Elektro:
- Drift detection = change point detection di time series
- PSI = spectral comparison (seperti THD analysis)
- KS test = comparing probability distributions
- Monitoring = SCADA system untuk model health
"""
print(drift_methods)


# ===========================================================
# BAGIAN 2: Drift Detection Implementation
# ===========================================================
# Drift detector membandingkan distribusi reference data
# (training data atau golden dataset) dengan current data.
#
# BEST PRACTICES:
# - Reference data harus representative dan "bersih".
# - Current data harus di-sample secara random untuk menghindari bias.
# - Multiple detection methods bisa digunakan untuk robustness.
# - Alert threshold harus di-tune berdasarkan false positive rate.

class DriftDetector:
    """
    Drift detection untuk model monitoring.
    
    Parameters:
    -----------
    reference_data : np.ndarray atau pd.DataFrame
        Baseline data (training data atau golden dataset).
    method : str, default 'ks'
        Detection method: 'ks', 'psi', 'wasserstein'.
    threshold : float, default 0.05
        Threshold untuk drift alert (p-value atau PSI).
        
    Notes:
    ------
    - Reference data = distribution yang dianggap "normal"
    - Current data = distribution yang sedang di-monitor
    - Alert jika drift terdeteksi
    - Pilih method berdasarkan tipe data dan use case
    
    Koneksi Teknik Elektro:
    - Reference = nominal operating condition
    - Current = current operating condition
    - Drift = deviation dari nominal
    - Alert = fault detection alarm
    """
    
    def __init__(self, reference_data, method='ks', threshold=0.05):
        self.reference = reference_data
        self.method = method
        self.threshold = threshold
        
    def detect_drift(self, current_data) -> Dict[str, any]:
        """
        Detect drift antara reference dan current data.
        
        Parameters:
        -----------
        current_data : np.ndarray atau pd.DataFrame
            Current data untuk compare.
            
        Returns:
        --------
        dict
            Drift detection results.
        """
        if self.method == 'ks':
            return self._ks_test(current_data)
        elif self.method == 'psi':
            return self._psi_test(current_data)
        elif self.method == 'wasserstein':
            return self._wasserstein_distance(current_data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _ks_test(self, current_data) -> Dict[str, any]:
        """Kolmogorov-Smirnov test."""
        statistic, p_value = stats.ks_2samp(self.reference, current_data)
        return {
            'method': 'KS',
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': p_value < self.threshold,
            'threshold': self.threshold
        }
    
    def _psi_test(self, current_data, bins=10) -> Dict[str, any]:
        """Population Stability Index."""
        # Create bins dari reference data
        min_val = min(self.reference.min(), current_data.min())
        max_val = max(self.reference.max(), current_data.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Calculate percentages
        ref_counts, _ = np.histogram(self.reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current_data, bins=bin_edges)
        
        ref_pct = ref_counts / len(self.reference)
        cur_pct = cur_counts / len(current_data)
        
        # Add small constant untuk avoid division by zero
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)
        
        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        return {
            'method': 'PSI',
            'psi_value': psi,
            'drift_detected': psi > self.threshold,
            'threshold': self.threshold,
            'severity': 'stable' if psi < 0.1 else 'warning' if psi < 0.25 else 'drift'
        }
    
    def _wasserstein_distance(self, current_data) -> Dict[str, any]:
        """Wasserstein distance (Earth Mover's Distance)."""
        distance = stats.wasserstein_distance(self.reference, current_data)
        return {
            'method': 'Wasserstein',
            'distance': distance,
            'drift_detected': distance > self.threshold,
            'threshold': self.threshold
        }


# ===========================================================
# BAGIAN 3: Model Performance Monitor
# ===========================================================
# Model performance monitor melacak metrics over time
# dan mendeteksi degradation trends.
#
# BEST PRACTICES:
# - Monitor multiple metrics, bukan hanya accuracy.
# - Gunakan moving average untuk menghaluskan noise.
# - Alert threshold harus consider seasonality.
# - Always compare dengan baseline (model sebelumnya).

class ModelMonitor:
    """
    Monitor model performance over time.
    
    Parameters:
    -----------
    model_name : str
        Name dari model yang di-monitor.
    metrics : list
        List of metrics untuk track.
        
    Notes:
    ------
    - Track performance metrics secara periodic
    - Detect degradation trends
    - Alert jika performance di bawah threshold
    - Degradation = perubahan yang signifikan dan sustained
    
    Koneksi Teknik Elektro:
    - Model monitor = performance meter
    - Metrics = efficiency indicators
    - Alerts = trip signals
    - Trends = degradation analysis
    """
    
    def __init__(self, model_name: str, metrics: List[str]):
        self.model_name = model_name
        self.metrics = metrics
        self.history = {metric: [] for metric in metrics}
        self.timestamps = []
        
    def log_performance(self, timestamp: datetime, **kwargs):
        """
        Log performance metrics.
        
        Parameters:
        -----------
        timestamp : datetime
            Timestamp dari measurement.
        **kwargs : dict
            Metric names dan values.
            
        Notes:
        ------
        - Log setiap kali model serving atau batch prediction.
        - Simpan juga input, prediction, dan ground truth
          untuk debugging dan retraining.
        """
        self.timestamps.append(timestamp)
        for metric, value in kwargs.items():
            if metric in self.history:
                self.history[metric].append(value)
    
    def check_degradation(self, metric: str, window: int = 7,
                          threshold: float = 0.1) -> bool:
        """
        Check if performance degraded.
        
        Parameters:
        -----------
        metric : str
            Metric untuk check.
        window : int, default 7
            Window size untuk moving average.
        threshold : float, default 0.1
            Degradation threshold (relative change).
            
        Returns:
        --------
        bool
            True jika degradation terdeteksi.
            
        Notes:
        ------
        - Compare recent window dengan previous window.
        - Threshold = relative change (e.g., 0.1 = 10% drop).
        - Gunakan moving average untuk reduce noise.
        """
        values = self.history[metric]
        if len(values) < window * 2:
            return False
        
        recent = np.mean(values[-window:])
        baseline = np.mean(values[-window*2:-window])
        
        change = (baseline - recent) / baseline if baseline != 0 else 0
        return change > threshold
    
    def plot_trends(self):
        """Plot performance trends."""
        fig, axes = plt.subplots(len(self.metrics), 1,
                                 figsize=(12, 3*len(self.metrics)))
        if len(self.metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, self.metrics):
            values = self.history[metric]
            ax.plot(range(len(values)), values, 'b-', linewidth=2)
            ax.set_ylabel(metric)
            ax.set_xlabel('Time')
            ax.set_title(f'{self.model_name} - {metric}')
            ax.grid(True, alpha=0.3)
            
            # Add threshold line jika ada
            if len(values) > 0:
                ax.axhline(y=np.mean(values), color='r',
                          linestyle='--', alpha=0.5, label='Mean')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig('01_model_monitoring.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("PLOT Saved: 01_model_monitoring.png")


# ===========================================================
# BAGIAN 4: Demo Drift Detection
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 4: DEMO DRIFT DETECTION")
print("="*60)

# Generate reference data (normal distribution)
reference = np.random.normal(100, 10, 1000)

# Generate current data dengan drift
current_normal = np.random.normal(100, 10, 1000)
current_drift = np.random.normal(120, 15, 1000)  # Mean shifted

# Detect drift
detector_ks = DriftDetector(reference, method='ks', threshold=0.05)
detector_psi = DriftDetector(reference, method='psi', threshold=0.25)

print("\n=== Drift Detection Results ===")
print("\nCase 1: No Drift")
result = detector_ks.detect_drift(current_normal)
print(f"  KS: statistic={result['statistic']:.4f}, p-value={result['p_value']:.4f}, "
      f"drift={result['drift_detected']}")

result = detector_psi.detect_drift(current_normal)
print(f"  PSI: value={result['psi_value']:.4f}, "
      f"drift={result['drift_detected']}, severity={result['severity']}")

print("\nCase 2: With Drift")
result = detector_ks.detect_drift(current_drift)
print(f"  KS: statistic={result['statistic']:.4f}, p-value={result['p_value']:.4f}, "
      f"drift={result['drift_detected']}")

result = detector_psi.detect_drift(current_drift)
print(f"  PSI: value={result['psi_value']:.4f}, "
      f"drift={result['drift_detected']}, severity={result['severity']}")

# Visualisasi
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(reference, bins=30, alpha=0.5, label='Reference', density=True)
axes[0].hist(current_normal, bins=30, alpha=0.5, label='Current (No Drift)', density=True)
axes[0].set_title('No Drift Detected')
axes[0].legend()

axes[1].hist(reference, bins=30, alpha=0.5, label='Reference', density=True)
axes[1].hist(current_drift, bins=30, alpha=0.5, label='Current (Drift)', density=True)
axes[1].set_title('Drift Detected!')
axes[1].legend()

plt.tight_layout()
plt.savefig('02_drift_visualization.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nPLOT Saved: 02_drift_visualization.png")


# ===========================================================
# BAGIAN 5: Monitoring Dashboard Concept
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 5: MONITORING DASHBOARD")
print("="*60)

monitoring_dashboard = """
TARGET KEY DASHBOARD COMPONENTS:

1. PERFORMANCE OVERVIEW
   - Current accuracy vs baseline
   - Trend over time (last 7, 30, 90 days)
   - Comparison dengan previous model version
   
   DETAIL:
   - Baseline = performance saat model pertama kali deploy.
   - Trend menunjukkan apakah performance menurun gradually.
   - Comparison dengan previous version untuk A/B test analysis.

2. DATA DRIFT PANEL
   - PSI values per feature
   - Distribution plots (reference vs current)
   - Drift alerts dan timeline
   
   DETAIL:
   - Feature-level drift detection menunjukkan feature mana yang berubah.
   - Distribution plots membantu understand jenis drift.
   - Timeline menunjukkan kapan drift mulai terjadi.

3. PREDICTION DISTRIBUTION
   - Score distribution over time
   - Calibration plots
   - Class distribution
   
   DETAIL:
   - Score distribution yang berubah bisa menunjukkan drift.
   - Calibration plot menunjukkan apakah probabilities reliable.
   - Class distribution menunjukkan label drift.

4. SYSTEM HEALTH
   - Latency percentiles
   - Throughput (QPS)
   - Error rate
   - Resource utilization
   
   DETAIL:
   - Latency p99 lebih informatif daripada mean.
   - Error rate spike bisa menunjukkan infrastructure issues.
   - Resource utilization membantu capacity planning.

5. BUSINESS IMPACT
   - Revenue impact
   - Cost savings
   - User engagement metrics
   - Conversion rates
   
   DETAIL:
   - Business impact adalah metrik paling penting.
   - Tie ML metrics ke business outcomes.
   - Track ROI dari ML system.

TARGET ALERTING RULES:

Critical Alerts (immediate action):
  - Accuracy drops > 10% dari baseline
  - PSI > 0.25 untuk key features
  - Error rate > 1%
  - Latency p99 > 500ms

Warning Alerts (investigate soon):
  - Accuracy drops 5-10%
  - PSI 0.1-0.25
  - Error rate 0.1-1%
  - Latency p99 200-500ms

Info Alerts (monitor):
  - Accuracy drops < 5%
  - PSI < 0.1
  - Feature correlations change
  - Data volume anomalies

TARGET RESPONSE PLAYBOOK:

Drift Detected:
  1. Verify: apakah drift real atau data pipeline issue?
  2. Analyze: feature mana yang drift?
  3. Impact: seberapa besar impact ke performance?
  4. Action:
     - Minor drift: monitor closely
     - Moderate drift: trigger retraining
     - Severe drift: rollback ke previous model
     
Performance Degradation:
  1. Check: apakah correlated dengan drift?
  2. Investigate: data quality issues?
  3. Action:
     - Data issue: fix pipeline
     - Model issue: retrain atau rollback
     - System issue: infrastructure fix

Koneksi Teknik Elektro:
- Dashboard = SCADA HMI (Human-Machine Interface)
- Alerts = relay protection trips
- Playbook = emergency procedures
- Trends = condition monitoring
"""
print(monitoring_dashboard)


# ===========================================================
# LATIHAN 23: Build Monitoring System
# ===========================================================
"""
TARGET Learning Objectives:
   - Membangun drift detection system
   - Mengimplementasikan performance monitoring
   - Membuat alerting mechanism

PANDUAN LANGKAH-LANGKAH:

STEP 1: Implement Multi-Feature Drift Detection
-----------------------------------------------
   a) Extend DriftDetector untuk support multiple features:
      - Input: DataFrame dengan multiple columns
      - Output: drift report per feature
      - Aggregate: overall drift score
      
   b) Methods:
      - KS test untuk continuous features
      - Chi-square untuk categorical features
      - PSI untuk risk features
      - Wasserstein untuk multimodal features
      
   c) Report generation:
      - Drift summary table
      - Feature importance ranking
      - Visualization: before/after distributions


STEP 2: Performance Monitoring Pipeline
---------------------------------------
   a) Log predictions dan ground truth:
      - Timestamp, input, prediction, actual, confidence
      - Store di database (PostgreSQL, BigQuery)
      
   b) Periodic evaluation:
      - Daily/weekly batch evaluation
      - Compute metrics: accuracy, precision, recall, F1
      - Compare dengan baseline
      
   c) Trend analysis:
      - Moving averages
      - Change point detection
      - Seasonal decomposition


STEP 3: Alerting System
-----------------------
   a) Define alert rules:
      - Threshold-based: metric > threshold
      - Statistical: metric outside confidence interval
      - Trend-based: degradation over time
      
   b) Alert channels:
      - Email
      - Slack/Teams
      - PagerDuty (critical)
      - Dashboard notifications
      
   c) Alert management:
      - Deduplication
      - Escalation policies
      - Alert history
      - Resolution tracking


STEP 4: Visualization Dashboard
-------------------------------
   a) Metrics over time:
      - Line charts untuk trends
      - Bar charts untuk comparisons
      - Heatmaps untuk feature drift
      
   b) Real-time monitoring:
      - Live updating charts
      - Current status indicators
      - Recent alerts
      
   c) Interactive exploration:
      - Time range selection
      - Feature filtering
      - Drill-down capabilities


TIPS:
   - Use Evidently AI untuk comprehensive drift reports
   - Prometheus + Grafana untuk system metrics
   - MLflow untuk experiment tracking
   - Great Expectations untuk data quality
   - Pandas untuk time series analysis

PERINGATAN COMMON MISTAKES:
   - Monitor hanya accuracy (ignore business metrics)
   - Alert threshold terlalu sensitive (alert fatigue)
   - Tidak track prediction latency
   - No baseline untuk comparison
   - Ignore data quality issues
   - No action playbook untuk alerts

TARGET EXPECTED OUTPUT:
   - Multi-feature drift detection system
   - Performance monitoring dengan trend analysis
   - Alerting system dengan multiple channels
   - Interactive dashboard
   - Response playbook
"""


# ===========================================================
# CHALLENGE: Production Monitoring System
# ===========================================================
"""
TARGET Learning Objectives:
   - Membangun production-grade monitoring system
   - Mengintegrasikan dengan existing infrastructure
   - Implementasi automated response

PANDUAN LANGKAH-LANGKAH:

STEP 1: Design Monitoring Architecture
--------------------------------------
   Konteks: Manufacturing defect detection model
   
   Components:
   - Prediction logger: log setiap prediction
   - Drift detector: analyze feature distributions
   - Performance tracker: compute metrics
   - Alert manager: send notifications
   - Dashboard: visualize metrics
   - Action executor: automated responses


STEP 2: Implementation
----------------------
   a) Prediction Logger:
      - FastAPI endpoint untuk log predictions
      - Store: BigQuery/PostgreSQL
      - Schema: timestamp, image_id, features, prediction,
        confidence, actual_label
      
   b) Drift Detector:
      - Daily batch job (Airflow)
      - Compare: last 7 days vs training data
      - Methods: PSI, KS, Wasserstein
      - Output: drift report
      
   c) Performance Tracker:
      - Weekly evaluation
      - Metrics: precision, recall, F1 per defect type
      - Business metrics: false positive cost,
        missed defect cost
      
   d) Alert Manager:
      - Slack notifications untuk warnings
      - Email untuk daily summaries
      - PagerDuty untuk critical alerts
      - Escalation: auto-create Jira ticket


STEP 3: Automated Response
--------------------------
   a) Minor drift:
      - Log dan monitor
      - Notify data science team
      
   b) Moderate drift:
      - Trigger data collection
      - Schedule retraining
      - Increase monitoring frequency
      
   c) Severe drift:
      - Auto-rollback ke previous model
      - Notify stakeholders
      - Switch ke heuristic/fallback
      - Emergency retraining


STEP 4: Dashboard dan Reporting
-------------------------------
   a) Executive dashboard:
      - Model accuracy trend
      - Business impact metrics
      - Alert summary
      
   b) Operational dashboard:
      - Real-time predictions
      - Feature distributions
      - System health metrics
      
   c) Weekly report:
      - Performance summary
      - Drift analysis
      - Recommendations
      - Action items


TIPS:
   - Use microservices architecture
   - Event-driven dengan Kafka
   - Containerize dengan Docker
   - Monitor the monitors (meta-monitoring)
   - Test alerts (alert testing)
   - Document runbooks

PERINGATAN COMMON MISTAKES:
   - Monitoring tanpa action
   - Alert fatigue (terlalu banyak alerts)
   - No escalation policy
   - Single point of failure
   - Tidak test failover
   - No documentation

TARGET EXPECTED OUTPUT:
   - Production monitoring system
   - Automated drift detection
   - Multi-channel alerting
   - Interactive dashboards
   - Automated response capabilities
   - Comprehensive documentation

Monitoring adalah kunci untuk production ML success!
"""

print("\n" + "="*50)
print("OK Modul selesai! Lanjut ke: 08-production-ml/03_llm_engineering.py")
print("="*50)
