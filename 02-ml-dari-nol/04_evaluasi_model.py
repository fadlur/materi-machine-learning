"""
=============================================================
FASE 2 — MODUL 4: EVALUASI MODEL — THE COMPLETE GUIDE
=============================================================
"A model is only as good as its evaluation."

Ini modul yang sering di-skip di tutorial, padahal:
- Evaluasi yang salah → keputusan yang salah → deployment gagal
- Data scientist yang jago evaluasi > yang jago bikin model

Setelah modul ini, kamu akan bisa:
1. Memilih metrik yang tepat untuk problem yang tepat
2. Mendeteksi data leakage
3. Melakukan proper cross-validation
4. Membandingkan model secara statistik valid

Durasi target: 2-3 jam
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ===========================================================
# 📖 BAGIAN 1: Regression Metrics Lengkap
# ===========================================================

def regression_metrics(y_true, y_pred):
    """
    Menghitung semua metrik regresi yang penting.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth values.
        
    y_pred : np.ndarray
        Predicted values.
        
    Returns:
    --------
    dict
        Dictionary dengan keys: 'mse', 'rmse', 'mae', 'r2'.
        
    Notes:
    ------
    Metrik yang dihitung:
    - MSE (Mean Squared Error): rata-rata squared errors.
      Penalti besar untuk error besar (outliers).
    - RMSE (Root MSE): akar MSE. Satuan sama dengan target.
    - MAE (Mean Absolute Error): rata-rata absolute errors.
      Robust terhadap outliers.
    - MAPE (Mean Absolute Percentage Error): error dalam persen.
    - R²: proporsi variance yang di-explain oleh model.
    - Adjusted R²: R² yang dikoreksi untuk jumlah fitur.
    
    Koneksi Teknik Elektro:
    - MSE = mean squared error di signal processing
    - RMSE = RMS (Root Mean Square) value
    - R² = SNR-like measure (signal explained / total signal)
    """
    residuals = y_true - y_pred
    
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals / (y_true + 1e-8))) * 100
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res / ss_tot
    
    n = len(y_true)
    # Adjusted R² — penalti untuk fitur yang banyak
    # p = jumlah fitur (kita estimasi, default 1)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)
    
    print("=== Regression Metrics ===")
    print(f"MSE:  {mse:.4f}  (penalti besar untuk error besar)")
    print(f"RMSE: {rmse:.4f}  (dalam satuan yang sama dengan y)")
    print(f"MAE:  {mae:.4f}  (robust terhadap outlier)")
    print(f"MAPE: {mape:.2f}% (persentase error)")
    print(f"R²:   {r2:.4f}  (proporsi variance yang di-explain)")
    print(f"Adj R²: {adj_r2:.4f} (R² yang adjust untuk jumlah fitur)")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


# Demo
y_true = np.random.randn(100) * 3 + 10
y_pred_good = y_true + np.random.randn(100) * 0.5
y_pred_bad = y_true + np.random.randn(100) * 3

print("Model BAIK:")
metrics_good = regression_metrics(y_true, y_pred_good)
print("\nModel BURUK:")
metrics_bad = regression_metrics(y_true, y_pred_bad)

# Residual analysis — PENTING untuk diagnosa model
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Actual vs Predicted
axes[0, 0].scatter(y_true, y_pred_good, alpha=0.5, s=20)
axes[0, 0].plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()], 'r--')
axes[0, 0].set_xlabel('Actual')
axes[0, 0].set_ylabel('Predicted')
axes[0, 0].set_title('Actual vs Predicted (titik harus di garis merah)')

# Residual plot
residuals = y_true - y_pred_good
axes[0, 1].scatter(y_pred_good, residuals, alpha=0.5, s=20)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Residual')
axes[0, 1].set_title('Residual Plot (harus random, no pattern!)')

# Residual distribution
axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Residual Distribution (harus normal)')
axes[1, 0].set_xlabel('Residual')

# QQ plot manual
sorted_residuals = np.sort(residuals)
theoretical = np.sort(np.random.randn(len(residuals)))
axes[1, 1].scatter(theoretical, sorted_residuals, alpha=0.5, s=20)
axes[1, 1].plot([-3, 3], [-3, 3], 'r--')
axes[1, 1].set_xlabel('Theoretical Quantiles')
axes[1, 1].set_ylabel('Sample Quantiles')
axes[1, 1].set_title('QQ Plot (titik harus di garis)')

plt.tight_layout()
plt.savefig('01_regression_diagnostics.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n📊 Saved: 01_regression_diagnostics.png")


# ===========================================================
# 📖 BAGIAN 2: Classification Metrics Lengkap
# ===========================================================

def classification_report(y_true, y_pred, y_proba=None, class_names=None):
    """
    Complete classification report dari nol.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth labels.
        
    y_pred : np.ndarray
        Predicted labels.
        
    y_proba : np.ndarray, optional
        Predicted probabilities (untuk AUC, dll).
        
    class_names : list of str, optional
        Nama kelas untuk display.
        
    Returns:
    --------
    np.ndarray
        Confusion matrix.
        
    Notes:
    ------
    Metrics per class:
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)
    - F1 = 2 * P * R / (P + R)
    - Support = jumlah samples per kelas
    
    Weighted average:
    - Rata-rata weighted by support (jumlah samples per kelas)
    """
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    if class_names is None:
        class_names = [f'Class {c}' for c in classes]
    
    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        i = np.where(classes == t)[0][0]
        j = np.where(classes == p)[0][0]
        cm[i][j] += 1
    
    print("=== Confusion Matrix ===")
    header = "         " + "  ".join(f"Pred_{c}" for c in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = f"True_{name:>6}: " + "  ".join(f"{cm[i][j]:5d}" for j in range(n_classes))
        print(row)
    
    # Per-class metrics
    print("\n=== Per-Class Metrics ===")
    print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 50)
    
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    for i, name in enumerate(class_names):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = cm[i, :].sum()
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        
        print(f"{name:<10} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10d}")
    
    # Weighted average
    total = sum(supports)
    w_precision = sum(p * s for p, s in zip(precisions, supports)) / total
    w_recall = sum(r * s for r, s in zip(recalls, supports)) / total
    w_f1 = sum(f * s for f, s in zip(f1s, supports)) / total
    
    print("-" * 50)
    print(f"{'Weighted':<10} {w_precision:>10.4f} {w_recall:>10.4f} {w_f1:>10.4f} {total:>10d}")
    print(f"\nAccuracy: {np.mean(y_true == y_pred):.4f}")
    
    return cm


# Demo
y_true_mc = np.array([0]*30 + [1]*30 + [2]*40)
y_pred_mc = np.array(
    [0]*25 + [1]*3 + [2]*2 +     # class 0
    [0]*5 + [1]*20 + [2]*5 +     # class 1
    [0]*2 + [1]*3 + [2]*35       # class 2
)
cm = classification_report(y_true_mc, y_pred_mc,
                            class_names=['Normal', 'Warning', 'Fault'])


# ===========================================================
# 📖 BAGIAN 3: Proper Cross-Validation
# ===========================================================

class CrossValidator:
    """
    Proper cross-validation — menghindari data leakage.
    
    Methods:
    --------
    k_fold_indices : static method
        Generate k-fold indices.
    stratified_k_fold_indices : static method
        Stratified k-fold yang preserves class distribution.
    time_series_split : static method
        Time series split untuk data temporal.
        
    Notes:
    ------
    - K-Fold CV: data dibagi ke k fold, tiap fold jadi test sekali
    - Stratified: proporsi kelas dijaga di setiap fold
    - Time Series: training hanya dari masa lalu, test dari masa depan
    - Koneksi Teknik Elektro: mirip dengan k-fold averaging
      di measurement uncertainty analysis
    """
    
    @staticmethod
    def k_fold_indices(n, k=5, seed=42):
        """
        Generate k-fold indices.
        
        Parameters:
        -----------
        n : int
            Total jumlah samples.
        k : int, default 5
            Jumlah fold.
        seed : int, default 42
            Random seed.
            
        Returns:
        --------
        list of np.ndarray
            List berisi k array of indices.
        """
        np.random.seed(seed)
        indices = np.random.permutation(n)
        fold_sizes = np.full(k, n // k)
        fold_sizes[:n % k] += 1
        folds = []
        current = 0
        for size in fold_sizes:
            folds.append(indices[current:current + size])
            current += size
        return folds
    
    @staticmethod
    def stratified_k_fold_indices(y, k=5, seed=42):
        """
        Stratified k-fold — preserves class distribution.
        
        Parameters:
        -----------
        y : np.ndarray
            Target labels.
        k : int, default 5
            Jumlah fold.
        seed : int, default 42
            Random seed.
            
        Returns:
        --------
        list of np.ndarray
            List berisi k array of indices.
            
        Notes:
        ------
        - Setiap fold memiliki proporsi kelas yang sama dengan data asli
        - Penting untuk imbalanced datasets
        - Implementasi: split per class, lalu combine
        """
        np.random.seed(seed)
        classes = np.unique(y)
        class_indices = [np.where(y == c)[0] for c in classes]
        
        folds = [[] for _ in range(k)]
        for idx_list in class_indices:
            np.random.shuffle(idx_list)
            fold_sizes = np.full(k, len(idx_list) // k)
            fold_sizes[:len(idx_list) % k] += 1
            current = 0
            for i, size in enumerate(fold_sizes):
                folds[i].extend(idx_list[current:current + size])
                current += size
        
        return [np.array(f) for f in folds]
    
    @staticmethod
    def time_series_split(n, n_splits=5, min_train=None):
        """
        Time Series Split — WAJIB untuk data temporal!
        
        Parameters:
        -----------
        n : int
            Total jumlah samples.
        n_splits : int, default 5
            Jumlah splits.
        min_train : int, optional
            Minimum jumlah training samples untuk split pertama.
            
        Returns:
        --------
        list of tuple
            List berisi (train_idx, test_idx) untuk setiap split.
            
        Notes:
        ------
        - Training set selalu dari masa lalu
        - Test set selalu dari masa depan
        - Ini mencegah data leakage dari future ke past
        - Jangan random shuffle untuk time series!
        
        Kenapa? Karena data masa depan TIDAK BOLEH dipakai
        untuk melatih model yang memprediksi masa depan.
        Ini sumber data leakage paling umum!
        """
        if min_train is None:
            min_train = n // (n_splits + 1)
        
        splits = []
        test_size = (n - min_train) // n_splits
        for i in range(n_splits):
            train_end = min_train + i * test_size
            test_end = min(train_end + test_size, n)
            splits.append((np.arange(train_end), np.arange(train_end, test_end)))
        return splits


# Visualisasi time series split
n = 100
splits = CrossValidator.time_series_split(n, n_splits=5)

fig, ax = plt.subplots(figsize=(12, 4))
for i, (train_idx, test_idx) in enumerate(splits):
    ax.barh(i, len(train_idx), left=0, height=0.3, color='blue', alpha=0.5)
    ax.barh(i, len(test_idx), left=len(train_idx), height=0.3, color='red', alpha=0.5)
ax.set_yticks(range(len(splits)))
ax.set_yticklabels([f'Split {i+1}' for i in range(len(splits))])
ax.set_xlabel('Sample Index')
ax.set_title('Time Series Cross-Validation (🟦 Train, 🟥 Test)')
ax.legend(['Train', 'Test'])
plt.tight_layout()
plt.savefig('02_time_series_cv.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n📊 Saved: 02_time_series_cv.png")


# ===========================================================
# 📖 BAGIAN 4: Data Leakage — Kesalahan Paling Berbahaya
# ===========================================================

print("\n" + "="*50)
print("⚠️  DATA LEAKAGE — The Silent Killer")
print("="*50)
print("""
Data leakage = informasi dari test set bocor ke training process.
Hasil: model terlihat BAGUS di evaluation, tapi GAGAL di production.

CONTOH LEAKAGE UMUM:

1. ❌ Normalize SEMUA data sebelum split
   → Test data ikut menentukan mean/std
   ✅ Normalize HANYA di training data, apply ke test

2. ❌ Feature selection menggunakan SEMUA data
   → Test data ikut menentukan fitur mana yang penting
   ✅ Feature selection HANYA di training fold

3. ❌ Random split pada time series data
   → Model "melihat" masa depan
   ✅ Pakai time series split (lihat di atas)

4. ❌ Duplicate data yang tersebar di train dan test
   → Model hapal, bukan belajar
   ✅ Deduplicate sebelum split

5. ❌ Target encoding di SEMUA data
   → Test labels bocor ke features
   ✅ Target encoding HANYA di training fold

Ingat: Leakage tidak akan tertangkap oleh cross-validation biasa
jika preprocessing dilakukan SEBELUM split!
""")

# Demo leakage
n = 200
X_demo = np.random.randn(n, 5)
y_demo = np.random.randint(0, 2, n)  # RANDOM labels — tidak ada pattern!

# ❌ WRONG: Normalize semua data dulu, baru split
X_leaked = (X_demo - X_demo.mean(axis=0)) / X_demo.std(axis=0)

# ✅ RIGHT: Split dulu, baru normalize
train_idx = np.arange(int(0.8 * n))
test_idx = np.arange(int(0.8 * n), n)

X_train = X_demo[train_idx]
X_test = X_demo[test_idx]

# Normalize menggunakan HANYA statistik training
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
X_train_proper = (X_train - train_mean) / train_std
X_test_proper = (X_test - train_mean) / train_std  # pakai mean/std TRAINING!

print("Demo: efek leakage pada normalisasi")
print(f"  Leaked test mean:  {X_leaked[test_idx].mean(axis=0).round(3)}")
print(f"  Proper test mean:  {X_test_proper.mean(axis=0).round(3)}")
print("  → Leaked mean ≈ 0 (informasi bocor), Proper mean ≠ 0 (realistic)")


# ===========================================================
# 📖 BAGIAN 5: Hyperparameter Tuning
# ===========================================================

def grid_search_cv(X, y, param_grid, model_class, k=5):
    """
    Manual grid search with cross-validation.
    
    Parameters:
    -----------
    X : np.ndarray
        Features.
    y : np.ndarray
        Targets.
    param_grid : list of dict
        List parameter combinations.
    model_class : class
        Model class.
    k : int, default 5
        Jumlah fold.
        
    Returns:
    --------
    best_params : dict
        Parameter terbaik.
    results : list
        List hasil untuk setiap parameter combination.
        
    Notes:
    ------
    - Untuk setiap parameter combination:
      a) Jalankan k-fold CV
      b) Hitung mean dan std score
    - Pilih parameter dengan mean score tertinggi
    - Proper normalization inside CV loop!
    """
    best_score = -np.inf
    best_params = None
    results = []
    
    folds = CrossValidator.k_fold_indices(len(X), k)
    
    for params in param_grid:
        fold_scores = []
        
        for i in range(k):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Proper normalization inside CV loop!
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0) + 1e-8
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std
            
            model = model_class(**params)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        results.append((params, mean_score, std_score))
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
        
        print(f"  {params}: {mean_score:.4f} ± {std_score:.4f}")
    
    print(f"\n  Best: {best_params} → {best_score:.4f}")
    return best_params, results


# ===========================================================
# 🏋️ EXERCISE 7: Evaluasi Lengkap
# ===========================================================
"""
🎯 Learning Objectives:
   - Membangun proper ML pipeline end-to-end
   - Menerapkan nested cross-validation
   - Mengimplementasikan bootstrap confidence interval

📋 LANGKAH-LANGKAH:

STEP 1: Implementasi Proper ML Pipeline
───────────────────────────────────────
Buat function run_pipeline(X, y, model_class, model_params) yang:

   a) Load/generate data
   b) Train-test split (80/20) dengan random seed
   c) Proper normalization:
      - Fit scaler pada X_train ONLY
      - Transform X_train dan X_test
      
   d) Train model
   e) Evaluate dengan SEMUA metrik yang relevan:
      - Regression: MSE, RMSE, MAE, R²
      - Classification: Accuracy, Precision, Recall, F1, AUC
      
   f) Visualisasi:
      - Actual vs Predicted (regression)
      - Confusion matrix (classification)
      - Residual plot (regression)

   💡 KENAPA pipeline penting?
     - Mencegah data leakage
     - Reproducible (dengan random seed)
     - Mudah di-extend untuk berbagai model


STEP 2: Implementasi Nested Cross-Validation
─────────────────────────────────────────────
Nested CV = CV di dalam CV.

   Struktur:
   - Outer loop: 5-fold untuk estimasi performance
   - Inner loop: 3-fold untuk hyperparameter tuning
   
   💡 Apa yang harus dilakukan:
     a) Bagi data ke 5 fold (outer)
     b) Untuk setiap outer fold:
        - Outer test = fold i
        - Outer train = fold lainnya
        - Bagi outer train ke 3 fold (inner)
        - Untuk setiap parameter:
          - Inner CV: train pada 2 inner fold, test pada 1 inner fold
          - Hitung mean inner score
        - Pilih best parameter berdasarkan inner CV
        - Train final model pada outer train dengan best parameter
        - Evaluate pada outer test
        
     c) Report: mean ± std dari 5 outer scores
     
   💡 KENAPA nested CV?
     - Single CV bisa overestimate performance
     - Karena hyperparameter dipilih berdasarkan CV score yang SAMA
     - Nested CV memberikan unbiased estimate
     - Gold standard untuk small datasets

   ⚠️ Hati-hati:
     - Jangan pakai outer test untuk tuning hyperparameter!
     - Inner CV HANYA pada outer train
     - Computationally expensive


STEP 3: Implementasi Bootstrap Confidence Interval
───────────────────────────────────────────────────
Bootstrap = resampling untuk estimasi uncertainty.

   💡 Apa yang harus dilakukan:
     a) Setelah model di-train, dapatkan test predictions
     b) Resample test set dengan replacement sebanyak 1000x
     c) Untuk setiap resample:
        - Hitung metric (accuracy, F1, dll.)
     d) Dari 1000 metrics:
        - Sort
        - 95% CI = [percentile 2.5, percentile 97.5]
        
   💡 KENAPA bootstrap?
     - Memberikan confidence interval untuk metrics
     - Menunjukkan stabilitas model
     - Membantu decision: apakah improvement signifikan?
     
   Contoh output:
   ```
   Accuracy: 0.85
   95% CI: [0.82, 0.88]
   ```
   Artinya: kita 95% yakin true accuracy ada di [0.82, 0.88]


💡 HINTS:
   - Untuk nested CV, gunakan 2 level loop
   - Untuk bootstrap, gunakan np.random.choice(n, size=n, replace=True)
   - Simpan semua scores untuk analisis variance
   - Gunakan box plot untuk visualisasi distribution

⚠️ COMMON MISTAKES:
   - Data leakage di normalization step
   - Menggunakan test set untuk hyperparameter tuning
   - Bootstrap tanpa replacement
   - Confidence interval menggunakan standard error, bukan percentiles

🎯 EXPECTED OUTPUT:
   - Pipeline yang proper dan reusable
   - Nested CV score yang realistic (biasanya lebih rendah dari simple CV)
   - Bootstrap CI yang narrow untuk dataset besar, wide untuk dataset kecil
   - Analisis: apakah model A significantly better dari model B?
"""


# ===========================================================
# 🔥 CHALLENGE: Benchmarking Framework
# ===========================================================
"""
🎯 Learning Objectives:
   - Membangun reusable benchmarking framework
   - Melakukan statistical comparison antar model
   - Membuat visualisasi comprehensive untuk model comparison

📋 LANGKAH-LANGKAH:

STEP 1: Design Benchmarking Framework
──────────────────────────────────────
Buat class ModelBenchmark dengan interface:

   class ModelBenchmark:
       def __init__(self, models_dict, dataset, k=5):
           '''
           models_dict: {'ModelName': (ModelClass, params)}
           dataset: (X, y)
           '''
           
       def run_cv(self):
           '''Jalankan proper CV untuk semua model'''
           
       def statistical_test(self):
           '''Paired t-test antar model'''
           
       def plot_results(self):
           '''Generate visualisasi'''
           
       def generate_report(self):
           '''Output summary report'''


STEP 2: Implementasi .run_cv()
─────────────────────────────
   Untuk setiap model:
   a) Jalankan k-fold CV
   b) Simpan scores per fold
   c) Hitung mean, std, min, max, median
   d) Catat training time per fold
   
   Output: dictionary dengan semua metrics


STEP 3: Implementasi .statistical_test()
────────────────────────────────────────
   Paired t-test antar setiap pasangan model:
   
   H0: μ_A - μ_B = 0 (tidak ada perbedaan signifikan)
   H1: μ_A - μ_B ≠ 0 (ada perbedaan signifikan)
   
   Gunakan scipy.stats.ttest_rel(scores_A, scores_B)
   
   💡 KENAPA paired t-test?
     - Scores dari fold yang sama = paired
     - Paired test lebih powerful dari independent test
     - Menunjukkan apakah improvement signifikan secara statistik


STEP 4: Implementasi .plot_results()
────────────────────────────────────
Buat figure dengan 4 subplot:

   a) Box plot: performance distribution per model
   b) Bar plot: mean score dengan error bar (std)
   c) Heatmap: p-values dari paired t-test
      - Rows dan columns = model names
      - Cell [i,j] = p-value(model_i vs model_j)
      - Color: green (p > 0.05, not significant), red (p < 0.05, significant)
      
   d) Learning curves: convergence per model (kalau ada)


STEP 5: Implementasi .generate_report()
───────────────────────────────────────
Output text report:

   ```
   === Model Benchmark Report ===
   
   Dataset: X (n_samples, n_features)
   CV: 5-fold
   
   Ranking:
   1. Model A: 0.92 ± 0.03
   2. Model B: 0.89 ± 0.04
   3. Model C: 0.85 ± 0.05
   
   Statistical Significance (p < 0.05):
   - Model A significantly better than Model B (p = 0.02)
   - Model B significantly better than Model C (p = 0.01)
   - Model A vs Model C: p = 0.001
   
   Recommendation:
   Deploy Model A. Significantly better than alternatives
   with low variance across folds.
   ```


💡 HINTS:
   - Gunakan scipy.stats untuk statistical tests
   - Gunakan seaborn untuk heatmap
   - Simpan semua raw scores untuk post-hoc analysis
   - Box plot lebih informatif dari bar plot untuk distributions

⚠️ COMMON MISTAKES:
   - Independent t-test instead of paired t-test
   - Tidak koreksi untuk multiple comparisons (Bonferroni)
   - Menginterpretasi p-value sebagai effect size
   - Tidak melihat variance (hanya mean)

🎯 EXPECTED OUTPUT:
   - Class ModelBenchmark yang reusable
   - Comprehensive visualisasi dalam satu figure
   - Statistical report dengan recommendation
   - Tools yang bisa dipakai di semua project selanjutnya

Ini akan jadi TOOL yang kamu pakai di semua project selanjutnya!
Investasi waktu sekarang = efisiensi di masa depan.
"""

print("\n" + "="*50)
print("🎉 FASE 2 SELESAI!")
print("="*50)
print("""
Kamu sekarang bisa:
✅ Bangun Linear Regression dari nol
✅ Bangun Logistic Regression dari nol
✅ Implementasi berbagai varian Gradient Descent
✅ Evaluasi model dengan proper methodology

Sebelum lanjut ke Fase 3, pastikan:
1. Semua exercise selesai
2. Minimal 2 challenge sudah dicoba
3. Project 1 di folder projects/ sudah dimulai

Lanjut ke: 03-classical-ml/01_supervised_learning.py
""")
