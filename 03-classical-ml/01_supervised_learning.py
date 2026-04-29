"""
=============================================================
FASE 3 - MODUL 1: SUPERVISED LEARNING (sklearn)
=============================================================
Sekarang kamu sudah paham isi perut ML - saatnya pakai sklearn.

TAPI: setiap kali pakai model sklearn, kamu harus bisa jelaskan
cara kerjanya karena sudah pernah bangun dari nol.

Modul ini mencakup:
- Decision Trees & Random Forest
- SVM (Support Vector Machines)
- Ensemble Methods (Bagging, Boosting)
- Model Comparison framework

Durasi target: 4-5 jam
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)


# ===========================================================
# BAGIAN 1: Decision Trees
# ===========================================================
# Decision tree = model non-linear yang mudah diinterpretasi.
# Cara kerja: split data berdasarkan fitur yang paling "informative"
# Metric: Gini impurity atau Information Gain (entropy)
#
# Matematika Gini Impurity:
#   Gini = 1 - Sum(p_i**2)
#   dimana p_i = proporsi kelas i di node tersebut.
#   Gini = 0 berarti node murni (semua sample sama kelasnya).
#   Gini mendekati 1 berarti node sangat impure (campuran kelas).
#
# Information Gain (Entropy):
#   Entropy = -Sum(p_i * log2(p_i))
#   Information Gain = Entropy(parent) - weighted_sum(Entropy(children))
#   Split yang lebih informatif menghasilkan Information Gain yang lebih besar.
#
# Kenapa Decision Tree penting?
# - Interpretable: kita bisa trace path dari root ke leaf untuk
#   menjelaskan kenapa suatu prediksi dibuat.
# - Non-parametric: tidak membuat asumsi distribusi data.
# - No need for feature scaling: split berdasarkan threshold,
#   bukan jarak.
#
# Kelebihan: interpretable, no need for feature scaling
# Kekurangan: mudah overfit!
#
# Hyperparameter kunci:
# - max_depth: kedalaman maksimum tree. Semakin dalam, semakin kompleks.
#   Default None = tree tumbuh sampai pure leaf atau min_samples_split.
# - min_samples_split: minimum sample yang diperlukan untuk split node.
#   Default 2. Naikkan untuk mencegah overfit.
# - min_samples_leaf: minimum sample di leaf node.
#   Naikkan untuk smoothing prediksi.
# - criterion: 'gini' (default, lebih cepat) atau 'entropy'.
#   Praktis hasilnya serupa.
#
# Edge case & numerical stability:
# - Jika fitur memiliki nilai yang identik, tree tidak bisa split
#   pada fitur tersebut.
# - Jika ada kelas dengan sangat sedikit sample, tree mungkin
#   membuat leaf yang overfit ke minority class.
# - Dengan data kontinu, split threshold dipilih dari nilai unik,
#   yang bisa memperlambat training jika sample sangat banyak.

X, y = make_classification(n_samples=500, n_features=10,
                            n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Demonstrasi overfitting pada Decision Tree
print("=== Decision Tree: Overfitting Demo ===")
depths = [1, 3, 5, 10, None]  # None = unlimited depth
for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_acc = dt.score(X_train, y_train)
    test_acc = dt.score(X_test, y_test)
    flag = "PERINGATAN OVERFIT!" if train_acc - test_acc > 0.1 else "OK"
    print(f"  Depth={str(depth):>4}: Train={train_acc:.4f}, Test={test_acc:.4f}  {flag}")


# ===========================================================
# BAGIAN 2: Random Forest
# ===========================================================
# Random Forest = ensemble of decision trees
# Kenapa lebih baik dari single tree?
# -> Bagging (bootstrap aggregating) + random feature subset
# -> Mengurangi variance (overfit) tanpa menambah bias secara signifikan
#
# Matematika Bagging:
# - Training: untuk setiap tree, ambil bootstrap sample (sampling
#   dengan replacement dari training data). Secara teoritis, ~63.2%
#   sample unik per tree (karena sampling with replacement).
# - Prediction: voting majority untuk klasifikasi, rata-rata untuk regresi.
# - Variance reduction: Var(ensemble) = rho * sigma**2 + (1-rho)/m * sigma**2
#   dimana rho = correlation antar tree, m = jumlah tree, sigma**2 = variance tree.
#   Random feature subset menurunkan rho.
#
# Random Feature Subset:
# - Setiap split hanya mempertimbangkan subset fitur (default sqrt(n_features)
#   untuk klasifikasi, n_features/3 untuk regresi).
# - Ini menurunkan korelasi antar tree, sehingga ensemble lebih diverse.
#
# Hyperparameter kunci:
# - n_estimators: jumlah tree. Biasanya 100-500. More is better (diminishing returns).
# - max_features: jumlah fitur yang dipertimbangkan per split.
#   'sqrt' (default klasifikasi), 'log2', atau float (fraction).
# - max_depth: kedalaman maksimum per tree. None = full growth.
#   Batasi untuk mencegah overfit individual trees.
# - min_samples_leaf: naikkan untuk smoothing.
#
# Analogi: Wisdom of Crowds - banyak model "biasa" yang
# di-combine bisa lebih baik dari 1 model "expert"
#
# Koneksi Teknik Elektro:
# - Bagging = diversity combining (seperti antenna diversity)
# - Random feature subset = frequency hopping (mengurangi interferensi/korelasi)
# - Voting = majority logic decoder

print("\n=== Random Forest vs Decision Tree ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest: Train={rf.score(X_train, y_train):.4f}, "
      f"Test={rf.score(X_test, y_test):.4f}")

# Feature importance - salah satu keunggulan RF
# Importance dihitung sebagai rata-rata decrease in impurity (Gini/Entropy)
# untuk setiap fitur di seluruh tree.
# TIPS: Feature importance bisa biased ke fitur kardinalitas tinggi
# (fitur dengan banyak nilai unik). Gunakan permutation importance
# untuk hasil yang lebih reliable.
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 4))
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [f'Feature {i}' for i in indices], rotation=45)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('01_feature_importance.png', dpi=100, bbox_inches='tight')
plt.close()
print("OK Saved: 01_feature_importance.png")


# ===========================================================
# BAGIAN 3: SVM (Support Vector Machines)
# ===========================================================
# SVM mencari hyperplane yang memaksimalkan MARGIN antara kelas.
#
# Matematika SVM:
# - Primal form: minimize 0.5 * ||w||**2 + C * Sum(xi_i)
#   subject to: y_i(w*x_i + b) >= 1 - xi_i, xi_i >= 0
# - w = weight vector (normal ke hyperplane)
# - C = regularization parameter: tradeoff antara margin lebar dan classification error.
#   C kecil = margin lebar, lebih toleran terhadap misclassification (lebih generalize).
#   C besar = margin sempit, kurang toleran (mungkin overfit).
# - xi_i = slack variables untuk data yang tidak perfectly separable.
#
# Kernel Trick:
# - Jika data tidak linearly separable, SVM memetakan data ke
#   higher-dimensional space menggunakan kernel function.
# - Kernel menghitung inner product di high-dim space tanpa
#   secara eksplisit melakukan transformasi (kernel trick).
#
# Kernel Functions:
# 1. Linear: K(x,y) = x*y. Cocok untuk data dengan banyak fitur (text).
# 2. RBF (Radial Basis Function): K(x,y) = exp(-gamma * ||x-y||**2).
#    gamma = 1 / (2 * sigma**2). gamma besar = influence range kecil (overfit).
#    gamma kecil = influence range besar (underfit).
# 3. Polynomial: K(x,y) = (gamma * x*y + coef0)**degree.
#
# Hyperparameter kunci:
# - C: regularization. Default 1.0. Tune dengan logspace (0.001, 0.01, 0.1, 1, 10, 100).
# - kernel: 'linear', 'rbf', 'poly', 'sigmoid'. RBF paling umum.
# - gamma: 'scale' (default, 1/(n_features * X.var())), 'auto' (1/n_features),
#   atau float. Sangat sensitif!
#
# Edge case & numerical stability:
# - SVM tidak scale-invariant: fitur dengan scale besar akan mendominasi.
#   WAJIB standardize sebelum SVM!
# - RBF dengan gamma terlalu besar bisa menyebabkan singular matrix.
# - SVM bisa sangat lambat untuk dataset besar (O(n^2) - O(n^3) complexity).
# - Class imbalance: pertimbangkan class_weight='balanced'.
#
# Koneksi Teknik Elektro:
# - Kernel trick = mengubah representasi sinyal (mirip transform domain)
# - RBF kernel = Gaussian filter di feature space
# - Support vectors = titik-titik paling "kritis" yang menentukan boundary
# - Margin maximization = optimal decision boundary (seperti matched filter)

X_moon, y_moon = make_moons(n_samples=300, noise=0.2, random_state=42)
X_moon_train, X_moon_test, y_moon_train, y_moon_test = train_test_split(
    X_moon, y_moon, test_size=0.2)

print("\n=== SVM: Kernel Comparison ===")
kernels = ['linear', 'rbf', 'poly']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, kernel in zip(axes, kernels):
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_moon_train, y_moon_train)
    test_acc = svm.score(X_moon_test, y_moon_test)
    
    # Plot decision boundary
    h = 0.02
    x_min, x_max = X_moon[:, 0].min() - 0.5, X_moon[:, 0].max() + 0.5
    y_min, y_max = X_moon[:, 1].min() - 0.5, X_moon[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_moon_test[:, 0], X_moon_test[:, 1], c=y_moon_test,
               cmap='RdYlBu', edgecolors='black', s=30)
    ax.set_title(f'SVM ({kernel}): {test_acc:.2f}')

plt.tight_layout()
plt.savefig('02_svm_kernels.png', dpi=100, bbox_inches='tight')
plt.close()
print("OK Saved: 02_svm_kernels.png")


# ===========================================================
# BAGIAN 4: Gradient Boosting
# ===========================================================
# Boosting: train model secara SEKUENSIAL, setiap model berikutnya
# fokus pada ERROR model sebelumnya.
#
# Matematika Gradient Boosting (GBM):
# - Model akhir: F_M(x) = Sum(f_m(x)) dimana f_m adalah weak learner ke-m.
# - Training: di setiap iterasi m, tambahkan weak learner yang
#   meminimalkan loss function L(y, F_{m-1}(x) + f_m(x)).
# - Untuk klasifikasi/regresi, ini equivalent dengan fitting
#   negative gradient (pseudo-residuals) dari loss function.
#   Sehingga dinamakan "Gradient" Boosting.
#
# Weak Learner:
# - Biasanya Decision Tree dengan max_depth rendah (3-5).
# - Disebut "weak" karena secara individual performanya biasa saja,
#   tapi kombinasi banyak weak learner menjadi kuat.
#
# Learning Rate (shrinkage):
# - Setiap tree dikalikan dengan learning_rate < 1 (biasanya 0.01-0.3).
# - Ini mencegah model terlalu cepat overfit dan memungkinkan
#   lebih banyak tree untuk di-train.
# - Tradeoff: learning_rate kecil + n_estimators besar = lebih akurat
#   tapi training lebih lambat.
#
# Beda dengan Random Forest (Bagging):
# - RF: train PARALLEL, reduce VARIANCE
# - GB: train SEQUENTIAL, reduce BIAS
#
# GradientBoosting biasanya performanya TERBAIK untuk tabular data!
#
# Hyperparameter kunci:
# - n_estimators: jumlah boosting rounds. Tune dengan early stopping.
# - learning_rate: shrinkage. Default 0.1. Lebih kecil = lebih lambat tapi lebih baik.
# - max_depth: kedalaman tree. 3-5 biasanya cukup. Lebih dalam = overfit risk.
# - subsample: fraction data untuk training tiap tree. < 1.0 = stochastic gradient boosting.
# - min_samples_split, min_samples_leaf: regularization tambahan.
#
# Edge case & numerical stability:
# - GBM sangat sensitif terhadap outliers (karena fitting residuals).
# - Learning rate terlalu besar bisa menyebabkan divergence.
# - Class imbalance: gunakan scale_pos_weight atau sample weights.
# - Multicollinearity bisa membuat feature importance tidak stabil.

print("\n=== Gradient Boosting ===")
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                 max_depth=3, random_state=42)
gb.fit(X_train, y_train)
print(f"Gradient Boosting: Train={gb.score(X_train, y_train):.4f}, "
      f"Test={gb.score(X_test, y_test):.4f}")


# ===========================================================
# BAGIAN 5: Grand Comparison - Proper Methodology
# ===========================================================
# Cross-validation (CV) adalah gold standard untuk evaluasi model.
# Mengapa? Karena single train/test split bisa memberikan estimate
# yang tidak representative (tergantung keberuntungan split).
#
# K-Fold CV:
# - Data dibagi ke K fold (biasanya 5 atau 10).
# - Setiap fold sekali menjadi validation set, sisanya training.
# - Final score = rata-rata score dari K runs.
# - Std dev memberikan sense of variance/uncertainty.
#
# Mengapa Pipeline penting?
# - Preprocessing (scaling, selection) harus di-fit HANYA pada training data
#   di setiap fold. Pipeline menjamin ini terjadi secara otomatis.
# - Tanpa pipeline, preprocessing di-fit pada seluruh data sebelum CV
#   = DATA LEAKAGE! Hasil CV akan terlalu optimistik.
#
# Statistical Significance:
# - Mean CV score lebih tinggi belum tentu berarti model lebih baik.
# - Perlu paired t-test atau Wilcoxon signed-rank test untuk
#   membuktikan perbedaan signifikan secara statistik.
# - Confidence interval lebih informatif dari point estimate.

print("\n=== Model Comparison (5-fold CV) ===")
print(f"{'Model':<25} {'Mean CV Score':>15} {'Std':>10}")
print("-" * 50)

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    'KNN (k=5)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ]),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', random_state=42))
    ]),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results[name] = scores
    print(f"{name:<25} {scores.mean():>15.4f} {scores.std():>10.4f}")

# Box plot comparison
fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot(results.values(), labels=results.keys())
ax.set_ylabel('Accuracy')
ax.set_title('Model Comparison (5-Fold CV)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('03_model_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("OK Saved: 03_model_comparison.png")


# ===========================================================
# BAGIAN 6: Hyperparameter Tuning (GridSearch)
# ===========================================================
# Hyperparameter vs Parameter:
# - Parameter: di-learn dari data (weights model).
# - Hyperparameter: di-set sebelum training (max_depth, learning_rate, C).
#
# GridSearchCV:
# - Mencoba SEMUA kombinasi dari parameter grid.
# - Untuk setiap kombinasi, lakukan cross-validation.
# - Pilih kombinasi dengan score CV tertinggi.
# - Complexity: O(|grid| * K) dimana |grid| = jumlah kombinasi, K = fold.
#
# RandomizedSearchCV:
# - Alternatif lebih cepat: sampling random dari distribusi parameter.
# - Lebih efisien jika search space besar.
# - Bisa menemukan kombinasi bagus dengan lebih sedikit iterasi.
#
# Nested CV (untuk unbiased evaluation):
# - Outer CV: evaluasi performa akhir.
# - Inner CV: hyperparameter tuning.
# - Mencegah optimistik bias dari hyperparameter tuning.
# - PERINGATAN: GridSearchCV.score() masih sedikit optimistik karena
#   hyperparameter di-tune pada data yang sama.
#
# Tips Tuning:
# 1. Mulai dengan grid kasar, lalu fine-tune di sekitar best params.
# 2. Gunakan logspace untuk parameter dengan range besar (C, gamma).
# 3. Prioritaskan parameter yang paling sensitif (contoh: C untuk SVM,
#    max_depth dan learning_rate untuk GBM).
# 4. Monitor training time: GridSearch bisa sangat lambat!

print("\n=== Hyperparameter Tuning: Random Forest ===")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
}

# PENTING: Pipeline dengan scaler di dalam CV!
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")


# ===========================================================
# LATIHAN 8: Model Selection Pipeline
# ===========================================================
"""
TARGET Learning Objectives:
   - Membangun reusable experiment runner
   - Melakukan proper model comparison dengan statistical testing
   - Mengimplementasikan Bayesian Optimization

PANDUAN LANGKAH-LANGKAH:

STEP 1: Buat function run_experiment()
---------------------------------------
Buat function yang menerima:
- X, y (data)
- models_dict: dictionary model names dan instances
- cv: jumlah fold

Yang dilakukan:
1. Untuk setiap model:
   a. Jalankan cross-validation
   b. Hitung mean, std, min, max score
   c. Catat training time
   
2. Output:
   a. DataFrame ranking model
   b. Box plot perbandingan
   c. Best model + best hyperparameters (GridSearch)

TIPS KENAPA reusable function?
  - Tidak perlu copy-paste code untuk setiap experiment
  - Konsistent methodology
  - Mudah di-extend


STEP 2: Statistical Comparison
------------------------------
Setelah mendapatkan CV scores untuk semua model:

1. Paired t-test antar setiap pasangan model
2. Buat heatmap p-values
3. Identifikasi: mana yang significantly better?

TIPS KENAPA statistical test?
  - Mean score lebih tinggi belum tentu significantly better
  - Perlu bukti statistik untuk claim "model A lebih baik"
  - Mencegah over-interpretasi random fluctuations


STEP 3: (Bonus) Bayesian Optimization
-------------------------------------
GridSearch mencoba SEMUA kombinasi -> inefficient!
Bayesian Optimization lebih pintar:

1. Mulai dengan random samples
2. Build surrogate model (Gaussian Process)
3. Pilih next point yang maximize "expected improvement"
4. Repeat

Library: scikit-optimize (skopt)
```python
from skopt import BayesSearchCV
search = BayesSearchCV(model, param_space, n_iter=50, cv=5)
```

TIPS KENAPA Bayesian Optimization?
  - Lebih efisien dari GridSearch (less iterations)
  - Meng-handle continuous parameters
  - Better untuk expensive evaluations

PERINGATAN Hati-hati:
  - Bayesian Optimization juga bisa overfit ke validation set
  - Nested CV masih diperlukan untuk unbiased estimate


TIPS HINTS:
   - Gunakan time.time() untuk mencatat training time
   - Gunakan pd.DataFrame untuk menyimpan results
   - Simpan raw scores untuk post-hoc analysis
   - Gunakan scipy.stats.ttest_rel untuk paired t-test

PERINGATAN COMMON MISTAKES:
   - Data leakage di preprocessing pipeline
   - Membandingkan model dengan hyperparameters yang tidak fair
   - Tidak menggunakan nested CV untuk final evaluation
   - Mengabaikan training time (deployability!)

TARGET EXPECTED OUTPUT:
   - Reusable run_experiment function
   - Statistical comparison report
   - Clear recommendation: "Deploy model X"
"""


# ===========================================================
# CHALLENGE: Multi-class Sensor Fault Classification
# ===========================================================
"""
TARGET Learning Objectives:
   - Mengaplikasikan supervised learning ke domain EE
   - Menangani imbalanced classes di data sensor
   - Melakukan feature importance analysis per kelas

PANDUAN LANGKAH-LANGKAH:

STEP 1: Generate Synthetic Sensor Data
---------------------------------------
Buat dataset klasifikasi fault motor listrik dengan 5 kelas:
- Normal
- Bearing fault
- Stator fault
- Rotor fault
- External interference

   Spesifikasi:
   - 1000 samples, 20 features (sensor readings)
   - Imbalanced classes: normal: 60%, bearing: 15%, stator: 10%, rotor: 10%, external: 5%
   - Beberapa fitur berkorelasi tinggi (realistis untuk sensor data)
   - Noise level: moderate

   TIPS KENAPA imbalanced?
     - Fault jarang terjadi di dunia nyata
     - Model cenderung bias ke kelas majority
     - Perlu strategi khusus untuk evaluation


STEP 2: Full EDA
----------------
   a) Distribusi kelas
   b) Correlation matrix
   c) Pair plot untuk fitur paling penting
   d) Box plot per kelas untuk fitur kunci


STEP 3: Feature Engineering (Domain Knowledge EE!)
--------------------------------------------------
   Gunakan domain knowledge untuk membuat fitur baru:
   
   a) Statistical features: mean, std, RMS per window
   b) Frequency domain features: dominant frequency, THD
   c) Cross-sensor features: correlation between sensors
   d) Physical features: power = V*I, power factor, etc.
   
   TIPS KENAPA domain features?
     - Fitur berbasis fisika lebih interpretable
     - Biasanya lebih diskriminatif dari raw features
     - Menunjukkan expertise di domain


STEP 4: Compare Minimal 5 Model
-------------------------------
   a) Logistic Regression (baseline)
   b) Decision Tree (interpretable)
   c) Random Forest (ensemble)
   d) Gradient Boosting (best for tabular)
   e) SVM dengan RBF kernel
   
   Evaluation metrics:
   - Accuracy (tapi hati-hati dengan imbalanced!)
   - Precision, Recall, F1 per kelas
   - Macro-average F1 (tidak terpengaruh imbalance)
   - Weighted-average F1


STEP 5: Hyperparameter Tuning untuk Best Model
-----------------------------------------------
   Gunakan GridSearchCV untuk model terbaik.
   
   PERINGATAN Hati-hati:
   - Gunakan stratified CV untuk imbalanced data
   - Primary metric: macro F1 atau weighted F1
   - Jangan optimize accuracy untuk imbalanced data!


STEP 6: Analisis Feature Importance per Kelas
---------------------------------------------
   Untuk Random Forest atau Gradient Boosting:
   
   a) Global feature importance (default dari sklearn)
   b) Per-class feature importance:
      - Untuk setiap kelas, lihat samples yang benar diprediksi
      - Hitung mean feature value untuk samples tersebut
      - Bandingkan dengan kelas lain
      
   c) Visualisasi:
      - Bar plot global importance
      - Heatmap: feature x class (mean value)


STEP 7: Classification Report Lengkap
-------------------------------------
   Untuk test set:
   - Confusion matrix
   - Per-class precision, recall, F1
   - Macro dan weighted averages
   - Analisis: kelas mana yang paling sulit diprediksi? Kenapa?


TIPS HINTS:
   - Gunakan class_weight='balanced' untuk handle imbalance
   - Gunakan SMOTE untuk oversampling (opsional)
   - StratifiedKFold untuk CV imbalanced data
   - classification_report dengan output_dict=True untuk analisis

PERINGATAN COMMON MISTAKES:
   - Optimizing accuracy untuk imbalanced data
   - Tidak stratified split -> test set tanpa minority class
   - Feature engineering sebelum split -> data leakage
   - Tidak analyze per-class performance

TARGET EXPECTED OUTPUT:
   - Model dengan macro F1 > 0.80
   - Feature importance analysis dengan insight domain EE
   - Clear understanding: fitur mana yang membedakan setiap fault?
   - Recommendation untuk deployment

Simpan hasilnya di projects/project_02_klasifikasi_sinyal/
"""

print("\n" + "="*50)
print("OK Modul selesai! Lanjut ke: 03-classical-ml/02_unsupervised_learning.py")
print("="*50)
