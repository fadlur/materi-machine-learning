"""
=============================================================
FASE 3 — MODUL 1: SUPERVISED LEARNING (sklearn)
=============================================================
Sekarang kamu sudah paham isi perut ML — saatnya pakai sklearn.

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
# 📖 BAGIAN 1: Decision Trees
# ===========================================================
# Decision tree = model non-linear yang mudah diinterpretasi.
# Cara kerja: split data berdasarkan fitur yang paling "informative"
# Metric: Gini impurity atau Information Gain (entropy)
#
# Kelebihan: interpretable, no need for feature scaling
# Kekurangan: mudah overfit!

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
    print(f"  Depth={str(depth):>4}: Train={train_acc:.4f}, Test={test_acc:.4f}"
          f"  {'⚠️ OVERFIT!' if train_acc - test_acc > 0.1 else '✅'}")


# ===========================================================
# 📖 BAGIAN 2: Random Forest
# ===========================================================
# Random Forest = ensemble of decision trees
# Kenapa lebih baik dari single tree?
# → Bagging (bootstrap aggregating) + random feature subset
# → Mengurangi variance (overfit) tanpa menambah bias
#
# Analogi: Wisdom of Crowds — banyak model "biasa" yang
# di-combine bisa lebih baik dari 1 model "expert"

print("\n=== Random Forest vs Decision Tree ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest: Train={rf.score(X_train, y_train):.4f}, "
      f"Test={rf.score(X_test, y_test):.4f}")

# Feature importance — salah satu keunggulan RF
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 4))
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [f'Feature {i}' for i in indices], rotation=45)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('01_feature_importance.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 01_feature_importance.png")


# ===========================================================
# 📖 BAGIAN 3: SVM (Support Vector Machines)
# ===========================================================
# SVM mencari hyperplane yang memaksimalkan MARGIN antara kelas.
#
# Koneksi Teknik Elektro:
# - Kernel trick = mengubah representasi sinyal (mirip transform domain)
# - RBF kernel = Gaussian filter di feature space
# - Support vectors = titik-titik paling "kritis" yang menentukan boundary

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
print("📊 Saved: 02_svm_kernels.png")


# ===========================================================
# 📖 BAGIAN 4: Gradient Boosting
# ===========================================================
# Boosting: train model secara SEKUENSIAL, setiap model berikutnya
# fokus pada ERROR model sebelumnya.
#
# Beda dengan Random Forest (Bagging):
# - RF: train PARALLEL, reduce VARIANCE
# - GB: train SEQUENTIAL, reduce BIAS
#
# GradientBoosting biasanya performanya TERBAIK untuk tabular data!

print("\n=== Gradient Boosting ===")
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                 max_depth=3, random_state=42)
gb.fit(X_train, y_train)
print(f"Gradient Boosting: Train={gb.score(X_train, y_train):.4f}, "
      f"Test={gb.score(X_test, y_test):.4f}")


# ===========================================================
# 📖 BAGIAN 5: Grand Comparison — Proper Methodology
# ===========================================================

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
print("📊 Saved: 03_model_comparison.png")


# ===========================================================
# 📖 BAGIAN 6: Hyperparameter Tuning (GridSearch)
# ===========================================================

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
# 🏋️ EXERCISE 8: Model Selection Pipeline
# ===========================================================
"""
Buat fungsi run_experiment(X, y, models_dict, cv=5) yang:

1. Untuk setiap model:
   a. Jalankan cross-validation
   b. Hitung mean, std, min, max score
   c. Catat training time

2. Output:
   a. Tabel ranking model
   b. Box plot perbandingan
   c. Best model + best hyperparameters (GridSearch)

3. Bonus: implementasi Bayesian Optimization untuk
   hyperparameter tuning (lebih efisien dari GridSearch)
   Hint: gunakan Gaussian Process sebagai surrogate model

Gunakan fungsi ini untuk semua project selanjutnya!
"""


# ===========================================================
# 🔥 CHALLENGE: Multi-class Sensor Fault Classification
# ===========================================================
"""
Buat sistem klasifikasi fault motor listrik dengan 5 kelas:
- Normal
- Bearing fault
- Stator fault
- Rotor fault
- External interference

Generate synthetic data yang realistis:
- 1000 samples, 20 features (sensor readings)
- Imbalanced classes (normal: 60%, bearing: 15%, stator: 10%, rotor: 10%, external: 5%)
- Beberapa fitur berkorelasi tinggi (realistis untuk sensor data)

Tasks:
1. Full EDA
2. Feature engineering (domain knowledge dari EE!)
3. Bandingkan minimal 5 model
4. Hyperparameter tuning untuk best model
5. Analisis: feature mana yang paling penting per kelas fault?
6. Buat classification report lengkap

Simpan hasilnya di projects/project_02_klasifikasi_sinyal/
"""

print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 03-classical-ml/02_unsupervised_learning.py")
print("="*50)


# ===========================================================
# MILESTONE ASSESSMENT — 3.1 Supervised Learning
# ===========================================================
# Referensi lengkap: ASSESSMENT.md (Fase 3, bagian 3.1)
#
# Level 1 — Bisa Dikerjakan (timer: 45 menit):
#   [ ] Train + evaluate: Decision Tree, Random Forest, SVM, GBM
#   [ ] Plot feature importance dari Random Forest
#   [ ] GridSearchCV untuk hyperparameter tuning
#
# Level 2 — Bisa Dijelaskan:
#   [ ] Bagging (RF) vs Boosting (XGBoost) — gambar diagram
#   [ ] Kapan SVM vs tree-based? Sebaliknya?
#   [ ] max_depth dan hubungannya dengan overfitting
#   [ ] Kenapa RF lebih robust dari single Decision Tree?
#
# Level 3 — Bisa Improvisasi (timer: 60 menit):
#   [ ] Bandingkan 5+ model dengan proper CV + tabel + plot
#   [ ] Analisis: model mana untuk data kecil/besar/noisy?
#   [ ] Bayesian Optimization untuk hyperparameter
#
# SKOR: ___/30
# TARGET PD: minimal 20/30 (rata-rata 2.0)
# ===========================================================
