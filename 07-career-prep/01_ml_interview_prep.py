"""
=============================================================
FASE 7 — MODUL 1: ML CODING INTERVIEW PREP
=============================================================
Interview untuk ML/AI Engineer biasanya terdiri dari:
1. ML Coding (Python + NumPy/Pandas) — 30-45 min
2. ML Theory / Concept — 30 min
3. ML System Design — 45-60 min
4. Behavioral — 30 min

Modul ini fokus pada #1 dan #2.
Background backend kamu sudah kuat di algorithms & system design,
tinggal adaptasi ke ML context.

Durasi target: 1 minggu (intensif latihan)
=============================================================
"""

import numpy as np
import pandas as pd

# ===========================================================
# 📖 BAGIAN 1: Python & Algorithms for ML
# ===========================================================
# Interview coding untuk ML sering lebih "practical" dari
# software engineering murni. Fokusnya: data manipulation,
vectorization, dan algoritma dasar ML.

print("""
╔══════════════════════════════════════════════════════════╗
║     ML CODING INTERVIEW: APA YANG DIUJI?               ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  1. NumPy Vectorization                                  ║
║     - Jangan pakai loop kalau bisa vectorized            ║
║     - Broadcasting, advanced indexing                    ║
║     - Matrix operations (dot, transpose, inverse)        ║
║                                                          ║
║  2. Pandas Data Manipulation                             ║
║     - groupby, merge, pivot                              ║
║     - Time series resampling                             ║
║     - Missing data handling                              ║
║                                                          ║
║  3. SQL (sering muncul!)                                 ║
║     - JOIN, GROUP BY, WINDOW functions                   ║
║     - Subqueries, CTEs                                   ║
║                                                          ║
║  4. Statistics & Probability                             ║
║     - Mean, variance, correlation                        ║
║     - Probability distributions                          ║
║     - Hypothesis testing (concept)                       ║
║                                                          ║
║  5. Simple ML Algorithms                                 ║
║     - Implementasi from scratch (LR, k-NN, k-Means)      ║
║     - Loss functions, gradients                          ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

# ===========================================================
# 🏋️ EXERCISE 1: NumPy Vectorization
# ===========================================================
"""
Implementasikan tanpa Python loops (for/while):

1. Diberikan matrix X (n, d) dan vector y (n,), hitung
   gradient untuk Linear Regression: grad = X^T (Xw - y) / n
   
2. Diberikan array 2D, normalize setiap row (mean=0, std=1)

3. Diberikan array 1D, temukan local maxima (elemen lebih besar
   dari kedua tetangganya)

4. Efficiently compute pairwise Euclidean distance antara
   setiap row di matrix X (n, d) dan Y (m, d)
   
5. Implementasi softmax function yang numerically stable
"""

def exercise_1_gradient(X, y, w):
    """Compute gradient for linear regression. NO LOOPS."""
    # TODO: implement
    pass

def exercise_1_normalize_rows(X):
    """Normalize each row to mean=0, std=1. NO LOOPS."""
    # TODO: implement
    pass

def exercise_1_local_maxima(arr):
    """Find local maxima. NO LOOPS."""
    # TODO: implement
    pass

def exercise_1_pairwise_distance(X, Y):
    """Pairwise Euclidean distance. NO LOOPS."""
    # TODO: implement
    pass

def exercise_1_softmax(z):
    """Numerically stable softmax. NO LOOPS."""
    # TODO: implement
    pass


# ===========================================================
# 🏋️ EXERCISE 2: Pandas Data Manipulation
# ===========================================================
"""
Diberikan DataFrame sales dengan kolom:
- date, product_id, category, region, units_sold, revenue

Jawab dengan Pandas (1-liner atau chain method):

1. Top 5 products by total revenue per category
2. Month-over-month growth rate per region
3. Detect outliers in revenue (z-score > 3) per category
4. Fill missing revenue with category median
5. Pivot table: region vs category, agg = total revenue
"""

# Sample data generator
def generate_sales_data(n=1000):
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    return pd.DataFrame({
        'date': dates,
        'product_id': np.random.randint(1, 100, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'region': np.random.choice(['East', 'West', 'North', 'South'], n),
        'units_sold': np.random.poisson(50, n),
        'revenue': np.random.exponential(1000, n)
    })


# ===========================================================
# 🏋️ EXERCISE 3: SQL untuk ML Engineer
# ===========================================================
"""
SQL sering diuji di ML Engineer interview karena data pipeline
sering berinteraksi dengan database.

Tulis SQL query untuk:

1. Dari tabel users (user_id, signup_date, country) dan
   tabel events (user_id, event_date, event_type, amount),
   hitung retention rate (%) per country untuk Day 7.
   
2. Dari tabel transactions (user_id, txn_date, amount),
   temukan users dengan running total > 10000 dalam
   30 hari terakhir (window function!).
   
3. Dari tabel experiments (user_id, experiment_group, conversion),
   hitung conversion rate dan confidence interval per group.
   
4. Deduplicate tabel logs berdasarkan user_id dan event_date,
   ambil record dengan timestamp terbaru.
   
5. Dari tabel hierarchical (employee_id, manager_id, salary),
   temukan salary median per department (self-join/CTE).
"""

sql_queries = {
    'retention': """
    -- TODO: Tulis query retention rate Day 7 per country
    """,
    'running_total': """
    -- TODO: Tulis query running total window
    """,
    'experiment': """
    -- TODO: Tulis query conversion rate + CI
    """,
    'dedup': """
    -- TODO: Tulis query deduplication
    """,
    'hierarchy': """
    -- TODO: Tulis query median salary
    """
}


# ===========================================================
# 🏋️ EXERCISE 4: ML Theory Interview Questions
# ===========================================================
"""
Jawab pertanyaan berikut dalam 2-3 menit (seperti di interview):

1. "Apa bedanya L1 dan L2 regularization? Kapan pakai yang mana?"
   
2. "Kenapa gradient descent bisa diverge? Bagaimana fix-nya?"
   
3. "Apa bedanya precision dan recall? Kapan prioritize yang mana?"
   
4. "Kenapa kita perlu validation set SELAIN test set?"
   
5. "Apa yang terjadi kalau learning rate terlalu tinggi?"
   
6. "Bagaimana handle class imbalance? Sebutkan 3 metode."
   
7. "Apa bedanya bagging dan boosting? Sebutkan contoh algoritma."
   
8. "Kenapa CNN works well untuk images? Jelaskan konvolusi."
   
9. "Apa vanishing gradient problem di RNN? Bagaimana LSTM memperbaiki?"
   
10. "Apa itu attention mechanism? Kenapa lebih baik dari RNN untuk sequence?"
    
11. "Bagaimana evaluasi model recommendation system?"
    
12. "Apa bedanya online learning dan batch learning?"
    
13. "Bagaimana detect dan handle data leakage?"
    
14. "Jelaskan bias-variance tradeoff dengan kata-kata sendiri."
    
15. "Kenapa feature scaling penting untuk SVM dan neural networks?"
"""


# ===========================================================
# 🏋️ EXERCISE 5: Implementasi From Scratch (Interview Favorite!)
# ===========================================================
"""
Di ML interview, sering diminta implementasi simple algorithm.
Ini test pemahaman fundamental, bukan hafalan API.

Implementasi dari nol (hanya NumPy, NO sklearn):
"""

def k_nearest_neighbors(X_train, y_train, X_test, k=3):
    """
    Implementasi k-NN classifier.
    Input: X_train (n, d), y_train (n,), X_test (m, d)
    Output: predictions (m,)
    """
    # TODO: implement
    pass

def k_means_clustering(X, k, max_iters=100):
    """
    Implementasi k-Means clustering.
    Input: X (n, d), k clusters
    Output: centroids (k, d), labels (n,)
    """
    # TODO: implement
    pass

def logistic_regression_sgd(X, y, lr=0.01, epochs=1000):
    """
    Implementasi Logistic Regression dengan SGD.
    Input: X (n, d), y (n,) binary {0,1}
    Output: weights (d,), loss_history
    """
    # TODO: implement
    pass


# ===========================================================
# 🏋️ EXERCISE 6: Debugging ML Code
# ===========================================================
"""
Di interview, sering diberikan kode yang "hampir benar" tapi
ada bug. Ini test debugging skill — sangat penting di production.

Baca kode di bawah, temukan 3+ bugs, dan perbaiki.
"""

def buggy_train_test_split(X, y, test_size=0.2):
    """Ada beberapa bugs di sini. Temukan dan perbaiki!"""
    n = len(X)
    test_n = int(n * test_size)
    # BUG 1: ???
    X_test = X[:test_n]
    y_test = y[:test_n]
    X_train = X[test_n:]
    y_train = y[test_n:]
    return X_train, X_test, y_train, y_test

def buggy_normalize(X):
    """Ada bugs di sini. Temukan dan perbaiki!"""
    mean = X.mean()
    std = X.std()
    # BUG 2: ???
    return (X - mean) / std

def buggy_accuracy(y_true, y_pred):
    """Ada bugs di sini. Temukan dan perbaiki!"""
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    # BUG 3: ???
    return correct


# ===========================================================
# 🔥 CHALLENGE: Mock Interview Simulation
# ===========================================================
"""
Lakukan mock interview dengan timer:

Setup:
- Timer: 45 menit
- Interviewer: teman, mentor, atau AI (ChatGPT/Claude)
- Format: 2 coding problems + 3 theory questions

Round 1 (15 min): NumPy/Pandas coding
Round 2 (15 min): Algorithm from scratch
Round 3 (15 min): Theory Q&A

Lakukan minimal 5x mock interview sebelum apply.
Record diri sendiri untuk review body language & clarity.
"""


# ===========================================================
# ✅ SOLUTIONS (Jangan dibaca sebelum mencoba!)
# ===========================================================
# Scroll ke bawah untuk solutions...
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

def solution_gradient(X, y, w):
    n = X.shape[0]
    return X.T @ (X @ w - y) / n

def solution_normalize_rows(X):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

def solution_local_maxima(arr):
    return np.where((arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:]))[0] + 1

def solution_pairwise_distance(X, Y):
    return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))

def solution_softmax(z):
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 07-career-prep/02_ml_system_design.py")
print("="*50)
