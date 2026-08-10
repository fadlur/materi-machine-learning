"""
=============================================================
FASE 8 - MODUL 1: ML INTERVIEW PREPARATION
=============================================================
Interview ML Engineer biasanya mencakup:
1. Coding interview (algorithms + ML implementation)
2. ML theory (algorithms, math, statistics)
3. System design (ML systems at scale)
4. Behavioral (experience, collaboration)

Koneksi Teknik Elektro:
- Coding = problem solving (seperti troubleshooting)
- ML theory = system dynamics dan signals
- System design = system architecture
- Behavioral = project management experience

Durasi target: 4-5 jam (latihan intensif)
============================================================="""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Tuple, Dict

np.random.seed(42)


# ===========================================================
# BAGIAN 1: Coding Interview - ML Implementation
# ===========================================================
print("="*60)
print("BAGIAN 1: CODING INTERVIEW PATTERNS")
print("="*60)

def vectorized_softmax(x: np.ndarray) -> np.ndarray:
    """
    Implementasi softmax yang numerically stable.
    
    Parameters:
    -----------
    x : np.ndarray
        Input array.
        
    Returns:
    --------
    np.ndarray
        Softmax probabilities.
        
    Notes:
    ------
    - Subtract max untuk numerical stability
    - Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    - Output: probabilities yang sum to 1
    - Kenapa subtract max? Karena exp(x) bisa overflow jika x besar.
      Dengan subtract max, nilai terbesar jadi 0, sehingga exp(0)=1
      dan nilai lainnya <= 1.
    
    Koneksi Teknik Elektro:
    - Softmax = normalized exponential (seperti power allocation)
    - Numerical stability = dynamic range compression
    """
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication dari scratch.
    
    Parameters:
    -----------
    A : np.ndarray, shape (m, n)
        First matrix.
    B : np.ndarray, shape (n, p)
        Second matrix.
        
    Returns:
    --------
    np.ndarray, shape (m, p)
        Product matrix.
        
    Complexity: O(m*n*p)
    
    Notes:
    ------
    - Implementasi naive triple loop
    - Untuk production, selalu gunakan np.dot atau BLAS library
    - Matrix multiply adalah operasi fundamental di deep learning
    """
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "Matrix dimensions incompatible"
    
    result = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i, j] += A[i, k] * B[k, j]
    return result


def k_fold_cross_validation(X: np.ndarray, y: np.ndarray,
                             k: int = 5) -> List[Tuple]:
    """
    Generate k-fold cross validation splits.
    
    Parameters:
    -----------
    X : np.ndarray
        Features.
    y : np.ndarray
        Labels.
    k : int, default 5
        Number of folds.
        
    Returns:
    --------
    List[Tuple]
        List of (X_train, y_train, X_val, y_val) tuples.
        
    Notes:
    ------
    - Setiap fold digunakan sebagai validation sekali
    - Data di-shuffle sebelum split untuk menghindari bias urutan
    - k=5 atau k=10 adalah pilihan umum
    - Stratified k-fold: menjaga proporsi class di setiap fold
    """
    n = len(X)
    fold_size = n // k
    indices = np.random.permutation(n)
    
    folds = []
    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n
        
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        folds.append((X[train_idx], y[train_idx], X[val_idx], y[val_idx]))
    
    return folds


# Test
print("\n=== Coding Tests ===")
x = np.array([1.0, 2.0, 3.0])
print(f"Softmax({x}) = {vectorized_softmax(x)}")
print(f"Sum: {vectorized_softmax(x).sum():.4f}")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"\nMatrix multiply:\n{matrix_multiply(A, B)}")

X_dummy = np.arange(10).reshape(-1, 1)
y_dummy = np.arange(10)
folds = k_fold_cross_validation(X_dummy, y_dummy, k=3)
print(f"\nK-fold: {len(folds)} folds generated")


# ===========================================================
# BAGIAN 2: ML Theory Questions
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 2: ML THEORY Q&A")
print("="*60)

ml_theory = """
TARGET LINEAR REGRESSION:

Q: Kenapa Linear Regression pakai MSE bukan MAE?
A: MSE differentiable everywhere, punya closed-form solution,
   dan punya probabilistic interpretation (MLE dengan Gaussian noise).
   MAE lebih robust terhadap outlier tapi tidak differentiable di 0.
   
   DETAIL:
   - MSE = (1/n) * sum((y_pred - y_true)^2)
   - Gradient MSE = (2/n) * X^T(Xw - y) -> linear in error
   - MAE = (1/n) * sum(|y_pred - y_true|)
   - Gradient MAE = sign(error) -> konstan, tidak memberikan informasi
     magnitude error untuk gradient descent.

Q: Apa bedanya Ridge dan Lasso?
A: Ridge = L2 penalty (shrinks coefficients, tidak sparse).
   Lasso = L1 penalty (sparse, feature selection).
   Elastic Net = kombinasi L1 dan L2.
   
   DETAIL:
   - Ridge loss = MSE + lambda * sum(w^2)
   - Lasso loss = MSE + lambda * sum(|w|)
   - Lasso bisa membuat coefficients exactly zero (feature selection)
   - Ridge lebih stabil untuk multicollinearity

Q: Kenapa perlu normalize features?
A: Agar regularization fair (tidak bias ke features dengan scale besar),
   gradient descent converge lebih cepat, dan numerical stability.
   
   DETAIL:
   - Tanpa normalization, feature dengan scale besar akan mendominasi
   - Learning rate yang sama untuk semua features menjadi tidak optimal
   - Normalization juga membantu interpretasi coefficients

TARGET LOGISTIC REGRESSION:

Q: Kenapa pakai sigmoid? Kenapa tidak linear?
A: Output probability harus dalam [0,1]. Sigmoid maps (-inf,inf) -> (0,1).
   Linear regression bisa predict <0 atau >1.
   
   DETAIL:
   - Sigmoid(x) = 1 / (1 + exp(-x))
   - Interpretasi: log-odds = linear combination of features
   - Alternatif: probit function (CDF dari Gaussian)

Q: Kenapa loss function BCE bukan MSE?
A: BCE = convex untuk logistic regression (guaranteed convergence).
   MSE = non-convex untuk classification (banyak local minima).
   
   DETAIL:
   - BCE menghasilkan gradient yang lebih besar saat prediction salah
   - MSE gradient bisa mendekati 0 saat prediction ekstrem (vanishing gradient)

Q: Apa itu odds ratio?
A: odds = p/(1-p). Logistic regression models log(odds) = linear.
   Odds ratio = exp(coefficient) = change in odds per unit change.

TARGET NEURAL NETWORKS:

Q: Kenapa ReLU lebih baik dari sigmoid?
A: ReLU: no vanishing gradient, computationally efficient, sparse activation.
   Sigmoid: gradient saturates (->0), computationally expensive.
   
   DETAIL:
   - ReLU(x) = max(0, x), gradient = 1 untuk x > 0
   - Sigmoid gradient = sigmoid(x) * (1 - sigmoid(x)), maksimum = 0.25
   - ReLU bisa "mati" jika semua input negatif (dead ReLU)
   - Variants: LeakyReLU, PReLU, GELU

Q: Apa itu backpropagation?
A: Chain rule applied to compute gradients of loss w.r.t. each parameter.
   Forward pass computes activations, backward pass computes gradients.
   
   DETAIL:
   - Backprop = efficient computation of gradients menggunakan chain rule
   - Complexity O(n) untuk n parameters (bukan exponential!)
   - Memerlukan computational graph (PyTorch autograd, TensorFlow)

Q: Kenapa butuh activation function?
A: Tanpa activation, neural network = linear model (composition of linear).
   Activation introduces non-linearity, enabling universal approximation.
   
   DETAIL:
   - Composition of linear functions is still linear
   - Non-linearity memungkinkan model belajar decision boundary yang kompleks
   - Universal approximation theorem: NN dengan 1 hidden layer bisa
     approximate any continuous function (dengan enough neurons)

Q: Apa bedanya batch norm dan layer norm?
A: BatchNorm: normalize across batch dimension.
   LayerNorm: normalize across feature dimension (per sample).
   LayerNorm lebih baik untuk RNN/transformer (variable length sequences).
   
   DETAIL:
   - BatchNorm: mean/variance per feature dihitung dari batch
   - LayerNorm: mean/variance dihitung dari semua features per sample
   - BatchNorm bergantung pada batch size, LayerNorm tidak
   - LayerNorm lebih stabil untuk inference

TARGET CNN:

Q: Kenapa convolution lebih baik dari fully connected untuk images?
A: Local connectivity (sparse connections), weight sharing (fewer params),
   translation invariance (detect features anywhere in image).
   
   DETAIL:
   - FC untuk 224x224x3 image = 150k input features -> terlalu banyak params
   - Convolution: kernel kecil (3x3, 5x5) yang di-slide di seluruh image
   - Weight sharing: kernel yang sama dipakai di semua lokasi
   - Translation invariance: feature bisa didetect di mana saja

Q: Apa fungsi pooling?
A: Reduce spatial dimensions, provide translation invariance,
   reduce computational cost, prevent overfitting.
   
   DETAIL:
   - Max pooling: ambil nilai maksimum di window -> lebih robust
   - Average pooling: ambil rata-rata -> lebih smooth
   - Modern networks (ResNet) sering pakai stride convolution daripada pooling

Q: Kenapa padding digunakan?
A: Maintain spatial dimensions, prevent information loss di borders,
   enable deeper networks.
   
   DETAIL:
   - Same padding: output size = input size
   - Valid padding: output size < input size
   - Padding juga memastikan pixels di border diproses sama banyaknya

TARGET RNN/LSTM:

Q: Apa masalah vanishing gradients di RNN?
A: Gradients diumpulkan melalui banyak time steps -> product of small numbers.
   Early layers hampir tidak belajar. LSTM solves ini dengan cell state.
   
   DETAIL:
   - RNN: h_t = tanh(W*x_t + U*h_{t-1})
   - Gradient melibatkan U^T di-multiply berkali-kali
   - Jika eigenvalues < 1: vanishing gradients
   - Jika eigenvalues > 1: exploding gradients
   - LSTM cell state: additive updates (bukan multiplicative)

Q: Apa bedanya LSTM dan GRU?
A: LSTM: 3 gates (forget, input, output), cell state.
   GRU: 2 gates (reset, update), no cell state.
   GRU simpler, faster, comparable performance.
   
   DETAIL:
   - LSTM lebih expressive, tapi lebih banyak parameters
   - GRU lebih cepat train, cocok untuk dataset lebih kecil
   - Pilihan biasanya empirical: coba keduanya, pilih yang lebih baik

Q: Kenapa Transformer lebih baik dari RNN?
A: Parallel processing (faster training), long-range dependencies,
   no vanishing gradients, global receptive field.
   
   DETAIL:
   - RNN sequential: O(n) time per layer
   - Transformer parallel: O(1) time per layer (dengan enough compute)
   - Self-attention: setiap token bisa directly attend ke token lain
   - Scalability: Transformer bisa di-scale ke model yang jauh lebih besar

TARGET OPTIMIZATION:

Q: Apa bedanya SGD dan Adam?
A: SGD: simple, might need careful tuning.
   Adam: adaptive learning rate, momentum, bias correction.
   Adam generally faster convergence, but SGD bisa generalize better.
   
   DETAIL:
   - SGD: w = w - lr * gradient
   - Adam: adaptive lr per parameter menggunakan first dan second moments
   - Adam dengan weight decay (AdamW) lebih baik untuk regularization
   - Untuk large models, SGD dengan momentum masih competitive

Q: Kenapa learning rate decay digunakan?
A: Large LR di awal untuk fast convergence, small LR di akhir
   untuk fine-tuning dan avoid oscillation.
   
   DETAIL:
   - Step decay: turunkan LR setiap N epochs
   - Cosine annealing: LR mengikuti cosine curve
   - Warmup: mulai dari LR kecil, naik ke target LR
   - One-cycle policy: naik lalu turun sekali dalam training

Q: Apa itu gradient clipping?
A: Limit gradient magnitude untuk prevent exploding gradients.
   Common di RNN/LSTM training.
   
   DETAIL:
   - Gradient norm: clip jika ||g|| > threshold
   - Gradient value: clip setiap element jika |g_i| > threshold
   - Critical untuk training RNN/LSTM/Transformer yang stabil

TARGET REGULARIZATION:

Q: Apa bedanya L1 dan L2 regularization?
A: L1: sparse weights (feature selection), convex but not smooth.
   L2: small weights (weight decay), smooth, ridge regression.
   
   DETAIL:
   - L1: bisa membuat weights exactly zero -> sparse model
   - L2: weights mendekati zero tapi jarang exactly zero
   - L1 lebih sulit optimize karena tidak differentiable di 0
   - Elastic Net menggabungkan keduanya

Q: Kenapa dropout bekerja?
A: Prevents co-adaptation of neurons. Ensemble effect:
   training = training many thinned networks, inference = averaging.
   
   DETAIL:
   - Dropout rate: probabilitas neuron di-disable (biasanya 0.2-0.5)
   - At inference: weights di-scale dengan (1 - dropout_rate)
   - Dropout juga bisa dianggap sebagai data augmentation di input

Q: Apa itu early stopping?
A: Stop training ketika validation loss mulai naik.
   Prevents overfitting tanpa hyperparameter tuning.
   
   DETAIL:
   - Monitor validation metric setiap epoch
   - Patience: berapa epoch menunggu sebelum stop jika tidak improve
   - Restore best weights: simpan weights terbaik selama training
   - Very practical dan efektif untuk mencegah overfitting

TARGET EVALUATION:

Q: Apa bedanya precision dan recall?
A: Precision: dari predicted positive, berapa yang benar?
   Recall: dari actual positive, berapa yang tertangkap?
   F1 = harmonic mean dari precision dan recall.
   
   DETAIL:
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - F1 = 2 * (Precision * Recall) / (Precision + Recall)
   - Pilih precision jika FP mahal, recall jika FN mahal

Q: Kapan menggunakan ROC-AUC vs PR-AUC?
A: ROC-AUC: balanced datasets, tidak sensitive ke class imbalance.
   PR-AUC: imbalanced datasets, focus pada positive class.
   
   DETAIL:
   - ROC: plot TPR vs FPR
   - PR: plot Precision vs Recall
   - Imbalanced datasets: ROC bisa terlalu optimistic
   - PR-AUC lebih informatif untuk minority class

Q: Apa itu cross-validation dan kenapa penting?
A: Split data ke k folds, train pada k-1, validate pada 1.
   Reduces variance dalam performance estimate, better generalization estimate.
   
   DETAIL:
   - k=5 atau k=10 adalah standar
   - Stratified k-fold: menjaga proporsi class
   - Leave-one-out: k = n (untuk dataset sangat kecil)
   - Time series: time-based split (bukan random)

TARGET BIAS-VARIANCE TRADEOFF:

Q: Apa itu bias dan variance?
A: Bias: error dari assumptions yang salah (underfitting).
   Variance: error dari sensitivity ke small fluctuations (overfitting).
   Total error = Bias^2 + Variance + Irreducible Error.
   
   DETAIL:
   - High bias: model terlalu simple, tidak bisa capture pattern
   - High variance: model terlalu complex, mempelajari noise
   - Irreducible error: noise di data yang tidak bisa dihilangkan

Q: Bagaimana reduce bias?
A: More complex model, more features, reduce regularization.

Q: Bagaimana reduce variance?
A: More data, regularization, simpler model, ensemble methods.
"""
print(ml_theory)


# ===========================================================
# BAGIAN 3: Math for ML
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 3: MATEMATIKA UNTUK ML")
print("="*60)

math_for_ml = """
TARGET LINEAR ALGEBRA:

- Vector: direction and magnitude
- Matrix: linear transformation
- Eigenvalue/Eigenvector: A*v = lambda*v (invariant directions)
  DETAIL: Eigenvectors adalah directions yang tidak berubah oleh transformasi A,
  hanya di-scale oleh eigenvalue lambda.
- SVD: A = U * Sigma * V^T (decomposition into orthogonal matrices)
  DETAIL: Singular Value Decomposition = generalisasi eigendecomposition untuk
  matriks non-square. Digunakan di PCA, collaborative filtering, dll.
- PCA = eigendecomposition of covariance matrix
  DETAIL: PCA mencari directions (principal components) yang menjelaskan
  variance maksimum di data.

TARGET CALCULUS:

- Gradient: grad f = vector of partial derivatives (direction of steepest ascent)
  DETAIL: Gradient menunjukkan arah di mana fungsi naik paling cepat.
  Untuk minimization, kita berjalan BERLAWANAN arah gradient.
- Chain rule: partial f / partial x = partial f / partial g * partial g / partial x (backpropagation!)
  DETAIL: Chain rule adalah fondasi dari backpropagation. Setiap layer
  menghitung local gradient, lalu di-chain bersama.
- Jacobian: matrix of all first-order partial derivatives
  DETAIL: Jacobian J memberikan linear approximation dari fungsi multidimensi.
- Hessian: matrix of second-order partial derivatives
  DETAIL: Hessian H digunakan untuk second-order optimization methods seperti Newton's method.
  H_ij = partial^2 f / partial x_i partial x_j.

TARGET PROBABILITY & STATISTICS:

- Bayes' Theorem: P(A|B) = P(B|A)*P(A) / P(B)
  DETAIL: Bayes' theorem memungkinkan kita update belief (posterior) berdasarkan evidence.
  P(A) = prior, P(B|A) = likelihood, P(A|B) = posterior.
- MLE: maximize P(data|theta) -> find best parameters
  DETAIL: Maximum Likelihood Estimation mencari parameter yang membuat data
  paling probable. Contoh: MLE untuk Gaussian mean = sample mean.
- MAP: maximize P(theta|data) -> MLE + prior
  DETAIL: Maximum A Posteriori menambahkan prior belief ke parameters.
  Equivalent dengan regularization (L2 = Gaussian prior).
- Gaussian: N(x; mu, sigma^2) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))
  DETAIL: Gaussian distribution muncul secara alami karena Central Limit Theorem.
- KL Divergence: measure difference antara distributions
  DETAIL: KL(P||Q) = sum P(x) * log(P(x)/Q(x)). Tidak symmetric!
  Digunakan di VAE, variational inference, information theory.

TARGET OPTIMIZATION:

- Convex function: f(lambda*x + (1-lambda)*y) <= lambda*f(x) + (1-lambda)*f(y)
  DETAIL: Convex function punya global minimum yang unik.
  Gradient descent untuk convex function guaranteed converge.
- Gradient Descent: theta = theta - alpha * grad J(theta)
  DETAIL: Update rule paling fundamental. Alpha = learning rate.
  Variants: SGD, Momentum, AdaGrad, RMSprop, Adam.
- Lagrange Multipliers: optimize dengan constraints
  DETAIL: Mengubah constrained optimization jadi unconstrained dengan
  menambahkan Lagrange multipliers.
- Convex Optimization: garanti global minimum
  DETAIL: Linear programming, quadratic programming, SDP adalah convex.
  Deep learning optimization biasanya NON-convex!
"""
print(math_for_ml)


# ===========================================================
# BAGIAN 4: SQL for Data Science
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 4: SQL UNTUK DATA SCIENCE")
print("="*60)

sql_questions = """
TARGET COMMON SQL PATTERNS:

Q: Find duplicate records
A: SELECT id, COUNT(*) FROM table GROUP BY id HAVING COUNT(*) > 1
   DETAIL: GROUP BY mengelompokkan records, HAVING memfilter groups.

Q: Running total / cumulative sum
A: SELECT date, amount, SUM(amount) OVER (ORDER BY date) AS running_total
   FROM transactions
   DETAIL: Window functions (OVER) memungkinkan aggregation tanpa collapsing rows.

Q: Rank rows per group
A: SELECT *, RANK() OVER (PARTITION BY category ORDER BY sales DESC)
   FROM sales
   DETAIL: PARTITION BY membagi data ke groups independent.
   RANK() memberikan rank dengan gaps untuk ties.
   DENSE_RANK() tanpa gaps.

Q: Moving average
A: SELECT date, value,
      AVG(value) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
      AS moving_avg
   FROM time_series
   DETAIL: ROWS BETWEEN mendefinisikan window sliding.
   Bisa juga RANGE BETWEEN untuk logical ranges.

Q: Find top N per group
A: SELECT * FROM (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY sales DESC) AS rn
      FROM sales
   ) WHERE rn <= 3
   DETAIL: ROW_NUMBER() memberikan unique rank (no ties).
   CTE (WITH clause) juga bisa digunakan untuk readability.

Q: Self-join untuk hierarchical data
A: SELECT e.name, m.name AS manager
   FROM employees e LEFT JOIN employees m ON e.manager_id = m.id
   DETAIL: Self-join menghubungkan table dengan dirinya sendiri.
   LEFT JOIN memastikan employees tanpa manager tetap muncul.

Q: Pivot / unpivot
A: SELECT * FROM (
      SELECT month, product, sales FROM sales_data
   ) PIVOT (SUM(sales) FOR month IN ('Jan', 'Feb', 'Mar'))
   DETAIL: Pivot mengubah rows jadi columns.
   Unpivot mengubah columns jadi rows.
   Di PostgreSQL gunakan crosstab, di MySQL gunakan CASE.
"""
print(sql_questions)


# ===========================================================
# LATIHAN 19: Interview Practice
# ===========================================================
"""
TARGET Learning Objectives:
   - Melatih coding interview skills
   - Menjawab ML theory questions dengan confident
   - Memahami common patterns dan pitfalls

PANDUAN LANGKAH-LANGKAH:

STEP 1: Coding Practice
-----------------------
   Implementasikan dari scratch (NumPy only):
   
   a) Linear Regression (closed-form dan gradient descent)
   b) Logistic Regression (SGD)
   c) K-Means clustering
   d) PCA (eigendecomposition)
   e) Neural Network (2-layer, backpropagation)
   f) k-Nearest Neighbors
   
   Setiap implementasi:
   - < 30 menit
   - Unit test dengan assert
   - Compare dengan sklearn
   
   DETAIL:
   - Interview coding biasanya 30-45 menit per problem
   - Mulai dari brute force, lalu optimize
   - Always test dengan edge cases
   - Jelaskan time dan space complexity


STEP 2: Theory Practice
-----------------------
   Untuk setiap topik di atas, latihan:
   
   a) Jelaskan dengan kata-kata sendiri (2-3 menit)
   b) Derive key equations (on paper)
   c) Give concrete examples
   d) Explain tradeoffs
   e) Answer follow-up questions
   
   TIPS:
     - Record yourself (cek clarity)
     - Practice dengan peer (mock interview)
     - Focus pada "why" bukan hanya "what"
     - Siapkan 2-3 contoh real-world untuk setiap algoritma


STEP 3: SQL Practice
--------------------
   Latihan query untuk scenarios:
   
   a) E-commerce: orders, customers, products
   b) IoT: sensor readings, devices, alerts
   c) Social media: users, posts, interactions
   d) Finance: transactions, accounts, fraud
   
   Tools: LeetCode, HackerRank, Mode Analytics
   
   DETAIL:
   - SQL sering di-test di data science interviews
   - Focus pada window functions, JOINs, subqueries
   - Latihan explain query execution plan (EXPLAIN)


STEP 4: Mock Interview
----------------------
   a) Timed practice (45-60 menit per session)
   b) Verbal explanation (think out loud)
   c) Handle edge cases
   d) Optimize dari O(n^2) ke O(n)
   e) Test dengan contoh input
   
   TIPS Common interview structure:
     - Clarify problem (2-3 menit)
     - Discuss approach (5 menit)
     - Implement (20-30 menit)
     - Test dan optimize (10 menit)
     - Discuss extensions (5 menit)


TIPS:
   - Always ask clarifying questions
   - Start dengan brute force, lalu optimize
   - Test dengan small examples
   - Discuss time/space complexity
   - Mention tradeoffs dan alternatives

PERINGATAN COMMON MISTAKES:
   - Langsung coding tanpa understand problem
   - Tidak test dengan examples
   - Ignore edge cases (empty input, single element, dll)
   - Tidak discuss complexity
   - Panic saat stuck -> take a breath, break down problem

TARGET EXPECTED OUTPUT:
   - Bisa implementasi 6+ algorithms dari scratch dalam <30 menit
   - Confident menjelaskan ML theory
   - Fluent dalam SQL queries
   - Experience dengan mock interviews

Latihan membuat perfect!
"""


# ===========================================================
# 🔥 CHALLENGE: 100 ML Questions
# ===========================================================
"""
TARGET Learning Objectives:
   - Menguasai 100+ pertanyaan interview ML
   - Membangun intuition yang deep
   - Siap untuk interview di top tech companies

PANDUAN LANGKAH-LANGKAH:

STEP 1: Master 100 Questions
----------------------------
   Kategorisasi pertanyaan:
   
   Supervised Learning (20 questions):
   - Linear/Logistic Regression
   - Decision Trees, Random Forest, Gradient Boosting
   - SVM, KNN, Naive Bayes
   - Neural Networks (basics)
   
   Unsupervised Learning (15 questions):
   - K-Means, DBSCAN, Hierarchical
   - PCA, t-SNE, UMAP
   - Gaussian Mixture Models
   - Anomaly Detection
   
   Deep Learning (25 questions):
   - CNN, RNN, LSTM, GRU
   - Transformer, Attention, BERT
   - Optimization, Regularization
   - Generative Models
   
   MLOps & Engineering (15 questions):
   - Feature Engineering
   - Model Deployment
   - Monitoring, Drift
   - Scaling, Performance
   
   Math & Statistics (15 questions):
   - Linear Algebra
   - Probability
   - Calculus
   - Optimization
   
   System Design (10 questions):
   - Recommendation Systems
   - Search Ranking
   - Fraud Detection
   - A/B Testing


STEP 2: Create Flashcards
-------------------------
   Untuk setiap pertanyaan:
   - Front: Pertanyaan
   - Back: Jawaban singkat (1-2 menit explanation)
   - Tags: kategori dan difficulty
   
   Tools: Anki, Quizlet, atau physical cards
   Review: spaced repetition (daily, weekly, monthly)
   
   DETAIL:
   - Spaced repetition adalah metode belajar paling efektif
   - Review kartu yang sulit lebih sering
   - Review kartu yang mudah lebih jarang


STEP 3: Practice with Peers
---------------------------
   a) Find study group (LinkedIn, local meetups)
   b) Schedule mock interviews (weekly)
   c) Give dan receive feedback
   d) Track progress (which areas need more work?)
   
   TIPS Platforms:
     - Pramp (free mock interviews)
     - Interviewing.io (anonymous practice)
     - LeetCode (coding)
     - System Design Primer (GitHub)


STEP 4: Company-Specific Prep
-----------------------------
   Research interview process per company:
   
   Google:
   - 4-5 rounds: coding, system design, ML theory, behavioral
   - Focus: scalability, algorithms
   
   Meta:
   - 3-4 rounds: coding, ML design, behavioral
   - Focus: product sense, practical ML
   
   Amazon:
   - 4-5 rounds: LP (Leadership Principles), coding, system design
   - Focus: customer obsession, ownership
   
   Startups:
   - 2-3 rounds: practical, end-to-end
   - Focus: versatility, shipping speed
   
   TIPS Resources:
     - Blind (interview experiences)
     - LeetCode Discuss (company tags)
     - Glassdoor (interview questions)


TIPS:
   - Consistency > intensity (latihan 1 jam/hari > 8 jam sekaligus)
   - Focus pada weak areas
   - Simulate real interview conditions (timed, verbal)
   - Review dan reflect setelah setiap practice

PERINGATAN COMMON MISTAKES:
   - Hanya baca tanpa practice
   - Skip mock interviews
   - Ignore behavioral questions
   - Tidak research company culture
   - Underprepare untuk system design

TARGET EXPECTED OUTPUT:
   - 100+ questions mastered
   - Flashcard deck lengkap
   - 10+ mock interviews completed
   - Confident untuk interview di target companies

Persistence is key - jangan menyerah!
"""

print("\n" + "="*50)
print("OK Modul selesai! Lanjut ke: 08-career-prep/02_ml_system_design.py")
print("="*50)
