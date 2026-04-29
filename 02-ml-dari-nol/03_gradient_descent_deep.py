"""
=============================================================
FASE 2 - MODUL 3: GRADIENT DESCENT DEEP DIVE
=============================================================
Gradient descent adalah JANTUNG dari semua ML modern.
Kalau kamu benar-benar paham GD + variannya, kamu bisa
debug model apa pun.

Koneksi Teknik Elektro:
- GD = steepest descent (optimization di control theory)
- Learning rate = step size
- Momentum = inersia (analogi fisika yang persis)
- Adam = PID controller untuk optimization!

Durasi target: 3-4 jam
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ===========================================================
# BAGIAN 1: Visualisasi Loss Landscape
# ===========================================================
# Sebelum optimasi, kita harus "lihat" apa yang dioptimasi.
# Loss landscape = surface yang ingin kita minimalkan.
# Visualisasi membantu memahami kenapa optimizer tertentu
# bekerja lebih baik dari yang lain.
#
# DETAIL MATEMATIKA:
# Loss landscape untuk model dengan parameter theta adalah
# surface L(theta) di ruang parameter.
# - Local minimum: titik dimana gradient = 0 dan Hessian positive definite
# - Global minimum: local minimum dengan nilai loss terendah
# - Saddle point: gradient = 0 tapi Hessian indefinite (ada eigenvalue positif dan negatif)
# - Plateau: region dengan gradient hampir 0 (optimizer bisa "stuck")
#
# Koneksi Teknik Elektro:
# - Loss landscape = error surface di adaptive control
# - Local minima = stable equilibrium points
# - Saddle points = unstable equilibrium
# - Gradient descent = system dynamics yang converge ke equilibrium

def rosenbrock(x, y):
    """
    Rosenbrock function - classic optimization test.
    
    Parameters:
    -----------
    x, y : np.ndarray atau scalar
        Input variables.
        
    Returns:
    --------
    np.ndarray atau scalar
        f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        
    Notes:
    ------
    - Global minimum: (1, 1) dengan f(1,1) = 0
    - Valley yang sempit dan melengkung -> sulit di-navigate
    - Sering disebut "banana function" karena bentuk contour-nya
    - Standard benchmark untuk testing optimization algorithms
    - Hessian di minimum = [[802, -400], [-400, 200]]
      Condition number ~ 2500 (sangat ill-conditioned!)
    
    Koneksi Teknik Elektro:
    - Mirip dengan non-convex optimization di adaptive control
    - Valley sempit = ill-conditioned system
    - Memerlukan optimizer dengan momentum untuk converge cepat
    """
    return (1 - x)**2 + 100 * (y - x**2)**2


def simple_quadratic(x, y):
    """
    Fungsi sederhana: f(x,y) = x^2 + 10*y^2
    
    Parameters:
    -----------
    x, y : np.ndarray atau scalar
        Input variables.
        
    Returns:
    --------
    np.ndarray atau scalar
        f(x,y) = x^2 + 10*y^2
        
    Notes:
    ------
    - Global minimum: (0, 0)
    - Elliptical contours (ill-conditioned)
    - Lebih mudah dari Rosenbrock, tapi masih menantang untuk vanilla GD
    - Condition number = 10 (ratio eigenvalues Hessian)
    - Hessian = [[2, 0], [0, 20]]
    - Gradient di x-direction jauh lebih kecil dari y-direction
      -> GD akan zig-zag (oscillate di y, lambat di x)
    """
    return x**2 + 10 * y**2


# Plot landscape
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x_range = np.linspace(-3, 3, 200)
y_range = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x_range, y_range)
# np.meshgrid membuat grid 2D dari 1D arrays
# X[i,j] = x_range[j], Y[i,j] = y_range[i]

# Quadratic (mudah)
Z1 = simple_quadratic(X, Y)
axes[0].contour(X, Y, Z1, levels=30, cmap='viridis')
axes[0].set_title('Quadratic: x^2 + 10*y^2 (mudah)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].plot(0, 0, 'r*', markersize=15, label='Minimum')
axes[0].legend()

# Rosenbrock (sulit - valley yang sempit)
x_range2 = np.linspace(-2, 2, 200)
y_range2 = np.linspace(-1, 3, 200)
X2, Y2 = np.meshgrid(x_range2, y_range2)
Z2 = rosenbrock(X2, Y2)
axes[1].contour(X2, Y2, Z2, levels=np.logspace(-1, 3, 30), cmap='viridis')
axes[1].set_title('Rosenbrock (sulit - narrow valley)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].plot(1, 1, 'r*', markersize=15, label='Minimum')
axes[1].legend()

plt.tight_layout()
plt.savefig('01_loss_landscapes.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 01_loss_landscapes.png")


# ===========================================================
# BAGIAN 2: Vanilla GD vs SGD vs Mini-batch
# ===========================================================
# Ada 3 varian utama gradient descent:
# 1. Full-batch GD: update menggunakan seluruh dataset
# 2. Stochastic GD (SGD): update per sample
# 3. Mini-batch GD: update per subset (batch)
#
# DETAIL MATEMATIKA:
# Full-batch: theta := theta - lr * grad(L_full)
#   - grad(L_full) = (1/n) * Sum grad(L_i)
#   - Gradient paling akurat (true gradient)
#   - Tapi mahal untuk dataset besar
#
# SGD: theta := theta - lr * grad(L_i)
#   - grad(L_i) = gradient untuk satu sample
#   - Noisy tapi unbiased estimator dari true gradient
#   - E[grad(L_i)] = grad(L_full)
#   - Variance tinggi tapi computation per step murah
#
# Mini-batch: theta := theta - lr * grad(L_batch)
#   - grad(L_batch) = (1/batch_size) * Sum grad(L_i) untuk batch
#   - Kompromi antara akurasi dan kecepatan
#   - Batch size = tradeoff antara variance dan computation
#
# Koneksi Teknik Elektro:
# - Full-batch = offline processing (semua data tersedia)
# - SGD = real-time processing (sample-by-sample)
# - Mini-batch = frame-based processing (DSP buffer)

def generate_regression_data(n=500):
    """
    Generate synthetic regression data.
    
    Parameters:
    -----------
    n : int, default 500
        Jumlah samples.
        
    Returns:
    --------
    X : np.ndarray, shape (n, 2)
        Features.
    y : np.ndarray, shape (n,)
        Targets dengan linear relationship + noise.
        
    Notes:
    ------
    - y = 3*x1 + 2*x2 + 1 + noise
    - True weights = [3, 2], true bias = 1
    - Noise Gaussian N(0, 0.5^2)
    """
    X = np.random.randn(n, 2)
    y = 3 * X[:, 0] + 2 * X[:, 1] + 1 + 0.5 * np.random.randn(n)
    return X, y


X_data, y_data = generate_regression_data(500)


class GDOptimizer:
    """
    Implementasi berbagai varian Gradient Descent.
    
    Attributes:
    -----------
    X : np.ndarray
        Training features.
    y : np.ndarray
        Training targets.
    n : int
        Jumlah samples.
        
    Notes:
    ------
    - Vanilla GD: update menggunakan seluruh dataset
    - SGD: update per sample
    - Mini-batch: update per subset
    - Koneksi Teknik Elektro: mirip dengan batch processing vs
      real-time processing di signal processing
    """
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n = len(X)
    
    def compute_gradient(self, X_batch, y_batch, w, b):
        """
        Menghitung gradient MSE untuk linear regression.
        
        Parameters:
        -----------
        X_batch : np.ndarray
            Batch features.
        y_batch : np.ndarray
            Batch targets.
        w : np.ndarray
            Current weights.
        b : float
            Current bias.
            
        Returns:
        --------
        dw : np.ndarray
            Gradient untuk weights.
        db : float
            Gradient untuk bias.
            
        Notes:
        ------
        Gradient MSE = (2/n_batch) * X_batch.T @ (y_pred - y_batch)
        Untuk SGD (batch_size=1): dw = 2 * x_i * (y_pred - y_i)
        Untuk full-batch: dw = (2/n) * X.T @ (y_pred - y)
        """
        y_pred = X_batch @ w + b
        error = y_pred - y_batch
        n = len(X_batch)
        dw = (2/n) * (X_batch.T @ error)
        db = (2/n) * np.sum(error)
        return dw, db
    
    def compute_loss(self, w, b):
        """
        Menghitung MSE loss untuk seluruh dataset.
        
        Parameters:
        -----------
        w : np.ndarray
            Weights.
        b : float
            Bias.
            
        Returns:
        --------
        float
            Mean Squared Error.
        """
        y_pred = self.X @ w + b
        return np.mean((y_pred - self.y) ** 2)
    
    def vanilla_gd(self, lr=0.01, n_iter=100):
        """
        Full-Batch Gradient Descent.
        
        Parameters:
        -----------
        lr : float, default 0.01
            Learning rate.
        n_iter : int, default 100
            Jumlah iterasi.
            
        Returns:
        --------
        w, b : np.ndarray, float
            Final parameters.
        history : dict
            Dictionary dengan 'loss' dan 'w' history.
            
        Notes:
        ------
        - Menggunakan seluruh dataset untuk setiap update
        - Gradient paling stabil (low variance)
        - Tapi lambat untuk dataset besar
        - Memory requirement tinggi jika dataset besar
        - Convergence teoritis: O(1/epsilon) untuk convex function
        """
        w = np.zeros(self.X.shape[1])
        b = 0.0
        history = {'loss': [], 'w': []}
        
        for _ in range(n_iter):
            dw, db = self.compute_gradient(self.X, self.y, w, b)
            w -= lr * dw
            b -= lr * db
            history['loss'].append(self.compute_loss(w, b))
            history['w'].append(w.copy())
        return w, b, history
    
    def sgd(self, lr=0.01, n_iter=100):
        """
        Stochastic Gradient Descent (1 sample per update).
        
        Parameters:
        -----------
        lr : float, default 0.01
            Learning rate.
        n_iter : int, default 100
            Jumlah epochs.
            
        Returns:
        --------
        w, b : np.ndarray, float
            Final parameters.
        history : dict
            Dictionary dengan 'loss' dan 'w' history.
            
        Notes:
        ------
        - Update per sample
        - Gradient sangat noisy tapi computation per update cepat
        - Noise bisa membantu escape local minima
        - Require smaller learning rate
        - Setiap epoch = 1 pass melalui seluruh dataset (n updates)
        - Convergence teoritis: O(1/epsilon^2) untuk convex
        """
        w = np.zeros(self.X.shape[1])
        b = 0.0
        history = {'loss': [], 'w': []}
        
        for _ in range(n_iter):
            # Shuffle setiap epoch
            indices = np.random.permutation(self.n)
            for i in indices:
                xi = self.X[i:i+1]
                yi = self.y[i:i+1]
                dw, db = self.compute_gradient(xi, yi, w, b)
                w -= lr * dw
                b -= lr * db
            
            history['loss'].append(self.compute_loss(w, b))
            history['w'].append(w.copy())
        return w, b, history
    
    def mini_batch_gd(self, lr=0.01, n_iter=100, batch_size=32):
        """
        Mini-Batch Gradient Descent.
        
        Parameters:
        -----------
        lr : float, default 0.01
            Learning rate.
        n_iter : int, default 100
            Jumlah epochs.
        batch_size : int, default 32
            Ukuran batch.
            
        Returns:
        --------
        w, b : np.ndarray, float
            Final parameters.
        history : dict
            Dictionary dengan 'loss' dan 'w' history.
            
        Notes:
        ------
        - Kompromi antara vanilla GD dan SGD
        - Batch size = tradeoff antara stability dan speed
        - 32-256 adalah range yang umum dipakai
        - GPU-friendly (parallel processing per batch)
        - Batch size kecil -> lebih noisy tapi generalization lebih baik
        - Batch size besar -> lebih stabil tapi bisa stuck di sharp minima
        - Convergence teoritis: di antara GD dan SGD
        """
        w = np.zeros(self.X.shape[1])
        b = 0.0
        history = {'loss': [], 'w': []}
        
        for _ in range(n_iter):
            indices = np.random.permutation(self.n)
            for start in range(0, self.n, batch_size):
                end = min(start + batch_size, self.n)
                batch_idx = indices[start:end]
                X_batch = self.X[batch_idx]
                y_batch = self.y[batch_idx]
                dw, db = self.compute_gradient(X_batch, y_batch, w, b)
                w -= lr * dw
                b -= lr * db
            
            history['loss'].append(self.compute_loss(w, b))
            history['w'].append(w.copy())
        return w, b, history


# Compare
opt = GDOptimizer(X_data, y_data)
print("\nTraining 3 varian GD...")
_, _, hist_gd = opt.vanilla_gd(lr=0.1, n_iter=50)
_, _, hist_sgd = opt.sgd(lr=0.01, n_iter=50)
_, _, hist_mb = opt.mini_batch_gd(lr=0.05, n_iter=50, batch_size=32)

plt.figure(figsize=(10, 5))
plt.plot(hist_gd['loss'], label='Full-Batch GD', linewidth=2)
plt.plot(hist_sgd['loss'], label='SGD', linewidth=2)
plt.plot(hist_mb['loss'], label='Mini-Batch (32)', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('GD vs SGD vs Mini-Batch')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.savefig('02_gd_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 02_gd_comparison.png")


# ===========================================================
# BAGIAN 3: Momentum
# ===========================================================
# Masalah: GD lambat di valley yang sempit (zig-zag)
# Solusi: Momentum - gunakan "inersia" dari update sebelumnya
#
# DETAIL MATEMATIKA:
# Update rule dengan momentum:
# v_t = beta * v_{t-1} + (1-beta) * gradient
# theta_t = theta_{t-1} - lr * v_t
#
# Atau dalam variasi populer:
# v_t = beta * v_{t-1} + gradient
# theta_t = theta_{t-1} - lr * v_t
#
# beta (momentum coefficient) = 0.9 adalah nilai yang umum.
# - beta = 0: tidak ada momentum (vanilla GD)
# - beta dekat 1: momentum sangat kuat (inersia besar)
#
# Efek momentum:
# - Arah konsisten -> velocity bertambah (accelerate)
# - Arah berubah -> velocity berkurang (damp oscillation)
# - Di valley sempit: mengurangi zig-zag
#
# Koneksi Teknik Elektro:
# - Momentum = inersia di mechanical systems
# - Mirip dengan damping di control systems
# - Velocity = state variable yang menyimpan history
# - beta = damping coefficient

class MomentumOptimizer:
    """
    Gradient Descent dengan Momentum.
    
    Attributes:
    -----------
    func : callable
        Fungsi objective yang akan di-minimize.
    grad : callable
        Function yang menghitung gradient.
        
    Notes:
    ------
    - Momentum = exponential moving average dari gradients
    - beta (momentum) = 0.9 adalah nilai yang umum
    - Mempercepat convergence di arah konsisten
    - Mengurangi oscillation di arah yang tidak konsisten
    - Koneksi Teknik Elektro: mirip dengan inersia di mechanical systems
      atau damping di control systems
    """
    
    def __init__(self, func, grad_func):
        self.func = func
        self.grad = grad_func
    
    def optimize(self, start, lr=0.01, momentum=0.9, n_iter=100):
        """
        Optimasi dengan momentum.
        
        Parameters:
        -----------
        start : array-like
            Starting point [x, y].
        lr : float, default 0.01
            Learning rate.
        momentum : float, default 0.9
            Momentum coefficient (0 = no momentum, 1 = infinite momentum).
            Nilai umum: 0.8 - 0.99.
        n_iter : int, default 100
            Jumlah iterasi.
            
        Returns:
        --------
        pos : np.ndarray
            Final position.
        history : np.ndarray
            Trajectory of positions.
            
        Notes:
        ------
        Update rule:
        v_t = beta * v_{t-1} + (1-beta) * g_t
        theta_t = theta_{t-1} - lr * v_t
        
        Interpretasi:
        - v adalah "velocity" (kecepatan)
        - beta adalah "friction" (mengurangi velocity dari iterasi sebelumnya)
        - (1-beta)*g_t adalah "acceleration" dari gradient saat ini
        - Semakin besar beta -> semakin banyak mempertahankan velocity lama
        """
        pos = np.array(start, dtype=float)
        velocity = np.zeros_like(pos)
        history = [pos.copy()]
        
        for _ in range(n_iter):
            g = self.grad(pos)
            # Update velocity: v = beta*v + (1-beta)*g
            velocity = momentum * velocity + (1 - momentum) * g
            # Update position: theta = theta - lr * v
            pos = pos - lr * velocity
            history.append(pos.copy())
        
        return pos, np.array(history)


# Quadratic function untuk demo
def quad_func(p):
    return p[0]**2 + 10 * p[1]**2


def quad_grad(p):
    """
    Gradient dari f(x,y) = x^2 + 10*y^2.
    
    df/dx = 2*x
    df/dy = 20*y
    """
    return np.array([2 * p[0], 20 * p[1]])


optimizer = MomentumOptimizer(quad_func, quad_grad)

# Compare GD vs Momentum
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, momentum, title in [(axes[0], 0.0, 'Vanilla GD (no momentum)'),
                              (axes[1], 0.9, 'With Momentum (beta=0.9)')]:
    _, history = optimizer.optimize([2.5, 2.5], lr=0.05, momentum=momentum, n_iter=50)
    
    # Contour
    x_r = np.linspace(-3, 3, 100)
    y_r = np.linspace(-3, 3, 100)
    Xg, Yg = np.meshgrid(x_r, y_r)
    Zg = Xg**2 + 10 * Yg**2
    ax.contour(Xg, Yg, Zg, levels=20, cmap='viridis', alpha=0.5)
    
    # Trajectory
    ax.plot(history[:, 0], history[:, 1], 'ro-', markersize=3, linewidth=1)
    ax.plot(history[0, 0], history[0, 1], 'gs', markersize=10, label='Start')
    ax.plot(0, 0, 'r*', markersize=15, label='Minimum')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('03_momentum.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 03_momentum.png")


# ===========================================================
# BAGIAN 4: Adam Optimizer
# ===========================================================
# Adam = Adaptive Moment Estimation
# Kombinasi dari Momentum + RMSProp
# Ini DEFAULT optimizer di deep learning!
#
# DETAIL MATEMATIKA:
# Adam menggunakan dua moment:
# - First moment (m): exponential moving average dari gradients
#   m_t = beta1 * m_{t-1} + (1-beta1) * g_t
# - Second moment (v): exponential moving average dari squared gradients
#   v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2
#
# Bias correction (penting di awal training!):
# - m_hat_t = m_t / (1 - beta1^t)
# - v_hat_t = v_t / (1 - beta2^t)
# - Tanpa bias correction, m dan v mendekati 0 di awal (karena diinisialisasi 0)
#
# Update rule:
# theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)
#
# Interpretasi:
# - m_hat_t / sqrt(v_hat_t) = signal-to-noise ratio
# - Semakin besar variance gradient -> step size lebih kecil
# - Semakin konsisten gradient -> step size lebih besar
# - Epsilon mencegah division by zero
#
# Analogi PID Controller:
# - Momentum (first moment) -> "I" (integral, akumulasi error)
# - RMSProp (second moment) -> adaptive gain per parameter
# - Bias correction -> transient response handling
# - lr -> overall controller gain

class AdamOptimizer:
    """
    Adam optimizer from scratch.
    
    Attributes:
    -----------
    lr : float
        Learning rate (default 0.001).
    beta1 : float
        Exponential decay rate untuk first moment (default 0.9).
    beta2 : float
        Exponential decay rate untuk second moment (default 0.999).
    eps : float
        Epsilon untuk numerical stability (default 1e-8).
        
    Notes:
    ------
    - Adam = Adaptive Moment Estimation
    - First moment (m) = exponential moving average of gradients
    - Second moment (v) = exponential moving average of squared gradients
    - Bias correction penting di awal training!
    - Default di PyTorch, TensorFlow, dll.
    - Koneksi Teknik Elektro: mirip dengan adaptive PID controller
      dengan auto-tuning gains
    """
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
    
    def optimize(self, grad_func, start, n_iter=100):
        """
        Optimasi menggunakan Adam.
        
        Parameters:
        -----------
        grad_func : callable
            Function yang menghitung gradient di posisi tertentu.
        start : array-like
            Starting point.
        n_iter : int, default 100
            Jumlah iterasi.
            
        Returns:
        --------
        pos : np.ndarray
            Final position.
        history : np.ndarray
            Trajectory.
            
        Notes:
        ------
        Algoritma Adam per iterasi t:
        1. g_t = gradient
        2. m_t = beta1 * m_{t-1} + (1-beta1) * g_t  (first moment)
        3. v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2  (second moment)
        4. m_hat_t = m_t / (1-beta1^t)  (bias correction)
        5. v_hat_t = v_t / (1-beta2^t)  (bias correction)
        6. theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)
        
        Kenapa bias correction penting?
        - m_0 = 0, v_0 = 0
        - Iterasi 1: m_1 = (1-beta1)*g_1 -> sangat kecil (karena 1-beta1 ~ 0.1)
        - Tanpa correction: step size sangat kecil di awal
        - Dengan correction: m_hat_1 = g_1 (seperti vanilla GD)
        """
        pos = np.array(start, dtype=float)
        m = np.zeros_like(pos)  # first moment (mean of gradients)
        v = np.zeros_like(pos)  # second moment (mean of squared gradients)
        history = [pos.copy()]
        
        for t in range(1, n_iter + 1):
            g = grad_func(pos)
            
            # Update biased moments
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g**2
            
            # Bias correction (penting di awal training!)
            # tanpa bias correction, m dan v mendekati 0 di awal
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)
            
            # Update
            pos = pos - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            history.append(pos.copy())
        
        return pos, np.array(history)


# Compare semua optimizer pada Rosenbrock
def rosenbrock_grad(p):
    """
    Gradient dari Rosenbrock function.
    
    df/dx = -2*(1-x) + 400*x*(x^2-y)
    df/dy = 200*(y-x^2)
    """
    x, y = p
    dx = -2*(1-x) + 400*x*(x**2 - y)
    dy = 200*(y - x**2)
    return np.array([dx, dy])


start = [-1.5, 1.5]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

optimizers = [
    ('Vanilla GD', lambda: MomentumOptimizer(rosenbrock, rosenbrock_grad).optimize(start, lr=0.001, momentum=0.0, n_iter=500)),
    ('Momentum (beta=0.9)', lambda: MomentumOptimizer(rosenbrock, rosenbrock_grad).optimize(start, lr=0.001, momentum=0.9, n_iter=500)),
    ('Adam', lambda: AdamOptimizer(lr=0.01).optimize(rosenbrock_grad, start, n_iter=500)),
]

x_r = np.linspace(-2, 2, 200)
y_r = np.linspace(-1, 3, 200)
Xg, Yg = np.meshgrid(x_r, y_r)
Zg = rosenbrock(Xg, Yg)

for ax, (name, opt_fn) in zip(axes, optimizers):
    final, history = opt_fn()
    ax.contour(Xg, Yg, Zg, levels=np.logspace(-1, 3, 30), cmap='viridis', alpha=0.5)
    ax.plot(history[:, 0], history[:, 1], 'r-', linewidth=1, alpha=0.7)
    ax.plot(history[0, 0], history[0, 1], 'gs', markersize=10)
    ax.plot(1, 1, 'r*', markersize=15)
    ax.set_title(f'{name}\nFinal: ({final[0]:.3f}, {final[1]:.3f})')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 3)

plt.tight_layout()
plt.savefig('04_optimizers_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 04_optimizers_comparison.png")

print("\n=== Optimizer Summary ===")
print("SGD:      Simpel, tapi zig-zag di landscape yang ill-conditioned")
print("Momentum: Lebih smooth, tapi perlu tune beta")
print("Adam:     Almost always works! Default choice untuk deep learning")
print("         Tapi bisa overshoot di beberapa kasus (lihat AdamW)")


# ===========================================================
# BAGIAN 5: Learning Rate - Efek & Scheduling
# ===========================================================
# Learning rate adalah hyperparameter PALING PENTING di GD.
# Efeknya lebih besar dari pilihan optimizer!
#
# DETAIL MATEMATIKA:
# Update: theta := theta - lr * gradient
# - lr terlalu kecil: convergence sangat lambat, bisa stuck di local minimum
# - lr terlalu besar: divergence (oscillate atau NaN)
# - lr "just right": smooth convergence
#
# Learning rate scheduling:
# - Step decay: lr = lr0 * gamma^(epoch // step_size)
# - Exponential decay: lr = lr0 * exp(-k * epoch)
# - Cosine annealing: lr = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*epoch/max_epoch))
# - Warmup: lr naik linear dari 0 ke lr_target di awal training
#
# Kenapa scheduling penting?
# - LR besar di awal: cepat converge ke region optimal
# - LR kecil di akhir: fine-tuning untuk presisi
# - Warmup: menghindari "thermal shock" ke model di awal training

print("\n=== Learning Rate Experiment ===")
opt = GDOptimizer(X_data, y_data)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
learning_rates = [0.001, 0.1, 0.5]
titles = ['Too small (lr=0.001)', 'Good (lr=0.1)', 'Too large (lr=0.5)']

for ax, lr, title in zip(axes, learning_rates, titles):
    try:
        _, _, hist = opt.vanilla_gd(lr=lr, n_iter=100)
        losses = hist['loss']
        if any(np.isnan(losses)) or any(np.isinf(losses)):
            raise ValueError("Diverged")
        ax.plot(losses)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True)
    except Exception:
        ax.text(0.5, 0.5, 'DIVERGED!', transform=ax.transAxes,
                fontsize=20, ha='center', color='red')
        ax.set_title(title)

plt.tight_layout()
plt.savefig('05_learning_rate_effect.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: 05_learning_rate_effect.png")


# ===========================================================
# LATIHAN 6: Implementasi Optimizer
# ===========================================================
"""
TARGET Learning Objectives:
   - Memahami RMSProp dan Nesterov Accelerated Gradient
   - Mengimplementasikan learning rate warmup
   - Membandingkan performa optimizer pada fungsi yang sulit

PANDUAN LANGKAH-LANGKAH:

STEP 1: Implementasi RMSProp dari nol
-------------------------------------
RMSProp = Root Mean Square Propagation.

   Update rule:
   v_t = beta * v_{t-1} + (1-beta) * g^2
   theta_t = theta_{t-1} - lr * g / (sqrt(v_t) + epsilon)
   
   TIPS: Apa yang harus dilakukan:
     a) Buat class RMSPropOptimizer
     b) Simpan state v (second moment)
     c) Update seperti rumus di atas
     
   TIPS: KENAPA RMSProp?
     - Adaptive learning rate per parameter
     - Parameter dengan gradient besar -> lr lebih kecil
     - Parameter dengan gradient kecil -> lr lebih besar
     - Mengatasi vanishing gradient untuk beberapa parameter
     
   PERINGATAN: Hati-hati:
     - beta umumnya 0.9 atau 0.99
     - v diinisialisasi ke 0
     - Tanpa bias correction (tidak seperti Adam)
     - Epsilon penting untuk mencegah division by zero


STEP 2: Implementasi Nesterov Accelerated Gradient (NAG)
--------------------------------------------------------
NAG = Momentum dengan "look-ahead".

   Update rule:
   g = grad(f)(theta - beta * v_{t-1})  <- gradient di posisi "lookahead"
   v_t = beta * v_{t-1} + lr * g
   theta_t = theta_{t-1} - v_t
   
   TIPS: Apa yang harus dilakukan:
     a) Buat class NAGOptimizer
     b) Hitung gradient di posisi theta - beta*v (bukan di theta)
     c) Update velocity dan position
     
   TIPS: KENAPA NAG?
     - "Melihat" ke depan sebelum melangkah
     - Lebih stable dari momentum sederhana
     - Convergence lebih cepat di convex functions
     - Mengurangi oscillation lebih efektif
     
   PERINGATAN: Hati-hati:
     - Gradient dihitung di posisi yang berbeda dari theta
     - Implementasi bisa tricky kalau tidak hati-hati
     - Perlu two-step update: lookahead lalu gradient


STEP 3: Implementasi Learning Rate Warmup
-----------------------------------------
Warmup = mulai dari lr kecil, naik linear ke lr target.

   TIPS: Schedule:
   - Epoch 1-N_warmup: lr naik linear dari lr_min ke lr_max
   - Epoch N_warmup+: cosine decay dari lr_max ke lr_min
   
   Formula warmup:
   lr = lr_min + (lr_max - lr_min) * (epoch / N_warmup)
   
   Formula cosine decay:
   lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
   
   TIPS: KENAPA warmup?
     - Di awal training, gradients bisa besar dan noisy
     - Warmup menghindari "thermal shock" ke model
     - Standard di training Transformer/BERT
     - Membantu stabilitas di awal training


STEP 4: Bandingkan Semua Optimizer
----------------------------------
Gunakan Rosenbrock function untuk perbandingan.

   Optimizer yang diuji:
   1. Vanilla GD
   2. Momentum
   3. NAG
   4. RMSProp
   5. Adam
   
   Metrics:
   - Final distance ke global minimum (1,1)
   - Number of iterations to converge
   - Stability (variance di trajectory)
   
   Visualisasi:
   - Plot trajectory di contour plot (satu figure, 5 subplot)
   - Plot loss curve (satu figure, semua optimizer)


TIPS HINTS:
   - Untuk RMSProp, v diinisialisasi ke zeros
   - Untuk NAG, hati-hati dengan posisi lookahead
   - Untuk warmup, simpan lr_history untuk plotting
   - Gunakan learning rate yang sama untuk fair comparison
   - Rosenbrock: lr kecil (~0.001) untuk GD, lebih besar untuk Adam (~0.01)

PERINGATAN COMMON MISTAKES:
   - RMSProp tanpa epsilon -> division by zero
   - NAG menghitung gradient di posisi salah
   - Warmup terlalu pendek -> tidak ada efek
   - Membandingkan dengan hyperparameters yang tidak fair

TARGET EXPECTED OUTPUT:
   - Adam dan RMSProp converge paling cepat
   - Vanilla GD zig-zag di valley
   - NAG lebih stabil dari Momentum
   - Warmup membuat training awal lebih smooth
"""


# ===========================================================
# CHALLENGE: Gradient Descent Visualizer
# ===========================================================
"""
TARGET Learning Objectives:
   - Membangun interactive visualization untuk optimizers
   - Memperdalam intuisi tentang optimization landscape
   - Mengembangkan tools untuk debugging dan teaching

PANDUAN LANGKAH-LANGKAH:

STEP 1: Design Interactive Visualizer
-------------------------------------
Buat script Python yang menerima input dari user:

   Input parameters:
   - Fungsi target: 'quadratic', 'rosenbrock', 'beale', 'himmelblau'
   - Optimizer: 'gd', 'momentum', 'rmsprop', 'adam'
   - Learning rate
   - Starting point [x, y]
   - Number of iterations
   
   TIPS: Fungsi tambahan yang bisa diuji:
   - Beale: f(x,y) = (1.5-x+x*y)^2 + (2.25-x+x*y^2)^2 + (2.625-x+x*y^3)^2
   - Himmelblau: f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2 (4 global minima!)


STEP 2: Implementasi Real-Time Visualization
--------------------------------------------
Gunakan matplotlib.animation untuk animasi:

   a) Buat figure dengan 2 subplot:
      - Left: contour plot + trajectory (real-time update)
      - Right: loss curve (real-time update)
      
   b) FuncAnimation untuk update per frame:
      - Frame i = iterasi ke-i
      - Update trajectory line
      - Update loss curve
      - Update current position marker
      
   c) Simpan animasi sebagai .gif atau .mp4


STEP 3: Add Gradient Visualization
----------------------------------
Tambahkan visualisasi gradient:

   a) Arrow dari current position menunjukkan arah gradient
   b) Arrow length proportional to gradient magnitude
   c) Color-coded: red = large gradient, blue = small gradient
   
   TIPS: KENAPA visualisasi gradient?
     - Memahami kenapa optimizer "zig-zag"
     - Melihat arah steepest descent
     - Memahami efek momentum (gradient vs velocity)


STEP 4: Add Comparison Mode
---------------------------
Tampilkan 2-4 optimizer secara bersamaan:

   a) Same figure, different colors untuk trajectory
   b) Synchronized animation (same frame = same iteration)
   c) Final comparison table: distance to minimum, iterations, path length


STEP 5: Export dan Dokumentasi
------------------------------
   a) Simpan animasi sebagai file
   b) Buat README dengan:
      - Cara menjalankan
      - Penjelasan setiap fungsi test
      - Interpretasi hasil
      - Tips untuk tuning hyperparameters


TIPS HINTS:
   - from matplotlib.animation import FuncAnimation
   - ani.save('optimizer_demo.gif', writer='pillow', fps=10)
   - Gunakan plt.pause(0.01) untuk real-time update tanpa animasi
   - Clear axis dengan ax.clear() sebelum redraw

PERINGATAN COMMON MISTAKES:
   - Tidak mengatur axis limits -> plot "jump"
   - Frame rate terlalu tinggi -> animasi terlalu cepat
   - Lupa normalize arrow length -> arrow terlalu besar/kecil
   - Tidak handle optimizer yang diverge -> NaN di animasi

TARGET EXPECTED OUTPUT:
   - Animasi .gif yang menunjukkan perbedaan optimizer
   - File Python yang bisa di-run dengan berbagai konfigurasi
   - README dengan penjelasan mendalam
   - Tools yang bisa dipakai untuk presentasi/teaching

Ini akan memperdalam intuisi tentang optimization landscape
dan kenapa Adam biasanya bekerja paling baik.
"""

print("\n" + "="*50)
print("OK Modul selesai! Lanjut ke: 02-ml-dari-nol/04_evaluasi_model.py")
print("="*50)
