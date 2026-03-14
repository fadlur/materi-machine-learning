"""
=============================================================
FASE 2 — MODUL 3: GRADIENT DESCENT DEEP DIVE
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
# 📖 BAGIAN 1: Visualisasi Loss Landscape
# ===========================================================
# Sebelum optimasi, kita harus "lihat" apa yang dioptimasi.

def rosenbrock(x, y):
    """Rosenbrock function — classic optimization test"""
    return (1 - x)**2 + 100 * (y - x**2)**2

def simple_quadratic(x, y):
    """Fungsi sederhana: f(x,y) = x² + 10y²"""
    return x**2 + 10 * y**2

# Plot landscape
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x_range = np.linspace(-3, 3, 200)
y_range = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x_range, y_range)

# Quadratic (mudah)
Z1 = simple_quadratic(X, Y)
axes[0].contour(X, Y, Z1, levels=30, cmap='viridis')
axes[0].set_title('Quadratic: x² + 10y² (mudah)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].plot(0, 0, 'r*', markersize=15, label='Minimum')
axes[0].legend()

# Rosenbrock (sulit — valley yang sempit)
x_range2 = np.linspace(-2, 2, 200)
y_range2 = np.linspace(-1, 3, 200)
X2, Y2 = np.meshgrid(x_range2, y_range2)
Z2 = rosenbrock(X2, Y2)
axes[1].contour(X2, Y2, Z2, levels=np.logspace(-1, 3, 30), cmap='viridis')
axes[1].set_title('Rosenbrock (sulit — narrow valley)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].plot(1, 1, 'r*', markersize=15, label='Minimum')
axes[1].legend()

plt.tight_layout()
plt.savefig('01_loss_landscapes.png', dpi=100, bbox_inches='tight')
plt.close()
print("📊 Saved: 01_loss_landscapes.png")


# ===========================================================
# 📖 BAGIAN 2: Vanilla GD vs SGD vs Mini-batch
# ===========================================================

def generate_regression_data(n=500):
    X = np.random.randn(n, 2)
    y = 3 * X[:, 0] + 2 * X[:, 1] + 1 + 0.5 * np.random.randn(n)
    return X, y

X_data, y_data = generate_regression_data(500)


class GDOptimizer:
    """Implementasi berbagai varian Gradient Descent"""

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n = len(X)

    def compute_gradient(self, X_batch, y_batch, w, b):
        """Gradient MSE untuk linear regression"""
        y_pred = X_batch @ w + b
        error = y_pred - y_batch
        n = len(X_batch)
        dw = (2/n) * (X_batch.T @ error)
        db = (2/n) * np.sum(error)
        return dw, db

    def compute_loss(self, w, b):
        y_pred = self.X @ w + b
        return np.mean((y_pred - self.y) ** 2)

    def vanilla_gd(self, lr=0.01, n_iter=100):
        """Full-Batch Gradient Descent"""
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
        """Stochastic Gradient Descent (1 sample per update)"""
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
        """Mini-Batch Gradient Descent"""
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
print("Training 3 varian GD...")
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
print("📊 Saved: 02_gd_comparison.png")


# ===========================================================
# 📖 BAGIAN 3: Momentum
# ===========================================================
# Masalah: GD lambat di valley yang sempit (zig-zag)
# Solusi: Momentum — gunakan "inersia" dari update sebelumnya
#
# v_t = β * v_{t-1} + (1-β) * gradient
# θ = θ - lr * v_t
#
# Analogi: bola yang menggelinding — semakin cepat di arah konsisten

class MomentumOptimizer:
    def __init__(self, func, grad_func):
        self.func = func
        self.grad = grad_func

    def optimize(self, start, lr=0.01, momentum=0.9, n_iter=100):
        pos = np.array(start, dtype=float)
        velocity = np.zeros_like(pos)
        history = [pos.copy()]

        for _ in range(n_iter):
            g = self.grad(pos)
            velocity = momentum * velocity + (1 - momentum) * g
            pos = pos - lr * velocity
            history.append(pos.copy())

        return pos, np.array(history)


# Quadratic function untuk demo
def quad_func(p):
    return p[0]**2 + 10 * p[1]**2

def quad_grad(p):
    return np.array([2 * p[0], 20 * p[1]])

optimizer = MomentumOptimizer(quad_func, quad_grad)

# Compare GD vs Momentum
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, momentum, title in [(axes[0], 0.0, 'Vanilla GD (no momentum)'),
                              (axes[1], 0.9, 'With Momentum (β=0.9)')]:
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
print("📊 Saved: 03_momentum.png")


# ===========================================================
# 📖 BAGIAN 4: Adam Optimizer
# ===========================================================
# Adam = Adaptive Moment Estimation
# Kombinasi dari Momentum + RMSProp
# Ini DEFAULT optimizer di deep learning!
#
# Analogi: Adam seperti PID controller untuk optimization:
# - Momentum (first moment) → "I" (integral, akumulasi)
# - RMSProp (second moment) → adaptive gain per parameter
# - Bias correction → transient response handling

class AdamOptimizer:
    """Adam optimizer from scratch"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def optimize(self, grad_func, start, n_iter=100):
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
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)

            # Update
            pos = pos - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            history.append(pos.copy())

        return pos, np.array(history)


# Compare semua optimizer pada Rosenbrock
def rosenbrock_grad(p):
    x, y = p
    dx = -2*(1-x) + 400*x*(x**2 - y)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

start = [-1.5, 1.5]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

optimizers = [
    ('Vanilla GD', lambda: MomentumOptimizer(rosenbrock, rosenbrock_grad).optimize(start, lr=0.001, momentum=0.0, n_iter=500)),
    ('Momentum (β=0.9)', lambda: MomentumOptimizer(rosenbrock, rosenbrock_grad).optimize(start, lr=0.001, momentum=0.9, n_iter=500)),
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
print("📊 Saved: 04_optimizers_comparison.png")

print("\n=== Optimizer Summary ===")
print("SGD:      Simpel, tapi zig-zag di landscape yang ill-conditioned")
print("Momentum: Lebih smooth, tapi perlu tune β")
print("Adam:     Almost always works! Default choice untuk deep learning")
print("         Tapi bisa overshoot di beberapa kasus (lihat AdamW)")


# ===========================================================
# 📖 BAGIAN 5: Learning Rate — Efek & Scheduling
# ===========================================================

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
print("📊 Saved: 05_learning_rate_effect.png")


# ===========================================================
# 🏋️ EXERCISE 6: Implementasi Optimizer
# ===========================================================
"""
1. Implementasi RMSProp dari nol:
   v_t = β * v_{t-1} + (1-β) * g²
   θ = θ - lr * g / (√v_t + ε)

2. Implementasi Nesterov Accelerated Gradient (NAG):
   Look-ahead step: g = ∇f(θ - β*v_{t-1})
   v_t = β * v_{t-1} + lr * g
   θ = θ - v_t

3. Implementasi Learning Rate Warmup:
   - Mulai dari lr yang sangat kecil
   - Naikkan secara linear sampai lr target dalam N step pertama
   - Lalu gunakan cosine decay

4. Bandingkan semua optimizer pada Rosenbrock function.
   Plot trajectory dan convergence curve.
"""


# ===========================================================
# 🔥 CHALLENGE: Gradient Descent Visualizer
# ===========================================================
"""
Buat interactive visualizer (bisa pakai matplotlib animation):

1. User bisa pilih:
   - Fungsi target (quadratic, rosenbrock, atau custom)
   - Optimizer (GD, Momentum, RMSProp, Adam)
   - Learning rate
   - Starting point

2. Visualisasi real-time:
   - Contour plot dengan trajectory
   - Loss curve
   - Gradient magnitude over time
   - Learning rate effective (untuk adaptive methods)

3. Buat animasi (matplotlib.animation.FuncAnimation)
   yang menunjukkan step-by-step optimization

Ini akan memperdalam intuisi tentang optimization landscape
dan kenapa Adam biasanya bekerja paling baik.
"""

print("\n" + "="*50)
print("✅ Modul selesai! Lanjut ke: 02-ml-dari-nol/04_evaluasi_model.py")
print("="*50)


# ===========================================================
# MILESTONE ASSESSMENT — 2.3 Gradient Descent Deep Dive
# ===========================================================
# Referensi lengkap: ASSESSMENT.md (Fase 2, bagian 2.3)
#
# Level 1 — Bisa Dikerjakan (timer: 30 menit):
#   [ ] Implementasi vanilla GD, SGD, dan mini-batch GD
#   [ ] Plot loss curve perbandingan ketiganya
#   [ ] Implementasi momentum
#
# Level 2 — Bisa Dijelaskan:
#   [ ] Kenapa SGD lebih zig-zag tapi bisa escape local minima?
#   [ ] Peran momentum? Analogikan dengan fisika/kontrol
#   [ ] Adam: apa yang di-track? Kenapa butuh bias correction?
#
# Level 3 — Bisa Improvisasi (timer: 45 menit):
#   [ ] Adam optimizer dari scratch
#   [ ] Learning rate scheduler: cosine annealing
#   [ ] Visualisasi trajectory di 2D loss surface
#
# SKOR: ___/27
# TARGET PD: minimal 18/27 (rata-rata 2.0)
# ===========================================================
