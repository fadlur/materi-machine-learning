"""
=============================================================
FASE 0: SETUP ENVIRONMENT
=============================================================
Jalankan file ini untuk memastikan semua tools siap.

Kita akan pakai:
- Python 3.10+
- NumPy, Pandas, Matplotlib, Seaborn (data & visualisasi)
- scikit-learn (classical ML)
- PyTorch (deep learning) — dipilih karena lebih "Pythonic" dan
  banyak dipakai di riset (relevan dengan background S2)
- Jupyter / VS Code Interactive (opsional, tapi recommended)
"""

import sys
import subprocess

def check_python():
    v = sys.version_info
    print(f"Python version: {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print("⚠️  Recommended: Python 3.10+")
    else:
        print("✅ Python version OK")

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {name} ({version})")
        return True
    except ImportError:
        print(f"❌ {name} — belum terinstall")
        return False

def main():
    print("=" * 50)
    print("🔍 Checking Environment")
    print("=" * 50)

    check_python()
    print()

    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("torch (PyTorch)", "torch"),
        ("jupyter", "jupyter"),
    ]

    missing = []
    for name, imp in packages:
        if not check_package(name, imp):
            missing.append(name)

    print()
    if missing:
        print("📦 Install yang belum ada:")
        print("   pip install numpy pandas matplotlib seaborn scikit-learn torch jupyter")
        print()
        print("   Untuk PyTorch, cek versi yang sesuai di: https://pytorch.org/get-started/locally/")
    else:
        print("🎉 Semua package sudah terinstall! Lanjut ke Fase 1.")

    print()
    print("=" * 50)
    print("📁 Struktur Direktori")
    print("=" * 50)
    print("""
    machine learning/
    ├── README.md                      ← Roadmap (baca ini dulu!)
    ├── 00-setup/                      ← Kamu di sini
    ├── 01-fondasi-data/               ← NumPy, Pandas, Visualisasi
    ├── 02-ml-dari-nol/                ← Bangun ML dengan NumPy only
    ├── 03-classical-ml/               ← sklearn & model selection
    ├── 04-deep-learning/              ← Neural nets & PyTorch
    ├── 05-advanced/                   ← Transfer learning, NLP, Generative
    ├── 06-expert/                     ← Paper impl, MLOps, Production
    └── projects/                      ← Proyek mandiri (portfolio!)
    """)

if __name__ == "__main__":
    main()
