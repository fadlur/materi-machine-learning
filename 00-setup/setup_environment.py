"""
=============================================================
FASE 0: SETUP ENVIRONMENT — ML Engineer Track
=============================================================
Jalankan file ini untuk memastikan semua tools siap.

Kita akan pakai:
- Python 3.10+
- NumPy, Pandas, Matplotlib, Seaborn (data & visualisasi)
- scikit-learn, XGBoost (classical ML)
- PyTorch (deep learning)
- Hugging Face Transformers (NLP & LLM)
- MLflow, W&B (experiment tracking)
- FastAPI, Docker (deployment)
- VS Code Interactive / Jupyter (recommended)

Untuk ML Engineer track, MLOps tools adalah WAJIB — leverage
pengalaman backend kamu!
=============================================================
"""

import sys
import subprocess


def check_python():
    """
    Memeriksa versi Python yang sedang berjalan.
    
    Function ini mengambil informasi versi Python dari sys.version_info
    dan memverifikasi apakah versinya memenuhi minimum requirement (3.10+).
    
    Parameters:
    -----------
    Tidak ada parameter. Function ini menggunakan sys.version_info global.
    
    Returns:
    --------
    None
        Function ini hanya mencetak hasil ke console, tidak mengembalikan nilai.
        
    Notes:
    ------
    - sys.version_info mengembalikan named tuple (major, minor, micro, releaselevel, serial)
    - Python 3.10 dipilih karena banyak fitur modern seperti pattern matching
      dan improved type hints yang sering dipakai di ML libraries
    """
    v = sys.version_info
    print(f"Python version: {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print("⚠️  Recommended: Python 3.10+")
    else:
        print("✅ Python version OK")


def check_package(name, import_name=None, optional=False):
    """
    Memeriksa apakah sebuah Python package sudah terinstall.
    
    Function ini mencoba meng-import package dan mengambil versinya.
    Jika import berhasil, package dianggap terinstall.
    Jika import gagal (ImportError), package dianggap belum terinstall.
    
    Parameters:
    -----------
    name : str
        Nama package yang akan ditampilkan di output.
        Contoh: "numpy", "scikit-learn", "torch"
        
    import_name : str, optional
        Nama module yang digunakan untuk import. Biasanya sama dengan 'name',
        tapi bisa berbeda (contoh: "scikit-learn" → import "sklearn").
        Default None, yang berarti menggunakan 'name' sebagai import_name.
        
    optional : bool, optional
        Jika True, package dianggap opsional dan tidak wajib.
        Jika False, package dianggap wajib.
        Default False.
        
    Returns:
    --------
    bool
        True jika package berhasil di-import (terinstall).
        False jika package gagal di-import (belum terinstall).
        
    Notes:
    ------
    - Function ini menggunakan __import__() untuk dynamic import
    - getattr(mod, '__version__', 'unknown') mengambil atribut __version__ 
      dari module. Jika tidak ada, menggunakan 'unknown' sebagai fallback.
    - Koneksi ke Teknik Elektro: mirip dengan pengecekan dependency 
      di embedded systems sebelum flashing firmware.
    """
    if import_name is None:
        import_name = name
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        tag = "✅" if not optional else "🟡"
        print(f"{tag} {name} ({version})" + (" [optional]" if optional else ""))
        return True
    except ImportError:
        tag = "❌" if not optional else "⚪"
        print(f"{tag} {name} — belum terinstall" + (" [optional]" if optional else ""))
        return False


def main():
    """
    Fungsi utama yang menjalankan seluruh pengecekan environment.
    
    Function ini mengorkestrasi pengecekan Python version dan
    semua package yang dibutuhkan untuk ML Engineer track.
    Package dikelompokkan berdasarkan kategori:
    - Core Data Science (Wajib)
    - Deep Learning (Wajib)
    - NLP & LLM (Wajib untuk AI Engineer)
    - MLOps & Production (Wajib — leverage backend exp!)
    - Data Validation & Monitoring (Recommended)
    - LLM Engineering (Recommended)
    - Interactive Environment (Optional)
    
    Parameters:
    -----------
    Tidak ada parameter.
    
    Returns:
    --------
    None
        Function ini hanya mencetak hasil ke console.
        
    Notes:
    ------
    - Setiap kategori package di-iterasi menggunakan for loop
    - Package yang missing dicatat dalam list untuk summary akhir
    - all_missing menggabungkan semua package yang belum terinstall
      dari semua kategori wajib
    """
    print("=" * 60)
    print("🔍 Checking Environment — ML Engineer Track")
    print("=" * 60)

    # Step 1: Cek versi Python terlebih dahulu
    check_python()
    print()

    # ===========================================================
    # 📦 CORE DATA SCIENCE (Wajib)
    # ===========================================================
    # Package ini adalah fondasi dari semua data science & ML di Python.
    # Tanpa package ini, hampir tidak mungkin melakukan ML di Python.
    print("📦 CORE DATA SCIENCE (Wajib)")
    print("-" * 40)
    core_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost", True),
        ("lightgbm", "lightgbm", True),
    ]
    core_missing = []
    for name, imp, *opt in core_packages:
        optional = opt[0] if opt else False
        if not check_package(name, imp, optional):
            core_missing.append(name)
    print()

    # ===========================================================
    # 📦 DEEP LEARNING (Wajib)
    # ===========================================================
    # PyTorch adalah framework deep learning yang paling populer di riset
    # dan semakin banyak dipakai di industri.
    print("📦 DEEP LEARNING (Wajib)")
    print("-" * 40)
    dl_packages = [
        ("torch (PyTorch)", "torch"),
        ("torchvision", "torchvision", True),
        ("torchaudio", "torchaudio", True),
    ]
    dl_missing = []
    for name, imp, *opt in dl_packages:
        optional = opt[0] if opt else False
        if not check_package(name, imp, optional):
            dl_missing.append(name)
    print()

    # ===========================================================
    # 📦 NLP & LLM (Wajib untuk AI Engineer track)
    # ===========================================================
    # Transformers dan datasets dari Hugging Face adalah tools utama
    # untuk NLP modern dan LLM engineering.
    print("📦 NLP & LLM (Wajib untuk AI Engineer)")
    print("-" * 40)
    nlp_packages = [
        ("transformers", "transformers"),
        ("datasets", "datasets", True),
        ("tokenizers", "tokenizers", True),
    ]
    nlp_missing = []
    for name, imp, *opt in nlp_packages:
        optional = opt[0] if opt else False
        if not check_package(name, imp, optional):
            nlp_missing.append(name)
    print()

    # ===========================================================
    # 📦 MLOps & PRODUCTION (Wajib — Ini kekuatanmu!)
    # ===========================================================
    # Package ini adalah yang membedakan ML Engineer dari Data Scientist.
    # Background backend kamu sangat membantu di sini!
    print("📦 MLOps & PRODUCTION (Wajib — Ini kekuatanmu!)")
    print("-" * 40)
    mlops_packages = [
        ("mlflow", "mlflow"),
        ("wandb", "wandb", True),
        ("dvc", "dvc", True),
        ("hydra-core", "hydra", True),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("streamlit", "streamlit", True),
        ("gradio", "gradio", True),
    ]
    mlops_missing = []
    for name, imp, *opt in mlops_packages:
        optional = opt[0] if opt else False
        if not check_package(name, imp, optional):
            mlops_missing.append(name)
    print()

    # ===========================================================
    # 📦 DATA VALIDATION & MONITORING (Recommended)
    # ===========================================================
    # Package ini untuk memastikan data quality dan monitoring di production.
    print("📦 DATA VALIDATION & MONITORING (Recommended)")
    print("-" * 40)
    mon_packages = [
        ("evidently", "evidently", True),
        ("pandera", "pandera", True),
        ("great_expectations", "great_expectations", True),
    ]
    mon_missing = []
    for name, imp, *opt in mon_packages:
        optional = opt[0] if opt else False
        if not check_package(name, imp, optional):
            mon_missing.append(name)
    print()

    # ===========================================================
    # 📦 LLM ENGINEERING (Recommended)
    # ===========================================================
    # Package ini untuk membangun aplikasi LLM seperti RAG dan agents.
    print("📦 LLM ENGINEERING (Recommended)")
    print("-" * 40)
    llm_packages = [
        ("langchain", "langchain", True),
        ("langchain-openai", "langchain_openai", True),
        ("openai", "openai", True),
        ("chromadb", "chromadb", True),
    ]
    llm_missing = []
    for name, imp, *opt in llm_packages:
        optional = opt[0] if opt else False
        if not check_package(name, imp, optional):
            llm_missing.append(name)
    print()

    # ===========================================================
    # 📦 INTERACTIVE ENVIRONMENT
    # ===========================================================
    # Jupyter/IPython memungkinkan eksplorasi data secara interaktif.
    print("📦 INTERACTIVE ENVIRONMENT")
    print("-" * 40)
    check_package("jupyter", "jupyter", optional=True)
    check_package("ipykernel", "ipykernel", optional=True)
    print()

    # ===========================================================
    # 📋 SUMMARY
    # ===========================================================
    # Menggabungkan semua package yang belum terinstall untuk summary.
    print("=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)

    all_missing = core_missing + dl_missing + nlp_missing + mlops_missing

    if all_missing:
        print("\n📦 Install semua yang belum ada:")
        print("   pip install numpy pandas matplotlib seaborn scikit-learn")
        print("   pip install torch torchvision torchaudio")
        print("   pip install transformers datasets")
        print("   pip install mlflow fastapi uvicorn streamlit")
        print("\n   Untuk PyTorch, cek versi yang sesuai di:")
        print("   https://pytorch.org/get-started/locally/")
        print("\n   Optional (tapi recommended untuk ML Engineer track):")
        print("   pip install xgboost lightgbm wandb dvc hydra-core")
        print("   pip install evidently pandera")
        print("   pip install langchain langchain-openai openai chromadb")
    else:
        print("🎉 Semua package wajib sudah terinstall!")
        print("   Kamu siap mulai ML Engineer Track.")

    print()
    print("=" * 60)
    print("📁 Struktur Direktori — ML Engineer Track")
    print("=" * 60)
    print("""
    machine-learning/
    ├── README.md                      ← Roadmap utama (baca dulu!)
    ├── 90-day-action-plan.md          ← Jadwal harian 90 hari
    ├── AGENTS.md                      ← Panduan untuk AI agent
    ├── 00-setup/                      ← Kamu di sini
    ├── 01-fondasi-data/               ← NumPy, Pandas, Visualisasi
    ├── 02-ml-dari-nol/                ← Bangun ML dengan NumPy only
    ├── 03-classical-ml/               ← sklearn & model selection
    ├── 04-deep-learning/              ← Neural nets & PyTorch
    ├── 05-advanced/                   ← Transfer learning, NLP, Generative
    ├── 06-expert/                     ← Paper impl, MLOps, Production
    ├── 07-career-prep/                ← Interview, Resume, System Design ← NEW
    ├── 08-production-ml/              ← Feature Store, Monitoring, LLM ← NEW
    └── projects/                      ← Proyek mandiri (portfolio!)
    """)

    print("=" * 60)
    print("🎯 Next Step")
    print("=" * 60)
    print("""
    1. Install semua package yang belum ada
    2. Baca 90-day-action-plan.md untuk jadwal harian
    3. Mulai dari 01-fondasi-data/01_numpy_essentials.py
    4. Setiap selesai modul → commit ke git
    """)


if __name__ == "__main__":
    main()
