"""
=============================================================
SOLUSI REFERENSI - 01_NUMPY_ESSENTIALS (EXERCISE 1)
=============================================================
Referensi jawaban untuk EXERCISE 1 di
01-fondasi-data/01_numpy_essentials.py.

⚠️  JANGAN dibuka sebelum mencoba sendiri!
    Coba dulu:  python 01_numpy_essentials.py --exercise

Jalankan solusi ini untuk melihat semua test lulus:
    python 01_numpy_solutions.py
=============================================================
"""

import numpy as np


def batch_normalize(X):
    """
    Normalisasi per kolom: X_norm = (X - mean) / std.

    Catatan penting:
    - axis=0 menghitung statistik per kolom (bukan keseluruhan array).
    - keepdims=True mempertahankan shape (1, D) agar broadcasting
      ke (N, D) bekerja dengan benar.
    - epsilon 1e-8 mencegah division by zero saat std = 0.
    """
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return (X - mean) / (std + 1e-8)


def cosine_similarity(a, b):
    """
    cos(theta) = (a . b) / (||a|| * ||b||)

    Catatan penting:
    - Denominator bisa 0 jika salah satu vektor adalah zero vector.
      Solusi: kembalikan 0.0 (atau raise error yang jelas).
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return np.dot(a, b) / denom


def one_hot_encode(labels, num_classes):
    """
    Ubah label integer -> matrix one-hot (N, num_classes).

    Catatan penting:
    - Fancy indexing: result[np.arange(N), labels] = 1
      men-set elemen baris i, kolom labels[i] menjadi 1.
    """
    result = np.zeros((len(labels), num_classes))
    result[np.arange(len(labels)), labels] = 1
    return result


# ===========================================================
# Verifikasi: semua test harus PASS
# ===========================================================
if __name__ == "__main__":
    print("=" * 50)
    print("🧪 Verifikasi solusi NumPy...")
    print("=" * 50)

    X_test = np.random.randn(50, 3)
    X_norm = batch_normalize(X_test)
    assert np.allclose(X_norm.mean(axis=0), 0, atol=1e-10), "Mean harus ~0"
    assert np.allclose(X_norm.std(axis=0), 1, atol=1e-10), "Std harus ~1"
    print("  ✅ PASS: batch_normalize (mean ~ 0, std ~ 1)")

    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    assert abs(cosine_similarity(a, b)) < 1e-10, "Orthogonal harus cos = 0"
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-10, "Identik harus cos = 1"
    print("  ✅ PASS: cosine_similarity (orthogonal = 0, identik = 1)")

    labels = np.array([0, 2, 1, 0])
    oh = one_hot_encode(labels, 3)
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    assert np.array_equal(oh, expected), f"Expected:\n{expected}\nGot:\n{oh}"
    print("  ✅ PASS: one_hot_encode (output sesuai ekspektasi)")

    print("=" * 50)
    print("🎉 Semua solusi lulus!")
    print("=" * 50)
