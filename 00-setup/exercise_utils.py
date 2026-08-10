"""
=============================================================
FASE 0 - UTIL: EXERCISE TEST RUNNER (PASS/FAIL)
=============================================================
Helper untuk mengecek jawaban exercise secara otomatis.
Memberikan feedback langsung: ✅ PASS / ❌ FAIL per test case,
sehingga kamu tahu persis bagian mana yang belum benar.

Cara pakai di modul (contoh: 01_numpy_essentials.py):

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "00-setup"))
    from exercise_utils import safe_check, summary, run_exercise_mode

    def run_exercise_tests():
        safe_check("batch_normalize: mean ~ 0",
                   lambda: np.allclose(batch_normalize(X).mean(axis=0), 0))
        summary()

    if run_exercise_mode():
        run_exercise_tests()

Kemudian jalankan dari folder modul:
    python 01_numpy_essentials.py --exercise

Filosofi:
- Fungsi yang belum diimplementasi men-raise NotImplementedError.
- safe_check menangkap error tersebut dan melaporkannya sebagai FAIL
  (bukan crash), jadi kamu bisa melihat semua test yang belum lulus.
=============================================================
"""

import sys

_passed = 0
_failed = 0


def safe_check(name, fn, detail=""):
    """
    Jalankan satu test case dan cetak PASS/FAIL.

    Parameters
    ----------
    name : str
        Deskripsi test (muncul di output).
    fn : callable -> bool
        Fungsi yang mengembalikan True jika test lulus.
        Exception apa pun (termasuk NotImplementedError) dianggap FAIL.
    detail : str
        Informasi tambahan saat FAIL (opsional).
    """
    global _passed, _failed
    try:
        ok = bool(fn())
    except NotImplementedError:
        ok = False
        detail = detail or "— fungsi belum diimplementasi (raise NotImplementedError)"
    except Exception as e:
        ok = False
        detail = f"— error: {type(e).__name__}: {e}"

    if ok:
        _passed += 1
        print(f"  ✅ PASS: {name}")
    else:
        _failed += 1
        print(f"  ❌ FAIL: {name} {detail}")


def summary():
    """
    Cetak ringkasan hasil semua test.
    Return True jika semua lulus, False jika ada yang gagal.
    """
    total = _passed + _failed
    print("=" * 50)
    if total == 0:
        print("📊 Tidak ada test yang dijalankan.")
    else:
        print(f"📊 Hasil exercise: {_passed}/{total} lulus")
        if _failed == 0:
            print("🎉 Semua test lulus! Lanjut ke materi berikutnya.")
        else:
            print(f"⚠️  Masih ada {_failed} test gagal. "
                  "Perbaiki kode lalu jalankan ulang dengan --exercise.")
    print("=" * 50)
    return _failed == 0


def run_exercise_mode():
    """
    Deteksi mode exercise: python <modul>.py --exercise
    Return True jika flag --exercise ada di argumen baris perintah.
    """
    return "--exercise" in sys.argv
