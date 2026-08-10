# 📚 Solusi Referensi (Safety Net)

> **Aturan emas:** Coba dulu sendiri minimal 30-60 menit SEBELUM membuka folder ini.
> Filosofi kurikulum adalah _deliberate practice_ — belajar paling efektif saat
> kamu "berjuang" dengan masalahnya dulu. Folder ini hanya safety net saat buntu.

## Cara Pakai

1. Kerjakan exercise di modul sampai buntu (atau selesai).
2. Untuk exercise yang bisa diverifikasi otomatis, jalankan test runner-nya:
   ```bash
   cd 01-fondasi-data
   python 01_numpy_essentials.py --exercise
   ```
3. Kalau benar-benar buntu, buka solusi referensi di folder ini.
4. Setelah selesai, **tulis ulang solusinya dari ingatan tanpa melihat file ini**
   (spaced repetition — ini yang membuat materi melekat).

## Daftar Solusi

| File Solusi                         | Modul Terkait                                    | Keterangan                                                           |
| ----------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------- |
| `01_numpy_solutions.py`             | `01-fondasi-data/01_numpy_essentials.py`         | EXERCISE 1: `batch_normalize`, `cosine_similarity`, `one_hot_encode` |
| `02_linear_regression_solutions.py` | `02-ml-dari-nol/01_linear_regression_scratch.py` | Variants GD: Full-batch, SGD, Mini-batch, Lasso                      |

## Catatan

- **🔥 CHALLENGE bersifat open-ended** (mini-project), jadi TIDAK ada satu
  jawaban "benar". Solusi referensi hanya salah satu cara yang valid.
- Jika kamu menemukan bug di solusi, jangan ragu untuk memperbaikinya —
  itu juga latihan yang bagus!
- Konvensi penamaan: `{nomor_modul}_{nama}_solutions.py`.
