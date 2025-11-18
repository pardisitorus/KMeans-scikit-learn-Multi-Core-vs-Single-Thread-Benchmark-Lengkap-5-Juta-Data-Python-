# KMeans-scikit-learn-Multi-Core-vs-Single-Thread-Benchmark-Lengkap-5-Juta-Data-Python-
Benchmark KMeans scikit-learn multi-core vs single thread pada 5 juta sampel (10 fitur, 10 cluster). Menunjukkan cara mengaktifkan semua core CPU lewat OMP/MKL_NUM_THREADS + joblib, lengkap dengan perbandingan waktu eksekusi dan kode 100% reproducible.



## 1. Pertanyaan / Permasalahan

Pada proyek kecil ini saya ingin menjawab pertanyaan:

> **“Seberapa besar percepatan yang diperoleh ketika training K-Means dijalankan secara paralel (semua core CPU) dibandingkan secara serial (1 thread) pada dataset sangat besar di scikit-learn versi terbaru, di mana parameter `n_jobs` sudah tidak tersedia lagi?”**

Untuk menjawabnya, saya:
- Membuat **dataset sintetis** berisi **5.000.000 baris** dan **10 fitur**.
- Melatih **K-Means** dua kali:
  - Mode **serial** → memaksa semua pustaka numerik hanya memakai **1 thread**.
  - Mode **paralel** → mengizinkan pustaka numerik memakai **semua logical CPU (24 thread)**.
- Mengukur **waktu training**, **inertia**, dan **silhouette score**, lalu menghitung **speedup**.

````python

---

## 2. Langkah-Langkah Kode yang Lengkap

File utama: `kmeans_serial_vs_parallel.py`

### 2.1. Import Library dan Informasi CPU

```python
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1. Informasi CPU
print("Jumlah logical CPU:", os.cpu_count())
````

**Tujuan:**

* Mengimpor semua library yang dibutuhkan.
* Mengecek jumlah logical CPU untuk mengetahui berapa core yang bisa dipakai saat mode paralel.

---

### 2.2. Membuat Dataset Besar (5 Juta Sampel)

```python
# 2. Dataset BESAR & BERAT agar paralel terlihat jelas
X, y_true = make_blobs(
    n_samples=5_000_000,  # 5 juta baris
    n_features=10,        # 10 fitur numerik
    centers=10,           # 10 klaster sebenarnya
    cluster_std=1.5,
    random_state=42
)
print("Shape dataset:", X.shape)
```

**Tujuan:**

* Menghasilkan dataset sintetis yang besar:

  * Cukup berat untuk menguji manfaat paralelisme.
  * Tidak perlu file eksternal.

---

### 2.3. Fungsi Training K-Means

```python
# 3. Fungsi training (tanpa n_jobs → sudah kompatibel sklearn terbaru)
def train_kmeans(X, n_clusters=10, random_state=42):
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=20,            # naik jadi 20 inisialisasi (bagian paling berat)
        max_iter=300,
        random_state=random_state,
        algorithm='lloyd'
    )
    
    start = time.time()
    kmeans.fit(X)
    end = time.time()
    duration = end - start
    
    inertia = kmeans.inertia_
    
    # Silhouette hanya 10.000 sampel agar cepat
    sample_size = 10_000
    idx = np.random.choice(X.shape[0], sample_size, replace=False)
    sil_score = silhouette_score(X[idx], kmeans.labels_[idx])
    
    return kmeans, duration, inertia, sil_score
```

**Tujuan:**

* Membungkus proses training K-Means dalam satu fungsi.
* Mengembalikan:

  * **Model terlatih**
  * **Waktu eksekusi** (`duration`)
  * **Inertia** (total sum of squared distances ke centroid)
  * **Silhouette score** (kualitas pemisahan klaster) dari sampel kecil.

---

### 2.4. Mode SERIAL – Memaksa 1 Thread

```python
# =============================================================================
# MODE SERIAL (paksa 1 thread saja)
# =============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Tambahan kontrol joblib (kadang masih dipakai sklearn)
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=1):

    print("\n=== TRAINING SERIAL (1 thread) ===")
    model_serial, time_serial, inertia_serial, sil_serial = train_kmeans(X)
    
    print(f"Waktu training   : {time_serial:.2f} detik")
    print(f"Inertia          : {inertia_serial:.1f}")
    print(f"Silhouette score : {sil_serial:.4f}")
```

**Tujuan:**

* Mengatur beberapa environment variable (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, dll.) menjadi **1** untuk memaksa pustaka numerik hanya memakai satu thread.
* Menggunakan `parallel_backend('threading', n_jobs=1)` agar backend joblib (jika dipakai internal scikit-learn) juga mengikuti konfigurasi serial.
* Menjalankan fungsi `train_kmeans` dan menyimpan hasilnya sebagai baseline **mode serial**.

---

### 2.5. Mode PARALEL – Menggunakan Semua Logical Core

```python
# =============================================================================
# MODE PARALEL (gunakan semua core)
# =============================================================================
n_cores = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)

with parallel_backend('threading', n_jobs=n_cores):

    print("\n=== TRAINING PARALEL (semua core) ===")
    model_parallel, time_parallel, inertia_parallel, sil_parallel = train_kmeans(X)
    
    print(f"Waktu training   : {time_parallel:.2f} detik")
    print(f"Inertia          : {inertia_parallel:.1f}")
    print(f"Silhouette score : {sil_parallel:.4f}")
```

**Tujuan:**

* Mengatur environment variable agar pustaka numerik boleh memakai **semua logical CPU**.
* Menggunakan `parallel_backend('threading', n_jobs=n_cores)` untuk mengizinkan backend joblib memakai semua core.
* Menjalankan `train_kmeans` untuk mode **paralel** dan mengambil metrik yang sama.

---

### 2.6. Menghitung Speedup Serial vs Paralel

```python
# =============================================================================
# Hasil Perbandingan
# =============================================================================
speedup = time_serial / time_parallel
print(f"\n>>> SPEEDUP : {speedup:.2f}x lebih cepat dengan paralel!")
print(f"    (Serial {time_serial:.2f}s → Paralel {time_parallel:.2f}s)")
```

**Tujuan:**

* Menghitung **rasio percepatan** (`speedup`) = waktu serial / waktu paralel.
* Menampilkan ringkasan hasil dalam satu baris yang mudah dibaca.

---

### 2.7. Cara Menjalankan

```bash
# 1. Buat environment (opsional, tapi disarankan)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependensi
pip install numpy matplotlib scikit-learn joblib

# 3. Jalankan skrip
python kmeans_serial_vs_parallel.py
```

> **Catatan:** Script ini butuh RAM yang cukup besar (dataset 5 juta baris x 10 fitur) dan waktu komputasi yang lumayan lama, tergantung spesifikasi mesin.

---

## 3. Analisis Hasil

### 3.1. Ringkasan Hasil Eksekusi

Dari salah satu eksekusi, log yang diperoleh (diringkas):

```text
Jumlah logical CPU: 24
Shape dataset: (5000000, 10)

=== TRAINING SERIAL (1 thread) ===
Waktu training   : 38.99 detik
Inertia          : 112519501.2
Silhouette score : 0.6572

=== TRAINING PARALEL (semua core) ===
Waktu training   : 37.87 detik
Inertia          : 112519501.2
Silhouette score : 0.6546

>>> SPEEDUP : 1.03x lebih cepat dengan paralel!
    (Serial 38.99s → Paralel 37.87s)
```

### 3.2. Interpretasi

1. **Kecepatan**

   * Mode paralel hanya memberikan **speedup sekitar 1.03×**.
   * Artinya, meskipun memakai 24 logical CPU, waktu training hanya sedikit lebih cepat (sekitar 1 detik lebih cepat) dibanding mode 1 thread.

2. **Kualitas Klaster**

   * **Inertia** pada serial dan paralel **identik** (`112519501.2`), artinya solusi klaster yang ditemukan sama.
   * **Silhouette score** sangat mirip (`0.6572` vs `0.6546`), perbedaan kecil dan bisa dianggap sama secara praktis.
   * Jadi, perbedaan mode eksekusi tidak mengubah kualitas model, hanya memengaruhi waktu komputasi.

3. **Mengapa Speedup Kecil?**
   Beberapa kemungkinan penyebab:

   * Implementasi K-Means di scikit-learn sudah **multi-threaded** di level pustaka numerik (BLAS/MKL). Jadi, meskipun kita memaksa serial vs paralel via environment, sebagian optimasi internal tetap berjalan sehingga perubahan konfigurasi tidak terlalu ekstrem.
   * Algoritma K-Means pada data ini mungkin sudah **mendekati bottleneck memori**:

     * Banyak operasi adalah baca/tulis memori besar (5 juta x 10), bukan hanya operasi aritmatika.
     * Ketika operasi dibagi ke banyak core, **bandwidth memori** menjadi batas utama, sehingga penambahan core tidak otomatis membuat waktu turun drastis.
   * Ada **overhead paralel**:

     * Sinkronisasi antar thread.
     * Manajemen task di joblib dan pustaka BLAS.
     * Overhead ini akan “memakan” sebagian keuntungan dari pemakaian banyak core.

4. **Pelajaran Penting**

   * Paralelisme tidak selalu memberikan speedup besar, terutama ketika:

     * Algoritma sudah dioptimasi.
     * Beban kerja per iterasi tidak terlalu berat dibanding overhead.
     * Bottleneck ada di memori, bukan CPU murni.

---

## 4. Kesimpulan Akhir

1. **Secara teknis**, komputasi paralel berhasil diaktifkan dengan mengatur environment variable dan backend joblib untuk memaksa jumlah thread yang digunakan saat training K-Means.
2. **Secara empiris**, pada dataset berukuran **5 juta sampel** dan **10 fitur**, mode paralel (semua core) hanya sedikit lebih cepat daripada mode serial (1 thread) dengan **speedup sekitar 1.03×**.
3. **Kualitas hasil klaster** (inertia dan silhouette score) hampir identik antara mode serial dan paralel, sehingga perbedaan utama hanya pada waktu eksekusi.
4. Eksperimen ini menunjukkan bahwa:

   * Menambah jumlah core **tidak selalu** menghasilkan percepatan besar,
     terutama jika algoritma sudah dioptimasi dan bottleneck berada di memori.
   * Untuk memahami performa model Machine Learning dalam praktik,
     kita perlu melihat **kombinasi algoritma, ukuran data, implementasi library,
     dan arsitektur hardware**, bukan hanya jumlah core CPU.

Proyek ini bisa dijadikan dasar untuk eksperimen lanjutan, misalnya:

* Mengubah ukuran dataset (lebih kecil / lebih besar).
* Mengubah jumlah klaster atau parameter `n_init`.
* Membandingkan dengan algoritma lain yang benar-benar sangat paralel (misalnya Random Forest dengan `n_estimators` besar).

```

