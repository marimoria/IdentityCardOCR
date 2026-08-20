# Identity Card OCR - Final Project Data Science Academy (DSA) COMPFEST 18

Pipeline Optical Character Recognition (OCR) untuk ekstraksi informasi terstruktur dari dokumen identitas (KTP) guna mendukung proses verifikasi Know Your Customer (KYC), dikembangkan sebagai Final Project Data Science Academy COMPFEST 18.

Repository ini berisi implementasi pipeline OCR yang dirancang untuk mengekstraksi informasi kunci dari dokumen identitas (KTP), meliputi nama, tanggal lahir, dan alamat. Proyek ini disusun sebagai bagian dari proses Know Your Customer (KYC) yang menjadi kebutuhan wajib bagi lembaga keuangan dan perbankan dalam rangka pemenuhan regulasi anti pencucian uang (Anti-Money Laundering/AML).

Pipeline dibangun untuk mengatasi tantangan variasi tinggi pada dokumen identitas, termasuk perbedaan kualitas gambar, kondisi pencahayaan, serta kondisi fisik dokumen yang tidak ideal (lusuh atau terlipat). Prosesnya mencakup tahapan business understanding, data understanding dan exploratory data analysis (EDA), data preprocessing, pemodelan dan optimisasi, evaluasi performa, hingga penyusunan rekomendasi bisnis berbasis hasil ekstraksi.

Sesuai ketentuan kompetisi, pipeline ini dibangun tanpa komponen berbasis Large Language Model (LLM) maupun Vision-Language Model (VLM), dan tanpa proses anotasi gambar secara manual. Evaluasi performa dilakukan dengan mengacu pada data ground truth yang sudah disediakan panitia.

> **Catatan tentang revisi README ini:** Versi ini disusun ulang dari pembacaan penuh source code di `1_EDA_and_Data_Understanding.ipynb`, `2_Feature_Engineering_Lib.ipynb`, dan `3_Modelling_and_Evaluation.ipynb`, baris per baris, bukan cuma dari ringkasan naratifnya. Beberapa fitur, ambang batas, dan mekanisme yang sebelumnya tidak disebutkan sudah ditambahkan di bawah, dan beberapa cuplikan kode yang sebelumnya disederhanakan sudah diganti dengan versi yang persis sama dengan isi notebook. **Notebook 4 (`4_Model_Comparison.ipynb` / `plan/4_model_comparison.py`) tidak ikut diunggah untuk pengecekan ini**, jadi bagian Notebook 4 di bawah masih apa adanya dari draft sebelumnya dan belum diverifikasi baris-per-baris seperti tiga notebook lainnya.

## Data Access

Dataset tidak dibagikan secara publik. Kolaborator bisa mengakses dataset lewat link drive yang dibagikan pihak COMPFEST, lalu unzip pada folder `dataset/` seperti berikut.

```
dataset/
├── ground_truth.csv
└── images/
    ├── image_001.jpg
    ├── image_002.jpg
    ├── ...
    └── image_732.jpg
plan/
├── 1_eda_and_data_understanding.py
├── 2_feature_engineering_lib.py
├── 3_modelling_and_evaluation.py
└── 4_model_comparison.py
```

Versi `.py` di folder `plan/` isinya sama persis dengan versi `.ipynb` di root, cuma dipisah dari format notebook supaya lebih gampang dibaca lewat editor kode biasa atau dijalankan lewat terminal.

Total data ground truth yang benar-benar dipakai di seluruh pipeline adalah **632 baris/gambar** (bukan 732 seperti nama file gambar terakhir di atas — ada gap penomoran file di dataset mentah).

### Berkas yang Dihasilkan Sepanjang Pipeline

Tidak semua berkas ini ada dari awal — sebagian besar adalah *artifact* yang dibuat dan di-cache oleh Notebook 1-3 supaya proses berat (OCR, DocAligner) tidak perlu diulang tiap kali notebook dijalankan ulang.

| Berkas / Folder | Dibuat oleh | Isi |
|---|---|---|
| `dataset/ground_truth_normalized.csv` | NB1 | CSV ground truth setelah normalisasi quoting, koreksi DOB, dan penambahan kolom `negara` |
| `dataset/image_quality_stats.csv` | NB1 | Metrik kualitas per gambar **original** (belum di-crop) |
| `dataset/fold_indices.pkl` | NB1 | Daftar index train/test untuk 5 fold StratifiedGroupKFold, dipakai ulang persis sama di NB3 |
| `dataset/baseline_ocr_cache.json` | NB1 | Cache hasil baseline OCR pada 5 gambar sampel (tanpa cropping) |
| `dataset/cropped_images_lib/` | NB2 | Gambar KTP hasil localize+warp dari DocAligner |
| `dataset/bbox_visualizations_lib/` | NB2 | Folder dibuat (`os.makedirs`) tapi **tidak pernah ditulis** di versi notebook ini — dead code, bukan bug fatal, cuma folder kosong yang tidak terpakai |
| `dataset/raw_ocr_v2_unfiltered.pkl` | NB2 | Cache hasil OCR mentah (`texts`, `scores`, `boxes`) per gambar hasil crop, pakai *cache invalidation* berbasis `mtime` file gambar |
| `dataset/text_origin_features.csv` | NB2 | Fitur kata kunci asal dokumen per gambar (`has_mykad_keyword`, dst) |
| `dataset/ocr_segments_features.csv` | NB2 | Satu baris per **segmen teks** OCR, dengan fitur spasial+tekstual+label lemah |
| `dataset/master_features.csv` | NB2 | Satu baris per **gambar**, gabungan seluruh fitur level dokumen |
| `dataset/shap_feature_importance.csv` | NB3 | Mean |SHAP value| per fitur per kelas label (dari fold terakhir) |
| `dataset/shap_{class}_bar.png`, `shap_{class}_beeswarm.png` | NB3 | Grafik SHAP per kelas (`name`, `birth_date`, `address`) |
| `dataset/crosscheck/Accepted/`, `Rejected/`, `Manual_Review/` | NB3 | Gambar KTP disalin ke folder sesuai keputusan routing akhir (hanya ditulis di fold terakhir agar tidak 5x duplikasi) |
| `dataset/crosscheck/ocr_predictions_crosscheck.csv` | NB3 | CSV hasil prediksi vs ground truth berdampingan untuk **seluruh 632 gambar** (out-of-fold, terkumpul dari 5 fold) |
| `dataset/semantic_visualizations/semantic_{filename}` | NB3 | Gambar KTP dengan kotak warna hasil klasifikasi field digambar ulang |
| `dataset/evaluation_results.csv` | NB3 | Rekam metrik evaluasi akhir per gambar dari seluruh fold |

## Alur Kerja Pipeline (Notebook 1 sampai 4)

```text
[Notebook 1: Pemahaman Data]
      │
      ├──> Normalisasi CSV ground truth (perbaiki quoting rusak)
      ├──> Koreksi nilai ground truth yang salah (DOB image_541-552)
      ├──> Penambahan kolom referensi 'negara' (21 negara, mapping manual per urutan file)
      ├──> Label lemah asal dokumen (Malaysia vs Luar Negeri) dari keberadaan alamat
      ├──> Analisis duplikasi identitas (satu orang bisa punya banyak foto)
      ├──> Hitung metrik kualitas gambar mentah (blur, brightness, contrast, edge density,
      │    dark pixel ratio, file size, kategori kualitas)
      ├──> Split data StratifiedGroupKFold (5 fold, dikunci ke file .pkl)
      ├──> Analisis pola teks (bin/binti, kata kunci MYKAD, karakter non-ASCII)
      └──> Baseline OCR tanpa cropping pada 5 sampel representatif
      │
      ▼
[Notebook 2: Lokalisasi & Rekayasa Fitur]
      │
      ├──> Potong KTP dari foto (DocAligner + perspective warp, dengan fallback)
      ├──> Hitung ulang kualitas gambar hasil potong (6 metrik) + skew angle
      ├──> Kompresi 4 metrik kualitas jadi 2 komponen PCA (quality_pc1, quality_pc2)
      ├──> Baca teks di atas kartu (PaddleOCR, model bahasa Portugis, dengan caching)
      ├──> Bangun fitur asal dokumen (has_mykad_keyword, has_ic_number_pattern,
      │    has_bin_binti_any_segment, n_segments, avg_segment_len)
      ├──> Beri label lemah ke tiap segmen teks (name/birth_date/address/other)
      │    lewat token overlap + pencocokan pola NRIC/MyTentera/tanggal terhadap ground truth
      ├──> Kalibrasi ambang batas triage berbasis pseudo-CER (khusus EDA, notebook 2)
      └──> Susun fitur spasial + tekstual per segmen, normalisasi, simpan ke CSV
      │
      ▼
[Notebook 3: Pemodelan & Evaluasi Akhir]
      │
      ├──> Latih Origin Classifier (TF-IDF + Logistic Regression) per fold
      ├──> Kalibrasi & latih Triage Classifier (Random Forest) per fold, ambang batas
      │    dihitung ulang dari data train fold itu sendiri (recall token >= 0.8)
      ├──> Latih Field Classifier (LightGBM + Optuna 5 trial, klasifikasi segmen teks)
      ├──> Deteksi format dokumen (Template Router: MRZ, EU License, MyKad, ~170 negara fallback)
      ├──> Ekstraksi field per template + penggabungan teks jadi nama/DOB/alamat
      ├──> Fallback scanner tanggal lahir kalau field classifier gagal menandai DOB
      ├──> Keputusan routing (AUTO_APPROVE / HUMAN_REVIEW / REJECT)
      ├──> Evaluasi 5-fold out-of-fold (CER, WER, F1 Token, Exact Accuracy, STP Rate,
      │    akurasi deteksi negara, pengecekan standar akurasi minimum)
      ├──> Analisis feature importance dengan SHAP (bar + beeswarm per kelas)
      ├──> Visualisasi semantik hasil ekstraksi (bounding box berwarna per field)
      └──> Ekspor folder crosscheck (Accepted/Rejected/Manual_Review) + CSV prediksi vs GT
      │
      ▼
[Notebook 4: Eksperimen Perbandingan] *(belum diverifikasi ulang pada revisi ini)*
      │
      ├──> Lokalisasi: DocAligner vs Heuristik CV Klasik (Canny + Contour)
      ├──> OCR Engine: PaddleOCR vs EasyOCR vs Tesseract
      ├──> Model Klasifikasi Field: LightGBM vs RandomForest vs XGBoost
      ├──> Paradigma Representasi: Spasial+LightGBM vs TF-IDF+LogReg vs Regex
      ├──> Dampak Triage, Ablation Study Fitur, dan Estimasi Latensi End-to-End
      └──> Ringkasan akhir seluruh perbandingan
```

---

## Ringkasan Komposisi Dataset

Kolom `negara` di `ground_truth_normalized.csv` **bukan fitur pelatihan** — ini label referensi manual yang cuma dipakai untuk validasi (EDA di Notebook 1, dan pengecekan akurasi deteksi negara di Notebook 3). Nilainya dipetakan lewat daftar hardcoded, diurutkan berdasarkan nomor urut nama file (`image_001.jpg` ... `image_632.jpg`):

| Negara | Jumlah Gambar | % dari Total |
|---|---:|---:|
| Malaysia | 112 | 17.7% |
| Germany | 60 | 9.5% |
| Czech Republic | 40 | 6.3% |
| Finland | 40 | 6.3% |
| Croatia | 40 | 6.3% |
| Ukraine | 40 | 6.3% |
| Brazil | 20 | 3.2% |
| Chile | 20 | 3.2% |
| China | 20 | 3.2% |
| Algeria | 20 | 3.2% |
| Spain | 20 | 3.2% |
| Greece | 20 | 3.2% |
| Hungary | 20 | 3.2% |
| Italy | 20 | 3.2% |
| Latvia | 20 | 3.2% |
| Macau | 20 | 3.2% |
| Moldova | 20 | 3.2% |
| Norway | 20 | 3.2% |
| Poland | 20 | 3.2% |
| Portugal | 20 | 3.2% |
| Turkey | 20 | 3.2% |
| **Total** | **632** | **100%** |

520 dari 632 gambar (82.3%) adalah dokumen luar negeri — angka inilah yang jadi alasan `PaddleOCR` di Notebook 2 dikonfigurasi dengan `lang='pt'` (Portugis) supaya charset Latin-extended (é, ã, ç, ñ) terbaca penuh tanpa mengorbankan bacaan teks Melayu.

```python
_NEGARA_LIST = (
    ['MALAYSIA'] * 112 + ['BRAZIL'] * 20 + ['CHILE'] * 20 + ['CHINA'] * 20 +
    ['CZECH REPUBLIC'] * 40 + ['GERMANY'] * 60 + ['ALGERIA'] * 20 + ['SPAIN'] * 20 +
    ['FINLAND'] * 40 + ['GREECE'] * 20 + ['CROATIA'] * 40 + ['HUNGARY'] * 20 +
    ['ITALY'] * 20 + ['LATVIA'] * 20 + ['MACAU'] * 20 + ['MOLDOVA'] * 20 +
    ['NORWAY'] * 20 + ['POLAND'] * 20 + ['PORTUGAL'] * 20 + ['TURKEY'] * 20 +
    ['UKRAINE'] * 40
)
_sorted_files = sorted(df['filename'].tolist(), key=lambda x: int(x.split('_')[1].split('.')[0]))
df['negara'] = df['filename'].map(dict(zip(_sorted_files, _NEGARA_LIST)))
```

---

## Penjelasan Detail Tiap Tahap

### Notebook 1: Pemahaman Data Mentah

Tahap ini penting untuk memahami isi data sebelum diproses lebih jauh, sekaligus memastikan pembagian data nanti benar-benar ketat supaya tidak ada foto orang yang sama bocor antara data latih dan data uji.

Semua metrik kualitas gambar yang dihitung di Notebook 1 dilakukan pada **gambar original**, sebelum proses cropping. Metrik yang sama dihitung ulang di Notebook 2 setelah kartu dipotong, dan versi hasil crop itulah yang benar-benar dipakai sebagai fitur model — Notebook 1 murni untuk eksplorasi awal.

**1.1 Normalisasi CSV Ground Truth**

CSV mentah dari panitia punya format tanda kutip yang berantakan, jadi tidak bisa langsung dibaca `pandas.read_csv` dengan aman. Kita membersihkannya dulu baris per baris sebelum diparse ulang pakai modul `csv` bawaan Python.

```python
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    raw_lines = f.readlines()

normalized_lines = []
for line in raw_lines:
    line = line.strip()
    if not line:
        continue
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]
        line = line.replace('""', '"')
    normalized_lines.append(line)

reader = csv.reader(io.StringIO('\n'.join(normalized_lines)))
rows = list(reader)
rows = [r + [''] * (4 - len(r)) if len(r) < 4 else r[:4] for r in rows]
```

Hasil normalisasi ini disimpan sebagai `ground_truth_normalized.csv` supaya proses berat ini cuma perlu dijalankan sekali (dicek dulu dengan `os.path.exists` sebelum dijalankan ulang).

**1.2 Koreksi Data yang Salah**

Selain masalah format, ada juga kesalahan isi data. Baris `image_541.jpg` sampai `image_552.jpg` punya tanggal lahir yang salah ketik (tanggalnya naik satu-satu secara acak padahal harusnya sama semua untuk orang yang sama — `image_533` sampai `image_540` sudah benar di `1971-05-12`). Kesalahan ini diperbaiki manual dan ditulis ulang ke CSV supaya koreksinya permanen dan tidak perlu diulang tiap kali notebook dijalankan.

```python
_bad_dob = [f'image_{i}.jpg' for i in range(541, 553)]
_fixed   = df['filename'].isin(_bad_dob)
if _fixed.any():
    df.loc[_fixed, 'birth_date'] = '1971-05-12'
    df.to_csv(NORMALIZED_CSV, index=False)
```

**1.3 Penambahan Kolom Referensi `negara`**

Setelah koreksi DOB, ditambahkan satu kolom lagi ke `ground_truth_normalized.csv`: `negara`, dipetakan dari daftar hardcoded 21 negara (lihat tabel komposisi dataset di atas), diurutkan berdasarkan angka di nama file. Kolom ini ditandai eksplisit di komentar kode sebagai **bukan fitur pelatihan** — cuma referensi validasi manual untuk EDA dan untuk mengecek akurasi Template Router di Notebook 3. Saat inference, asal negara ditentukan murni dari kata kunci hasil OCR, bukan dari kolom ini.

**1.4 Label Lemah Asal Dokumen**

Dataset ini berisi campuran KTP Malaysia dan dokumen identitas negara lain, dan keduanya butuh penanganan berbeda karena tata letak dan field yang tersedia jauh berbeda. Masalahnya, tidak ada kolom "negara" yang bisa dipercaya penuh untuk dijadikan target pelatihan (kolom `negara` yang ada sifatnya cuma referensi validasi manual, bukan untuk fitur pelatihan, seperti dijelaskan di atas). Jadi kita pakai pendekatan **weak labeling**, yaitu memakai sinyal tidak langsung yang cukup kuat sebagai pengganti label asli.

Sinyalnya adalah keberadaan kolom alamat. KTP Malaysia (MyKad) selalu mencantumkan alamat lengkap, sementara sebagian besar dokumen identitas negara lain di dataset ini tidak menyertakan alamat pada ground truth. Jadi:

```python
df['has_address'] = df['address'].notna()
df['doc_origin_weak_label'] = df['has_address'].map({True: 'Malaysia', False: 'Luar Negeri'})
df['identity_key'] = df['name'].astype(str) + '__' + df['birth_date'].astype(str)
```

`identity_key` di sini dipakai supaya kita bisa mengelompokkan semua foto yang milik orang yang sama, dipakai lagi nanti di proses split data. Kolom ini eksplisit ditandai hanya untuk keperluan EDA dan split data, dan tidak boleh dibocorkan ke model saat prediksi.

**1.5 Analisis Duplikasi Identitas**

Satu orang di dataset ini bisa punya puluhan foto berbeda (sudut pengambilan, pencahayaan, kondisi kartu berbeda-beda) — sampai puluhan foto per identitas, khususnya di kelompok Luar Negeri yang jumlah identitas uniknya jauh lebih sedikit dibanding jumlah fotonya. Kalau ini tidak diperhatikan, ada risiko besar data leakage, yaitu foto orang yang sama muncul baik di data latih maupun data uji, sehingga skor evaluasi jadi terlalu optimis dan tidak mencerminkan performa di dunia nyata. Analisis ini membuktikan kenapa strategi split data nanti wajib berbasis kelompok (group), bukan berbasis baris biasa.

**1.6 Analisis Format Tanggal Lahir**

Format tanggal lahir diklasifikasikan ke beberapa kategori (`YYYY-MM-DD`, `YYYY` saja, `DD.MM.YYYY`, kosong, atau format lain) untuk memastikan regex ekstraksi di tahap berikutnya bisa menangani seluruh variasi yang benar-benar muncul di ground truth.

**1.7 Distribusi Panjang Teks**

Panjang nama dan alamat dianalisis untuk merancang batasan saringan awal segmen teks. Nama pada dokumen Malaysia cenderung lebih panjang karena konvensi bin/binti, sementara alamat Malaysia bisa memuat puluhan karakter.

**1.8 Analisis Kualitas Gambar Original**

Setiap gambar original (belum di-crop) dihitung metriknya dan disimpan ke `image_quality_stats.csv`. Metrik yang dihitung di sini **lebih lengkap** dari yang disebut sebelumnya — bukan cuma 4 metrik, tapi 10 kolom:

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = img.shape[:2]
edges = cv2.Canny(gray, 50, 150)
_, thr = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

img_stats.append({
    'filename': fname,
    'width': w,
    'height': h,
    'aspect_ratio': w / h,
    'is_portrait': w < h,
    'blur_score': cv2.Laplacian(gray, cv2.CV_64F).var(),
    'brightness': float(gray.mean()),
    'contrast': float(gray.std()),
    'edge_density': float(edges.mean()),
    'dark_pixel_ratio': float((thr == 0).sum() / (h * w)),
    'file_kb': os.path.getsize(path) / 1024,
})
```

Selain itu ditambahkan kolom kategorikal `quality_cat` lewat binning `blur_score`:

```python
stats_df['quality_cat'] = pd.cut(
    stats_df['blur_score'],
    bins=[0, 50, 300, np.inf],
    labels=['Buram', 'Sedang', 'Tajam']
)
```

Penjelasan matematis tiap metrik:

- **Blur score (Laplacian variance).** Laplacian adalah turunan kedua dari intensitas piksel, jadi dia sensitif terhadap perubahan tajam (tepi objek). Gambar yang tajam punya banyak tepi dengan kontras tinggi, sehingga hasil Laplacian-nya bervariasi besar. Gambar yang buram tepinya halus dan hampir seragam, jadi variansnya kecil. Karena itu varians dari hasil Laplacian dipakai sebagai proxy ketajaman, makin tinggi nilainya makin tajam gambarnya. Rentang nilainya di dataset ini sangat ekstrem, dari sekitar 4 sampai 2704.
- **Brightness.** Cuma rata-rata nilai piksel grayscale (skala 0-255). Kalau nilainya rendah berarti gambar gelap.
- **Contrast.** Standar deviasi dari nilai piksel grayscale. Kalau nilainya rendah, piksel-pikselnya seragam semua (gambar flat, kurang detail).
- **Edge density.** Rata-rata hasil deteksi tepi Canny. Canny sendiri bekerja dengan mencari gradien intensitas yang melewati dua ambang batas (di sini 50 dan 150), lalu menyisakan tepi yang benar-benar kuat. Rata-rata dari peta tepi ini menunjukkan seberapa "ramai" tekstur di gambar, berguna untuk membedakan area kartu yang penuh tulisan dari latar belakang polos.
- **Dark pixel ratio.** Proporsi piksel yang jatuh di sisi gelap setelah thresholding biner sederhana (ambang 128). Dipakai belakangan juga sebagai salah satu fitur triage dan salah satu aturan keras penolakan gambar di Notebook 3 (`crop_dark_pixel_ratio > 0.80`).
- **File size (KB)** dan **aspect ratio** juga direkam, dipakai untuk analisis distribusi dan korelasi tapi tidak dibawa langsung sebagai fitur model.
- **Quality category.** Binning `blur_score` ke tiga kelas (`Buram` di bawah 50, `Sedang` 50-300, `Tajam` di atas 300) dipakai untuk visualisasi sampel dan analisis distribusi per asal dokumen, bukan sebagai fitur model langsung.

Rata-rata blur score dokumen luar negeri jauh lebih rendah dibanding Malaysia (sekitar 63 berbanding 757), diduga karena perbedaan sumber kamera pengambilan gambar.

**1.9 Distribusi dan Korelasi Metrik Kualitas**

Sebaran tiap metrik diplot (histogram dengan garis median), lalu dihitung matriks korelasi antar `blur_score`, `brightness`, `contrast`, `edge_density`, `dark_pixel_ratio`, `file_kb`, dan `aspect_ratio`. Korelasi tinggi antar sebagian metrik inilah yang jadi alasan dilakukannya kompresi PCA di Notebook 2 (lihat 2.3 di bawah).

**1.10 Sampel Visual per Kategori Kualitas**

Untuk tiap kategori (`Buram`, `Sedang`, `Tajam`), diambil 3 gambar representatif beserta histogram intensitas grayscale-nya, untuk memvalidasi secara visual bahwa hasil hitungan numerik memang sesuai dengan penilaian mata manusia.

**1.11 Orientasi Gambar**

Sebagian besar foto diambil dalam posisi portrait, yang berarti banyak area kosong di sekitar kartu (meja, latar belakang) ikut terekam kamera. Hal inilah yang jadi salah satu alasan utama kenapa tahap cropping presisi di Notebook 2 sangat diperlukan.

**1.12 Split Data dengan StratifiedGroupKFold**

Ini bagian paling krusial di Notebook 1. Kita butuh skema split yang memenuhi dua syarat sekaligus:

1. Foto milik orang (identitas) yang sama tidak boleh terpisah antara fold latih dan fold uji (mencegah data leakage yang sudah dibahas di poin 1.5).
2. Proporsi Malaysia vs Luar Negeri harus tetap seimbang di tiap fold, supaya tiap fold jadi representasi yang adil dari keseluruhan data.

`StratifiedGroupKFold` dari scikit-learn menyelesaikan dua syarat ini sekaligus, dengan `groups` menentukan unit yang tidak boleh dipecah dan `y` menentukan target yang proporsinya mau dijaga tetap seimbang antar fold.

```python
groups = df['identity_key'].values
y_stratify = df['doc_origin_weak_label'].values
skf = StratifiedGroupKFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(skf.split(df, y=y_stratify, groups=groups)):
    df_test = df.iloc[test_idx]
    df_train = df.iloc[train_idx]

    ids_train = set(df_train['identity_key'].unique())
    ids_test = set(df_test['identity_key'].unique())
    bocor = len(ids_train & ids_test)
```

Variabel `bocor` di sini dihitung untuk verifikasi, memastikan betul-betul tidak ada `identity_key` yang muncul di train dan test dalam fold yang sama. Kalau nilainya 0 di semua fold, berarti split-nya sudah bersih.

Hasil pembagian ini disimpan ke file `fold_indices.pkl` supaya bisa dipakai ulang persis sama di Notebook 3 dan Notebook 4. Ini penting supaya perbandingan model di Notebook 4 benar-benar adil, karena semua metode diuji dengan pembagian data yang identik.

```python
fold_indices = []
for fold, (train_idx, test_idx) in enumerate(skf.split(df, y=y_stratify, groups=groups)):
    fold_indices.append({'fold': fold + 1, 'train': train_idx.tolist(), 'test': test_idx.tolist()})

with open(FOLD_PATH, 'wb') as f:
    pickle.dump(fold_indices, f)
```

**1.13 Analisis Pola Penamaan (bin/binti)**

Kartu identitas Malaysia mencantumkan penanda silsilah (`BIN `, `BINTI `, `A/L `, `A/P `) yang tidak muncul sama sekali di dokumen luar negeri. Sinyal biner ini sangat diskriminatif, jadi dijadikan salah satu fitur pembeda asal dokumen.

**1.14 Analisis Pola Teks Pembeda Negara (level EDA)**

Sebagai eksplorasi awal (bukan fitur final — fitur final dihitung di Notebook 2 dari hasil OCR, bukan dari kolom `name` ground truth), dicoba deteksi kata kunci `MYKAD`/`WARGANEGARA`/`KAD PENGENALAN` dan pola nomor 12 digit langsung pada kolom `name` ground truth:

```python
df['has_mykad_kw'] = df['name'].str.contains('MYKAD|WARGANEGARA|KAD PENGENALAN', case=False, na=False)
df['has_ic_pattern'] = df['name'].str.contains(r'\b\d{6}\d{2}\d{4}\b', regex=True, na=False)
```

**1.15 Analisis Karakter Non-ASCII dan Diakritik**

Nama dokumen luar negeri sering mengandung diakritik (é, ñ, ç, dan sejenisnya) yang bisa memicu kesalahan pencocokan kalau tidak dinormalisasi. Fungsi `normalize_for_cer` memakai `unicodedata.normalize('NFKC', ...)` sebelum lowercase, dipakai lagi sebagai basis fungsi CER di Notebook 2 dan 3.

**1.16 Baseline OCR Tanpa Preprocessing**

Untuk menunjukkan kenapa tahap pemotongan kartu di Notebook 2 itu perlu, dijalankan baseline OCR pada foto **mentah langsung** (tanpa cropping) memakai `PaddleOCR` dengan `lang='en'` (bukan `lang='pt'` yang dipakai pipeline utama di Notebook 2 — baseline ini murni ilustrasi awal, konfigurasi bahasanya belum dioptimasi). Lima sampel dipilih secara representatif, bukan acak:

```python
sample_candidates = {
    'buram': my_blur.iloc[0]['filename'],                 # Malaysia paling buram
    'normal': my_blur.iloc[len(my_blur) // 2]['filename'], # Malaysia median blur
    'tajam': my_blur.iloc[-1]['filename'],                 # Malaysia paling tajam
    'luar_negeri': ln_blur.iloc[len(ln_blur) // 2]['filename'], # Luar negeri median blur
    'gelap': stats_df.sort_values('brightness').iloc[0]['filename'], # gambar paling gelap
}
```

Hasil OCR-nya divisualisasikan dengan bounding box (fungsi `draw_ocr_boxes`) supaya area di luar kartu yang ikut terbaca (meja, tangan, latar belakang) bisa terlihat jelas sebagai bukti visual kenapa cropping akurat wajib dilakukan sebelum OCR.

**1.17 Ringkasan Temuan Eksplorasi**

Notebook ditutup dengan tabel ringkasan 9 temuan utama beserta implikasi teknisnya:

| Temuan | Detail | Implikasi Teknis |
|---|---|---|
| Dataset multi-negara | Malaysia dan 20 negara lain | Ekstraksi address hanya aktif untuk dokumen Malaysia |
| Alamat MNAR (Missing Not At Random) | 82.3% gambar tidak punya alamat | Tidak boleh diimputasi, deteksi asal dokumen dulu |
| Duplikasi identitas | Hingga puluhan foto per orang | StratifiedGroupKFold wajib untuk cegah data leakage |
| Pola nama eksklusif | Pola BIN/BINTI/A-L/A-P hanya di Malaysia | Fitur nama sangat diskriminatif di field classifier |
| Format tanggal | Beragam format `YYYY-MM-DD` dan `YYYY` saja | Regex multi-format diperlukan |
| Kualitas gambar bervariasi | Blur score dari 4 hingga 2704 | Preprocessing adaptif dan Triage Gate ML wajib ada |
| Orientasi beragam | Banyak gambar portrait dan berlatar | Lokalisasi kartu contour/homography sangat kritis |
| Dokumen luar negeri buram | Rata-rata blur luar negeri 63 vs Malaysia 757 | Preprocessing harus adaptif |
| OCR mentah membaca latar | Teks latar ikut terdeteksi | Cropping akurat adalah prasyarat utama sebelum OCR |

---

### Notebook 2: Lokalisasi Dokumen dan Rekayasa Fitur

Tahap ini mengubah foto KTP mentah menjadi tabel angka yang bisa dipelajari algoritma, tanpa harus mengamati piksel gambar satu per satu secara langsung.

**2.1 Inisialisasi Model**

Dua model dipakai di sini, DocAligner untuk mencari posisi kartu, dan PaddleOCR untuk membaca tulisannya.

```python
ocr_engine = PaddleOCR(use_angle_cls=True, lang='pt', device='cpu', enable_mkldnn=False)

import turbojpeg
class DummyTurboJPEG:
    def __init__(self, *args, **kwargs):
        pass
turbojpeg.TurboJPEG = DummyTurboJPEG

from docaligner import DocAligner
doc_model = DocAligner()
```

Pemilihan `lang='pt'` (model bahasa Portugis) untuk PaddleOCR bukan salah ketik. Alasannya, sekitar 82.3% dataset ini adalah dokumen luar negeri dengan karakter Latin-extended yang berat (Ñ, é, ã, ç, dan sejenisnya), sementara model bahasa Portugis mendukung penuh charset ini tanpa mengorbankan akurasi baca teks Melayu yang notabene juga berbasis alfabet Latin standar.

Sebelum mengimpor `DocAligner`, modul `turbojpeg.TurboJPEG` di-patch dengan kelas dummy kosong. Ini trik praktis supaya `docaligner` tidak gagal saat startup kalau library sistem `libturbojpeg` tidak tersedia di environment eksekusi — DocAligner tetap bisa jalan tanpa akselerasi decode JPEG itu.

**2.2 Lokalisasi Kartu**

DocAligner mendeteksi 4 titik sudut kartu secara langsung dari piksel gambar (model deep learning untuk regresi koordinat sudut). Setelah 4 titik ditemukan, dilakukan **perspective warp**, yaitu transformasi geometris yang meluruskan kartu yang miring atau terfoto dari sudut pandang tidak tegak lurus menjadi persegi panjang rapi.

```python
def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_warp(img, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    max_width  = max(int(np.linalg.norm(tr - tl)), int(np.linalg.norm(br - bl)))
    max_height = max(int(np.linalg.norm(bl - tl)), int(np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [max_width-1, 0], [max_width-1, max_height-1], [0, max_height-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (max_width, max_height))
```

Logika `order_points` memakai trik geometri sederhana. Titik pojok kiri-atas punya jumlah koordinat (x+y) paling kecil, titik pojok kanan-bawah punya jumlah paling besar. Sementara selisih koordinat (y-x) dipakai untuk membedakan pojok kanan-atas (selisih paling kecil) dari pojok kiri-bawah (selisih paling besar). Setelah urutan 4 titik ini pasti, `cv2.getPerspectiveTransform` menghitung matriks transformasi 3x3 yang memetakan 4 titik sumber ke 4 titik tujuan (persegi panjang bersih), lalu `warpPerspective` menerapkan matriks itu ke seluruh gambar.

Setelah warp, kalau hasilnya lebih tinggi daripada lebar (`wh > ww`), gambar diputar 90 derajat searah jarum jam supaya orientasinya konsisten landscape.

Ada juga mekanisme pengaman kalau DocAligner gagal atau hasilnya tidak masuk akal (area crop terlalu kecil dibanding foto asli, atau rasio aspek terlalu ekstrem — DocAligner diketahui gagal secara diam-diam pada sekitar 48 kartu, menghasilkan warp yang nyaris kosong), gambar akan di-fallback ke gambar asli tanpa crop supaya proses tidak berhenti total.

```python
orig_area = img.shape[0] * img.shape[1]
warp_area = wh * ww
aspect    = max(ww, wh) / max(min(ww, wh), 1)
if warp_area < 0.10 * orig_area or aspect > 3.5:
    return img, 'fallback_original'
```

Metode lokalisasi yang terpakai (`docaligner`, `fallback_original`, atau `original` kalau gambar sumber tidak ditemukan sama sekali, atau `cached` kalau hasil crop sudah pernah ada) dicatat per gambar di kolom `localization_method` pada `master_df`.

**2.3 Kualitas Hasil Crop, PCA, dan Skew Angle**

Setelah crop, dihitung ulang metrik kualitas gambar — kali ini **6 metrik**, bukan cuma 4 seperti di Notebook 1:

```python
def compute_crop_quality(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w  = img.shape[:2]
    edges = cv2.Canny(gray, 50, 150)
    return {
        'crop_blur_score':       float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        'crop_brightness':       float(gray.mean()),
        'crop_contrast':         float(gray.std()),
        'crop_edge_density':     float(edges.mean()),
        'crop_dark_pixel_ratio': float((gray < 50).mean()),
        'crop_aspect_ratio':     w / max(h, 1),
    }
```

Perhatikan bahwa `crop_dark_pixel_ratio` di sini didefinisikan berbeda dari `dark_pixel_ratio` di Notebook 1 — di sini langsung dari ambang piksel grayscale `< 50`, bukan dari hasil `cv2.threshold` biner terpisah. Ditambah `skew_angle` untuk mengukur seberapa miring hasil crop-nya:

```python
def compute_skew_angle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords    = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    return float(angle)
```

Metode ini memakai thresholding Otsu untuk memisahkan piksel tulisan (terang) dari latar (gelap) secara otomatis tanpa perlu menentukan ambang batas manual, lalu `minAreaRect` mencari persegi panjang dengan luas minimum yang membungkus semua piksel tulisan tersebut. Sudut kemiringan persegi panjang itulah yang dipakai sebagai estimasi skew.

Setelah itu, `crop_blur_score` ditransformasi log (`crop_blur_score_log = np.log1p(crop_blur_score)`) untuk melembutkan distribusinya yang sangat skewed (sudah teridentifikasi sejak analisis distribusi di Notebook 1.9).

Empat metrik kualitas (`crop_blur_score_log`, `crop_brightness`, `crop_contrast`, `crop_edge_density`) lalu dikompresi jadi 2 komponen utama lewat **PCA**, setelah distandarisasi dulu dengan `StandardScaler`:

```python
pca_cols   = ['crop_blur_score_log', 'crop_brightness', 'crop_contrast', 'crop_edge_density']
scaler_pca = StandardScaler()
X_quality  = scaler_pca.fit_transform(master_df[pca_cols].fillna(0))
pca        = PCA(n_components=2, random_state=42)
pca_comps  = pca.fit_transform(X_quality)

master_df['quality_pc1'] = pca_comps[:, 0]
master_df['quality_pc2'] = pca_comps[:, 1]
```

`quality_pc1` inilah yang nantinya dipakai lagi sebagai salah satu dari 7 fitur `TRIAGE_FEATURES` di Notebook 3 (`quality_pc2` dihitung tapi tidak dipakai lebih lanjut — cuma untuk visualisasi 2D). Hasil kompresi ini divisualisasikan lewat scatter plot 2D berwarna berdasarkan tingkat blur logaritmik, untuk memverifikasi secara visual bahwa PCA memang berhasil memisahkan gambar tajam dari gambar buram.

**2.4 Ekstraksi Teks OCR**

Gambar hasil crop diperkecil dulu (maksimal sisi 1600 piksel) sebelum dibaca PaddleOCR, supaya proses lebih cepat tanpa mengorbankan keterbacaan teks. Koordinat bounding box hasil OCR kemudian dikembalikan lagi ke skala ukuran gambar aslinya.

```python
h, w = img.shape[:2]
max_side = 1600
scale = 1.0
if max(h, w) > max_side:
    scale = max_side / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

texts, scores, boxes = parse_ocr_result(ocr_engine.predict(img))

if scale < 1.0 and boxes:
    boxes = [[[float(pt[0] / scale), float(pt[1] / scale)] for pt in box] for box in boxes]
```

Perlu dicatat juga bahwa gambar dikirim ke OCR dalam bentuk BGR mentah tanpa preprocessing tambahan seperti CLAHE atau bilateral filter. Ini keputusan sadar, karena preprocessing semacam itu justru sering membuat gambar "kotor" di mata model deep learning modern yang sudah dilatih dengan variasi kondisi gambar luas, jadi input mentah lebih menghasilkan bacaan yang bersih dibanding input yang sudah "diproses berlebihan". Komentar di kode bahkan menyebut versi cache sebelumnya (yang masih pakai CLAHE/bilateral filter) sengaja diberi nama file cache berbeda (`raw_ocr_v2_unfiltered.pkl`) supaya cache lama tetap tersimpan sebagai backup saat bereksperimen dengan pendekatan baru ini.

Seluruh hasil OCR mentah (`texts`, `scores`, `boxes`) di-cache ke `raw_ocr_v2_unfiltered.pkl`, dengan mekanisme *cache invalidation* sederhana: kalau nama file sudah ada di cache **dan** file gambar hasil crop belum berubah sejak cache terakhir ditulis (dibandingkan lewat `os.path.getmtime`), OCR tidak dijalankan ulang untuk gambar itu.

**2.5 Fitur Sinyal Asal Dokumen**

Selain label lemah dari Notebook 1, dibangun juga fitur tambahan berbasis kata kunci hasil OCR yang sifatnya deterministik dan independen dari bias label alamat. Fungsi ini menghasilkan **5 fitur**, bukan 3:

```python
def build_origin_features(texts):
    joined_text = " ".join(texts).upper()

    has_mykad = int(any(kw in joined_text for kw in ["MYKAD", "WARGANEGARA", "KAD PENGENALAN"]))
    has_bin_binti = int(any(kw in joined_text for kw in ["BIN ", "BINTI ", "A/L ", "A/P "]))
    has_ic_pattern = int(bool(re.search(r'\b\d{12}\b|\b\d{6}-\d{2}-\d{4}\b', joined_text)))

    return {
        'has_mykad_keyword': has_mykad,
        'has_ic_number_pattern': has_ic_pattern,
        'has_bin_binti_any_segment': has_bin_binti,
        'n_segments': len(texts),
        'avg_segment_len': np.mean([len(t) for t in texts]) if texts else 0
    }
```

`n_segments` (jumlah segmen teks yang berhasil dibaca OCR di satu gambar) dan `avg_segment_len` (rata-rata panjang tiap segmen) ditambahkan sebagai sinyal kasar kepadatan tulisan pada kartu — nilai ini disimpan ke `text_origin_features.csv` tapi **tidak ikut dipakai** sebagai input Field Classifier ataupun Origin Classifier di Notebook 3 (keduanya tidak masuk `FEATURE_COLS`); fungsinya lebih sebagai bahan eksplorasi/diagnostik.

> **Catatan revisi kritik di kode (`REVISI KRITIK #4 & #5`):** versi sebelumnya dari notebook ini sempat membangun ulang kolom `doc_origin_weak_label` di sini juga, tapi ditemukan cacat secara semantik karena basisnya cuma merepresentasikan kelengkapan field alamat di ground truth — bukan sinyal independen. Kode saat ini secara eksplisit **menghapus** kolom `doc_origin_weak_label` dari `master_df` di titik ini (kalau ada), supaya model hilir (Origin Classifier di Notebook 3) tidak lagi bersandar diam-diam pada proxy yang bias itu, dan dipaksa mengandalkan fitur kata kunci eksplisit di atas plus TF-IDF dari teks OCR itu sendiri. Kolom target numerik `doc_origin_encoded_target` (dari `df_gt`, dihitung sebelum tahap ini, berbasis `has_address` juga) tetap dipertahankan karena itu satu-satunya target berlabel yang tersedia untuk melatih Origin Classifier — cuma representasi teks/label deskriptifnya yang dihapus, bukan target numeriknya.

```python
df_origin_feats.to_csv(os.path.join(DATASET_DIR, 'text_origin_features.csv'), index=False)
```

**2.6 Normalisasi Teks dan Definisi Fitur Segmen**

Sebelum masuk ke pelabelan, disiapkan dulu kamus dan fungsi pembantu untuk mengekstrak fitur tekstual dari tiap segmen:

```python
MALAYSIA_STATES = [
    'JOHOR','KEDAH','KELANTAN','MELAKA','NEGERI SEMBILAN','PAHANG',
    'PERAK','PERLIS','PULAU PINANG','SABAH','SARAWAK','SELANGOR',
    'TERENGGANU','KUALA LUMPUR','LABUAN','PUTRAJAYA',
    'W. PERSEKUTUAN','KL','SBH','SWK','PNG','TRG','PHG',
]

DATE_PATTERNS = [
    re.compile(r'\d{4}-\d{2}-\d{2}'),
    re.compile(r'\d{2}/\d{2}/\d{4}'),
    re.compile(r'\b\d{4}\b'),
]

def _has_addr_kw(t):  return int(any(k in t.upper() for k in [
    'JALAN','JLN','LORONG','LRG','KAMPUNG','KG','NO.','NO ','TAMAN',
    'TMN','PERSIARAN','LEBUH','BATU','TINGKAT','BLOK'
]))
```

**2.7 Pelabelan Lemah Segmen Teks**

Ini bagian paling penting di Notebook 2. Karena tidak ada anotasi manual, setiap potongan teks hasil OCR (segmen) perlu dilabeli otomatis sebagai `name`, `birth_date`, `address`, atau `other`, dengan cara mencocokkan isinya ke ground truth.

Sebelum pencocokan, label numerik yang menempel di depan teks (misalnya hasil OCR field bernomor seperti `"1. UZORAK"`) dibersihkan dulu khusus untuk pencocokan nama:

```python
txt_name_clean = re.sub(r'^\d+[a-c]?\.\s*', '', str(txt).strip())
```

Untuk `name` dan `address`, dipakai metrik **token overlap**, bukan CER biasa. Alasannya, satu segmen OCR bisa saja cuma berisi sepotong kecil dari field lengkap (misalnya cuma nama depan), atau malah tercampur kata lain di sekitarnya. Kalau dibandingkan pakai CER penuh terhadap keseluruhan field, hasilnya bisa salah menghukum segmen yang sebenarnya benar tapi cuma sepotong.

```python
def compute_token_overlap(seg_txt, full_gt):
    seg_nospace = str(seg_txt).upper().replace(' ', '')
    gt_nospace  = str(full_gt).upper().replace(' ', '')
    if len(gt_nospace) >= 4 and gt_nospace in seg_nospace:
        return 1.0
    if len(gt_nospace) >= 4 and lev.distance(seg_nospace, gt_nospace) <= 2:
        return 1.0

    seg_tokens = str(seg_txt).upper().split()
    gt_tokens  = str(full_gt).upper().split()
    matches = 0
    for s_tok in seg_tokens:
        for g_tok in gt_tokens:
            if (len(s_tok) >= 4 and lev.distance(s_tok, g_tok) <= 1) or (len(s_tok) < 4 and s_tok == g_tok):
                matches += 1
                break
    return matches / len(seg_tokens)
```

Cara bacanya, skor overlap dihitung sebagai pecahan dari jumlah token di segmen OCR yang berhasil cocok ke salah satu token di field ground truth, dengan toleransi jarak edit 1 karakter untuk kata panjang (menutupi typo kecil hasil OCR seperti O terbaca 0). Ada juga jalan pintas kalau ground truth (tanpa spasi) ternyata jadi substring dari segmen OCR (tanpa spasi), langsung dianggap cocok sempurna — atau kalau jarak edit antar keduanya (tanpa spasi) di bawah atau sama dengan 2. Ini menangani kasus OCR yang menggabungkan dua kata jadi satu, misalnya "SILVA COSTA" terbaca sebagai "SILVACOSTA".

Untuk `birth_date`, pendekatannya jauh lebih berlapis dibanding sekadar "pencocokan pola NRIC lalu CER":

1. **Pencocokan substring NRIC standar (`YYMMDD`).** Diambil digit tahun-bulan-tanggal dari ground truth (`birth_date[2:4]+birth_date[5:7]+birth_date[8:10]`), lalu dicari di setiap potongan 6-karakter berturutan dari teks segmen (setelah strip spasi dan tanda hubung) dengan toleransi jarak edit ≤ 1.
2. **Pencocokan pola "MyTentera IC".** Kalau langkah 1 gagal dan panjang teks segmen (bersih) minimal 10 karakter, dicoba juga cocokkan 4 karakter pertama terhadap `MMDD` (bulan-tanggal tanpa tahun) dari ground truth, dengan toleransi jarak edit ≤ 1 — menangani format nomor identitas militer/Tentera yang urutan digitnya berbeda dari NRIC sipil biasa.
3. **Fallback normalisasi tanggal + CER.** Kalau kedua pencocokan pola gagal, teks segmen dinormalisasi dulu lewat fungsi `_normalize_date_weak` (menangani format `DD.MM.YYYY`/`DD:MM:YYYY`/`DD-MM-YYYY`, `YYYY.MM.DD`, dan tanggal dengan nama bulan — termasuk bahasa Melayu "Mac"/"Ogos" dan Indonesia "Maret"/"Agustus"/"Desember"), lalu dianggap cocok kalau CER-nya di bawah 0.3.

```python
def _normalize_date_weak(t):
    t = str(t).upper().replace(',', '').strip()
    m = re.search(r'\b(\d{2})[.:/-](\d{2})[.:/-](\d{4})\b', t)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    m = re.search(r'\b(\d{4})[.:/-](\d{2})[.:/-](\d{2})\b', t)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    months = {
        'JAN': '01', 'JANUARI': '01', 'JANUARY': '01', 'FEB': '02', ..., 'MAC': '03',
        'OGOS': '08', 'AGUSTUS': '08', 'DIS': '12', 'DISEMBER': '12', 'DESEMBER': '12', ...
    }
    for mon_str, mon_num in months.items():
        if mon_str in t:
            m = re.search(r'(\d{1,2})\s+' + mon_str + r'\s+(\d{4})', t)
            if m:
                return f"{m.group(2)}-{mon_num}-{m.group(1).zfill(2)}"
    return t
```

Threshold overlap minimum ditentukan dari analisis sensitivitas (dicoba 0.3, 0.4, 0.5), dilengkapi dengan pemeriksaan manual pada segmen "borderline" (skor overlap antara 0.3 dan 0.5, khususnya yang skornya persis di sekitar 1/3, populasi terbesar) untuk memastikan pilihan angka **0.5** sebagai titik yang menyisakan proporsi segmen "signal" yang wajar tanpa terlalu banyak menerima segmen yang sebetulnya cuma kebetulan mirip.

```python
scores_map = {'name': ov_name, 'birth_date': overlap_dob, 'address': ov_addr}
best_label, best_score = max(scores_map.items(), key=lambda kv: kv[1])

if best_label == 'birth_date':
    # Kalau salah satu dari 3 pencocokan pola DOB di atas berhasil (overlap_dob == 1.0),
    # segmen langsung dianggap birth_date terlepas dari nilai threshold overlap.
    label = 'birth_date' if overlap_dob >= 1.0 else (
        best_label if best_score >= VALIDATED_OVERLAP_THRESHOLD else 'other'
    )
else:
    label = best_label if best_score >= VALIDATED_OVERLAP_THRESHOLD else 'other'
```

Perhatikan jalur khusus untuk `birth_date`: kalau salah satu dari tiga metode pencocokan pola tanggal di atas berhasil (`overlap_dob >= 1.0`), segmen langsung dilabeli `birth_date` tanpa perlu memenuhi ambang batas 0.5 — karena skor `overlap_dob` memang didesain biner (0.0 atau 1.0), bukan skala fraksi seperti `ov_name`/`ov_addr`.

**2.8 Fitur Spasial dan Tekstual per Segmen**

Setelah label ditentukan, setiap segmen diberi fitur numerik yang menggambarkan posisi dan bentuk teksnya di atas kartu.

```python
y_rel     = float(np.mean(pts[:, 1])) / max(img_h, 1)
x_rel     = float(np.mean(pts[:, 0])) / max(img_w, 1)
width_rel = (float(np.max(pts[:, 0])) - float(np.min(pts[:, 0]))) / max(img_w, 1)
```

`y_rel` dan `x_rel` adalah posisi tengah segmen dinormalisasi terhadap tinggi dan lebar gambar (0 sampai 1), sementara `width_rel` adalah lebar segmen relatif terhadap lebar gambar. Normalisasi relatif ini penting karena ukuran kartu hasil crop bisa berbeda-beda antar gambar, jadi posisi absolut dalam piksel tidak bisa dibandingkan langsung antar dokumen, sedangkan posisi relatif bisa.

Kalau file gambar hasil crop untuk suatu segmen ternyata hilang dari disk, dimensi gambar di-fallback ke 1000x1000 piksel dan ditandai lewat flag `is_spatial_dummy = 1` (kondisi normal `is_spatial_dummy = 0`) supaya baris-baris dengan fitur spasial yang tidak akurat ini bisa dikenali/difilter kalau perlu, disertai `warnings.warn` saat terjadi.

Daftar lengkap kolom yang disimpan per baris segmen ke `ocr_segments_features.csv`:

| Kolom | Deskripsi |
|---|---|
| `filename` | Nama file gambar asal segmen |
| `text` | Teks mentah hasil OCR untuk segmen ini |
| `ocr_conf` | Confidence score dari PaddleOCR |
| `y_rel`, `x_rel`, `width_rel` | Posisi & lebar relatif segmen (belum di-scale MinMax) |
| `is_spatial_dummy` | 1 kalau gambar crop hilang saat fitur spasial dihitung (fallback dimensi 1000x1000) |
| `text_len` | Panjang string teks |
| `word_count` | Jumlah kata |
| `alpha_ratio` | Rasio karakter huruf terhadap total karakter |
| `digit_ratio` | Rasio karakter angka terhadap total karakter |
| `is_all_caps` | 1 kalau seluruh teks huruf kapital |
| `has_date_pattern` | 1 kalau cocok salah satu dari 3 pola tanggal (`DATE_PATTERNS`) |
| `has_address_kw` | 1 kalau mengandung kata kunci alamat (JALAN, LORONG, dst) |
| `has_state_kw` | 1 kalau mengandung nama negara bagian Malaysia |
| `has_postcode` | 1 kalau mengandung pola 5 digit kode pos |
| `has_bin_binti` | 1 kalau mengandung BIN/BINTI/A-L/A-P |
| `line_rank` | Urutan baris dari atas ke bawah berdasarkan `y_rel` |
| `label` | Label lemah hasil pencocokan ground truth (`name`/`birth_date`/`address`/`other`) |
| `box` | Koordinat bounding box mentah (4 titik) |
| `has_mykad_keyword`, `has_ic_number_pattern` | Disalin dari fitur level dokumen (Notebook 2.5) ke tiap baris segmen |
| `quality_pc1`, `skew_angle` | Disalin dari fitur level dokumen (kualitas hasil crop) ke tiap baris segmen |

`line_rank` berguna karena pada kebanyakan format KTP, nama biasanya muncul di baris atas sementara alamat di baris bawah. Perlu dicatat bahwa `has_mykad_keyword`, `has_ic_number_pattern`, `quality_pc1`, dan `skew_angle` **ditempelkan di tiap baris segmen** tapi **tidak masuk daftar `FEATURE_COLS`** yang benar-benar dipakai Field Classifier di Notebook 3 — kolom-kolom ini tersimpan di CSV untuk keperluan eksplorasi/debugging lanjutan, bukan input model produksi.

**2.9 Diagnostik Cakupan Label dan Kalibrasi Triage (khusus EDA Notebook 2)**

Dua analisis tambahan dilakukan sebelum normalisasi akhir:

*Cakupan label per dokumen* — mengecek berapa persen dari seluruh 632 gambar yang punya minimal satu segmen berlabel `name`, `birth_date`, dan `address` masing-masing, sebagai sanity check bahwa pelabelan lemah di atas benar-benar menghasilkan sinyal yang cukup untuk melatih model.

*Kalibrasi ambang batas triage berbasis pseudo-CER* — analisis hubungan antara rata-rata confidence OCR (`avg_ocr_conf`, dibagi ke 10 bin) dan tingkat kegagalan baca nama, diukur lewat **pseudo-CER** (`1 - token recall` nama terhadap gabungan seluruh teks OCR di kartu itu). Dicari bin confidence terendah yang median pseudo-CER-nya masih di bawah 0.1, dengan fallback 0.85 kalau tidak ada bin yang memenuhi.

```python
ocr_tokens = set(normalize_for_cer(joined_ocr_text).split())
gt_tokens = set(normalize_for_cer(str(row_gt['name'])).split())
recall = len(gt_tokens.intersection(ocr_tokens)) / len(gt_tokens) if gt_tokens else 0.0
pseudo_cer_name = 1.0 - recall
```

**Penting:** perhitungan ambang batas ini secara eksplisit ditandai di kode sebagai **murni untuk visualisasi/EDA**, bukan angka final yang dipakai model. Kode mencetak peringatan berikut untuk menegaskan hal ini:

> "STATUS: Aman. Kalkulasi ini murni untuk visualisasi/EDA agar tidak menyesatkan analisa. Ambang batas Triage untuk model final dihitung ulang secara independen per-fold di Notebook 3 (CV)."

Metodologinya pun berbeda dari kalibrasi final di Notebook 3: di sini dipakai **pseudo-CER** (berbasis 1 dokumen dibanding seluruh gabungan teksnya sendiri, threshold median CER ≤ 0.1) dan dihitung sekali dari seluruh dataset, sedangkan Notebook 3 memakai **token recall langsung** (bukan 1 dikurangi recall) dengan threshold median recall ≥ 0.8, dan dihitung ulang terpisah di setiap fold train supaya tidak ada informasi dari fold test yang bocor ke ambang batas.

**2.10 Normalisasi Spasial Akhir dan Penyimpanan**

Kolom spasial (`y_rel`, `x_rel`, `width_rel`) dinormalisasi ulang dengan `MinMaxScaler` supaya rentang nilainya konsisten 0 sampai 1 di seluruh dataset, dan `line_rank` dinormalisasi per gambar (dibagi dengan rentang baris di gambar itu sendiri) supaya urutan barisnya bisa dibandingkan antar dokumen dengan jumlah baris berbeda-beda.

```python
mm_scaler = MinMaxScaler()
df_segments[['y_rel_scaled', 'x_rel_scaled', 'width_rel_scaled']] = \
    mm_scaler.fit_transform(df_segments[['y_rel', 'x_rel', 'width_rel']].fillna(0))

df_segments['line_rank_norm'] = df_segments.groupby('filename')['line_rank'].transform(
    lambda s: (s - s.min()) / max((s.max() - s.min()), 1)
)
```

Hasil akhirnya disimpan sebagai dua file, `ocr_segments_features.csv` (satu baris per segmen teks) dan `master_features.csv` (satu baris per gambar), keduanya jadi input utama Notebook 3.

---

### Notebook 3: Pemodelan dan Evaluasi Akhir

Notebook ini melatih tiga model secara berurutan (Origin Classifier, Triage Classifier, Field Classifier), lalu merangkai semuanya jadi satu pipeline ekstraksi lengkap yang dievaluasi lewat skema 5-fold cross validation.

**3.0 Konstanta dan Konfigurasi Global**

Beberapa konstanta dideklarasikan di awal notebook dan dipakai berulang kali di seluruh pipeline:

```python
FEATURE_COLS = [
    'y_rel_scaled', 'x_rel_scaled', 'width_rel_scaled',
    'text_len', 'word_count',
    'ocr_conf', 'alpha_ratio', 'digit_ratio', 'is_all_caps',
    'has_date_pattern', 'has_address_kw', 'has_state_kw',
    'has_postcode', 'has_bin_binti', 'line_rank_norm',
    'doc_origin_encoded',
]  # 16 fitur input Field Classifier (LightGBM)

TRIAGE_FEATURES = [
    'crop_blur_score_log', 'crop_brightness', 'crop_contrast',
    'crop_edge_density', 'crop_dark_pixel_ratio', 'skew_angle', 'quality_pc1',
]  # 7 fitur input Triage Classifier (RandomForest)

LABEL_MAP     = {'other': 0, 'name': 1, 'birth_date': 2, 'address': 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
COLOR_MAP = {  # warna BGR untuk visualisasi bounding box
    'name':       (0,   0, 255),   # merah
    'birth_date': (255, 0,   0),   # biru
    'address':    (0, 255,   0),   # hijau
    'other':      (128, 128, 128), # abu-abu
}
```

Saat data dimuat ulang di awal notebook, `master_features.csv` digabung ulang dengan `ground_truth_normalized.csv` yang terbaru untuk kolom `name`, `birth_date`, `address`, `negara` — bukan memakai nilai ground truth yang mungkin sudah usang tersimpan di `master_features.csv` dari eksekusi Notebook 2 sebelumnya (komentar kode menyebut ini sebagai perbaikan atas "Bug 10": ground truth di `master_features.csv` bisa basi kalau `ground_truth_normalized.csv` diedit manual setelah Notebook 2 dijalankan).

**3.1 Metrik Evaluasi Teks**

Sebelum melatih apa pun, didefinisikan dulu cara mengukur seberapa dekat hasil prediksi dengan ground truth.

```python
def compute_cer(hyp, ref):
    ref_str = str(ref).strip().lower()
    hyp_str = str(hyp).strip().lower()
    if len(ref_str) == 0:
        return 0.0 if len(hyp_str) == 0 else 1.0
    return lev.distance(hyp_str, ref_str) / len(ref_str)

def compute_wer(hyp, ref):
    ref_words = str(ref).strip().lower().split()
    hyp_words = str(hyp).strip().lower().split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return lev.distance(' '.join(hyp_words), ' '.join(ref_words)) / len(ref_words)

def compute_f1_token(hyp, ref):
    ref_set = set(str(ref).strip().lower().split())
    hyp_set = set(str(hyp).strip().lower().split())
    if not ref_set and not hyp_set:
        return 1.0
    if not ref_set or not hyp_set:
        return 0.0
    tp = len(ref_set & hyp_set)
    precision, recall = tp / len(hyp_set), tp / len(ref_set)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def exact_accuracy(hyp, ref):
    return float(str(hyp).strip().lower() == str(ref).strip().lower())
```

**CER (Character Error Rate)** dihitung dari jarak edit Levenshtein (jumlah minimum operasi tambah, hapus, atau ganti karakter untuk mengubah teks prediksi jadi teks ground truth) dibagi panjang karakter ground truth. Semakin kecil nilainya semakin dekat hasil prediksi ke jawaban benar, dan nilai 0 berarti sama persis.

**WER (Word Error Rate)** konsepnya sama tapi dihitung di level kata, bukan karakter, cocok untuk melihat kesalahan pada level susunan kata bukan cuma typo huruf.

**Token F1** dihitung dari precision dan recall pencocokan kata antara prediksi dan ground truth (berbasis himpunan kata unik, bukan urutan), lalu digabung jadi harmonic mean. Bedanya dengan CER, Token F1 tidak peduli urutan kata, jadi dua nama yang isinya sama tapi urutan kata depan-belakangnya tertukar tetap dianggap cocok sempurna di metrik ini walau CER-nya bisa tinggi.

**Exact Accuracy** adalah pencocokan string sama persis (case-insensitive, setelah `strip()`) — biner 1 atau 0.

**3.2 Normalisasi dan Validasi Hasil Prediksi**

Sebelum dibandingkan ke ground truth (dan sebelum digabungkan jadi field final), teks hasil ekstraksi dibersihkan dulu dari noise seperti label field ("NAMA:", "NAME:"), kata kunci non-nama ("WARGANEGARA", "ISLAM"), dan karakter aneh.

```python
def _normalize_name(text):
    text = text.upper().strip()
    for noise in ['NAMA:', 'NAME:', 'NAMA', 'NAME', 'PENUH', 'FULL']:
        text = text.replace(noise, '')
    text = re.sub(r'[^A-Z /\-]', ' ', text)
    noise_words = ['WARGANEGARA', 'MALAYSIA', 'LELAKI', 'PEREMPUAN', 'ISLAM', 'MYKAD', 'KAD', 'PENGENALAN']
    for noise in noise_words:
        text = re.sub(rf'\b{noise}\b', '', text)
    return re.sub(r'\s+', ' ', text).strip()
```

Untuk alamat, ada normalisasi serupa plus ekspansi singkatan lewat kamus `ADDR_ABBREV`:

```python
ADDR_ABBREV = {
    r'\bJLN\b': 'JALAN', r'\bLRG\b': 'LORONG', r'\bKG\b': 'KAMPUNG',
    r'\bTMN\b': 'TAMAN', r'\bPKR\b': 'PERAK',
}

def _normalize_address(text):
    text = text.upper().strip()
    for noise in ['ALAMAT:', 'ADDRESS:', 'ALAMAT', 'ADDRESS']:
        text = text.replace(noise, '')
    for abbr, full in ADDR_ABBREV.items():
        text = re.sub(abbr, full, text)
    return re.sub(r'\s+', ' ', text).strip()
```

Untuk tanggal, fungsi `_normalize_date` mencoba banyak pola secara berurutan sampai salah satunya cocok:

1. `DD.MM.YYYY` / `DD:MM:YYYY` / `DD-MM-YYYY`
2. `YYYY.MM.DD` / `YYYY:MM:DD` / `YYYY-MM-DD`
3. `DD MONTH YYYY` dengan nama bulan (termasuk bahasa Melayu "Mac"/"Ogos" dan Indonesia "Maret"/"Agustus"/"Desember")
4. Fallback pola MRZ paspor standar ICAO (posisi 13-19 dari string bersih) atau pola nomor IC (`YYMMDD` di tengah string angka+huruf)
5. Fallback generik `YYMMDD` murni 6 digit (hanya kalau panjang string setelah dibersihkan tepat 6 karakter, supaya tidak salah mengurai `DD.MM.YYYY` yang masih ada titiknya)

Untuk penentuan abad dari 2 digit tahun (`YY`), aturannya: kalau `YY > 25` (atau `> 26` tergantung cabang kode) dianggap `19XX`, selain itu dianggap `20XX`.

Hasil akhirnya juga divalidasi lewat aturan sederhana:

```python
def validate_name(text):
    if not text or len(text) < 2: return False, 'EMPTY'
    if any(c.isdigit() for c in text): return False, 'CONTAINS_DIGIT'
    if len(text) > 100: return False, 'TOO_LONG'
    return True, 'OK'

def validate_date(text):
    if not text: return False, 'EMPTY'
    m = re.match(r'^(\d{4})(?:-(\d{2})-(\d{2}))?$', text)
    if not m: return False, 'BAD_FORMAT'
    year = int(m.group(1))
    if not (1900 <= year <= 2015): return False, f'YEAR_OUT_OF_RANGE:{year}'
    if m.group(2):
        month, day = int(m.group(2)), int(m.group(3))
        if not (1 <= month <= 12): return False, f'BAD_MONTH:{month}'
        if not (1 <= day <= 31): return False, f'BAD_DAY:{day}'
    return True, 'OK'
```

Nama dianggap tidak valid kalau kosong, mengandung angka, atau kepanjangan (>100 karakter), sementara tanggal dianggap tidak valid kalau formatnya tidak sesuai pola `YYYY` atau `YYYY-MM-DD`, atau tahunnya di luar rentang wajar (1900-2015).

Ada juga fungsi kecil `clean_name_pred` yang membuang semua digit dari prediksi nama sebelum divalidasi — alasannya, digit di dalam field nama nyaris selalu adalah kesalahan OCR atau segmen nomor identitas yang salah diklasifikasikan sebagai nama; kalau setelah digit dibuang hasilnya kosong, nama otomatis dianggap tidak valid dan berakhir di `HUMAN_REVIEW`.

**3.3 Origin Classifier**

Model pertama menebak asal dokumen (Malaysia atau bukan) dari isi teks hasil OCR, dilatih ulang di setiap fold supaya tidak ada kebocoran data antara train dan test.

```python
train_texts = df_seg_train.groupby('filename')['text'].apply(lambda x: ' '.join([str(v).lower() for v in x]))
y_origin_train = df_train.set_index('filename').loc[train_texts.index, 'doc_origin_encoded_target'] == 1

origin_clf = make_pipeline(TfidfVectorizer(max_features=300), LogisticRegression(random_state=42, class_weight='balanced'))
origin_clf.fit(train_texts, y_origin_train)
```

Logikanya, TF-IDF (Term Frequency-Inverse Document Frequency, dibatasi maksimal 300 fitur kata) mengubah gabungan seluruh teks OCR di satu gambar jadi vektor angka yang menangkap kata mana yang sering muncul di satu dokumen tapi jarang muncul di dokumen lain, cocok untuk menangkap kata kunci pembeda negara seperti nama negara di paspor atau istilah MyKad. Logistic Regression (dengan `class_weight='balanced'` untuk mengompensasi ketidakseimbangan kelas Malaysia vs Luar Negeri) lalu belajar memisahkan dua kelas ini berdasarkan vektor TF-IDF tadi.

Prediksi model ini pada data **test** fold itu sendiri (bukan label aslinya) dipakai lagi sebagai salah satu fitur input Field Classifier di langkah berikutnya (`doc_origin_encoded`), supaya Field Classifier bisa ikut menyesuaikan cara membaca posisi field berdasarkan asal dokumen. Kalau ada nama file di test set yang gagal dipetakan (misal karena segmen OCR-nya kosong), nilainya di-default ke `1` (dianggap Malaysia) lewat `.fillna(1)`.

Performanya diukur lewat precision, recall, dan F1 biner per fold, lalu dirata-rata di akhir sebagai metrik `Origin Classifier F1`.

**3.4 Triage Classifier**

Tujuannya menyaring foto yang kualitasnya terlalu buruk untuk diproses sebelum sampai ke tahap ekstraksi field, supaya sistem tidak memberi keputusan yang salah dari input yang memang sudah tidak layak baca.

Ambang batas confidence OCR (`threshold_conf`) ditentukan secara dinamis **per fold**, dihitung ulang dari data **train fold itu sendiri saja** (bukan dari keseluruhan dataset seperti kalkulasi EDA di Notebook 2.9, supaya tidak ada informasi dari fold test yang bocor ke kalibrasi). Metodenya: rata-rata confidence OCR (`avg_ocr_conf`) dibagi ke dalam 10 kelompok (bin), lalu untuk tiap dokumen train dihitung **token recall** nama (bukan pseudo-CER) — proporsi token ground truth nama yang berhasil ditemukan di gabungan teks OCR, dengan toleransi jarak edit 1 untuk token ≥4 karakter dan pencocokan eksak untuk token lebih pendek. Dicari bin dengan confidence paling rendah yang median recall-nya masih di atas 0.8.

```python
max_edits = 1 if len(r_tok) >= 4 else 0
if lev.distance(r_tok, h_tok) <= max_edits:
    matched_count += 1
recall = matched_count / len(ref_tokens)
...
conf_bins = np.linspace(0, 1, 11)
median_recall_per_bin = df_triage_curve_train.groupby('conf_bin')['recall'].median()
threshold_conf = None
for interval, med_r in median_recall_per_bin.items():
    if pd.notna(med_r) and med_r >= 0.8:
        threshold_conf = interval.left
        break
```

Kalau tidak ada bin yang memenuhi syarat, dipakai nilai default 0.85 sebagai jaring pengaman (dengan `warnings.warn`). Threshold ini lalu dipakai membuat label `is_readable`, dan `RandomForestClassifier` (`n_estimators=100`, `class_weight='balanced'`) dilatih untuk memprediksi label ini dari 7 fitur kualitas gambar (`TRIAGE_FEATURES`).

Saat evaluasi di test set, keputusan triage dihitung lewat fungsi `get_triage_label` yang menerapkan **aturan keras dulu, baru model**: kalau `crop_blur_score < 30`, langsung ditolak (`0`) tanpa perlu bertanya ke model; baru kalau lolos aturan keras, prediksi diserahkan ke `triage_clf`.

Performa triage diukur lewat dua metrik kesalahan:
- **FRR (False Rejection Rate)**, proporsi dokumen yang sebenarnya bisa dibaca (label `is_readable=1` menurut threshold recall test) tapi salah ditolak sistem. Ini kerugian di sisi bisnis karena nasabah asli jadi ditolak.
- **FAR (False Acceptance Rate)**, proporsi dokumen yang sebenarnya tidak layak baca tapi malah diloloskan sistem. Ini kerugian di sisi risiko karena bisa meloloskan data yang salah baca tanpa disadari.

```python
frr = np.sum((y_t_test == 1) & (y_t_pred == 0)) / max(np.sum(y_t_test == 1), 1)
far = np.sum((y_t_test == 0) & (y_t_pred == 1)) / max(np.sum(y_t_test == 0), 1)
```

**3.5 Field Classifier**

Ini otak utama sistem, tugasnya mengklasifikasikan tiap segmen teks hasil OCR ke salah satu dari 4 kelas (`other`, `name`, `birth_date`, `address`) berdasarkan 16 fitur spasial dan tekstual di `FEATURE_COLS` (lihat 3.0 di atas) yang sudah dibangun di Notebook 2.

Algoritma yang dipakai adalah **LightGBM**, salah satu jenis gradient boosting yang membangun banyak decision tree secara bertahap, di mana tiap tree baru fokus memperbaiki kesalahan prediksi tree-tree sebelumnya. Dipilih karena ringan secara komputasi, cepat dilatih, dan cenderung sangat andal untuk data tabular campuran (numerik dan kategorikal sederhana) seperti fitur spasial di sini.

Hyperparameter modelnya dicari otomatis lewat **Optuna**, sebuah library optimisasi hyperparameter yang mencoba kombinasi parameter secara cerdas (`TPESampler`, bukan grid search brute force), dievaluasi lewat 3-fold `StratifiedKFold` di dalam data latih tiap fold, dengan target memaksimalkan macro F1. Pencarian dibatasi **5 trial per fold** (`study.optimize(..., n_trials=5)`) — jumlah yang cukup kecil karena pencarian ini diulang untuk tiap satu dari 5 fold CV luar, jadi total ada 25 kombinasi hyperparameter yang dicoba di seluruh proses evaluasi.

```python
def optuna_objective(trial, X, y):
    params = {
        'objective': 'multiclass', 'num_class': 4, 'class_weight': 'balanced',
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 40),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 30),
        'random_state': 42, 'verbose': -1
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in cv.split(X, y):
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        scores.append(f1_score(y.iloc[val_idx], clf.predict(X.iloc[val_idx]), average='macro'))
    return np.mean(scores)

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(lambda trial: optuna_objective(trial, X_seg_train, y_seg_train), n_trials=5)
```

**Catatan diketahui (dari komentar di notebook): segmen nomor IC vs label DOB.** Pada KTP Malaysia, nomor IC mengandung tanggal lahir tertanam di 6 digit pertama (format `YYMMDD-XX-XXXX`, contoh: `001230-11-0470` → lahir 30 Des 2000), yang secara teknis adalah sumber DOB paling andal di MyKad. Namun LightGBM kadang memprediksi segmen ini sebagai `other`, karena rasio digitnya (`digit_ratio ≈ 0.86`) mirip dengan segmen numerik lain (kode pos, nomor referensi), dan `FEATURE_COLS` tidak punya fitur eksplisit untuk pola `YYMMDD-XX-XXXX`. **Dampak operasional tetap nihil**, karena `extract_fields()` punya fallback scanner (lihat 3.7 di bawah) yang menyapu semua segmen — termasuk yang berlabel `other` — mencari pola tanggal valid, sehingga DOB tetap terekstrak benar meski model field classifier salah label. Perbaikan yang disarankan di komentar kode: tambahkan fitur `has_ic_pattern` (regex `\d{6}-\d{2}-\d{4}`) ke `FEATURE_COLS` supaya model mendapat sinyal eksplisit untuk kelas `birth_date`.

**3.6 Template Router**

Bukan semua dokumen identitas punya tata letak yang bisa dibaca dengan pendekatan spasial generik. Paspor dengan MRZ, SIM Eropa dengan penomoran field baku, dan kartu identitas dari puluhan negara berbeda punya konvensi penempatan nama/tanggal lahir yang sangat berbeda-beda. Karena itu dibangun sistem deteksi format dokumen (template) yang bekerja bertingkat, dicek dari sinyal yang paling spesifik ke yang paling umum.

```python
def detect_template(segs_df):
    texts = segs_df['text'].dropna().astype(str).tolist()
    all_text = ' '.join(texts).upper()

    # 1. MRZ -- baris lebar tetap, padat karakter '<'
    mrz_pattern = re.compile(r'^[A-Z0-9<]{28,44}$')
    for t in texts:
        t_clean = t.replace(' ', '').upper()
        if mrz_pattern.match(t_clean) and t_clean.count('<') >= 5:
            return 'mrz'

    # 2. SIM Eropa dengan penomoran field "1." "2." "3."
    numbered = re.compile(r'^\d[a-c]?\.\s*\S')
    if sum(1 for t in texts if numbered.match(t.strip())) >= 3:
        return 'eu_license'

    # 3. Kartu identitas Eropa dengan pola posisi khas, misal "1.68 PRT"
    if re.search(r'\b\d\.\d{2}\s+[A-Z]{3}\b', all_text):
        return 'eu_id_positional'

    # 4. MyKad
    if any(kw in all_text for kw in ['MYKAD', 'KAD PENGENALAN', 'WARGANEGARA']):
        return 'mykad'

    # 5. Kamus kata kunci nama negara (~170 negara) sebagai fallback
    for clean_country, aliases in COUNTRY_ALIASES.items():
        if any(_country_match(alias, all_text) for alias in aliases):
            return f'fallback_{clean_country}'

    return 'unknown'
```

Kalau template terdeteksi, dipakai fungsi ekstraksi khusus format itu. Kalau tidak (`unknown`, atau `eu_id_positional` yang belum punya extractor khusus), sistem jatuh kembali ke pendekatan generik berbasis prediksi Field Classifier lewat `_assemble_field`.

**Kamus `COUNTRY_ALIASES` — sekitar 170 negara.** Dikelompokkan per kawasan (Asia-Pasifik, Eropa, Amerika, Afrika, Pasifik, dan kelompok "Other" berisi negara lintas kawasan seperti Australia, Kanada, negara Timur Tengah, dan Asia Selatan), tiap negara dipetakan ke satu atau lebih alias — nama dalam bahasa Inggris, nama dalam bahasa lokal negara itu sendiri (mis. `DEUTSCHLAND` untuk Germany, `SUOMI` untuk Finland, `HRVATSKA` untuk Croatia), dan/atau kode 3-huruf ISO. Contoh potongan untuk negara-negara yang benar-benar ada di dataset ini:

```python
COUNTRY_ALIASES = {
    'CROATIA':        ['CROATIA', 'HRVATSKA'],                          # dataset: 40 gambar
    'CZECH REPUBLIC': ['CZECH REPUBLIC', 'CZECHIA', 'CESKA REPUBLIKA'],  # dataset: 40 gambar
    'FINLAND':        ['FINLAND', 'SUOMI'],                             # dataset: 40 gambar
    'GERMANY':        ['GERMANY', 'DEUTSCHLAND'],                       # dataset: 60 gambar
    'GREECE':         ['GREECE', 'HELLAS', 'HELLENIC'],                 # dataset: 20 gambar
    'SPAIN':          ['SPAIN', 'ESPANA', 'REINO DE ESPANA'],           # dataset: 20 gambar
    # ...dan puluhan negara lain di luar dataset ini sebagai jaring pengaman generalisasi
}
```

Pencocokan alias memakai **regex word-boundary** (`\b...\b`), bukan substring `in` biasa. Ini perbaikan penting yang didokumentasikan langsung di komentar kode, dengan contoh kegagalan nyata kalau memakai substring biasa:

- `'KOR'` (alias untuk South Korea) adalah substring dari `'HENKILOKORTTI'` (bahasa Finlandia untuk "kartu identitas") → kartu Finlandia salah terdeteksi sebagai Korea Selatan.
- `'AND'` (alias untuk Andorra) adalah substring dari `'RHEINLAND'` (nama wilayah di Jerman) → kartu Jerman salah terdeteksi sebagai Andorra.
- `'ARM'` (alias untuk Armenia) adalah substring dari `'ARMAS'` (kata dalam bahasa Spanyol) → kartu Spanyol salah terdeteksi sebagai Armenia.

Karena itu, aturan yang dipegang di kode: **alias yang lebih pendek dari 5 karakter tidak boleh masuk kamus ini sama sekali**, dan seluruh pencocokan alias yang tersisa tetap diwajibkan lewat regex `\b` supaya tidak menangkap potongan kata di tengah string lain.

**3.7 Fungsi Ekstraksi per Template**

Fungsi generik `_assemble_field` dipakai sebagai basis untuk template `mykad`, `fallback_<negara>`, dan `unknown`/`eu_id_positional`. Sebelum menggabungkan teks, dilakukan beberapa tahap pembersihan:

```python
def _assemble_field(segs_df, label, is_malaysia):
    subset = segs_df[segs_df['pred_label_str'] == label].copy()
    if subset.empty:
        return '', 0.0

    # Buang segmen dengan confidence OCR di bawah CONF_FLOOR (0.70) --
    # menyaring tanda tangan, watermark, dan noise OCR lain.
    subset = subset[subset['ocr_conf'] >= CONF_FLOOR]
    if subset.empty:
        return '', 0.0

    # Deduplikasi bounding box yang tumpang tindih (dibulatkan 2 desimal)
    # supaya kata yang sama tidak digabung dobel.
    subset['_y_rounded'] = subset['y_rel_scaled'].round(2)
    subset['_x_rounded'] = subset['x_rel_scaled'].round(2)
    subset = subset.drop_duplicates(subset=['_y_rounded', '_x_rounded'])

    if label == 'birth_date':
        # Deduplikasi MRZ: kalau ada beberapa kandidat DOB, ambil yang paling
        # atas (y_rel terkecil), supaya baris MRZ di bagian bawah kartu
        # tidak salah terpilih ketimbang tanggal lahir yang tercetak biasa.
        subset = subset.sort_values('y_rel')
        best_row = subset.iloc[0]
        return _normalize_date(str(best_row['text'])), float(best_row['ocr_conf'])

    if label in ['name', 'address']:
        # kelompokkan jadi baris berdasarkan kedekatan posisi vertikal
        subset = subset.sort_values('y_rel_scaled')
        line_ids, current_line_id, last_y = [], 0, -100
        for y in subset['y_rel_scaled']:
            if y - last_y > 0.02:
                current_line_id += 1
                last_y = y
            line_ids.append(current_line_id)
        subset['line_id'] = line_ids
        subset = subset.sort_values(['line_id', 'x_rel_scaled'])

        if label == 'name':
            return _normalize_name(' '.join(subset['text'].astype(str))), avg_conf
        else:
            if not is_malaysia:
                return '', 0.0
            return _normalize_address(' '.join(subset['text'].astype(str))), avg_conf
```

Angka ambang `0.02` di sini artinya dua segmen dianggap satu baris yang sama kalau posisi vertikalnya (yang sudah dinormalisasi 0-1) berselisih kurang dari 2% tinggi kartu. `CONF_FLOOR` sendiri sengaja diturunkan dari 0.85 ke **0.70** — komentar kode menjelaskan bahwa PaddleOCR rutin memberi skor 0.75-0.85 untuk segmen bahasa asing meski bacaannya sudah benar, sementara gerbang triage (`crop_blur_score < 30`) di tahap sebelumnya sudah cukup untuk menyaring gambar yang benar-benar tidak layak baca, jadi `CONF_FLOOR` yang terlalu ketat di titik ini justru membuang informasi yang sebenarnya valid.

Untuk template `mrz` (`extract_mrz`), dipakai parsing khusus format MRZ standar ICAO 9303. Fungsi ini menangani beberapa variasi jumlah baris MRZ yang berhasil terdeteksi (1, 2, atau ≥3 baris — paspor TD3 punya 2 baris MRZ, tapi kadang OCR memecahnya jadi 3 baris terpisah), dan dua format posisi tanggal lahir yang berbeda tergantung tipe dokumen (TD3 paspor: digit DOB ada di posisi karakter 13-19 dari baris kedua; TD1 kartu ID: digit DOB ada di posisi 0-6). Baris nama diambil dari format `SURNAME<<GIVEN<NAMES<<<<<<<`, dipisah berdasarkan tanda `<<` pertama untuk memisahkan nama keluarga dari nama depan, dengan prefiks 5 karakter tipe dokumen+kode negara di awal baris dibuang lebih dulu.

Untuk template `eu_license` (`extract_eu_license`), field-nya sudah punya penomoran baku (field 1 = nama keluarga, field 2 = nama depan, field 3 = tanggal lahir), dicocokkan lewat regex penomoran `^(\d[a-c]?)\.?\s+(.*)`. Titik setelah nomor field dibuat opsional karena OCR kadang tidak membaca tanda titiknya (misal terbaca `"2 Anne"`, bukan `"2. Anne"`). Ada juga penanganan khusus kasus OCR yang melebur nomor field ke dalam nilai tanggal lahir (misal `"328.02.64"` sebenarnya adalah field `3` + tanggal `28.02.64`), dan nama tidak dipaksa collapse jadi nama keluarga saja kalau nama depannya kosong — keduanya digabung kalau tersedia, atau dipakai salah satu yang ada.

Untuk template `mykad` (`extract_mykad`), ekstraksi cukup langsung memanggil `_assemble_field` untuk ketiga field (`name`, `birth_date`, `address`) dengan `is_malaysia=True`.

Untuk template `fallback_<negara>` (`extract_fallback_country`), ada dua penanganan tambahan khusus:

1. **Pembuangan kata label field.** Sebelum nama dikembalikan, token-token yang cocok dengan kamus `_FIELD_LABEL_WORDS` (label field tercetak dalam berbagai bahasa — Spanyol: `NOMBRES`/`APELLIDOS`; Prancis: `NOM`/`PRENOM`; Italia: `COGNOME`/`NOME`; Portugis: `SOBRENOME`; Jerman: `FAMILIENNAME`/`VORNAME`; Polandia: `IMIE`/`NAZWISKO`; Kroasia/Serbia: `IME`/`PREZIME`; ditambah kata sifat kewarganegaraan seperti `CHILENA`/`BRASILEIRO`/`ALGERIENNE`) dibuang dari hasil, termasuk varian dengan tanda titik dua di belakangnya.
2. **Pembalikan urutan nama keluarga dan nama depan.** Untuk negara-negara yang konvensi penulisan kartunya menampilkan nama keluarga lebih dulu (didaftar eksplisit dalam himpunan `surname_first_countries` yang mencakup mayoritas negara Asia Timur dan hampir seluruh negara Eropa serta sebagian Amerika Latin dan Afrika di kamus `COUNTRY_ALIASES`), token terakhir dianggap nama depan dan sisanya (bisa lebih dari satu kata) dianggap nama keluarga majemuk, lalu digabung ulang dengan urutan "nama depan nama keluarga".

**Fallback scanner tanggal lahir.** Fungsi publik `extract_fields` yang memanggil salah satu dari fungsi-fungsi di atas sesuai template, punya satu pengaman terakhir: kalau setelah ekstraksi per-template DOB masih kosong (misalnya karena Field Classifier salah melabeli semua segmen tanggal sebagai `other`, seperti kasus nomor IC yang dibahas di 3.5), sistem **menyapu ulang seluruh segmen** di dokumen itu (termasuk yang berlabel `other`) mencari teks yang setelah dinormalisasi lewat `_normalize_date` menghasilkan format `YYYY-MM-DD` yang valid, lalu memakai hasil pertama yang ketemu.

```python
if not dob:
    for _, row in segs_df.dropna(subset=['text']).iterrows():
        norm = _normalize_date(str(row['text']))
        if norm and re.match(r'^\d{4}-\d{2}-\d{2}$', norm):
            dob = norm
            dob_conf = float(row['ocr_conf']) if pd.notna(row['ocr_conf']) else 0.5
            break

name = clean_name_pred(name)
```

**3.8 Keputusan Routing**

Setelah nama, tanggal lahir, dan alamat berhasil diekstrak, sistem memutuskan status akhir tiap dokumen lewat aturan berlapis, dari pengecekan paling murah dan pasti (aturan keras) sampai yang paling mahal (model dan confidence).

```python
CONF_AUTO_APPROVE = 0.75
CONF_HUMAN_REVIEW = 0.45  # dideklarasikan tapi tidak dirujuk lagi di route_decision — ambang cadangan
CONF_FLOOR        = 0.70  # dipakai di _assemble_field, bukan di route_decision

def route_decision(pred_name, pred_dob, pred_addr, is_malaysia, triage_feats, triage_clf,
                    crop_blur_score, crop_brightness, crop_dark_pixel_ratio,
                    conf_name, conf_dob, conf_addr):
    if crop_blur_score < 30 or crop_brightness < 30 or crop_dark_pixel_ratio > 0.80:
        return 'REJECT'

    name_valid, _ = validate_name(pred_name)
    dob_valid, _  = validate_date(pred_dob)
    if not name_valid or not dob_valid:
        return 'HUMAN_REVIEW'

    if triage_feats is not None and triage_clf.predict(triage_feats)[0] == 0:
        return 'REJECT'

    min_conf = min(conf_name, conf_dob)
    if is_malaysia:
        min_conf = min(min_conf, conf_addr)
    if min_conf >= CONF_AUTO_APPROVE:
        return 'AUTO_APPROVE'
    return 'HUMAN_REVIEW'
```

Urutannya sengaja disusun dari yang paling murah dan deterministik dulu (cek angka kualitas gambar), baru ke validasi format hasil ekstraksi, baru ke model Triage Classifier, dan terakhir baru ke ambang batas confidence gabungan. `CONF_AUTO_APPROVE` diset di 0.75, artinya dokumen baru bisa lolos otomatis penuh (`AUTO_APPROVE`) kalau confidence OCR paling lemah dari semua field yang diekstrak masih di atas 0.75. Kalau tidak memenuhi tapi tetap valid formatnya, statusnya jadi `HUMAN_REVIEW` (perlu dicek petugas), dan kalau kualitas gambarnya memang terlalu buruk dari awal, langsung `REJECT`. Perlu dicatat, `CONF_HUMAN_REVIEW = 0.45` dideklarasikan sebagai konstanta di kode tapi **tidak pernah dirujuk** di dalam `route_decision` itu sendiri — kemungkinan sisa dari iterasi desain sebelumnya atau disiapkan untuk pengembangan lanjutan (misalnya ambang tambahan buat memisahkan `HUMAN_REVIEW` dari `REJECT` berdasarkan confidence, bukan cuma validitas format).

**3.9 Loop Pelatihan dan Evaluasi 5-Fold**

Seluruh langkah di atas (Origin Classifier → Triage Classifier → Field Classifier + Optuna → Template Router → ekstraksi field → routing) dijalankan berurutan **di dalam satu loop per fold**, memakai fold yang sama persis dari `fold_indices.pkl` yang dikunci di Notebook 1. Setiap gambar dievaluasi tepat sekali sebagai bagian dari fold uji (out-of-fold), sehingga akumulasi kelima fold mencakup seluruh 632 gambar tanpa duplikasi.

Selain metrik evaluasi numerik, loop ini juga menghasilkan dua output tambahan yang bermanfaat untuk audit manual, **ditulis hanya pada fold terakhir** supaya tidak tertulis 5 kali:

1. **Folder crosscheck** (`dataset/crosscheck/Accepted/`, `Rejected/`, `Manual_Review/`) — gambar KTP hasil crop disalin ke folder sesuai status routing akhirnya (`REJECT` → folder ditolak triage; `AUTO_APPROVE` → diterima; selain itu → perlu tinjauan manual).
2. **CSV crosscheck** (`dataset/crosscheck/ocr_predictions_crosscheck.csv`) — satu baris per gambar (akumulasi dari `all_predictions_log` di seluruh 5 fold, jadi totalnya 632 baris), berisi kolom `filename`, `routing`, `pred_name`/`gt_name`, `pred_dob`/`gt_dob`, `pred_addr`/`gt_addr`, `cer_name`, `cer_dob`, dan `template`, diurutkan supaya baris `REJECT` dan `HUMAN_REVIEW` muncul lebih dulu — memudahkan reviewer manusia fokus ke kasus yang paling butuh perhatian.

**3.10 Ringkasan Evaluasi Menyeluruh**

Metrik akhir yang dilaporkan mencakup:

- **F1 Origin Classifier** — seberapa akurat model menebak asal dokumen, dirata-rata dari precision/recall/F1 tiap fold.
- **Triage FRR/FAR** — dijelaskan di bagian 3.4.
- **CER dan F1 Token untuk nama** — seberapa dekat hasil ekstraksi nama ke ground truth.
- **Exact Accuracy dan Extracted Rate untuk tanggal lahir** — seberapa sering tanggal lahir terbaca sama persis, dan seberapa sering field ini berhasil ditemukan sama sekali (`dob_extracted`, dari `bool(pred_dob.strip())`).
- **CER dan F1 untuk alamat** (khusus dokumen Malaysia, karena dokumen lain tidak punya alamat di ground truth).
- **STP Rate (Straight-Through Processing Rate)** — persentase dokumen yang lolos `AUTO_APPROVE` tanpa perlu campur tangan manusia sama sekali, dihitung dari seluruh baris `all_metrics_df` (termasuk yang di-REJECT triage).
- **Country Detection Accuracy** — khusus untuk dokumen yang template-nya terdeteksi lewat jalur `fallback_<negara>`, dibandingkan `country_detected` terhadap kolom referensi `negara` di ground truth. Metrik ini secara eksplisit hanya menilai jalur fallback country-alias, bukan MRZ/EU license/MyKad.
- Evaluasi juga dipecah per subset **Malaysia vs Luar Negeri** secara terpisah, supaya performa tidak tertutupi rata-rata gabungan.

Selain angka mentah, notebook juga menjalankan **pengecekan standar akurasi minimum** eksplisit terhadap tiga ambang batas keberterimaan bisnis:

```python
print(f"  CER Nama < 15%   : {'LULUS' if cer_nama < 0.15 else 'GAGAL'} ({cer_nama:.1%})")
print(f"  Exact DOB > 70%  : {'LULUS' if exact_dob > 0.70 else 'GAGAL'} ({exact_dob:.1%})")
print(f"  CER Alamat < 25% : {'LULUS' if cer_alamat < 0.25 else 'GAGAL'} ({cer_alamat:.1%})")
```

**Tiga catatan insight bisnis** juga didokumentasikan langsung sebagai komentar di kode pada titik ini, menjelaskan sumber-sumber error yang sifatnya inheren pada fisik dokumen (bukan bug pipeline):

1. **Embossed text ceiling.** Nama yang dicetak timbul (laser-personalized embossing) mengandalkan bayangan fisik yang bertabrakan secara optik dengan latar microprint yang rata, menghasilkan "sup karakter" pada field nama meski teks boilerplate di sekitarnya terbaca sempurna. CER nama karena itu punya batas bawah fisik yang tidak bisa dihilangkan murni lewat preprocessing software.
2. **Fitur keamanan template.** Kartu spesimen (mis. template "Erika Mustermann") konsisten berkumpul di skor confidence rendah (0.37-0.85) karena memakai fitur anti-reproduksi keamanan tinggi seperti pola guilloche yang memang didesain untuk mengalahkan pemindaian optik. Ini perilaku yang diharapkan, bukan kegagalan pipeline.
3. **Silau hologram & kerusakan fisik.** Gambar dengan silau pencahayaan parah atau blur berat adalah kegagalan pengambilan gambar fisik yang memang diharapkan. Pipeline berhasil mengidentifikasi dan mengarahkan gambar-gambar yang tidak layak baca ini ke folder crosscheck Rejected/Manual_Review, menghemat waktu proses dan mencegah korupsi data di tahap hilir — ini harus dibingkai sebagai pipeline bekerja dengan benar, bukan sebagai error sistem.

**3.11 Matriks Kebingungan Klasifikasi**

Menggunakan model dan data dari fold terakhir sebagai sampel, dicetak `classification_report` (precision/recall/F1 per kelas: `other`, `name`, `birth_date`, `address`) dan digambar confusion matrix 4x4 untuk melihat kelas mana yang paling sering tertukar oleh Field Classifier.

**3.12 Analisis Feature Importance dengan SHAP**

Bagian ini menganalisis fitur numerik apa yang paling diprioritaskan Field Classifier untuk tiap kelas prediksi, memakai `shap.TreeExplainer` pada model LightGBM dari fold terakhir.

Notebook mencatat penjelasan eksplisit soal multikolinearitas: korelasi tinggi antar fitur (misalnya `y_rel_scaled` dan `line_rank_norm` yang sama-sama mengukur posisi vertikal) **tidak membatalkan** validitas SHAP untuk model berbasis tree, karena SHAP menghitung kontribusi marginal tiap fitur dari semua kemungkinan subset fitur sehingga nilainya tetap konsisten dan additive — dampaknya cuma kepentingan bisa "terbagi" antara dua fitur yang berkorelasi tinggi, sehingga keduanya tampak kurang penting secara individual walau sebenarnya kelompok fiturnya (mis. "posisi vertikal" secara keseluruhan) sangat menentukan.

Kode juga menormalisasi format output SHAP supaya konsisten antar versi library (`shap` versi baru mengembalikan array 3D `(n_samples, n_features, n_classes)`, versi lama mengembalikan list per kelas, kadang dengan kolom bias tambahan yang perlu dipotong).

Untuk tiap kelas (kecuali `other`), dihasilkan dua visualisasi:
- **Bar chart** — mean |SHAP value| per fitur, diurutkan dari paling penting.
- **Beeswarm plot** — menunjukkan arah pengaruh sekaligus besarannya (titik merah = nilai fitur tinggi, biru = nilai fitur rendah, memengaruhi probabilitas kelas itu ke arah mana).

Seluruh nilai mean |SHAP| per fitur per kelas diekspor ke `shap_feature_importance.csv`, dan tiap grafik disimpan sebagai PNG (`shap_{class}_bar.png`, `shap_{class}_beeswarm.png`).

**3.13 Visualisasi Semantik Hasil Ekstraksi**

Untuk validasi visual akhir, dipilih sampel gambar yang **representatif lintas status routing dan asal dokumen** — untuk tiap status (`AUTO_APPROVE`, `HUMAN_REVIEW`, `REJECT`), diambil satu contoh dari Malaysia dan satu dari luar negeri kalau tersedia (bukan sampel acak).

Pada tiap gambar sampel digambar ulang:
1. Seluruh segmen yang diprediksi `other` sebagai kotak abu-abu tipis (kontras minimal, sekadar penanda).
2. Kotak bounding box gabungan (mengelilingi seluruh segmen penyusun) untuk tiap field final (`name`, `birth_date`, `address`) dengan warna sesuai `COLOR_MAP` (merah/biru/hijau), dilabeli dengan **teks hasil ekstraksi akhir** (bukan cuma nama kelasnya) — kalau field classifier gagal menandai segmen manapun untuk field itu tapi fallback scanner tetap berhasil mengekstrak (misal DOB dari fallback tanggal), sistem mencari mundur segmen `other` mana yang teksnya cocok dengan hasil akhir itu supaya kotaknya tetap bisa digambar.

Karena kolom `box` di CSV berisi representasi string dari array numpy yang bisa terpotong (numpy mencetak `...` untuk array panjang, sehingga cuma sudut pertama dan terakhir yang tersimpan penuh), ada fungsi pengurai khusus `_parse_box` yang merekonstruksi 4 sudut persegi panjang dari 2 sudut yang selamat, dengan penanganan `try/except` kalau parsing tetap gagal.

Hasilnya disimpan sebagai gambar ke `dataset/semantic_visualizations/semantic_{filename}` sekaligus ditampilkan inline di notebook, berjudul `[asal dokumen] [status routing] {filename}`.

**3.14 Menyimpan Laporan Akhir**

Seluruh baris metrik dari 5 fold (`all_metrics_df`) disimpan ke `dataset/evaluation_results.csv`, ditutup dengan cetak ringkas jumlah rekaman dan STP Rate final ke layar.

---

### Notebook 4: Eksperimen Perbandingan

> Notebook ini **tidak diunggah** untuk pengecekan detail README kali ini, jadi bagian di bawah masih memuat ringkasan dari draft README sebelumnya dan belum diverifikasi ulang baris-per-baris seperti Notebook 1-3 di atas. Kalau ingin bagian ini juga dijamin 100% akurat, unggah `4_Model_Comparison.ipynb` (atau `plan/4_model_comparison.py`) dan saya bisa cek ulang dengan cara yang sama.

Tujuan notebook ini membuktikan kenapa pilihan metode di pipeline utama (Notebook 1-3) memang lebih baik dibanding alternatif lain yang lebih umum atau lebih sederhana, dengan perbandingan yang adil (fitur sama, fold sama, data sama).

**4.1 Lokalisasi: DocAligner vs Heuristik CV Klasik**

Sebagai pembanding DocAligner, dibangun metode klasik berbasis computer vision tanpa deep learning, yaitu deteksi tepi Canny diikuti pencarian kontur.

```python
def localize_card_classical(img, target_ratio=1.586):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)
    edges   = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.1 * img_area or area > 0.98 * img_area:
            continue
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            best_quad = approx.reshape(4, 2).astype(np.float32)
            break
```

`approxPolyDP` menyederhanakan bentuk kontur jadi polygon dengan lebih sedikit titik, dengan toleransi galat 2% dari keliling kontur. Kalau hasil sederhananya persis 4 titik, bentuk itu dianggap kandidat kartu. Ini pendekatan klasik yang umum dipakai di aplikasi scanner dokumen sebelum era deep learning, tapi punya kelemahan mendasar, dia mengasumsikan kartu punya sudut yang tajam dan tepi yang kontras jelas dengan latar belakang, sehingga sering gagal di foto dengan latar belakang senada warna kartu atau kartu dengan sudut membulat.

Perbandingan dilakukan dari beberapa sisi, tingkat keberhasilan menemukan kontur kartu, kualitas hasil crop (blur, edge density, contrast), dan yang paling penting, dampaknya ke akurasi OCR di tahap berikutnya (CER nama).

**4.2 OCR Engine: PaddleOCR vs EasyOCR vs Tesseract**

Ketiga mesin OCR dijalankan pada gambar crop yang sama supaya perbandingannya adil, lalu diukur CER, WER, dan waktu proses. Karena beberapa engine bisa membaca lebih banyak teks di luar field target (misalnya ikut membaca nomor IC atau watermark), dipakai teknik **best match window** supaya perbandingan tidak merugikan engine yang "terlalu rajin" membaca.

```python
def _best_match_window(hyp, ref_str, distance_fn):
    hyp_words = str(hyp).strip().lower().split()
    n = max(1, len(ref_str.split()))
    best = float('inf')
    for i in range(len(hyp_words) - n + 1):
        window = ' '.join(hyp_words[i:i+n])
        best = min(best, distance_fn(window, ref_str))
    return best
```

Cara bacanya, dicari potongan kata berturutan sepanjang jumlah kata ground truth dari seluruh hasil OCR yang paling mirip, baru dihitung CER-nya dari potongan terbaik itu, bukan dari keseluruhan hasil OCR yang panjang.

**4.3 Model Klasifikasi Field: LightGBM vs RandomForest vs XGBoost**

Perbandingan paling apple-to-apple di notebook ini, karena ketiga model dilatih di fitur (`FEATURE_COLS`) dan fold split yang identik dengan Notebook 3, jadi selisih performa murni berasal dari algoritmanya, bukan dari perbedaan data.

**4.4 Paradigma Representasi: Spasial+LightGBM vs TF-IDF+LogReg vs Regex**

Pertanyaan yang mau dijawab di sini, apakah isi teksnya sendiri sudah cukup untuk menebak jenis field, atau posisi dan bentuk teks yang justru lebih menentukan. Ditambahkan juga baseline aturan tangan murni tanpa machine learning sama sekali sebagai batas bawah pembanding.

```python
def regex_classify_segment(text):
    t = str(text).strip().upper()
    if re.search(r'\d{4}-\d{2}-\d{2}', t) or re.fullmatch(r'\d{4}', t):
        return 'birth_date'
    addr_kw = ['JALAN', 'JLN', 'LORONG', 'LRG', 'KAMPUNG', 'KG', 'TAMAN', 'TMN', 'SELANGOR', 'JOHOR']
    if any(kw in t for kw in addr_kw):
        return 'address'
    alpha_ratio = sum(c.isalpha() for c in t) / max(len(t), 1)
    digit_ratio = sum(c.isdigit() for c in t) / max(len(t), 1)
    if alpha_ratio > 0.7 and digit_ratio < 0.1 and len(t) >= 4:
        return 'name'
    return 'other'
```

Hipotesis di balik pendekatan spasial, posisi field di kartu identitas jauh lebih konsisten dibanding isi katanya, karena nama orang dari negara berbeda-beda tidak punya pola kata yang seragam, sementara posisinya di kartu (biasanya di bagian atas, alamat di bagian bawah) jauh lebih dapat diprediksi lintas negara setelah kartu berhasil diluruskan lewat proses lokalisasi.

**4.5 Dampak Triage terhadap Kualitas Crop**

Membandingkan berapa persen gambar hasil crop yang bakal ditolak sistem triage (blur score di bawah 30) antara DocAligner dan metode klasik, untuk menunjukkan bahwa kualitas pemotongan berdampak langsung ke jumlah dokumen yang harus direview manual.

**4.6 Ablation Study Fitur**

Melatih ulang LightGBM sambil mematikan kelompok fitur tertentu secara bergantian untuk melihat seberapa besar F1 score turun, dibandingkan dengan model yang memakai semua fitur.

```python
spatial_features = ['y_rel_scaled', 'x_rel_scaled', 'width_rel_scaled', 'line_rank_norm']
regex_features = ['has_date_pattern', 'has_address_kw', 'has_state_kw', 'has_postcode', 'has_bin_binti', ...]
```

Dicoba tiga kondisi, model dengan semua fitur, model tanpa fitur spasial, dan model tanpa fitur berbasis pola/regex. Selisih F1 antara model penuh dan model yang kehilangan satu kelompok fitur menunjukkan seberapa penting kontribusi kelompok fitur itu terhadap performa keseluruhan.

**4.7 Estimasi Latensi End-to-End**

Menjumlahkan waktu proses tiap tahap (lokalisasi, OCR, inferensi model klasifikasi) untuk mendapat estimasi kasar berapa lama sistem memproses satu KTP dari awal sampai akhir.

**4.8 Ringkasan Akhir**

Semua hasil perbandingan di atas dirangkum jadi satu tabel ringkas, mencantumkan metode mana yang menang di tiap aspek beserta angka pendukungnya, supaya bisa langsung dikutip di laporan atau slide presentasi final project.

---

## Lampiran A — Daftar Lengkap Fitur yang Direkayasa

Ini jawaban langsung untuk pertanyaan "apakah rekayasa fitur sudah mencakup semua fitur yang dibuat": berikut inventaris **semua** kolom yang dihasilkan proses rekayasa fitur di Notebook 1 dan 2, dan status pemakaiannya di model Notebook 3.

### A.1 Level Dokumen (`master_features.csv`, dari Notebook 1 & 2)

| Fitur | Dibuat di | Dipakai sebagai input model? |
|---|---|---|
| `has_address`, `doc_origin_weak_label` (versi awal di NB1) | NB1 | Tidak — `doc_origin_weak_label` dihapus lagi di NB2 (lihat 2.5) |
| `doc_origin_encoded_target` | NB1/NB2 (dari `df_gt`) | **Ya** — target Origin Classifier |
| `identity_key` | NB1 | Tidak — cuma untuk grouping split data |
| `negara` | NB1 | Tidak — referensi validasi saja (dipakai di evaluasi NB3, bukan fitur) |
| `localization_method` | NB2 | Tidak — log diagnostik |
| `crop_blur_score`, `crop_brightness`, `crop_contrast`, `crop_edge_density`, `crop_dark_pixel_ratio`, `crop_aspect_ratio` | NB2 | Sebagian — 4 di antaranya (log-blur, brightness, contrast, edge density) masuk ke PCA; `crop_dark_pixel_ratio` masuk `TRIAGE_FEATURES` dan aturan keras `route_decision` |
| `skew_angle` | NB2 | **Ya** — `TRIAGE_FEATURES` |
| `crop_blur_score_log` | NB2 | **Ya** — `TRIAGE_FEATURES`, dan input PCA |
| `quality_pc1` | NB2 (PCA) | **Ya** — `TRIAGE_FEATURES` |
| `quality_pc2` | NB2 (PCA) | Tidak — cuma untuk visualisasi 2D |
| `has_mykad_keyword`, `has_ic_number_pattern`, `has_bin_binti_any_segment` | NB2 | Tidak masuk `FEATURE_COLS`/`TRIAGE_FEATURES` — tersimpan di CSV untuk eksplorasi |
| `n_segments`, `avg_segment_len` | NB2 | Tidak — diagnostik saja |
| `avg_ocr_conf` | NB2 | Tidak dipakai langsung sebagai fitur model, tapi jadi basis kalibrasi ambang batas triage |

### A.2 Level Segmen Teks (`ocr_segments_features.csv`, dari Notebook 2)

| Fitur | Dipakai di `FEATURE_COLS` (Field Classifier)? |
|---|---|
| `y_rel_scaled`, `x_rel_scaled`, `width_rel_scaled` | **Ya** |
| `text_len`, `word_count` | **Ya** |
| `ocr_conf` | **Ya** |
| `alpha_ratio`, `digit_ratio` | **Ya** |
| `is_all_caps` | **Ya** |
| `has_date_pattern`, `has_address_kw`, `has_state_kw`, `has_postcode`, `has_bin_binti` | **Ya** |
| `line_rank_norm` | **Ya** |
| `doc_origin_encoded` (prediksi Origin Classifier, ditempel per fold) | **Ya** |
| `y_rel`, `x_rel`, `width_rel` (versi belum di-scale) | Tidak — cuma basis sebelum `MinMaxScaler` |
| `is_spatial_dummy` | Tidak — flag kualitas data saja |
| `line_rank` (versi belum dinormalisasi) | Tidak |
| `label` | Tidak — ini target (`y`), bukan fitur |
| `box`, `text` | Tidak — dipakai untuk parsing template/visualisasi, bukan fitur numerik model |
| `has_mykad_keyword`, `has_ic_number_pattern`, `quality_pc1`, `skew_angle` (disalin dari level dokumen) | Tidak masuk `FEATURE_COLS` walau tersimpan di tiap baris segmen |

**Total fitur yang benar-benar masuk ke Field Classifier: 16** (`FEATURE_COLS`). **Total fitur yang masuk ke Triage Classifier: 7** (`TRIAGE_FEATURES`). Sisanya adalah fitur pendukung, diagnostik, atau bahan EDA yang direkayasa tapi sengaja tidak diumpankan ke model manapun.

## Lampiran B — Konstanta Ambang Batas Penting

| Konstanta | Nilai | Lokasi | Fungsi |
|---|---|---|---|
| `VALIDATED_OVERLAP_THRESHOLD` | 0.5 | NB2 | Ambang skor token overlap minimum supaya segmen dilabeli `name`/`address` (bukan `other`) |
| Threshold blur EDA (pseudo-CER) | dinamis, fallback 0.85 | NB2.9 | Cuma untuk visualisasi, tidak dipakai model |
| Threshold confidence triage per-fold | dinamis, fallback 0.85 | NB3.4 | Dasar label `is_readable` untuk melatih Triage Classifier |
| Aturan keras reject (blur) | `crop_blur_score < 30` | NB3.4 & NB3.8 | Auto-reject sebelum model triage ditanya |
| Aturan keras reject (gelap) | `crop_brightness < 30` | NB3.8 | Bagian dari `route_decision` |
| Aturan keras reject (dominan gelap) | `crop_dark_pixel_ratio > 0.80` | NB3.8 | Bagian dari `route_decision` |
| `CONF_FLOOR` | 0.70 | NB3.7 | Ambang minimum confidence OCR segmen sebelum ikut digabung jadi field |
| `CONF_AUTO_APPROVE` | 0.75 | NB3.8 | Ambang confidence gabungan minimum untuk status `AUTO_APPROVE` |
| `CONF_HUMAN_REVIEW` | 0.45 | NB3.8 | Dideklarasikan, **tidak dipakai** di `route_decision` saat ini |
| Rentang tahun valid | 1900-2015 | NB3.2 (`validate_date`) | Validasi kewajaran tahun lahir |
| Ambang pengelompokan baris | selisih `y_rel_scaled` > 0.02 | NB2.7 & NB3.7 | Menentukan dua segmen ada di baris yang sama atau tidak |
| Optuna | 5 trial × 5 fold = 25 total | NB3.5 | Pencarian hyperparameter LightGBM |
| Standar akurasi minimum (acceptance criteria) | CER Nama < 15%, Exact DOB > 70%, CER Alamat < 25% | NB3.10 | Ambang lulus/gagal yang dicetak eksplisit di laporan evaluasi |