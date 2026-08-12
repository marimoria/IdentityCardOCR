# Identity Card OCR - Final Project Data Science Academy (DSA) COMPFEST 18
 
Pipeline Optical Character Recognition (OCR) untuk ekstraksi informasi terstruktur dari dokumen identitas (KTP) guna mendukung proses verifikasi Know Your Customer (KYC), dikembangkan sebagai Final Project Data Science Academy COMPFEST 18.
 
Repository ini berisi implementasi pipeline Optical Character Recognition (OCR) yang dirancang untuk mengekstraksi informasi kunci dari dokumen identitas (KTP), meliputi nama, tanggal lahir, dan alamat. Proyek ini disusun sebagai bagian dari proses Know Your Customer (KYC) yang menjadi kebutuhan wajib bagi lembaga keuangan dan perbankan dalam rangka pemenuhan regulasi anti pencucian uang (Anti-Money Laundering/AML).
 
Pipeline dibangun untuk mengatasi tantangan variasi tinggi pada dokumen identitas, termasuk perbedaan kualitas gambar, kondisi pencahayaan, serta kondisi fisik dokumen yang tidak ideal (lusuh atau terlipat). Proses pengembangan mencakup tahapan business understanding, data understanding dan exploratory data analysis (EDA), data preprocessing, pemodelan dan optimisasi, evaluasi performa, hingga penyusunan rekomendasi bisnis berbasis hasil ekstraksi.
 
Sesuai dengan ketentuan kompetisi, pipeline pada proyek ini dibangun tanpa menggunakan komponen berbasis Large Language Model (LLM) maupun Vision-Language Model (VLM), serta tanpa proses anotasi gambar secara manual. Evaluasi performa model dilakukan dengan mengacu pada data ground truth yang telah disediakan.
 
## Data Access

Dataset tidak dibagikan secara publik. Kolaborator dapat mengakses dataset melalui link drive yang dibagikan oleh pihak COMPFEST lalu unzip pada folder `dataset/` seperti berikut:

```
dataset/
├── ground_truth.csv
└── images/
    ├── image_001.jpg
    ├── image_002.jpg
    ├── ...
    └── image_732.jpg
plan/
```

## Proses Exploratory Data Analysis (EDA)

Tahap Exploratory Data Analysis dilakukan pada file `1_EDA_and_Data_Understanding.ipynb` untuk memahami struktur dan kualitas dataset sebelum membangun model. Berikut adalah langkah utama yang dilakukan pada tahap ini:

1. **Normalisasi Data Tabular**: Memperbaiki format file ground truth agar mudah diproses, terutama memisahkan baris yang mengandung alamat dengan format yang tidak standar.
2. **Identifikasi Asal Dokumen**: Menemukan bahwa dataset terdiri dari dokumen Malaysia dan luar negeri. Hanya dokumen Malaysia yang memiliki informasi alamat.
3. **Analisis Kualitas Gambar**: Mengekstrak metrik numerik seperti tingkat blur, kecerahan, dan kontras dari setiap gambar. Hal ini penting untuk menentukan teknik preprocessing yang sesuai nantinya.
4. **Analisis Pola Teks**: Mempelajari format tanggal lahir dan pola penamaan khas (seperti penggunaan kata BIN atau BINTI) yang akan sangat berguna untuk melatih model klasifikasi teks.
5. **Strategi Pembagian Data**: Mengidentifikasi adanya banyak foto untuk satu orang yang sama. Pembagian data latih dan uji kemudian dirancang dengan memisahkan identitas secara ketat agar model dievaluasi secara adil tanpa menghafal wajah atau nama.
6. **Evaluasi OCR Dasar**: Menjalankan mesin OCR langsung pada gambar mentah tanpa modifikasi. Hasilnya membuktikan bahwa program perlu memotong area latar belakang dan menjernihkan gambar agar hasil pembacaan teks menjadi akurat.
