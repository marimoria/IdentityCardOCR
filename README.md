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
```
