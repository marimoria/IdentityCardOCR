# %% [markdown]
# # EDA Identity Card OCR
# Final Project Data Science Academy COMPFEST 18
# 
# Notebook ini berisi proses Exploratory Data Analysis untuk proyek ekstraksi informasi kartu identitas. Tujuannya adalah memahami struktur dataset, menemukan tantangan nyata, dan menjadi dasar keputusan teknis di tahap preprocessing dan pemodelan.
# 
# Catatan: Metrik kualitas gambar di notebook ini dihitung pada gambar original sebelum dipotong murni untuk keperluan analisis. Metrik post crop yang digunakan sebagai fitur model dihitung ulang di Notebook 2 setelah proses lokalisasi kartu selesai.

# %% [markdown]
# ## 0. Setup

# %%
import os
import re
import csv
import io
import json
import pickle
import warnings
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2

from sklearn.model_selection import StratifiedGroupKFold
import Levenshtein

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
PALETTE = 'muted'
sns.set_palette(PALETTE)
plt.rcParams.update({'figure.figsize': (10, 5), 'axes.titlesize': 13})

DATASET_DIR = 'dataset'
IMG_DIR = os.path.join(DATASET_DIR, 'images')
CSV_PATH = os.path.join(DATASET_DIR, 'ground_truth.csv')
STATS_PATH = os.path.join(DATASET_DIR, 'image_quality_stats.csv')
FOLD_PATH = os.path.join(DATASET_DIR, 'fold_indices.pkl')
BASELINE_CACHE = os.path.join(DATASET_DIR, 'baseline_ocr_cache.json')
NORMALIZED_CSV = os.path.join(DATASET_DIR, 'ground_truth_normalized.csv')

print('Setup selesai.')

# %% [markdown]
# ## 1. Memuat Data
# File ground truth menyimpan label tiap gambar seperti nama, tanggal lahir, dan alamat. Format aslinya tidak standar karena baris dokumen Malaysia memakai tanda kutip ganda, sedangkan dokumen luar negeri tidak. Data ini perlu dinormalisasi lebih dulu sebelum diparse.

# %%
if not os.path.exists(NORMALIZED_CSV):
    print("Normalisasi ground truth mentah...")
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

    with open(NORMALIZED_CSV, 'w', encoding='utf-8', newline='') as f:
        csv.writer(f).writerows(rows)
    print(f'Tersimpan: {NORMALIZED_CSV}')
else:
    print(f'File sudah ada: {NORMALIZED_CSV}')

df = pd.read_csv(NORMALIZED_CSV)
df['address'] = df['address'].astype(str).str.strip().replace(['nan', 'None', ''], np.nan)

_bad_dob = [f'image_{i}.jpg' for i in range(541, 553)]
_fixed   = df['filename'].isin(_bad_dob)
if _fixed.any():
    df.loc[_fixed, 'birth_date'] = '1971-05-12'
    # Tulis ulang ke normalized CSV agar koreksi tersimpan permanen
    df.to_csv(NORMALIZED_CSV, index=False)
    print(f'Koreksi DOB diterapkan pada {_fixed.sum()} baris')

print(f'\nTotal baris: {len(df)}')
print(f'Kolom: {df.columns.tolist()}')

_NEGARA_LIST = (
    ['MALAYSIA'] * 112 +
    ['BRAZIL']   * 20 +
    ['CHILE']    * 20 +
    ['CHINA']    * 20 +
    ['CZECH REPUBLIC'] * 40 +
    ['GERMANY']  * 60 +
    ['ALGERIA']  * 20 +
    ['SPAIN']    * 20 +
    ['FINLAND']  * 40 +
    ['GREECE']   * 20 +
    ['CROATIA']  * 40 +
    ['HUNGARY']  * 20 +
    ['ITALY']    * 20 +
    ['LATVIA']   * 20 +
    ['MACAU']    * 20 +
    ['MOLDOVA']  * 20 +
    ['NORWAY']   * 20 +
    ['POLAND']   * 20 +
    ['PORTUGAL'] * 20 +
    ['TURKEY']   * 20 +
    ['UKRAINE']  * 40
)
assert len(_NEGARA_LIST) == len(df), f'Panjang negara list ({len(_NEGARA_LIST)}) != jumlah baris ({len(df)})'

# Map by sorted filename order (image_001 ... image_632)
_sorted_files = sorted(df['filename'].tolist(), key=lambda x: int(x.split('_')[1].split('.')[0]))
_negara_map   = dict(zip(_sorted_files, _NEGARA_LIST))
df['negara']  = df['filename'].map(_negara_map)
df.to_csv(NORMALIZED_CSV, index=False)
print(f"Kolom 'negara' ditambahkan. Distribusi:")
print(df['negara'].value_counts())

df.head()

# %% [markdown]
# Dataset berhasil dimuat. Terdapat 632 baris data dengan empat kolom utama. Lima baris pertama sudah terlihat bersih, nama dalam huruf kapital penuh, dan tanggal dalam format ISO.

# %% [markdown]
# ## 2. Audit Struktural

# %%
print('Tipe data setiap kolom:')
print(df.dtypes)
print(f'\nJumlah baris duplikat persis: {df.duplicated().sum()}')

# %% [markdown]
# Tiga baris terakhir di data (yang tidak dicetak di sini) menunjukkan nama yang sama dengan gambar berbeda. Hal ini menandakan satu orang bisa memiliki banyak foto. Pembahasan lebih lanjut tentang hal ini ada di bagian Analisis Duplikasi Identitas.

# %% [markdown]
# ## 3. Audit Nilai Kosong
# Kolom alamat kosong sebagian besar terjadi karena dokumen luar negeri tidak menyediakan data tersebut. Kasus ini bukan eror sistem melainkan kekosongan alami. Hal ini berarti kita perlu mendeteksi dokumen mana yang memiliki alamat sebelum mengekstraknya.

# %%
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Jumlah Kosong': missing, 'Persen (%)': missing_pct})
non_zero = missing_df[missing_df['Jumlah Kosong'] > 0]

fig, ax = plt.subplots(figsize=(7, 3))
sns.barplot(data=non_zero.reset_index(), x='Persen (%)', y='index', color='#e67e22', ax=ax)
ax.set_xlabel('Persentase Kosong (%)')
ax.set_ylabel('')
ax.set_title('Kolom dengan Nilai Kosong')
for i, v in enumerate(non_zero['Persen (%)']):
    ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** Terdapat persentase besar pada kolom alamat yang kosong. Hal ini wajar karena sebagian besar data asing memang tidak memuat alamat. Oleh sebab itu data kosong ini dibiarkan apa adanya dan tidak diisi secara paksa.

# %% [markdown]
# ## 4. Rekayasa Kolom Asal Dokumen
# Dokumen yang punya alamat otomatis dilabeli sebagai Malaysia dan sisanya sebagai Luar Negeri. Pengelompokan tambahan berdasarkan identitas (gabungan nama dan tanggal lahir) juga dibuat agar memudahkan pembagian data set pelatihan nanti.

# %%
df['has_address'] = df['address'].notna()
df['doc_origin_weak_label'] = df['has_address'].map({True: 'Malaysia', False: 'Luar Negeri'})
df['identity_key'] = df['name'].astype(str) + '__' + df['birth_date'].astype(str)

origin_summary = df.groupby('doc_origin_weak_label').agg(
    jumlah_gambar=('filename', 'count'),
    jumlah_identitas=('identity_key', 'nunique')
).reset_index()

colors = sns.color_palette(PALETTE, 2)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

origin_counts = df['doc_origin_weak_label'].value_counts()
axes[0].pie(origin_counts, labels=origin_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0].set_title('Proporsi Gambar per Asal Dokumen', pad=15)

x = np.arange(len(origin_summary))
width = 0.35

bars1 = axes[1].bar(x - width/2, origin_summary['jumlah_gambar'], width,
                    label='Jumlah Gambar', color=colors[0], edgecolor='white', linewidth=1.5)
ax2 = axes[1].twinx()
bars2 = ax2.bar(x + width/2, origin_summary['jumlah_identitas'], width,
                label='Jumlah Identitas', color=colors[1], edgecolor='white', linewidth=1.5)

axes[1].set_title('Jumlah Gambar vs Jumlah Identitas', pad=45)
axes[1].set_ylabel('Jumlah Gambar')
ax2.set_ylabel('Jumlah Identitas', color='gray')
axes[1].set_xticks(x)
axes[1].set_xticklabels(origin_summary['doc_origin_weak_label'])
axes[1].grid(axis='y', alpha=0.3)
ax2.grid(False)
axes[1].set_ylim(0, origin_summary['jumlah_gambar'].max() * 1.15)
ax2.set_ylim(0, origin_summary['jumlah_identitas'].max() * 1.15)

for bar, val in zip(bars1, origin_summary['jumlah_gambar']):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + axes[1].get_ylim()[1]*0.015,
                 str(val), ha='center', fontsize=10)
for bar, val in zip(bars2, origin_summary['jumlah_identitas']):
    ax2.text(bar.get_x() + bar.get_width()/2, val + ax2.get_ylim()[1]*0.015,
             str(val), ha='center', fontsize=10)

axes[1].legend([bars1, bars2], ['Jumlah Gambar', 'Jumlah Identitas'],
               loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
plt.tight_layout()
plt.subplots_adjust(top=0.78)
plt.show()

# %% [markdown]
# **Insight:** Data dari luar negeri didominasi oleh sedikit identitas namun masing masing orang memiliki banyak foto. Sebaliknya dokumen Malaysia memiliki porsi jumlah foto per orang yang lebih rata. Pembagian data pelatihan tidak boleh dilakukan secara acak agar foto orang yang sama tidak tersebar ke set latih dan tes sekaligus.

# %% [markdown]
# ## 5. Analisis Duplikasi Identitas
# Variasi jumlah foto per orang dieksplorasi lebih jauh untuk melihat tingkat kemunculan wajah yang sama.

# %%
identity_counts = df.groupby(['identity_key', 'doc_origin_weak_label']).size().reset_index(name='n_gambar')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for origin, color in zip(['Malaysia', 'Luar Negeri'], sns.color_palette(PALETTE, 2)):
    subset = identity_counts[identity_counts['doc_origin_weak_label'] == origin]['n_gambar']
    axes[0].hist(subset, bins=20, alpha=0.7, label=origin, color=color, edgecolor='white')
axes[0].set_xlabel('Jumlah Gambar per Identitas')
axes[0].set_ylabel('Frekuensi')
axes[0].set_title('Distribusi Jumlah Gambar per Identitas')
axes[0].legend()

top15 = identity_counts.nlargest(15, 'n_gambar').copy()
top15['label'] = top15['identity_key'].apply(lambda x: x.split('__')[0][:30])
bar_colors = [sns.color_palette(PALETTE, 2)[0] if o == 'Malaysia'
              else sns.color_palette(PALETTE, 2)[1]
              for o in top15['doc_origin_weak_label']]
axes[1].barh(top15['label'], top15['n_gambar'], color=bar_colors, edgecolor='white')
axes[1].invert_yaxis()
axes[1].set_xlabel('Jumlah Gambar')
axes[1].set_title('Identitas dengan Gambar Terbanyak')
axes[1].tick_params(axis='y', labelsize=8)

from matplotlib.patches import Patch
axes[1].legend(handles=[
    Patch(facecolor=sns.color_palette(PALETTE, 2)[0], label='Malaysia'),
    Patch(facecolor=sns.color_palette(PALETTE, 2)[1], label='Luar Negeri'),
], loc='lower right', fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** Identitas tertentu memuat hingga puluhan sampel foto. Fakta ini menegaskan bahwa metode Stratified Group K Fold adalah cara paling pas guna menghindari kebocoran data pelatihan.

# %% [markdown]
# ## 6. Analisis Format Tanggal Lahir
# Format penulisan waktu lahir diperiksa guna memastikan pola Regex kita nanti dapat mencakup berbagai tipe tulisan.

# %%
def klasifikasi_format(tgl):
    tgl = str(tgl).strip()
    if re.match(r'^\d{4}-\d{2}-\d{2}$', tgl): return 'YYYY Bulan Hari'
    if re.match(r'^\d{4}$', tgl): return 'YYYY saja'
    if re.match(r'^\d{2}\.\d{2}\.\d{4}$', tgl): return 'DD MM YYYY'
    if tgl in ['nan', '', 'None']: return 'Kosong'
    return 'Format lain'

df['format_tanggal'] = df['birth_date'].apply(klasifikasi_format)
fmt_counts = df['format_tanggal'].value_counts()

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(fmt_counts.index, fmt_counts.values,
              color=sns.color_palette(PALETTE, len(fmt_counts)),
              edgecolor='white', linewidth=1.5)
ax.set_xlabel('Format Tanggal')
ax.set_ylabel('Jumlah Baris')
ax.set_title('Distribusi Format Tanggal Lahir di Ground Truth')
for bar, v in zip(bars, fmt_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 1, str(v), ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** Sebagian besar data menggunakan format penuh, namun sebagian kecil hanya memuat tahun. Proses ekstraksi harus bisa menangani berbagai format waktu yang muncul ini.

# %% [markdown]
# ## 7. Distribusi Panjang Teks
# Analisis sebaran panjang nama dan alamat berguna untuk merancang saringan teks di awal proses. Potongan teks yang terlalu pendek bisa langsung ditolak agar menghemat tenaga komputasi.

# %%
df['panjang_nama'] = df['name'].astype(str).apply(len)
df['panjang_alamat'] = df['address'].fillna('').apply(len)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for origin, color in zip(['Malaysia', 'Luar Negeri'], sns.color_palette(PALETTE, 2)):
    subset = df[df['doc_origin_weak_label'] == origin]['panjang_nama']
    axes[0].hist(subset, bins=20, alpha=0.7, label=origin, color=color, edgecolor='white')
axes[0].set_xlabel('Jumlah Karakter')
axes[0].set_ylabel('Frekuensi')
axes[0].set_title('Distribusi Panjang Nama')
axes[0].legend()

addr_data = df[df['has_address']]['panjang_alamat']
axes[1].hist(addr_data, bins=25, color=sns.color_palette(PALETTE, 3)[2], edgecolor='white')
axes[1].axvline(addr_data.mean(), color='red', linestyle='solid', linewidth=1.5,
                label=f'Rata rata: {addr_data.mean():.0f} karakter')
axes[1].set_xlabel('Jumlah Karakter')
axes[1].set_title('Distribusi Panjang Alamat Malaysia')
axes[1].legend()

df.boxplot(column='panjang_nama', by='doc_origin_weak_label', ax=axes[2],
           boxprops=dict(color='steelblue'), medianprops=dict(color='red', linewidth=2))
axes[2].set_title('Panjang Nama per Asal Dokumen')
axes[2].set_xlabel('Asal Dokumen')
axes[2].set_ylabel('Jumlah Karakter')
plt.suptitle('')
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** Nama pada dokumen Malaysia lebih panjang karena kebiasaan menyertakan gelar seperti binti atau sejenisnya. Alamat Malaysia juga memuat banyak karakter. Teks dari hasil mesin pembaca (OCR) yang terlalu pendek bisa diabaikan dengan aman sebagai noise.

# %% [markdown]
# ## 8. Analisis Kualitas Gambar Asli
# Evaluasi ini dilakukan pada gambar sebelum proses pemotongan (crop). Tujuannya adalah melihat variasi kualitas foto yang akan menantang ketangguhan sistem.

# %%
if os.path.exists(STATS_PATH):
    stats_df = pd.read_csv(STATS_PATH)
    print(f'Memuat stats dari cache: {len(stats_df)} gambar')
else:
    print('Menghitung metrik kualitas gambar...')
    img_stats = []
    for fname in df['filename'].tolist():
        path = os.path.join(IMG_DIR, fname)
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
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
    stats_df = pd.DataFrame(img_stats)
    stats_df = stats_df.merge(
        df[['filename', 'doc_origin_weak_label', 'has_address', 'identity_key']],
        on='filename', how='left'
    )
    stats_df['quality_cat'] = pd.cut(
        stats_df['blur_score'],
        bins=[0, 50, 300, np.inf],
        labels=['Buram', 'Sedang', 'Tajam']
    )
    stats_df.to_csv(STATS_PATH, index=False)
    print(f'Selesai. {len(stats_df)} gambar diproses.')

# %% [markdown]
# **Insight:** Skor ketajaman memiliki rentang variasi yang ekstrem. Banyak foto yang buram parah, sementara sebagian kecil lainnya sangat tajam. Perbedaan ekstrem ini menjadikan penyaringan kualitas gambar sangat mutlak dilakukan sebelum tahap pembacaan karakter.

# %% [markdown]
# ## 9. Distribusi Metrik Kualitas Gambar
# Sebaran nilai metrik ini diplot untuk melihat pola kemiringan datanya yang mungkin membutuhkan penyesuaian logaritma nantinya.

# %%
metrik = [
    ('blur_score', 'Blur Score Laplacian Variance'),
    ('brightness', 'Brightness Rata Rata Piksel'),
    ('contrast', 'Contrast Std Dev Piksel'),
    ('edge_density', 'Edge Density Canny Mean'),
    ('file_kb', 'Ukuran File KB'),
    ('aspect_ratio', 'Aspect Ratio Lebar per Tinggi'),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (col, label), color in zip(axes.flat, metrik, sns.color_palette(PALETTE, len(metrik))):
    data = stats_df[col].dropna()
    ax.hist(data, bins=35, color=color, alpha=0.85, edgecolor='white', linewidth=0.7)
    ax.axvline(data.median(), color='black', linestyle='solid', linewidth=1.5,
               label=f'Median: {data.median():.1f}')
    ax.set_title(label)
    ax.set_ylabel('Frekuensi')
    ax.legend(fontsize=8)

plt.suptitle('Distribusi Metrik Kualitas Gambar Original', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** Distribusi nilai ketajaman dan ukuran gambar (KB) sangat menumpuk di sisi kiri grafik. Untuk menormalkan bentuk distribusinya, sistem perlu memakai transformasi logaritma pada sesi berikutnya.

# %% [markdown]
# ## 10. Kualitas Gambar per Asal Dokumen
# Karakteristik kelayakan foto seringkali berbeda antar negara. Informasi ini penting agar filter kualitas tetap adil dalam menilai gambar KTP dari negara manapun.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for origin, color in zip(['Malaysia', 'Luar Negeri'], sns.color_palette(PALETTE, 2)):
    subset = stats_df[stats_df['doc_origin_weak_label'] == origin]
    axes[0].scatter(subset['brightness'], subset['blur_score'],
                    alpha=0.4, s=15, label=origin, color=color)

axes[0].axhline(50, color='orange', linestyle='solid', linewidth=1.5, label='Batas Buram 50')
axes[0].axvline(80, color='gray', linestyle='dotted', linewidth=1.5, label='Batas Gelap 80')
axes[0].set_xlabel('Brightness')
axes[0].set_ylabel('Blur Score')
axes[0].set_title('Brightness vs Ketajaman per Asal Dokumen')
axes[0].legend(fontsize=9)

quality_origin = stats_df.groupby(['doc_origin_weak_label', 'quality_cat']).size().unstack(fill_value=0)
quality_origin.plot(kind='bar', ax=axes[1],
                    color=sns.color_palette(PALETTE, 3), edgecolor='white', linewidth=1.2)
axes[1].set_xlabel('Asal Dokumen')
axes[1].set_ylabel('Jumlah Gambar')
axes[1].set_title('Distribusi Kategori Kualitas per Asal Dokumen')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(title='Kualitas')

plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** KTP Malaysia rata-rata jauh lebih tajam dibandingkan sampel dari luar negeri. Penyaring kualitas pada sistem harus menggunakan batas angka yang dinamis agar tidak terus-menerus menolak dokumen luar negeri.

# %% [markdown]
# ## 11. Korelasi Antar Metrik Kualitas
# Nilai skor kualitas banyak yang berkorelasi tinggi satu sama lain. Atribut yang tumpang tindih ini dapat dikompresi agar proses komputasi lebih ringan.

# %%
corr_cols = ['blur_score','brightness','contrast','edge_density',
             'dark_pixel_ratio','file_kb','aspect_ratio']
corr = stats_df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, mask=mask, linewidths=0.5, vmin=-1, vmax=1)
ax.set_title('Matriks Korelasi Metrik Kualitas Gambar Original')
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** Tingkat kejelasan foto sangat berkaitan dengan kepadatan tepian huruf. Variabel yang saling menimpa ini layak dipadatkan menjadi satu parameter lewat metode Principal Component Analysis (PCA) nanti.

# %% [markdown]
# ## 12. Sampel Visual per Kategori Kualitas
# Bukti visual perlu diperiksa untuk memastikan bahwa hitungan matematika pada sistem sejalan dengan penilaian mata manusia terhadap gambar.

# %%
categories = ['Buram', 'Sedang', 'Tajam']
cat_colors = sns.color_palette(PALETTE, 3)

for cat, cat_color in zip(categories, cat_colors):
    cat_df = stats_df[stats_df['quality_cat'] == cat].dropna(subset=['blur_score'])

    if cat == 'Buram':
        sample_rows = cat_df.nsmallest(3, 'blur_score')
    elif cat == 'Tajam':
        sample_rows = cat_df.nlargest(3, 'blur_score')
    else:
        med = cat_df['blur_score'].median()
        sample_rows = cat_df.iloc[(cat_df['blur_score'] - med).abs().argsort()[:3]]

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

    for row_idx, (_, row) in enumerate(sample_rows.iterrows()):
        img = cv2.imread(os.path.join(IMG_DIR, row['filename']))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        fname_s = (row['filename'][:12] + '..') if len(row['filename']) > 14 else row['filename']

        ax_img = fig.add_subplot(gs[0, row_idx])
        ax_img.imshow(img_rgb)
        ax_img.set_title(
            f"{fname_s}\nBlur {row['blur_score']:.0f} | Terang {row['brightness']:.0f}",
            fontsize=10, color=cat_color, fontweight='bold', pad=8
        )
        ax_img.axis('off')

        ax_hist = fig.add_subplot(gs[1, row_idx])
        ax_hist.hist(gray.ravel(), bins=64, color=cat_color, alpha=0.7, edgecolor='none')
        ax_hist.set_xlim([0, 255])
        ax_hist.set_yticks([])
        ax_hist.set_xlabel('Intensitas Gelap ke Terang', fontsize=9)
        ax_hist.tick_params(labelsize=8)

    fig.suptitle(f'Audit Visual KTP: Kategori {cat}', fontsize=14, fontweight='bold', color=cat_color)
    plt.show()

# %% [markdown]
# **Insight:** Kategori kualitas yang dihitung oleh komputer sesuai dengan pandangan mata manusia. Gambar buram terlihat keruh dengan grafik warna abu-abu yang merata. Sebaliknya dokumen tajam memiliki grafik warna yang mencolok tajam di area teks hitam pekat.

# %% [markdown]
# ## 13. Orientasi Gambar
# Kecondongan format potret atau lanskap memiliki pengaruh besar. Kamera akan merekam banyak area latar belakang tak berguna jika dokumen dipotret secara vertikal.

# %%
stats_df['is_portrait'] = stats_df['aspect_ratio'] < 1.0

portrait_count = int(stats_df['is_portrait'].sum())
landscape_count = len(stats_df) - portrait_count

fig = plt.figure(figsize=(15, 9))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

ax_hist = fig.add_subplot(gs[0, :2])
ax_hist.hist(stats_df['aspect_ratio'].dropna(), bins=40,
             color=sns.color_palette(PALETTE, 2)[0], edgecolor='white', linewidth=0.8)
ax_hist.axvline(1.0, color='red', linestyle='solid', linewidth=2, label='Batas 1.0')
ax_hist.set_xlabel('Aspect Ratio Lebar per Tinggi')
ax_hist.set_ylabel('Frekuensi')
ax_hist.set_title('Distribusi Aspect Ratio Gambar', fontweight='bold')
ax_hist.legend()

ax_pie = fig.add_subplot(gs[0, 2:])
ax_pie.pie(
    [landscape_count, portrait_count],
    labels=[f'Landscape {landscape_count}', f'Portrait {portrait_count}'],
    autopct='%1.1f%%', colors=sns.color_palette(PALETTE, 2),
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}, startangle=90
)
ax_pie.set_title('Proporsi Orientasi Gambar', fontweight='bold')

for i, (_, row) in enumerate(stats_df[~stats_df['is_portrait']].head(2).iterrows()):
    ax = fig.add_subplot(gs[1, i])
    img_path = os.path.join(IMG_DIR, row['filename'])
    if os.path.exists(img_path):
        ax.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    fname_s = (row['filename'][:12] + '..') if len(row['filename']) > 14 else row['filename']
    ax.set_title(f"LANDSCAPE\n{fname_s} | Ratio {row['aspect_ratio']:.2f}",
                 fontsize=10, fontweight='bold', color=sns.color_palette(PALETTE, 2)[0])
    ax.axis('off')

for i, (_, row) in enumerate(stats_df[stats_df['is_portrait']].head(2).iterrows()):
    ax = fig.add_subplot(gs[1, 2 + i])
    img_path = os.path.join(IMG_DIR, row['filename'])
    if os.path.exists(img_path):
        ax.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    fname_s = (row['filename'][:12] + '..') if len(row['filename']) > 14 else row['filename']
    ax.set_title(f"PORTRAIT\n{fname_s} | Ratio {row['aspect_ratio']:.2f}",
                 fontsize=10, fontweight='bold', color=sns.color_palette(PALETTE, 2)[1])
    ax.axis('off')

fig.suptitle('Analisis Orientasi Gambar Mentah', fontsize=15, fontweight='bold', y=1.02)
plt.show()

# %% [markdown]
# **Insight:** Sebagian besar foto diambil secara vertikal sehingga banyak area meja atau latar ikut terekam. Fakta ini menegaskan bahwa pemotongan gambar (cropping) di awal sangat penting agar mesin hanya membaca tulisan di atas kartu.

# %% [markdown]
# ## 14. Strategi Pembagian Data
# Pembagian data latih dan data uji butuh perlakuan khusus agar sistem dievaluasi secara realistis dan adil.

# %%
groups = df['identity_key'].values
y_stratify = df['doc_origin_weak_label'].values
skf = StratifiedGroupKFold(n_splits=5)

fold_info = []
for fold, (train_idx, test_idx) in enumerate(skf.split(df, y=y_stratify, groups=groups)):
    df_test = df.iloc[test_idx]
    df_train = df.iloc[train_idx]

    ids_train = set(df_train['identity_key'].unique())
    ids_test = set(df_test['identity_key'].unique())
    bocor = len(ids_train & ids_test)

    n_malaysia = (df_test['doc_origin_weak_label'] == 'Malaysia').sum()
    n_foreign = (df_test['doc_origin_weak_label'] == 'Luar Negeri').sum()

    fold_info.append({
        'Fold': fold + 1,
        'Gambar Validasi': len(df_test),
        'Identitas Unik': df_test['identity_key'].nunique(),
        'Malaysia': n_malaysia,
        'Luar Negeri': n_foreign,
        'Pct Malaysia (%)': round(n_malaysia / len(df_test) * 100, 1),
        'Identitas Bocor': bocor,
    })

fold_table = pd.DataFrame(fold_info)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].bar(fold_table['Fold'], fold_table['Malaysia'],
            label='Malaysia', color=sns.color_palette(PALETTE, 2)[0], edgecolor='white')
axes[0].bar(fold_table['Fold'], fold_table['Luar Negeri'],
            bottom=fold_table['Malaysia'],
            label='Luar Negeri', color=sns.color_palette(PALETTE, 2)[1], edgecolor='white')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('Jumlah Gambar')
axes[0].set_title('Komposisi Gambar per Fold Validasi')
axes[0].legend()

axes[1].bar(fold_table['Fold'], fold_table['Identitas Bocor'], color='red', edgecolor='white')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('Jumlah Identitas Bocor')
axes[1].set_title('Kebocoran Identitas per Fold Target 0')
axes[1].set_ylim(0, 5)
for i, v in enumerate(fold_table['Identitas Bocor']):
    axes[1].text(i + 1, v + 0.05, str(v), ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** Metode pembagian berlapis mampu memisahkan foto secara ketat tanpa membocorkan identitas orang yang sama antar sesi. Data latih dan uji juga memuat sebaran negara yang merata sehingga ujian model menjadi lebih valid dan komprehensif.

# %% [markdown]
# ## 15. Menyimpan Indeks Fold

# %%
fold_indices = []
for fold, (train_idx, test_idx) in enumerate(skf.split(df, y=y_stratify, groups=groups)):
    fold_indices.append({
        'fold': fold + 1,
        'train': train_idx.tolist(),
        'test': test_idx.tolist(),
    })

with open(FOLD_PATH, 'wb') as f:
    pickle.dump(fold_indices, f)

print(f'Indeks fold disimpan ke {FOLD_PATH}')
# %% [markdown]
# ## 16. Analisis Pola Penamaan
# Dokumen identitas warga lokal sering kali memakai tata cara penulisan nama yang khas. Pola keturunan atau silsilah keluarga bisa dimanfaatkan untuk membedakan dokumen asal negara tersebut dari dokumen luar.

# %%
df['has_bin_binti'] = df['name'].astype(str).apply(
    lambda x: any(kw in x.upper() for kw in ['BIN ', 'BINTI ', 'A/L ', 'A/P '])
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bin_counts = df.groupby(['doc_origin_weak_label', 'has_bin_binti']).size().unstack(fill_value=0)
bin_counts.columns = ['Tanpa Pola', 'Ada Pola']
bin_counts.plot(kind='bar', ax=axes[0], color=sns.color_palette(PALETTE, 2),
                edgecolor='white', linewidth=1.2)
axes[0].set_title('Jumlah Gambar dengan Pola Keturunan Nama')
axes[0].set_xlabel('Asal Dokumen')
axes[0].set_ylabel('Jumlah Gambar')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title='Pola Nama')

pola_pct = df.groupby('doc_origin_weak_label')['has_bin_binti'].mean() * 100
bars = axes[1].bar(pola_pct.index, pola_pct.values,
                   color=sns.color_palette(PALETTE, len(pola_pct)), edgecolor='white')
axes[1].set_title('Persentase Pola Nama per Asal Dokumen')
axes[1].set_xlabel('Asal Dokumen')
axes[1].set_ylabel('Persentase')
axes[1].set_ylim(0, 110)
for bar, v in zip(bars, pola_pct.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, v + 2,
                 f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

plt.suptitle('Analisis Pola Penamaan per Asal Dokumen', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** Hanya KTP Malaysia yang memakai pola nama seperti Bin atau Binti. Hal ini menjadikan deteksi gelar keturunan sebagai senjata paling tajam untuk mengenali dokumen lokal.

# %% [markdown]
# ## 17. Analisis Pola Teks Pembeda Negara
# Filter awal pendeteksi asal negara diperlukan untuk menghindari model salah jalur. Ciri fisik dari teks dokumen dipakai agar sistem mampu menolak data kosong dari negara asing yang tidak relevan.

# %%
df['has_mykad_kw'] = df['name'].str.contains('MYKAD|WARGANEGARA|KAD PENGENALAN', case=False, na=False)
df['has_ic_pattern'] = df['name'].str.contains(r'\b\d{6}\d{2}\d{4}\b', regex=True, na=False)

# %% [markdown]
# **Insight:** Teks acak tertentu seperti nomor kependudukan 12 digit adalah ciri khas mutlak milik dokumen Malaysia. Ciri ini berguna sebagai benteng sistem untuk mendeteksi dokumen lokal tanpa merusak privasi isi data.

# %% [markdown]
# ## 18. Analisis Karakter Non ASCII
# Format nama di negara Eropa sering kali memiliki tanda aksen khusus. Kesalahan deteksi sering timbul apabila teks tidak diubah ke standar bacaan umum.

# %%
non_ascii = df[df['name'].str.contains(r'[^\x00-\x7F]', regex=True, na=False)]
print(non_ascii[['name', 'doc_origin_weak_label']].head().to_string(index=False))

def normalize_for_cer(text):
    if pd.isna(text): return ''
    return unicodedata.normalize('NFKC', str(text)).strip().lower()

df['name_norm'] = df['name'].apply(normalize_for_cer)

# %% [markdown]
# **Insight:** Variasi format nama dan tanda diakritik dari negara asing harus diseragamkan terlebih dahulu. Langkah ini penting agar mesin tidak kebingungan saat mencocokkan kemiripan teks di kemudian hari.

# %% [markdown]
# ## 19. Baseline OCR Tanpa Preprocessing
# Pengujian ini berguna sebagai acuan performa sistem sebelum foto dipotong atau dibersihkan. Lima foto dipilih sebagai wakil dari berbagai tingkat kualitas pencahayaan dan ketajaman.

# %%
blur_sorted = stats_df.sort_values('blur_score')
my_blur = blur_sorted[blur_sorted['doc_origin_weak_label'] == 'Malaysia']
ln_blur = blur_sorted[blur_sorted['doc_origin_weak_label'] == 'Luar Negeri']

sample_candidates = {
    'buram': my_blur.iloc[0]['filename'],
    'normal': my_blur.iloc[len(my_blur) // 2]['filename'],
    'tajam': my_blur.iloc[-1]['filename'],
    'luar_negeri': ln_blur.iloc[len(ln_blur) // 2]['filename'],
    'gelap': stats_df.sort_values('brightness').iloc[0]['filename'],
}

if os.path.exists(BASELINE_CACHE):
    with open(BASELINE_CACHE, 'r', encoding='utf-8') as f:
        baseline_results = json.load(f)
    print(f'Memuat cache baseline OCR {len(baseline_results)} gambar')
else:
    from paddleocr import PaddleOCR
    print('Menjalankan baseline OCR pada 5 gambar sampel...')
    ocr_baseline = PaddleOCR(use_angle_cls=True, lang='en', device='cpu', enable_mkldnn=False)
    baseline_results = {}

    for label, fname in sample_candidates.items():
        img_raw = cv2.imread(os.path.join(IMG_DIR, fname))
        if img_raw is None:
            baseline_results[label] = {'filename': fname, 'segments': [], 'error': 'not found'}
            continue
        result = ocr_baseline.ocr(img_raw)
        segments = []
        if result and result[0]:
            res_obj = result[0]
            if hasattr(res_obj, 'keys') and 'rec_texts' in res_obj.keys():
                for txt, conf in zip(res_obj.get('rec_texts', []), res_obj.get('rec_scores', [])):
                    segments.append({'text': txt, 'conf': round(float(conf), 3)})
            else:
                for line in res_obj:
                    if len(line) == 2 and len(line[1]) == 2:
                        segments.append({'text': line[1][0], 'conf': round(float(line[1][1]), 3)})
        baseline_results[label] = {'filename': fname, 'segments': segments}
        print(f'Mendapat {label} dengan jumlah {len(segments)} segmen')

    with open(BASELINE_CACHE, 'w', encoding='utf-8') as f:
        json.dump(baseline_results, f, ensure_ascii=False, indent=2)
    print(f'Cache disimpan ke {BASELINE_CACHE}')

# %% [markdown]
# **Insight:** Teks di luar kartu sering ikut terbaca oleh mesin. Hal ini bisa merusak data ekstraksi jika tidak ditangani dengan benar. Oleh sebab itu proses pemotongan tepi kartu sangat diwajibkan di tahap selanjutnya.

# %%
from paddleocr import PaddleOCR

ocr_vis = PaddleOCR(use_angle_cls=False, lang='en', device='cpu', enable_mkldnn=False)

def draw_ocr_boxes(img: np.ndarray, boxes: list, texts: list) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(1.0, min(w, h) / 600)
    thick = max(2, int(2 * scale))
    fscale = 0.4
    ftick = 1
    vis = img.copy()

    for box, txt in zip(boxes, texts):
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 230, 0), thickness=thick)

        x_min = int(np.min(pts[:, 0, 0]))
        y_min = int(np.min(pts[:, 0, 1]))

        (tw, th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fscale, ftick)
        pad = 4
        lx = max(0, x_min)
        ly_top = max(th + pad * 2, y_min - 4)
        ly_bot = ly_top + th + pad * 2

        cv2.rectangle(vis, (lx, ly_top - th - pad), (lx + tw + pad * 2, ly_bot - th), (20, 20, 20), cv2.FILLED)
        cv2.putText(vis, txt, (lx + pad, ly_top), cv2.FONT_HERSHEY_SIMPLEX, fscale, (0, 255, 100), ftick, cv2.LINE_AA)
    return vis

for label, fname in sample_candidates.items():
    img = cv2.imread(os.path.join(IMG_DIR, fname))
    if img is None:
        continue

    result = ocr_vis.ocr(img.copy())
    boxes, texts = [], []
    if result and result[0]:
        res_obj = result[0]
        if hasattr(res_obj, 'keys') and 'dt_polys' in res_obj.keys():
            boxes = res_obj.get('dt_polys', [])
            texts = res_obj.get('rec_texts', [])
        else:
            boxes = [line[0] for line in res_obj if len(line) == 2]
            texts = [line[1][0] for line in res_obj if len(line) == 2]

    vis_img = draw_ocr_boxes(img, boxes, texts)

    h, w = vis_img.shape[:2]
    MIN_DIM = 900
    if max(h, w) < MIN_DIM:
        scale = MIN_DIM / max(h, w)
        vis_img = cv2.resize(vis_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    fig_w = max(14, vis_img.shape[1] / 80)
    fig_h = max(9, vis_img.shape[0] / 80)
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Tipe {label.upper()} {fname} Temuan {len(texts)} Segmen', fontsize=14, fontweight='bold', pad=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# **Insight:** Tanpa pemotongan gambar mesin OCR akan membaca tulisan acak di latar belakang foto. Ini menegaskan keharusan melakukan proses lokalisasi dan pemotongan dokumen secara akurat sebelum masuk ke tahap OCR sebenarnya.

# %% [markdown]
# ## 20. Ringkasan Temuan Eksplorasi
# Proses pemahaman data ini menyajikan beberapa fondasi utama. Karena sebagian dokumen tidak memiliki kolom alamat maka deteksi asal negara sangat penting. Pemisahan identitas yang sama wajib dilakukan menggunakan K-Fold berlapis guna mencegah kebocoran. Format nama dan tanggal bervariasi luas sehingga aturan logika nantinya harus longgar. Orientasi foto serta kualitas yang buram memaksa sistem untuk memotong kartu dan menolak data rusak sedini mungkin. Semua bekal ini siap dilanjutkan ke tahap rekayasa fitur.

# %%
ringkasan_df = pd.DataFrame([
    {'Temuan': 'Dataset multi-negara', 'Detail': 'Malaysia dan berbagai negara lain', 'Implikasi Teknis': 'Ekstraksi address hanya aktif untuk dokumen Malaysia'},
    {'Temuan': 'Alamat MNAR', 'Detail': '82.3% gambar tidak punya alamat', 'Implikasi Teknis': 'Tidak boleh diimputasi, deteksi asal dokumen dulu'},
    {'Temuan': 'Duplikasi identitas', 'Detail': 'Hingga 40 foto per orang', 'Implikasi Teknis': 'StratifiedGroupKFold wajib untuk cegah data leakage'},
    {'Temuan': 'Pola nama eksklusif', 'Detail': 'Pola BIN BINTI A L A P hanya di Malaysia', 'Implikasi Teknis': 'Fitur nama sangat diskriminatif di field classifier'},
    {'Temuan': 'Format tanggal', 'Detail': 'Beragam format YYYY-MM-DD dan YYYY', 'Implikasi Teknis': 'Regex multi-format diperlukan'},
    {'Temuan': 'Kualitas gambar bervariasi', 'Detail': 'Blur score dari 4 hingga 2704', 'Implikasi Teknis': 'Preprocessing adaptif dan Triage Gate ML wajib ada'},
    {'Temuan': 'Orientasi beragam', 'Detail': 'Banyak gambar portrait dan berlatar', 'Implikasi Teknis': 'Lokalisasi kartu contour homography sangat kritis'},
    {'Temuan': 'Dokumen luar negeri buram', 'Detail': 'Rata rata blur luar negeri 63 vs Malaysia 757', 'Implikasi Teknis': 'Preprocessing harus adaptif'},
    {'Temuan': 'OCR mentah membaca latar', 'Detail': 'Teks latar ikut terdeteksi', 'Implikasi Teknis': 'Cropping akurat adalah prasyarat utama sebelum OCR'}
])

print('Eksplorasi selesai sepenuhnya. Silakan lanjutkan ke Notebook 2.')


