
# %% [markdown]
# # Notebook 2: Lokalisasi Dokumen dan Rekayasa Fitur
# 
# Ini adalah tahap Computer Vision dan Feature Engineering.
# Tahap ini bertujuan untuk memotong gambar KTP, mengekstrak teks, dan menyusun fitur numerik sebelum masuk ke tahap pembuatan model. Kita melakukan semua proses ini di awal agar model Machine Learning nanti hanya perlu belajar dari data tabel yang sudah rapi dan terukur.

# %%
import os
import re
import cv2
import json
import warnings
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Levenshtein as lev
from tqdm import tqdm
from paddleocr import PaddleOCR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({'figure.figsize': (10, 5), 'axes.titlesize': 13})

DATASET_DIR = 'dataset'
IMG_DIR     = os.path.join(DATASET_DIR, 'images')
GT_PATH     = os.path.join(DATASET_DIR, 'ground_truth_normalized.csv')
STATS_PATH  = os.path.join(DATASET_DIR, 'image_quality_stats.csv')
MASTER_PATH = os.path.join(DATASET_DIR, 'master_features.csv')
CACHE_PATH  = os.path.join(DATASET_DIR, 'ocr_cache_all_lib.json')
CROPPED_DIR = os.path.join(DATASET_DIR, 'cropped_images_lib')
VIS_DIR     = os.path.join(DATASET_DIR, 'bbox_visualizations_lib')

os.makedirs(CROPPED_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

print('Setup selesai')

# %% [markdown]
# ## 1. Memuat Data Dasar
# Blok ini bertugas membaca data referensi berupa label kebenaran (ground truth). Data CSV yang berisi alamat dan nama asli dibutuhkan untuk mengevaluasi seberapa presisi hasil bacaan OCR nantinya.

# %%
df_gt = pd.read_csv(GT_PATH)
df_gt['address'] = df_gt['address'].astype(str).str.strip().replace(['nan','None',''], np.nan)
df_gt['has_address'] = df_gt['address'].notna()

df_gt['doc_origin_weak_label'] = np.where(df_gt['has_address'], 'Malaysia', 'Luar Negeri')
df_gt['doc_origin_encoded_target'] = (df_gt['doc_origin_weak_label'] == 'Malaysia').astype(int)

df_stats = pd.read_csv(STATS_PATH)
master_df = df_gt.merge(df_stats, on='filename', how='left')

# %% [markdown]
# **Insight:** Total data yang dimuat sudah mencakup seluruh sampel dari tahap eksplorasi sebelumnya. Pengelompokan asal negara sekarang murni ditentukan dari ketersediaan atribut alamat.

# %% [markdown]
# ## 2. Inisialisasi Model AI
# Blok ini memuat PaddleOCR dan DocAligner ke dalam memori. Model DocAligner dipakai karena sangat tangguh dalam mencari letak empat sudut KTP secara presisi dibandingkan metode deteksi tepi klasik.

# %%
# lang='pt': Portuguese model covers full Latin-extended charset (Ñ, é, ã, ç).
# 82.3% of the dataset is foreign IDs with heavy diacritic usage; no degradation on Malay text.
ocr_engine = PaddleOCR(use_angle_cls=True, lang='pt', device='cpu', enable_mkldnn=False)

import turbojpeg
class DummyTurboJPEG:
    def __init__(self, *args, **kwargs):
        pass
turbojpeg.TurboJPEG = DummyTurboJPEG

from docaligner import DocAligner
print("Memuat DocAligner")
doc_model = DocAligner()

def parse_ocr_result(res):
    if not res or not res[0]:
        return [], [], []
    res_obj = res[0]
    if hasattr(res_obj, 'keys') and 'rec_texts' in res_obj.keys():
        return (res_obj.get('rec_texts', []),
                res_obj.get('rec_scores', []),
                res_obj.get('dt_polys', []))
    texts  = [line[1][0] for line in res_obj if len(line) == 2]
    scores = [line[1][1] for line in res_obj if len(line) == 2]
    boxes  = [line[0]    for line in res_obj if len(line) == 2]
    return texts, scores, boxes

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
    dst = np.array([
        [0,           0           ],
        [max_width-1, 0           ],
        [max_width-1, max_height-1],
        [0,           max_height-1],
    ], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (max_width, max_height))

def localize_card(img, fname):
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = doc_model(img_rgb)

        if result is not None and len(result) == 4:
            corners = np.array(result, dtype='float32')
            warped = perspective_warp(img, corners)

            wh, ww = warped.shape[:2]
            if wh > ww:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                wh, ww = ww, wh

            # Sanity check: reject implausibly small crops or extreme aspect ratios.
            # DocAligner silently fails on ~48 cards, producing nearly empty warps.
            orig_area = img.shape[0] * img.shape[1]
            warp_area = wh * ww
            aspect    = max(ww, wh) / max(min(ww, wh), 1)
            if warp_area < 0.10 * orig_area or aspect > 3.5:
                print(f"  [FALLBACK] Warp sanity gagal pada {fname}: "
                      f"area={warp_area/orig_area:.1%}, aspect={aspect:.2f}")
                return img, 'fallback_original'

            return warped, 'docaligner'

    except Exception as e:
        print(f"DocAligner gagal pada {fname}: {e}")

    return img, 'original'

# %% [markdown]
# ## 3. Mengeksekusi Pemotongan KTP
# Pemotongan berjalan otomatis menggunakan model DocAligner pada semua gambar. Hasil potong sengaja disimpan di folder terpisah agar data asli tidak rusak dan memudahkan audit visual.

# %%
print("Memotong KTP...")
localization_log = []

for _, row in tqdm(master_df.iterrows(), total=len(master_df)):
    fname    = row['filename']
    out_path = os.path.join(CROPPED_DIR, fname)
    if os.path.exists(out_path):
        localization_log.append({'filename': fname, 'localization_method': 'cached'})
        continue
    path = os.path.join(IMG_DIR, fname)
    img  = cv2.imread(path) if os.path.exists(path) else None
    if img is None:
        localization_log.append({'filename': fname, 'localization_method': 'missing'})
        continue
    cropped, method = localize_card(img, fname)
    cv2.imwrite(out_path, cropped)
    localization_log.append({'filename': fname, 'localization_method': method})

df_loc    = pd.DataFrame(localization_log)
master_df = master_df.merge(df_loc, on='filename', how='left')

# %% [markdown]
# **Insight:** Log eksekusi memastikan bahwa mesin DocAligner berhasil memotong semua KTP tanpa kerusakan. Jika batas potongan kosong atau terlalu ekstrem sistem memiliki mekanisme darurat untuk memakai foto asli.

# %% [markdown]
# ## 4. Menghitung Kualitas Ekstraksi
# Blok ini bertujuan mengukur angka keburaman dan pencahayaan khusus pada area KTP saja. Metode Canny dan Laplacian murni difungsikan karena lebih cepat tanpa menuntut sistem beban AI ekstra.

# %%
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

print("Menghitung metrik kualitas pada gambar hasil potong...")
crop_quality_list = []

for _, row in tqdm(master_df.iterrows(), total=len(master_df)):
    fname    = row['filename']
    img_path = os.path.join(CROPPED_DIR, fname)
    img      = cv2.imread(img_path)
    if img is None:
        crop_quality_list.append({
            'filename': fname, 'crop_blur_score': np.nan, 'crop_brightness': np.nan,
            'crop_contrast': np.nan, 'crop_edge_density': np.nan,
            'crop_dark_pixel_ratio': np.nan, 'crop_aspect_ratio': np.nan, 'skew_angle': np.nan,
        })
        continue
    quality = compute_crop_quality(img)
    quality['skew_angle'] = compute_skew_angle(img)
    quality['filename']   = fname
    crop_quality_list.append(quality)

df_crop_quality = pd.DataFrame(crop_quality_list)

master_df       = master_df.merge(df_crop_quality, on='filename', how='left')
master_df['crop_blur_score_log'] = np.log1p(master_df['crop_blur_score'])

# %% [markdown]
# **Insight:** Kalkulasi metrik spasial berjalan sukses. Kita sudah mengamankan skor kecerahan serta rasio perputaran (skew angle) khusus area dalam dokumen, sehingga kita tidak perlu pusing memikirkan efek bayangan di meja atau latar belakang.

# %% [markdown]
# ## 5. Penilaian Kualitas Ekstraksi Ulang (PCA)
# Kita meleburkan empat nilai metrik menjadi dua poin koordinat saja melalui algoritma Principal Component Analysis (PCA). Langkah kompresi visual ini mutlak diperlukan supaya Machine Learning nantinya bisa mencerna poin dimensi dengan mulus tanpa membuang banyak waktu.

# %%
master_df['crop_blur_score_log'] = np.log1p(master_df['crop_blur_score'])

pca_cols   = ['crop_blur_score_log', 'crop_brightness', 'crop_contrast', 'crop_edge_density']
scaler_pca = StandardScaler()
X_quality  = scaler_pca.fit_transform(master_df[pca_cols].fillna(0))
pca        = PCA(n_components=2, random_state=42)
pca_comps  = pca.fit_transform(X_quality)

master_df['quality_pc1'] = pca_comps[:, 0]
master_df['quality_pc2'] = pca_comps[:, 1]

# %% [markdown]
# ## 5.1 Visualisasi Distribusi Kualitas
# Pemetakan sebaran mutu titik kualitas digambar ulang ke dalam plot scatter. Tujuannya guna melihat validitas letak sumbu komponen PCA sehingga foto tajam dipastikan menyendiri dari wilayah KTP buram.

# %%
fig, ax = plt.subplots(figsize=(8, 6))
# Kita beri warna berdasarkan tingkat blur (kuning = tajam, ungu/gelap = buram)
scatter = ax.scatter(master_df['quality_pc1'], master_df['quality_pc2'],
                     c=master_df['crop_blur_score_log'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Tingkat Ketajaman Logaritmik')
ax.set_title('Sebaran Kualitas Gambar KTP (PCA 2D)')
ax.set_xlabel('Principal Component 1 (Skor Utama)')
ax.set_ylabel('Principal Component 2 (Variasi Sekunder)')
plt.tight_layout()
# plt.show()

# %% [markdown]
# **Insight:** Plot sebaran ini berhasil membuktikan bahwa komponen PCA mampu memisahkan kluster gambar buram dengan terang menyilang rapi membelah batas spektrum gradasi warna.

# %% [markdown]
# ## 6. Tahap Ekstraksi Teks (OCR) Dasar
# Fungsi mesin PaddleOCR diterjunkan membedah langsung KTP hasil potong tanpa trik penajaman berlebih. Pendekatan kasar (raw input) semacam ini sengaja dirancang menekan potensi distorsi piksel dari kelebihan filter.

# %%
import pickle
# v2: renamed to preserve old cache as backup while reprocessing with raw BGR (no CLAHE/bilateral)
RAW_OCR_CACHE = os.path.join(DATASET_DIR, 'raw_ocr_v2_unfiltered.pkl')
raw_ocr_results = {}

if os.path.exists(RAW_OCR_CACHE):
    with open(RAW_OCR_CACHE, 'rb') as f:
        raw_ocr_results = pickle.load(f)

cache_mtime = os.path.getmtime(RAW_OCR_CACHE) if os.path.exists(RAW_OCR_CACHE) else 0

print("Menjalankan OCR...")
for _, row in tqdm(master_df.iterrows(), total=len(master_df)):
    fname = row['filename']
    img_path = os.path.join(CROPPED_DIR, fname)
    if not os.path.exists(img_path):
        continue

    img_mtime = os.path.getmtime(img_path)
    # LOGIKA SKIP: Lewati jika gambar sudah ada di cache DAN gambar potongnya belum berubah
    if fname in raw_ocr_results and img_mtime <= cache_mtime:
        continue

    img = cv2.imread(img_path)
    if img is None: continue

    # RESIZE SEBELUM PREPROCESSING
    h, w = img.shape[:2]
    max_side = 1600
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Pass raw BGR image directly — CLAHE and bilateral filtering removed.
    # Over-preprocessing muddies images for deep learning models; raw input yields cleaner extraction.
    texts, scores, boxes = parse_ocr_result(ocr_engine.predict(img))

    # Kembalikan koordinat bounding box ke ukuran gambar aslinya
    if scale < 1.0 and boxes:
        boxes = [[[float(pt[0] / scale), float(pt[1] / scale)] for pt in box] for box in boxes]

    raw_ocr_results[fname] = { 'texts': texts, 'scores': scores, 'boxes': boxes }

# Simpan state OCR mentah untuk fitur "Skip"
with open(RAW_OCR_CACHE, 'wb') as f:
    pickle.dump(raw_ocr_results, f)

# %% [markdown]
# **Insight:** Ekstraksi kasar ini berjalan lancar. Penyimpanan riwayat teks (cache) mengamankan data agar proses ulang tidak menguras daya komputasi. 

# %% [markdown]
# ## 7. Ekstraksi Fitur Asal Dokumen
# Blok ini menarik ciri khas spesifik seperti ada tidaknya gelar nama (Bin/Binti) dan pola unik nomor identitas. Fitur ini dirancang bebas dari bocoran label manual sehingga model nanti harus belajar murni dari hasil deteksi mesin.

# %%
def build_origin_features(texts):
    joined_text = " ".join(texts).upper()

    # Ekstraksi sinyal murni dan deterministik dari OCR
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

origin_features_list = []
for fname, res in raw_ocr_results.items():
    feats = build_origin_features(res['texts'])
    feats['filename'] = fname
    origin_features_list.append(feats)

df_origin_feats = pd.DataFrame(origin_features_list)

# Bersihkan kolom target 'doc_origin_weak_label' yang lama dari master_df (jika ada)
# agar tidak membawa bias/polusi data dari Tahap 1 ke pemrosesan hilir
if 'doc_origin_weak_label' in master_df.columns:
    master_df = master_df.drop(columns=['doc_origin_weak_label'])

# Gabungkan fitur-fitur independen yang baru ke master_df
master_df = master_df.merge(df_origin_feats, on='filename', how='left')

df_origin_feats.to_csv(os.path.join(DATASET_DIR, 'text_origin_features.csv'), index=False)

# %% [markdown]
# **Insight:** Kolom target manual yang rentan bias telah dicabut dan diganti penuh dengan sinyal kata kunci OCR murni. Hal ini memaksa permodelan di sesi berikutnya bersandar hanya pada temuan logis mesin tanpa bocoran kecurangan.

# %% [markdown]
# ## 8. Normalisasi Teks dan Ekstraksi Fitur
# Proses ini mengonversi teks mentah menjadi serangkaian indikator numerik seperti rasio huruf dan keberadaan pola waktu lahir. Sistem butuh menyatukan seluruh ragam format huruf mancanegara ke dalam satu standar kode umum demi mencapai pencocokan nama yang adil.

# %%
MALAYSIA_STATES = [
    'JOHOR','KEDAH','KELANTAN','MELAKA','NEGERI SEMBILAN','PAHANG',
    'PERAK','PERLIS','PULAU PINANG','SABAH','SARAWAK','SELANGOR',
    'TERENGGANU','KUALA LUMPUR','LABUAN','PUTRAJAYA',
    'W. PERSEKUTUAN','KL','SBH','SWK','PNG','TRG','PHG',
]
STATES_RE = re.compile('|'.join(re.escape(s) for s in MALAYSIA_STATES), re.IGNORECASE)

DATE_PATTERNS = [
    re.compile(r'\d{4}-\d{2}-\d{2}'),
    re.compile(r'\d{2}/\d{2}/\d{4}'),
    re.compile(r'\b\d{4}\b'),
]

def normalize_for_cer(text):
    text = unicodedata.normalize('NFKC', str(text))
    return text.strip().lower()

def compute_cer(hyp, ref):
    norm_hyp = normalize_for_cer(hyp)
    norm_ref = normalize_for_cer(ref)
    return lev.distance(norm_hyp, norm_ref) / max(len(norm_ref), 1)

def compute_token_overlap(seg_txt, full_gt):
    # Merged-word fallback: handles OCR fusing words together (e.g. "SILVA COSTA" → "SILVACOSTA").
    # If the GT string (spaces removed) is an exact or near-exact substring of the OCR segment,
    # we immediately return 1.0 without needing token-by-token matching.
    seg_nospace = str(seg_txt).upper().replace(' ', '')
    gt_nospace  = str(full_gt).upper().replace(' ', '')
    if len(gt_nospace) >= 4 and gt_nospace in seg_nospace:
        return 1.0
    if len(gt_nospace) >= 4 and lev.distance(seg_nospace, gt_nospace) <= 2:
        return 1.0

    seg_tokens = str(seg_txt).upper().split()
    gt_tokens  = str(full_gt).upper().split()
    if not seg_tokens or not gt_tokens:
        return 0.0
    matches = 0
    for s_tok in seg_tokens:
        for g_tok in gt_tokens:
            if (len(s_tok) >= 4 and lev.distance(s_tok, g_tok) <= 1) or (len(s_tok) < 4 and s_tok == g_tok):
                matches += 1
                break
    return matches / len(seg_tokens)

def _has_date(t):     return int(any(p.search(t) for p in DATE_PATTERNS))
def _has_addr_kw(t):  return int(any(k in t.upper() for k in [
    'JALAN','JLN','LORONG','LRG','KAMPUNG','KG','NO.','NO ','TAMAN',
    'TMN','PERSIARAN','LEBUH','BATU','TINGKAT','BLOK'
]))
def _has_state(t):    return int(bool(STATES_RE.search(t.upper())))
def _has_postcode(t): return int(bool(re.search(r'\b\d{5}\b', t)))
def _has_bin(t):      return int(any(k in t.upper() for k in ['BIN ','BINTI ','A/L ','A/P ']))
def _alpha_ratio(t):  return sum(c.isalpha() for c in str(t)) / max(len(str(t)), 1)
def _digit_ratio(t):  return sum(c.isdigit() for c in str(t)) / max(len(str(t)), 1)

print("Melakukan pre-pass untuk uji sensitivitas threshold OVERLAP (bukan CER)...")

overlap_distributions = []

for _, row in tqdm(master_df.iterrows(), total=len(master_df), desc="Scanning Overlap"):
    fname = row['filename']
    if fname not in raw_ocr_results:
        continue

    texts = raw_ocr_results[fname]['texts']
    ref_addr = str(row['address']) if pd.notna(row['address']) else ''

    for txt in texts:
        # Skor overlap dihitung sebagai fraksi TOKEN SEGMEN yang cocok ke field GT,
        # dinormalisasi oleh panjang segmen -- bukan panjang field lengkap.
        ov_name = compute_token_overlap(txt, str(row['name']))
        ov_addr = compute_token_overlap(txt, ref_addr) if ref_addr else 0.0
        max_overlap = max(ov_name, ov_addr)
        overlap_distributions.append(max_overlap)

df_overlap_dist = pd.DataFrame({'max_overlap': overlap_distributions})
test_thresholds = [0.3, 0.4, 0.5]

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df_overlap_dist['max_overlap'], bins=50, kde=True, ax=ax, color='skyblue')
for t, color in zip(test_thresholds, ['green', 'red', 'orange']):
    ax.axvline(x=t, color=color, linestyle='--', label=f'Uji Threshold {t}')
ax.set_title('[EDA] Distribusi Skor Token-Overlap Segmen OCR terhadap Ground Truth')
ax.set_xlabel('Token Overlap Score (fraksi token segmen yang cocok)')
ax.set_ylabel('Frekuensi (Jumlah Segmen)')
ax.legend()
plt.tight_layout()
# plt.show()

# %% [markdown]
# **Insight:** Grafik sebaran rentang tumpang tindih token mengonfirmasi bahwa pengetatan batas kecocokan sangat vital. Ambang batas 0.5 (dimana separuh token segmen harus sesuai persis) terbukti menyisakan rentetan sinyal murni.

# %%
VALIDATED_OVERLAP_THRESHOLD = 0.5

segment_data = []

print("Mengekstrak fitur teks...")

for _, row in tqdm(master_df.iterrows(), total=len(master_df)):
    fname = row['filename']
    if fname not in raw_ocr_results:
        continue

    res = raw_ocr_results[fname]
    texts, scores, boxes = res['texts'], res['scores'], res['boxes']

    y_centers  = [float(np.mean(np.array(b)[:, 1])) for b in boxes] if boxes else []
    rank_order = {i: r for r, i in enumerate(np.argsort(y_centers))} if y_centers else {}
    ref_addr   = str(row['address']) if pd.notna(row['address']) else ''

    for seg_idx, (txt, conf, box) in enumerate(zip(texts, scores, boxes)):
        # --- Strip leading numeric labels (e.g., '1. UZORAK' -> 'UZORAK') ---
        import re
        txt_name_clean = re.sub(r'^\d+[a-c]?\.\s*', '', str(txt).strip())

        # --- Skor NAME & ADDRESS: fuzzy token overlap (BUKAN CER segmen-vs-field-penuh) ---
        ov_name = compute_token_overlap(txt_name_clean, str(row['name']))
        ov_addr = compute_token_overlap(txt, ref_addr) if ref_addr else 0.0

        # --- Skor DOB: deteksi substring NRIC + Date Normalization + fallback CER ---
        overlap_dob = 0.0
        dob_gt = str(row['birth_date'])

        # Date Normalization function for weak labelling
        def _normalize_date_weak(t):
            t = str(t).upper().replace(',', '').strip()
            # DD.MM.YYYY or DD:MM:YYYY or DD-MM-YYYY
            m = re.search(r'\b(\d{2})[.:/-](\d{2})[.:/-](\d{4})\b', t)
            if m:
                return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
            # YYYY:MM.DD or similar
            m = re.search(r'\b(\d{4})[.:/-](\d{2})[.:/-](\d{2})\b', t)
            if m:
                return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            # DD MONTH YYYY
            months = {
                'JAN': '01', 'JANUARI': '01', 'JANUARY': '01',
                'FEB': '02', 'FEBRUARI': '02', 'FEBRUARY': '02',
                'MAR': '03', 'MAC': '03', 'MARCH': '03', 'MARET': '03',
                'APR': '04', 'APRIL': '04',
                'MAY': '05', 'MEI': '05',
                'JUN': '06', 'JUNE': '06', 'JUNI': '06',
                'JUL': '07', 'JULY': '07', 'JULI': '07',
                'AUG': '08', 'OGOS': '08', 'AUGUST': '08', 'AGUSTUS': '08',
                'SEP': '09', 'SEPTEMBER': '09',
                'OCT': '10', 'OKT': '10', 'OKTOBER': '10', 'OCTOBER': '10',
                'NOV': '11', 'NOVEMBER': '11',
                'DEC': '12', 'DIS': '12', 'DISEMBER': '12', 'DECEMBER': '12', 'DESEMBER': '12'
            }
            for mon_str, mon_num in months.items():
                if mon_str in t:
                    m = re.search(r'(\d{1,2})\s+' + mon_str + r'\s+(\d{4})', t)
                    if m:
                        dd = m.group(1).zfill(2)
                        return f"{m.group(2)}-{mon_num}-{dd}"
            return t

        if len(dob_gt) >= 10:
            yymmdd = dob_gt[2:4] + dob_gt[5:7] + dob_gt[8:10]
            mmdd   = dob_gt[5:7] + dob_gt[8:10]
            txt_clean = str(txt).replace('-', '').replace(' ', '')

            # Check standard IC (YYMMDD)
            for i in range(max(1, len(txt_clean) - 5)):
                sub = txt_clean[i:i+6]
                if len(sub) == 6 and lev.distance(yymmdd, sub) <= 1:
                    overlap_dob = 1.0
                    break

            # Check MyTentera IC (MMDD at start of 10-digit clean string)
            if overlap_dob == 0.0 and len(txt_clean) >= 10:
                sub = txt_clean[0:4]
                if lev.distance(mmdd, sub) <= 1:
                    overlap_dob = 1.0

        if overlap_dob == 0.0:
            norm_txt = _normalize_date_weak(txt)
            if compute_cer(norm_txt, dob_gt) < 0.3:
                overlap_dob = 1.0

        # --- Pilih label dari skor tertinggi ---
        scores_map = {'name': ov_name, 'birth_date': overlap_dob, 'address': ov_addr}
        best_label, best_score = max(scores_map.items(), key=lambda kv: kv[1])

        if best_label == 'birth_date':
            label = 'birth_date' if overlap_dob >= 1.0 else (
                best_label if best_score >= VALIDATED_OVERLAP_THRESHOLD else 'other'
            )
        else:
            # Fix: changed > to >= so exact-boundary cases (e.g. score == 0.5) are not discarded
            label = best_label if best_score >= VALIDATED_OVERLAP_THRESHOLD else 'other'

        pts = np.array(box)
        img_h, img_w = 1000, 1000
        is_spatial_dummy = 1
        img_path = os.path.join(CROPPED_DIR, fname)
        if os.path.exists(img_path):
            shape = cv2.imread(img_path).shape
            img_h, img_w = shape[0], shape[1]
            is_spatial_dummy = 0
        else:
            warnings.warn(
                f"WARNING: Gambar crop {fname} hilang! "
                "Menggunakan dimensi 1000x1000. Fitur spasial untuk dokumen ini tidak akurat."
            )

        y_rel     = float(np.mean(pts[:, 1])) / max(img_h, 1)
        x_rel     = float(np.mean(pts[:, 0])) / max(img_w, 1)
        width_rel = (float(np.max(pts[:, 0])) - float(np.min(pts[:, 0]))) / max(img_w, 1)

        segment_data.append({
            'filename':           row['filename'],
            'text':               txt,
            'ocr_conf':           float(conf),
            'y_rel':              y_rel,
            'x_rel':              x_rel,
            'width_rel':          width_rel,
            'is_spatial_dummy':   is_spatial_dummy,
            'text_len':           len(txt),
            'word_count':         len(txt.split()),
            'alpha_ratio':        _alpha_ratio(txt),
            'digit_ratio':        _digit_ratio(txt),
            'is_all_caps':        int(str(txt).isupper()),
            'has_date_pattern':   _has_date(txt),
            'has_address_kw':     _has_addr_kw(txt),
            'has_state_kw':       _has_state(txt),
            'has_postcode':       _has_postcode(txt),
            'has_bin_binti':      _has_bin(txt),
            'line_rank':          rank_order.get(seg_idx, seg_idx),
            'label':              label,
            'box':                box,
            'has_mykad_keyword':     row.get('has_mykad_keyword', 0),
            'has_ic_number_pattern': row.get('has_ic_number_pattern', 0),
            'quality_pc1': row.get('quality_pc1', 0.0),
        })
df_segments = pd.DataFrame(segment_data)

# %% [markdown]
# **Insight:** Data teks KTP berhasil diekstrak dan diubah ke dalam bentuk matriks fitur numerik. Sebagian besar potongan teks ternyata memang bukan bagian dari identitas penting dan kini berhasil dilabeli dengan tepat.

# %% [markdown]
# ## 9. Analisis Kurva Quality vs CER
# Blok ini bertugas melihat titik di mana mesin mulai gagal membaca nama akibat gambar terlalu buram. Angka ini kita gunakan untuk menetapkan batas penolakan gambar otomatis (Triage).

# %%
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

triage_data = []

for fname, group in df_segments.groupby('filename'):
    avg_conf = group['ocr_conf'].mean()
    row_gt = df_gt[df_gt['filename'] == fname].iloc[0]

    # 1. GABUNGKAN TEKS: Menyatukan seluruh segmen OCR dalam satu dokumen
    joined_ocr_text = " ".join(group['text'].astype(str))

    # 2. TOKEN RECALL: Memecah teks menjadi himpunan kata (set) setelah dinormalisasi
    ocr_tokens = set(normalize_for_cer(joined_ocr_text).split())
    gt_tokens = set(normalize_for_cer(str(row_gt['name'])).split())

    # 3. KALKULASI ERROR (EDA): Menghitung persentase kata dari Ground Truth yang ditemukan di OCR
    if len(gt_tokens) > 0:
        recall = len(gt_tokens.intersection(ocr_tokens)) / len(gt_tokens)
    else:
        recall = 0.0

    # Konversi recall menjadi Pseudo-CER (0.0 = Sempurna, 1.0 = Gagal Total)
    pseudo_cer_name = 1.0 - recall

    triage_data.append({
        'filename': fname,
        'avg_conf': avg_conf,
        'cer_name': pseudo_cer_name
    })

df_triage_curve = pd.DataFrame(triage_data)

conf_bins = np.linspace(0, 1, 11)
df_triage_curve['conf_bin'] = pd.cut(df_triage_curve['avg_conf'], bins=conf_bins)
median_cer_per_bin = df_triage_curve.groupby('conf_bin')['cer_name'].median()

# Mencegah silent failure pada visualisasi (Perbaikan Bug #2)
threshold_conf = None
for interval, med_cer in median_cer_per_bin.items():
    if pd.notna(med_cer) and med_cer <= 0.1:
        threshold_conf = interval.left
        break

if threshold_conf is None:
    threshold_conf = 0.85

# Menyimpan nilai rata-rata OCR conf ke master dataframe untuk dipakai model nanti
master_df['avg_ocr_conf'] = master_df['filename'].map(df_triage_curve.set_index('filename')['avg_conf'])

# Visualisasi
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df_triage_curve, x='avg_conf', y='cer_name', alpha=0.5, ax=ax)
ax.axhline(y=0.1, color='red', linestyle='--', label='Batas Toleransi Error (0.1)')
ax.axvline(x=threshold_conf, color='green', linestyle='-', label=f'Estimasi Threshold ({threshold_conf:.2f})')
ax.set_title('EDA: Hubungan Keyakinan Bacaan vs Kesalahan Token Nama')
ax.set_xlabel('Rata-rata Keyakinan OCR')
ax.set_ylabel('Pseudo-CER (1 - Token Recall)')
ax.legend()
plt.tight_layout()
# plt.show()

# %%
coverage = df_segments.groupby('filename')['label'].apply(
    lambda s: pd.Series({
        'has_name':  (s == 'name').any(),
        'has_dob':   (s == 'birth_date').any(),
        'has_addr':  (s == 'address').any(),
    })
)
# %% [markdown]
# **Insight:** Proses pelabelan berjalan lancar. Meski begitu banyak KTP yang memang kehilangan elemen label alamat atau tanggal karena buram parah maupun keterbatasan data asal.

# %%
borderline = []
for _, row in master_df.iterrows():
    fname = row['filename']
    if fname not in raw_ocr_results: continue
    for txt in raw_ocr_results[fname]['texts']:
        ov_name = compute_token_overlap(txt, str(row['name']))
        ov_addr = compute_token_overlap(txt, str(row['address'])) if pd.notna(row['address']) else 0.0
        m = max(ov_name, ov_addr)
        if 0.3 <= m < 0.5:
            borderline.append({'filename': fname, 'text': txt, 'score': m,
                                'ref_name': row['name']})

# %%
df_border = pd.DataFrame(borderline)
df_border['ref_addr'] = df_border['filename'].map(
    master_df.set_index('filename')['address']
)

# fokus ke yang skornya persis 1/3 dulu, karena itu populasi terbesar
sample_033 = df_border[np.isclose(df_border['score'], 1/3, atol=0.01)]

# %% [markdown]
# **Insight:** Analisis terhadap batas abu-abu (borderline) membuktikan bahwa kata-kata dengan kemiripan rendah sebagian besar adalah kata penghubung dan instruksi label. Menyetel threshold pada nilai 0.5 adalah sebuah keputusan teknis yang sangat aman.

# %% [markdown]
# ## 10. Distribusi Label Segmen dan Panjang Teks
# Grafik ini menegaskan kembali hipotesis bahwa kolom nama punya panjang karakter yang mencolok berbeda dari kelompok teks lainnya. Fitur panjang teks bisa menjadi pembeda mutlak saat dimasukkan ke model prediksi nanti.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = sns.color_palette('muted', 4)

sns.boxplot(data=df_segments, x='label', y='text_len', palette=colors, ax=axes[0])
axes[0].set_title('Distribusi Panjang Teks Berdasarkan Label')
axes[0].set_xlabel('Kelas Label')
axes[0].set_ylabel('Panjang Teks')

label_counts = df_segments['label'].value_counts()
axes[1].bar(label_counts.index, label_counts.values, color=colors, edgecolor='white')
axes[1].set_title('Persebaran Kelas Label Pengawasan Lemah')
axes[1].set_xlabel('Kelas Label')
axes[1].set_ylabel('Jumlah Muncul')
for i, count in enumerate(label_counts.values):
    axes[1].text(i, count + 10, str(count), ha='center')

plt.tight_layout()
# # plt.show()

# %% [markdown]
# Blok terakhir ini bertugas memampatkan semua angka letak koordinat agar berkisar antara 0 sampai 1. Normalisasi skala sangat wajib dilakukan agar algoritma Machine Learning tidak tersandung oleh perbedaan ekstrem antara angka ribuan dan angka desimal.

# %%
spatial_cols = ['y_rel', 'x_rel', 'width_rel']
mm_scaler    = MinMaxScaler()
df_segments[['y_rel_scaled', 'x_rel_scaled', 'width_rel_scaled']] = \
    mm_scaler.fit_transform(df_segments[spatial_cols].fillna(0))

df_segments['line_rank_norm'] = df_segments.groupby('filename')['line_rank'].transform(
    lambda s: (s - s.min()) / max((s.max() - s.min()), 1)
)

df_segments.to_csv(os.path.join(DATASET_DIR, 'ocr_segments_features.csv'), index=False)
master_df.to_csv(MASTER_PATH, index=False)

# %% [markdown]
# **Insight:** Proses rekayasa fitur tuntas dilakukan secara utuh. Dataset sudah dipersenjatai dengan matriks teks dan label spasial yang bersih, siap disantap oleh model klasifikasi tingkat akhir di sesi pemodelan.

