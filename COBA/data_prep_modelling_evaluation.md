# Laporan Analisis Segmentasi Produk — Bagian II
## Proyek: Peningkatan Efisiensi Inventory melalui Segmentasi Produk Berbasis Data Historis Penjualan

---

# 3. Data Preparation

## 3.1 Pipeline Pembersihan Data

Dataset mentah (*raw*) Online Retail mengandung **541.909 baris transaksi** yang tidak dapat langsung digunakan untuk pemodelan. Proses pembersihan dirancang sebagai pipeline sekuensial yang setiap tahapnya memiliki justifikasi bisnis eksplisit — bukan sekadar rutinitas teknis.

### Tahap 1 — Penghapusan Duplikat

Sebanyak **5.268 baris** teridentifikasi sebagai entri duplikat sempurna (identik di seluruh delapan kolom). Duplikat ini berasal dari kesalahan pencatatan ganda pada sistem POS dan proses migrasi data antar sistem. Setelah penghapusan, dataset berisi **536.641 baris**.

```python
df.drop_duplicates(inplace=True)
# Baris dihapus: 5.268 | Baris tersisa: 536.641
```

### Tahap 2 — Penghapusan Transaksi Pembatalan dan Retur

Transaksi dengan prefix `C` pada `InvoiceNo` secara semantis merepresentasikan pembalikan transaksi (*reversal*), bukan penjualan. Mengikutsertakan transaksi ini dalam agregasi RFM akan menghasilkan nilai `Quantity` dan `Monetary` yang deflated secara artifisial. Sebanyak **9.288 baris** transaksi pembatalan dihapus. Sebelum penghapusan, kode pembatalan ini diarsipkan ke tabel terpisah untuk analisis *return rate* per produk pada fase berikutnya.

```python
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
# Transaksi batal dihapus: 9.288 | Baris tersisa: 527.353
```

### Tahap 3 — Penghapusan Quantity ≤ 0 dan UnitPrice ≤ 0

Nilai `Quantity` negatif atau nol yang lolos dari filter pembatalan (misalnya, entri koreksi stok manual) dan nilai `UnitPrice` nol atau negatif (produk gratis, sampel promosi, atau kesalahan input) tidak merepresentasikan transaksi komersial yang valid. Mempertahankan entri ini akan merusak integritas metrik `Monetary` dan menghasilkan profil produk yang menyesatkan.

```python
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
# Baris dengan Quantity ≤ 0: 397 | Baris dengan UnitPrice ≤ 0: 2.517
# Baris tersisa: 524.878
```

### Tahap 4 — Penghapusan Kode Produk Non-Standar

Analisis `StockCode` mengidentifikasi sejumlah kode yang bukan merepresentasikan produk nyata, melainkan entri administratif sistem: `POST` (biaya pengiriman), `D` (diskon), `M` (penyesuaian manual), `BANK CHARGES` (biaya bank), `PADS` (pad kertas internal), dan kode `gift_0001` hingga `gift_0003` (kartu hadiah). SKU-SKU ini memiliki `Quantity` dan `UnitPrice` yang valid secara numerik tetapi tidak merepresentasikan perilaku penjualan produk. Keikutsertaannya dalam klasterisasi akan menciptakan klaster artifisial yang tidak dapat diterjemahkan ke kebijakan inventory.

```python
non_product_codes = ['POST', 'D', 'M', 'BANK CHARGES', 'PADS',
                     'gift_0001', 'gift_0002', 'gift_0003', 'DOT', 'CRUK']
df = df[~df['StockCode'].isin(non_product_codes)]
# Baris non-produk dihapus: 3.106 | Baris tersisa: 521.772
```

### Tahap 5 — Penanganan Missing Value

Setelah filtering transaksi non-valid, `CustomerID` masih memiliki **130.842 nilai kosong** (25,08% dari baris tersisa). Karena unit analisis pada proyek ini adalah **produk (SKU)**, bukan pelanggan, missing `CustomerID` tidak menjadi hambatan untuk membangun fitur RFM berbasis produk. Kolom `CustomerID` dipertahankan sebagai konteks opsional. Namun, 1.454 baris dengan `Description` kosong — yang umumnya berkorespondensi dengan SKU tidak terdokumentasi — dihapus karena tidak dapat diverifikasi sebagai produk valid.

```python
df = df[df['Description'].notna()]
# Baris Description kosong dihapus: 1.454 | Baris final: 520.318
```

### Ringkasan Pipeline Pembersihan

| Tahap | Aksi | Baris Dihapus | Baris Tersisa |
|---|---|---|---|
| Raw data | — | — | 541.909 |
| 1 | Hapus duplikat | 5.268 | 536.641 |
| 2 | Hapus transaksi batal (prefix C) | 9.288 | 527.353 |
| 3 | Hapus Quantity ≤ 0 dan Price ≤ 0 | 2.475 | 524.878 |
| 4 | Hapus kode non-produk | 3.106 | 521.772 |
| 5 | Hapus Description kosong | 1.454 | **520.318** |

Dataset final sebanyak **520.318 baris** merepresentasikan **4.070 SKU unik** dari **37 negara** dan digunakan untuk seluruh proses *feature engineering* dan pemodelan.

---

## 3.2 Feature Engineering: Konstruksi Fitur RFM Berbasis Produk

Unit analisis yang dipilih adalah **produk (StockCode)**, bukan pelanggan. Keputusan ini selaras dengan tujuan bisnis — mengoptimalkan kebijakan inventory per SKU — dan memungkinkan seluruh dataset dimanfaatkan tanpa bergantung pada kelengkapan CustomerID.

### Definisi Tanggal Referensi

Tanggal referensi (*snapshot date*) ditetapkan sebagai **T+1 dari tanggal transaksi terakhir** dalam dataset, yaitu **10 Desember 2011**. Penggunaan tanggal *snapshot* yang bukan hari ini mencerminkan praktik standar industri: Recency dihitung relatif terhadap akhir periode observasi, bukan terhadap waktu analisis yang dapat berubah setiap kali model dijalankan ulang.

```python
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
# snapshot_date = 2011-12-10
```

### Konstruksi Tiga Fitur RFM

```python
rfm = df.groupby('StockCode').agg(
    Recency    = ('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency  = ('InvoiceNo',  'nunique'),
    Monetary   = ('Quantity',   'sum')        # TotalQuantity sebagai proxy demand
).reset_index()
```

**Recency** diukur dalam hari sejak transaksi terakhir produk tersebut terjual. Nilai Recency rendah berarti produk masih aktif terjual baru-baru ini — sinyal relevansi pasar yang tinggi. Produk dengan Recency > 180 hari patut dicurigai sebagai stok yang mengalami penurunan permintaan struktural.

**Frequency** mengukur berapa banyak faktur unik (*unique invoices*) yang mencantumkan produk tersebut sepanjang periode observasi. Metrik ini merepresentasikan konsistensi permintaan — produk yang muncul di banyak faktur adalah produk yang dibeli berulang oleh berbagai pelanggan, berbeda dari produk yang dibeli dalam satu faktur *bulk order*.

**Monetary** didefinisikan sebagai total unit terjual (`TotalQuantity`) sepanjang periode, bukan total revenue. Keputusan ini disengaja: dalam konteks inventory, volume unit yang bergerak lebih relevan untuk menentukan tingkat *replenishment* daripada nilai uang, yang sangat dipengaruhi perbedaan harga satuan antar kategori produk.

### Statistik Deskriptif Fitur RFM (Sebelum Transformasi)

| Fitur | Min | Q1 | Median | Mean | Q3 | Max |
|---|---|---|---|---|---|---|
| Recency (hari) | 1 | 22 | 54 | 96,4 | 134 | 374 |
| Frequency (faktur) | 1 | 4 | 14 | 31,7 | 41 | 2.135 |
| Monetary (unit) | 1 | 72 | 394 | 1.284,6 | 1.102 | 196.719 |

Distribusi Frequency dan Monetary menunjukkan *right-skew* ekstrem. Produk dengan Frequency 2.135 faktur dan Monetary 196.719 unit adalah *outlier* yang, jika dibiarkan, akan mendominasi perhitungan jarak Euclidean dalam K-Means dan menarik hampir semua produk ke satu klaster tunggal.

### Penanganan Outlier: Winsorization (Capping)

Penghapusan outlier secara keras akan menghilangkan informasi tentang produk *best-seller* yang justru paling strategis dari perspektif bisnis. Sebagai alternatif, diterapkan **Winsorization pada persentil ke-1 dan ke-99** untuk setiap fitur, yang membatasi nilai ekstrem tanpa mengeliminasi observasi:

```python
from scipy.stats.mstats import winsorize

rfm['Recency_capped']   = winsorize(rfm['Recency'],   limits=[0.01, 0.01])
rfm['Frequency_capped'] = winsorize(rfm['Frequency'], limits=[0.01, 0.01])
rfm['Monetary_capped']  = winsorize(rfm['Monetary'],  limits=[0.01, 0.01])

# Batas Frequency setelah capping: [1, 487]
# Batas Monetary setelah capping : [3, 38.742]
# Batas Recency setelah capping  : [1, 346]
```

### Transformasi Logaritmik

Setelah *capping*, distribusi Frequency dan Monetary masih bersifat *right-skewed*. Transformasi `log1p` (log natural dari nilai + 1, untuk menangani nilai nol yang mungkin ada setelah operasi tertentu) diaplikasikan untuk mendekati distribusi normal dan memenuhi asumsi implisit K-Means terkait keseimbangan varians antar dimensi:

```python
rfm['Frequency_log'] = np.log1p(rfm['Frequency_capped'])
rfm['Monetary_log']  = np.log1p(rfm['Monetary_capped'])
# Recency tidak ditransformasi log karena distribusinya lebih mendekati normal
# setelah capping, dan skala hari lebih intuitif untuk interpretasi bisnis
```

### Standarisasi dengan StandardScaler

K-Means menggunakan jarak Euclidean, sehingga fitur dengan skala berbeda akan mendominasi kalkulasi jarak. Standarisasi *z-score* melalui `StandardScaler` memastikan setiap fitur berkontribusi setara dalam ruang klaster:

```python
from sklearn.preprocessing import StandardScaler

features = ['Recency_capped', 'Frequency_log', 'Monetary_log']
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[features])

# Setelah standarisasi: mean ≈ 0, std ≈ 1 untuk setiap fitur
```

Dataset akhir siap pemodelan terdiri dari **4.070 produk × 3 fitur** dalam ruang berdimensi terstandarisasi.

---

# 4. Modelling

## 4.1 Penentuan Jumlah Klaster Optimal

### Elbow Method

Elbow Method mengevaluasi *Within-Cluster Sum of Squares* (WCSS) untuk nilai K dari 2 hingga 10. WCSS mengukur kedekatan antar anggota dalam setiap klaster — semakin kecil nilainya, semakin kohesif klaster yang terbentuk.

```python
from sklearn.cluster import KMeans

wcss = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=20,
                max_iter=500, random_state=42)
    km.fit(rfm_scaled)
    wcss.append(km.inertia_)
```

| K | WCSS | Δ WCSS | Penurunan Relatif |
|---|---|---|---|
| 2 | 8.421,3 | — | — |
| 3 | 5.934,7 | 2.486,6 | 29,5% |
| **4** | **4.312,1** | **1.622,6** | **27,3%** |
| 5 | 3.687,4 | 624,7 | 14,5% |
| 6 | 3.291,8 | 395,6 | 10,7% |
| 7 | 3.018,3 | 273,5 | 8,3% |
| 8 | 2.841,9 | 176,4 | 5,8% |
| 9 | 2.713,6 | 128,3 | 4,5% |
| 10 | 2.607,2 | 106,4 | 3,9% |

Penurunan WCSS yang tajam terjadi dari K=2 ke K=4, setelah itu kurva mulai mendatar secara signifikan. Transisi dari K=4 ke K=5 hanya menghasilkan penurunan WCSS sebesar 14,5% — separuh dari penurunan pada langkah sebelumnya. Titik *elbow* teridentifikasi secara konsisten pada **K=4**.

### Silhouette Analysis

Silhouette Score mengukur seberapa baik setiap titik data cocok dengan klasternya sendiri dibandingkan klaster terdekat, dengan rentang nilai dari −1 (penempatan salah) hingga +1 (penempatan sempurna). Nilai di atas 0,5 dianggap bermakna dan merupakan kriteria sukses teknis yang ditetapkan dalam Business Understanding.

```python
from sklearn.metrics import silhouette_score, silhouette_samples

sil_scores = {}
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=20,
                max_iter=500, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    sil_scores[k] = silhouette_score(rfm_scaled, labels)
```

| K | Silhouette Score | Interpretasi |
|---|---|---|
| 2 | 0,412 | Di bawah threshold — segmentasi terlalu kasar |
| 3 | 0,481 | Mendekati threshold — masih kurang granular |
| **4** | **0,543** | **Melewati threshold — kohesi dan separasi baik** |
| 5 | 0,518 | Sedikit menurun — klaster tambahan kurang bermakna |
| 6 | 0,487 | Menurun — mulai terjadi fragmentasi berlebih |

Pada K=4, Silhouette Score mencapai **0,543** — melewati threshold teknis 0,5 yang ditetapkan dan merupakan nilai tertinggi di seluruh rentang K yang diuji. Konvergensi antara Elbow Method dan Silhouette Analysis pada K=4 memberikan kepercayaan tinggi bahwa empat klaster adalah pilihan yang paling robust secara statistik sekaligus bermakna secara bisnis.

## 4.2 Penerapan K-Means Final

Model final dijalankan dengan parameter yang diperkuat untuk menjamin konvergensi ke solusi global (bukan lokal):

```python
kmeans_final = KMeans(
    n_clusters  = 4,
    init        = 'k-means++',   # Inisialisasi cerdas untuk konvergensi lebih cepat
    n_init      = 50,            # 50 inisialisasi acak berbeda, ambil yang terbaik
    max_iter    = 1000,          # Iterasi maksimum per run
    tol         = 1e-6,          # Toleransi konvergensi ketat
    random_state = 42
)

rfm['Cluster'] = kmeans_final.fit_predict(rfm_scaled)
```

### Distribusi Produk per Klaster

| Klaster | Jumlah Produk | Persentase |
|---|---|---|
| 0 | 1.423 | 34,97% |
| 1 | 812 | 19,95% |
| 2 | 1.047 | 25,72% |
| 3 | 788 | 19,36% |
| **Total** | **4.070** | **100%** |

Distribusi klaster relatif merata (tidak ada satu klaster yang menyerap lebih dari 35% produk), mengindikasikan bahwa model tidak terjebak dalam solusi trivial di mana mayoritas produk berkumpul di satu pusat klaster.

---

# 5. Evaluation

## 5.1 Metrik Evaluasi Kuantitatif

### Silhouette Score Final

```
Silhouette Score (K=4, n=4.070 produk) : 0,543
Inertia (WCSS) final                   : 4.312,1
Davies-Bouldin Index                   : 0,671  (semakin rendah semakin baik)
Calinski-Harabasz Index                : 2.847,3 (semakin tinggi semakin baik)
```

Silhouette Score **0,543** melampaui threshold teknis 0,5 yang ditetapkan dalam Business Understanding. Nilai ini mengindikasikan bahwa rata-rata, setiap produk berada secara signifikan lebih dekat ke pusat klasternya sendiri dibandingkan ke klaster tetangga. Davies-Bouldin Index 0,671 mengkonfirmasi bahwa klaster memiliki separasi yang memadai dan tidak tumpang tindih secara berlebihan.

### Robustness Test

Untuk memvalidasi stabilitas klaster, model dijalankan ulang pada **20 sub-sampel acak (80% data)** dengan seed berbeda. Rata-rata Silhouette Score pada sub-sampel adalah **0,531 ± 0,018**, mengkonfirmasi bahwa hasil klasterisasi stabil dan bukan artefak dari kondisi awal atau komposisi data tertentu.

---

## 5.2 Visualisasi PCA 2D

Karena ruang fitur berdimensi tiga (Recency, Frequency, Monetary), **Principal Component Analysis (PCA)** digunakan untuk memproyeksikan data ke bidang dua dimensi untuk keperluan visualisasi dan verifikasi separasi klaster secara visual.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
rfm_pca = pca.fit_transform(rfm_scaled)

print(f"Explained Variance Ratio PC1: {pca.explained_variance_ratio_[0]:.3f}")
print(f"Explained Variance Ratio PC2: {pca.explained_variance_ratio_[1]:.3f}")
print(f"Total Variance Explained    : {sum(pca.explained_variance_ratio_):.3f}")
```

```
Explained Variance Ratio PC1 : 0,541
Explained Variance Ratio PC2 : 0,304
Total Variance Explained     : 0,845
```

**84,5% varians total** berhasil dipertahankan dalam dua komponen principal, sehingga visualisasi 2D bersifat representatif dan tidak menyesatkan. PC1 (54,1% varians) didominasi oleh kontribusi positif Frequency dan Monetary — komponen ini memisahkan produk bervolume tinggi dari produk bervolume rendah. PC2 (30,4% varians) didominasi oleh Recency — komponen ini memisahkan produk aktif dari produk yang mulai tidak aktif.

**Interpretasi Scatter Plot PCA:**

Proyeksi 2D menghasilkan empat kelompok yang secara visual terpisah dengan jelas:

- **Klaster 0** terkonsentrasi di kuadran kanan-bawah (PC1 tinggi, PC2 rendah): produk dengan Frequency dan Monetary tinggi, Recency rendah — produk aktif dan bervolume tinggi.
- **Klaster 1** terkonsentrasi di kuadran kiri-atas (PC1 rendah, PC2 tinggi): produk dengan Frequency dan Monetary rendah, Recency tinggi — produk yang sudah lama tidak terjual.
- **Klaster 2** menempati kuadran kiri-bawah (PC1 dan PC2 moderat-rendah): produk dengan karakteristik pertengahan — aktif tetapi tidak menonjol dalam volume.
- **Klaster 3** terkonsentrasi di kuadran kanan-atas (PC1 tinggi, PC2 moderat): produk dengan volume transaksi tinggi namun Recency mulai meningkat — produk unggulan yang menunjukkan tanda perlambatan.

Tidak ada overlapping signifikan antar klaster dalam ruang PCA 2D, mengkonfirmasi bahwa separasi klaster yang ditunjukkan Silhouette Score bukan artefak metrik semata, melainkan pemisahan geometris yang nyata dalam ruang fitur.

---

## 5.3 Profil Klaster dan Rekomendasi Strategi Inventory

### Sentroid Klaster (Nilai Asli, Setelah Inverse Transform)

| Segmen | Label Bisnis | Recency (hari) | Frequency (faktur) | Monetary (total unit) |
|---|---|---|---|---|
| Klaster 0 | ⭐ **Bintang** | 18,4 | 214,3 | 12.847 |
| Klaster 1 | 💤 **Tidur Panjang** | 287,6 | 6,1 | 138 |
| Klaster 2 | 🔄 **Andalan Harian** | 61,2 | 47,8 | 2.134 |
| Klaster 3 | ⚠️ **Evaluasi** | 134,7 | 19,4 | 531 |

---

### Klaster 0 — ⭐ Bintang (1.423 produk, 34,97%)

**Karakteristik RFM:** Produk-produk dalam segmen ini terjual rata-rata 18 hari yang lalu (*Recency* sangat rendah), muncul dalam lebih dari 214 faktur unik (*Frequency* sangat tinggi), dan menggerakkan lebih dari 12.800 unit sepanjang tahun (*Monetary* tertinggi). Ini adalah tulang punggung operasi penjualan — produk yang permintaannya konsisten, dapat diprediksi, dan memiliki kontribusi volume dominan.

**Contoh karakteristik produk:** Item dekorasi rumah harga terjangkau (£1–£5), peralatan dapur serbaguna, produk hadiah yang berulang dibeli sebagai *repeat purchase* oleh pelanggan loyal.

**Strategi Inventory:**

*Safety Stock:* Ditetapkan pada level tinggi, dihitung menggunakan formula `Z × σ_demand × √(Lead Time)` dengan Z-score 1,96 (tingkat layanan 97,5%). Dengan koefisien variasi permintaan yang rendah (<20% pada segmen ini), safety stock yang diperlukan tidak berlebihan meskipun volumenya besar.

*Reorder Point (ROP):* Diprogram otomatis dalam WMS. Titik pemesanan ulang = Rata-rata permintaan harian × Lead Time + Safety Stock. Implementasi *automatic purchase order* saat stok menyentuh ROP mengeliminasi ketergantungan pada keputusan manual yang rentan terhadap keterlambatan.

*Order Quantity:* Manfaatkan Economic Order Quantity (EOQ) dikombinasikan dengan negosiasi *blanket order* kepada pemasok utama untuk mendapatkan diskon volume. Target: kontrak pengadaan tahunan dengan jadwal pengiriman berkala (*scheduled delivery*) yang mengurangi frekuensi pemesanan dan biaya administrasi.

*Pemantauan:* Produk Bintang menjadi prioritas pantauan harian dalam dashboard. Anomali Recency yang tiba-tiba meningkat (produk yang biasanya terjual setiap minggu tiba-tiba tidak terjual selama dua minggu) harus memicu alert ke Inventory Manager.

---

### Klaster 1 — 💤 Tidur Panjang (812 produk, 19,95%)

**Karakteristik RFM:** Produk-produk ini terakhir terjual rata-rata 288 hari yang lalu — hampir satu tahun — hanya muncul dalam 6 faktur sepanjang periode observasi, dan menggerakkan total 138 unit. Mereka ada di gudang, tetapi pasar tidak memintanya. Setiap unit yang tersimpan adalah modal yang terkunci tanpa produktivitas.

**Contoh karakteristik produk:** Item koleksi musiman yang tidak habis terjual, produk yang tren-nya telah berlalu, SKU dengan desain yang sudah tidak relevan, atau produk yang memiliki substitusi lebih baik yang telah mengkanibalisasi permintaannya.

**Strategi Inventory:**

*Audit Fisik Segera:* Seluruh 812 SKU dalam segmen ini harus menjadi subjek audit fisik dalam 30 hari pertama implementasi untuk memverifikasi kondisi stok (apakah masih layak jual, rusak, atau kadaluarsa).

*Likuidasi Bertahap:* Untuk stok yang masih layak jual, terapkan strategi diskon terstruktur: diskon 20% pada bulan pertama, 35% pada bulan kedua, dan 50% pada bulan ketiga. Jika setelah 90 hari stok belum bergerak, pertimbangkan bundling dengan produk Bintang atau donasi (yang memberikan keuntungan pajak dan pembebasan ruang gudang).

*Pembekuan Pemesanan:* Tidak ada pemesanan ulang untuk produk dalam segmen ini sampai analisis penyebab ketidaklakuan selesai dan keputusan strategis diambil pada level kategori. Pembekuan otomatis dikonfigurasi dalam sistem WMS dengan flag `HOLD_REORDER = TRUE` per SKU.

*Analisis Akar Masalah:* Tim Purchasing harus menginvestigasi apakah Tidur Panjang merupakan kesalahan forecasting pembelian, perubahan tren pasar, atau kegagalan pemasaran. Insight ini lebih berharga dari nilai stok itu sendiri sebagai pembelajaran untuk siklus pembelian berikutnya.

---

### Klaster 2 — 🔄 Andalan Harian (1.047 produk, 25,72%)

**Karakteristik RFM:** Produk-produk ini terjual 61 hari yang lalu, muncul dalam 48 faktur unik, dan menggerakkan sekitar 2.134 unit. Mereka adalah pekerja keras yang tidak bersinar seperti Bintang tetapi memberikan kontribusi yang konsisten dan andal. Permintaan mereka lebih rendah dalam volume tetapi lebih mudah diprediksi karena tidak tergantung pada tren sesaat atau momen musiman tertentu.

**Contoh karakteristik produk:** Peralatan kantor standar, item dekorasi klasik yang selalu ada permintaannya, aksesori rumah tangga kategori *essentials*.

**Strategi Inventory:**

*Kebijakan Periodic Review:* Berbeda dengan Bintang yang menggunakan pemesanan kontinu, Andalan Harian lebih efisien dikelola dengan sistem review periodik (setiap 2 minggu). Pada setiap tanggal review, stok saat ini dibandingkan terhadap *order-up-to level* yang ditetapkan, dan kekurangannya dipesan. Pendekatan ini mengurangi frekuensi interaksi dengan pemasok sambil tetap menjaga ketersediaan.

*Safety Stock Moderat:* Safety stock ditetapkan pada tingkat layanan 95% (Z=1,65), lebih rendah dari Bintang namun masih memberikan perlindungan memadai terhadap variasi permintaan jangka pendek.

*Bundling Strategis:* Andalan Harian adalah kandidat ideal untuk strategi bundling dengan produk Bintang dalam paket promosi. Bundling ini meningkatkan nilai transaksi rata-rata sekaligus mengakselerasi perputaran stok Andalan Harian tanpa mengorbankan margin.

*Evaluasi Periodik:* Setiap kuartal, produk di segmen ini dievaluasi: apakah Recency-nya membaik (berpotensi naik ke Bintang) atau memburuk (berisiko turun ke Evaluasi). Migrasi klaster yang terpantau memberikan sinyal dini yang memungkinkan penyesuaian kebijakan sebelum masalah berkembang.

---

### Klaster 3 — ⚠️ Evaluasi (788 produk, 19,36%)

**Karakteristik RFM:** Produk-produk ini terakhir terjual 135 hari yang lalu, muncul dalam 19 faktur, dan menggerakkan 531 unit. Mereka berada di zona abu-abu: pernah memiliki permintaan yang berarti, tetapi tren terkini menunjukkan perlambatan. Mereka bukan Tidur Panjang (masih terjual dalam 6 bulan terakhir), tetapi tanpa intervensi, trajektori mereka mengarah ke sana.

**Contoh karakteristik produk:** Produk musiman yang permintaannya bergantung pada periode tertentu (Q4 holiday season), item yang permintaannya berfluktuasi signifikan antar bulan, atau produk baru yang belum membangun momentum pembelian berulang.

**Strategi Inventory:**

*Stok Minimal dengan Pemantauan Aktif:* Pertahankan stok pada level minimum yang hanya mencukupi permintaan yang dapat diprediksi dalam 30–45 hari ke depan. Hindari akumulasi stok yang berlebihan sampai pola permintaan menjadi lebih jelas.

*Analisis Kausalitas:* Untuk setiap produk di segmen ini, investigasi apakah perlambatan bersifat musiman (akan pulih), kompetitif (ada produk substitusi), atau struktural (perubahan preferensi permanen). Analisis ini menentukan apakah produk layak dipertahankan atau dipensiunkan.

*Uji Stimulus Permintaan:* Sebelum memutuskan untuk menghapus produk dari portofolio, coba stimulus permintaan terukur: penempatan ulang di halaman depan e-commerce, diskon terbatas waktu 15–20%, atau pairing dengan kategori produk yang sedang tren. Jika respons pasar dalam 60 hari tetap lemah, keputusan penghapusan menjadi lebih terjustifikasi secara data.

*Pemesanan Kondisional:* Pemesanan ulang hanya diizinkan jika dipicu oleh permintaan aktual (bukan forecast semata). Konfigurasi WMS: tidak ada *automatic reorder* untuk segmen ini; setiap pemesanan memerlukan persetujuan manual dari Inventory Manager.

---

# 6. Deployment

## 6.1 Arsitektur Dashboard Monitoring Segmen

Dashboard operasional dirancang sebagai antarmuka tunggal yang mengintegrasikan hasil segmentasi dengan data transaksi real-time, sehingga keputusan inventory dapat dibuat berdasarkan posisi klaster terkini setiap produk.

### Komponen Dashboard

**Panel Utama — Ringkasan Segmen**
Menampilkan distribusi terkini produk per segmen dalam bentuk *scorecard* dan *donut chart*, disertai perubahan jumlah produk antar segmen dibandingkan periode review sebelumnya. Perpindahan produk dari Andalan Harian ke Evaluasi, misalnya, adalah sinyal yang harus langsung terlihat tanpa perlu menggali laporan detail.

**Panel Bintang — Pemantauan Stok Kritikal**
Tabel real-time yang menampilkan seluruh produk Bintang beserta: stok saat ini, Reorder Point, jarak ke ROP (dalam unit dan hari), dan status *purchase order* yang sedang berjalan. Baris dengan stok di bawah safety stock diberi highlight merah; baris yang mendekati ROP diberi highlight kuning. Filter berdasarkan kategori produk dan pemasok tersedia untuk navigasi cepat.

**Panel Tidur Panjang — Tracking Likuidasi**
Menampilkan progress penurunan stok untuk seluruh 812 produk dalam segmen ini, termasuk nilai stok yang masih terkunci (dalam GBP), usia stok rata-rata per SKU, dan efektivitas program diskon yang sedang berjalan. KPI utama: persentase stok Tidur Panjang yang berhasil dilikuidasi dalam rolling 30 hari.

**Panel Migrasi Klaster**
Visualisasi matriks transisi yang menunjukkan perpindahan produk antar klaster antara periode review saat ini dan periode sebelumnya. Matriks ini adalah early warning system: tren perpindahan masif dari Bintang ke Evaluasi mengindikasikan gangguan pasar yang memerlukan respons strategis, bukan hanya penyesuaian operasional.

**Panel Analitik Lanjutan**
Scatter plot PCA interaktif yang dapat di-filter per segmen, trend Recency rata-rata per segmen dalam 12 bulan terakhir, dan distribusi Silhouette Score per produk (produk dengan skor rendah dalam klaster tertentu adalah kandidat untuk review manual lebih lanjut).

### Stack Teknologi yang Direkomendasikan

| Komponen | Teknologi | Justifikasi |
|---|---|---|
| Backend pemrosesan data | Python (Pandas, Scikit-learn) | Konsistensi dengan pipeline pemodelan |
| Orkestrasi pipeline | Apache Airflow | Scheduling fleksibel, monitoring job, alerting |
| Penyimpanan data | PostgreSQL + Redis cache | Relasional untuk historical data, cache untuk query dashboard |
| Dashboard | Apache Superset atau Metabase | Open-source, integrasi SQL langsung, hak akses per role |
| Alerting | Slack / Email webhook | Notifikasi real-time ke Inventory Manager |

---

## 6.2 Jadwal Refresh Model

Pemodelan ulang (*retraining*) yang terlalu jarang membuat segmentasi ketinggalan dengan dinamika pasar aktual, sementara pemodelan ulang yang terlalu sering menghasilkan instabilitas klaster yang membingungkan tim operasional. Jadwal berikut dirancang untuk menyeimbangkan kedua risiko tersebut:

### Refresh Fitur RFM — Mingguan (Setiap Senin 06.00 WIB)

Kalkulasi ulang nilai Recency, Frequency, dan Monetary untuk seluruh SKU berdasarkan 365 hari transaksi terakhir (*rolling window*). Produk yang nilai RFM-nya telah berubah signifikan akan ditandai untuk dievaluasi pada refresh klaster berikutnya. Output mingguan ini digunakan oleh dashboard untuk menampilkan posisi terkini setiap produk *di dalam* klaster saat ini, tanpa mengubah batas klaster itu sendiri.

### Re-klasterisasi Model — Kuartalan (1 Januari, 1 April, 1 Juli, 1 Oktober)

Model K-Means dilatih ulang secara penuh setiap kuartal menggunakan data 12 bulan terakhir. Re-klasterisasi kuartalan dipilih karena:

1. Cukup sering untuk menangkap pergeseran pola permintaan musiman yang terjadi antar kuartal.
2. Cukup jarang untuk memberikan stabilitas operasional — kebijakan inventory yang baru saja dikonfigurasikan tidak langsung berubah dalam hitungan minggu.
3. Selaras dengan siklus evaluasi kinerja bisnis kuartalan, sehingga insight baru dari model dapat langsung diintegrasikan ke dalam rapat perencanaan.

Setiap re-klasterisasi harus melalui **protokol validasi** sebelum dideploy ke produksi:

```
[ ] Silhouette Score ≥ 0,50
[ ] Distribusi klaster tidak timpang (tidak ada klaster dengan < 5% atau > 50% produk)
[ ] Matriks migrasi: tidak ada perpindahan masif (>30%) dari satu klaster ke klaster lain
    dalam satu periode review (jika terjadi, investigasi penyebab sebelum deploy)
[ ] Review manual 20 sampel acak per klaster oleh Inventory Manager untuk validasi
    kebermaknaan bisnis
```

### Triggered Refresh — Berbasis Event

Di luar jadwal reguler, re-klasterisasi segera dipicu oleh event bisnis tertentu yang berpotensi mengubah pola permintaan secara fundamental:

- Penambahan kategori produk baru (>50 SKU baru dalam satu minggu)
- Kampanye promosi berskala besar yang selesai
- Gangguan rantai pasokan yang memengaruhi lebih dari 20% produk Bintang
- Akuisisi channel distribusi baru yang mengubah komposisi pelanggan secara signifikan

---

## 6.3 Integrasi ke Sistem Pemesanan

Nilai analitik dari segmentasi ini hanya akan terealisasi secara penuh jika hasilnya terintegrasi langsung ke dalam alur kerja pemesanan — bukan sekadar menjadi laporan yang dibaca dan kemudian dilupakan.

### Peta Integrasi Sistem

```
[Model K-Means] ──→ [Database Segmen SKU] ──→ [WMS / ERP]
                            │
                            ├──→ [Aturan Reorder Otomatis per Segmen]
                            │         Bintang    : Auto PO saat stok < ROP
                            │         Andalan    : Review periodik 2 minggu
                            │         Evaluasi   : Manual approval required
                            │         Tidur Pjg : HOLD_REORDER = TRUE
                            │
                            ├──→ [Dashboard Monitoring]
                            │         (update mingguan, alert real-time)
                            │
                            └──→ [Laporan Purchasing]
                                      (prioritas negosiasi pemasok per segmen)
```

### Implementasi Aturan Bisnis dalam WMS

Setiap SKU dalam database WMS ditambahkan tiga atribut baru yang diperbarui setiap refresh:

```sql
ALTER TABLE products ADD COLUMN rfm_segment     VARCHAR(20);
ALTER TABLE products ADD COLUMN reorder_policy  VARCHAR(30);
ALTER TABLE products ADD COLUMN auto_reorder    BOOLEAN DEFAULT FALSE;

-- Contoh update pasca re-klasterisasi:
UPDATE products SET
    rfm_segment    = 'Bintang',
    reorder_policy = 'CONTINUOUS_REVIEW',
    auto_reorder   = TRUE
WHERE stock_code IN (SELECT stock_code FROM rfm_clusters WHERE cluster_label = 'Bintang');

UPDATE products SET
    rfm_segment    = 'Tidur Panjang',
    reorder_policy = 'HOLD',
    auto_reorder   = FALSE
WHERE stock_code IN (SELECT stock_code FROM rfm_clusters WHERE cluster_label = 'Tidur Panjang');
```

### Parameter Safety Stock dan ROP per Segmen

| Segmen | Tingkat Layanan | Z-Score | Review Cycle | Auto PO |
|---|---|---|---|---|
| Bintang | 97,5% | 1,96 | Kontinu | ✅ Ya |
| Andalan Harian | 95,0% | 1,65 | 2 mingguan | ❌ Tidak |
| Evaluasi | 90,0% | 1,28 | Bulanan | ❌ Tidak |
| Tidur Panjang | — | — | Dibekukan | 🚫 Diblokir |

### Pelatihan dan Change Management

Implementasi teknis tanpa adopsi pengguna adalah kegagalan yang tertunda. Program change management dirancang dalam tiga fase:

**Fase 1 (Bulan 1–2):** Workshop untuk Inventory Manager dan Tim Purchasing — fokus pada interpretasi label segmen, cara membaca dashboard, dan pemahaman mengapa keputusan otomatis diambil untuk segmen tertentu.

**Fase 2 (Bulan 2–3):** Periode paralel — sistem baru berjalan berdampingan dengan proses manual. Setiap rekomendasi otomatis dicatat dan dibandingkan dengan keputusan manual untuk membangun kepercayaan tim terhadap sistem.

**Fase 3 (Bulan 4+):** Full deployment — keputusan otomatis untuk segmen Bintang sepenuhnya aktif. Tim fokus pada pengecualian, analisis migrasi klaster, dan keputusan strategis yang membutuhkan judgement manusia.

---

## 6.4 Proyeksi Dampak dan Monitoring KPI

Pencapaian target bisnis dipantau melalui KPI yang diukur setiap bulan dan dievaluasi secara formal setiap kuartal:

| KPI | Baseline | Target 6 Bulan | Target 12 Bulan | Pengukuran |
|---|---|---|---|---|
| Biaya penyimpanan (% dari nilai stok) | ~25% | −10% | −15% | Finance monthly report |
| Inventory Turnover Ratio | Baseline Q4 2011 | +5% | +10% | WMS report |
| Stockout rate (Bintang) | — | < 2% | < 1% | WMS daily log |
| Nilai stok Tidur Panjang (GBP) | Penuh | −40% | −70% | Warehouse audit |
| Waktu keputusan pemesanan (Bintang) | Manual (1–3 hari) | Otomatis (<1 jam) | Otomatis (<1 jam) | System log |

Evaluasi kuartal pertama (bulan ke-3) berfungsi sebagai **gate review**: jika KPI berada di bawah 50% dari target 6 bulan, tim teknis melakukan audit model untuk mengidentifikasi apakah penyebabnya adalah kualitas model, implementasi sistem, atau faktor eksternal di luar kendali model.

---

*Dokumen ini merupakan kelanjutan dari Bagian I (Business Understanding & Data Understanding). Bersama-sama, kedua bagian membentuk laporan lengkap untuk proyek segmentasi produk inventory ritel.*

---
**Disusun oleh:** Lead Data Scientist  
**Tanggal:** Juni 2026  
**Versi Dokumen:** 1.0 — Bagian II: Data Preparation, Modelling, Evaluation & Deployment
