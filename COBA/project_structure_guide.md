# Panduan Struktur Proyek Data Science Kelas Produksi
## Proyek: Peningkatan Efisiensi Inventory melalui Segmentasi Produk Berbasis Data Historis Penjualan
### Metodologi CRISP-DM | Model K-Means Clustering | Fitur RFM | Tim 4 Orang

---

# Daftar Isi

1. [Filosofi Desain Struktur Proyek](#1-filosofi-desain-struktur-proyek)
2. [Struktur Direktori Lengkap](#2-struktur-direktori-lengkap)
3. [Penjelasan Setiap Komponen](#3-penjelasan-setiap-komponen)
4. [Contoh Isi File Kunci](#4-contoh-isi-file-kunci)
5. [Praktik Terbaik](#5-praktik-terbaik)
6. [Alur Kerja Tim: Dari Kickoff hingga Deploy](#6-alur-kerja-tim-dari-kickoff-hingga-deploy)

---

# 1. Filosofi Desain Struktur Proyek

Struktur proyek ini dirancang berdasarkan tiga prinsip utama yang menjamin keberlangsungan dan kualitas proyek data science skala tim:

**Reproducibility First** — Setiap anggota tim baru harus dapat mereproduksi seluruh pipeline dari data mentah hingga model terlatih hanya dengan menjalankan tiga perintah: `conda env create`, `make data`, dan `make train`. Tidak ada langkah manual yang tersembunyi.

**Separation of Concerns** — Notebook eksplorasi (kode coba-coba) dipisahkan secara ketat dari skrip produksi (kode yang dijalankan di server). Kode yang belum melewati code review tidak pernah masuk ke direktori `src/`.

**Auditability** — Setiap perubahan data, parameter model, dan hasil evaluasi dicatat secara otomatis sehingga pertanyaan "mengapa model bulan lalu memberi hasil berbeda?" selalu dapat dijawab.

---

# 2. Struktur Direktori Lengkap

```
inventory-segmentation/
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # Pipeline CI otomatis saat PR ke main
│   │   └── model-refresh.yml         # Jadwal re-training model kuartalan
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md             # Template laporan bug terstandarisasi
│   │   └── feature_request.md        # Template permintaan fitur baru
│   └── pull_request_template.md      # Checklist wajib sebelum PR di-merge
│
├── config/
│   ├── config.yaml                   # Seluruh parameter proyek terpusat
│   ├── logging.yaml                  # Konfigurasi format dan level logging
│   └── great_expectations/
│       ├── expectations/
│       │   ├── raw_data_suite.json   # Aturan validasi data mentah
│       │   └── rfm_suite.json        # Aturan validasi fitur RFM hasil FE
│       └── checkpoints/
│           └── pipeline_checkpoint.yml # Titik validasi dalam pipeline
│
├── data/
│   ├── raw/                          # Data mentah — JANGAN pernah dimodifikasi
│   │   └── online_retail.xlsx        # File sumber dari UCI Repository
│   ├── interim/                      # Data setelah cleaning, belum final
│   │   └── online_retail_cleaned.parquet
│   ├── processed/                    # Data siap model — output feature engineering
│   │   ├── rfm_features.parquet      # Fitur RFM per SKU sebelum scaling
│   │   └── rfm_scaled.npy            # Fitur RFM setelah StandardScaler
│   └── external/                     # Data pendukung dari sumber luar
│       └── country_region_map.csv    # Pemetaan negara ke region bisnis
│
├── docs/
│   ├── business/
│   │   ├── business_understanding.md # Dokumen BA: latar belakang & tujuan
│   │   └── data_understanding.md     # Dokumen DA: kualitas & eksplorasi data
│   ├── technical/
│   │   ├── data_dictionary.md        # Kamus data lengkap semua kolom
│   │   ├── modeling_report.md        # Laporan: metodologi & hasil pemodelan
│   │   └── api_reference.md          # Dokumentasi fungsi publik src/
│   ├── reports/
│   │   ├── final_report.pdf          # Laporan PDF final untuk stakeholder
│   │   └── executive_summary.pdf     # Ringkasan satu halaman untuk manajemen
│   └── diagrams/
│       ├── pipeline_flow.png         # Diagram alur pipeline end-to-end
│       └── cluster_profiles.png      # Visualisasi profil 4 segmen produk
│
├── models/
│   ├── kmeans_v1.0.pkl               # Model K-Means terlatih versi produksi
│   ├── scaler_v1.0.pkl               # StandardScaler yang disesuaikan data train
│   └── metrics/
│       ├── silhouette_scores.json    # Skor Silhouette per nilai K
│       └── cluster_profiles.json     # Statistik sentroid per klaster
│
├── notebooks/
│   ├── 01_eda/
│   │   ├── 01.1_data_overview.ipynb  # Eksplorasi awal: shape, tipe, missing
│   │   ├── 01.2_distribution_analysis.ipynb # Distribusi Quantity, Price, negara
│   │   └── 01.3_temporal_analysis.ipynb     # Tren bulanan dan musiman
│   ├── 02_preprocessing/
│   │   ├── 02.1_cleaning_exploration.ipynb  # Eksperimen pembersihan data
│   │   └── 02.2_outlier_treatment.ipynb     # Analisis dan penanganan outlier
│   ├── 03_modeling/
│   │   ├── 03.1_rfm_construction.ipynb      # Konstruksi dan validasi fitur RFM
│   │   ├── 03.2_elbow_silhouette.ipynb      # Penentuan K optimal
│   │   └── 03.3_kmeans_final.ipynb          # Fitting model final & interpretasi
│   ├── 04_evaluation/
│   │   ├── 04.1_cluster_profiling.ipynb     # Profil RFM per klaster
│   │   └── 04.2_pca_visualization.ipynb     # Visualisasi PCA 2D hasil klaster
│   └── 05_reporting/
│       └── 05.1_dashboard_prototype.ipynb   # Prototipe visual dashboard Streamlit
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                 # Memuat data dari berbagai format sumber
│   │   ├── cleaner.py                # Logika pembersihan data tervalidasi
│   │   └── validator.py              # Integrasi Great Expectations ke pipeline
│   ├── features/
│   │   ├── __init__.py
│   │   ├── rfm_builder.py            # Konstruksi fitur Recency, Frequency, Monetary
│   │   ├── transformer.py            # Winsorization, log transform, StandardScaler
│   │   └── feature_store.py          # Menyimpan dan memuat fitur dari disk
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Melatih K-Means dengan parameter dari config
│   │   ├── evaluator.py              # Menghitung Silhouette, DB Index, visualisasi
│   │   └── predictor.py              # Memuat model tersimpan dan memprediksi klaster
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── eda_plots.py              # Fungsi plot untuk eksplorasi data
│   │   └── cluster_plots.py          # Fungsi plot untuk visualisasi klaster
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── dashboard.py              # Aplikasi Streamlit dashboard monitoring
│   │   └── alerting.py               # Pengiriman alert ke Slack webhook
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py          # Memuat dan memvalidasi config.yaml
│       └── logger.py                 # Inisialisasi logger terstandarisasi
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Fixtures pytest yang digunakan lintas test
│   ├── unit/
│   │   ├── test_cleaner.py           # Unit test fungsi pembersihan data
│   │   ├── test_rfm_builder.py       # Unit test konstruksi fitur RFM
│   │   └── test_transformer.py       # Unit test winsorization & transformasi
│   ├── integration/
│   │   ├── test_pipeline_e2e.py      # Test pipeline dari data mentah ke klaster
│   │   └── test_model_persistence.py # Test simpan/muat model .pkl
│   └── data/
│       └── sample_data.csv           # Dataset minimal untuk keperluan testing
│
├── scripts/
│   ├── run_pipeline.py               # Entry point: menjalankan pipeline end-to-end
│   ├── run_training.py               # Menjalankan hanya tahap training model
│   ├── run_evaluation.py             # Menghasilkan laporan evaluasi model
│   ├── run_monitoring.py             # Meluncurkan dashboard Streamlit
│   └── refresh_model.py              # Script re-training untuk jadwal kuartalan
│
├── airflow/
│   ├── dags/
│   │   ├── weekly_rfm_refresh.py     # DAG: refresh fitur RFM setiap Senin
│   │   └── quarterly_retrain.py      # DAG: re-training model setiap kuartal
│   └── plugins/
│       └── inventory_operators.py    # Custom Airflow operator untuk proyek ini
│
├── .dvc/
│   └── config                        # Konfigurasi remote storage DVC
│
├── .github/                          # (Didefinisikan di atas)
├── .gitignore                        # Daftar file yang diabaikan Git
├── .flake8                           # Konfigurasi linter PEP8
├── .pre-commit-config.yaml           # Hook pre-commit: black, flake8, isort
├── data.dvc                          # DVC tracking untuk folder data/
├── Dockerfile                        # Container image untuk deployment
├── docker-compose.yml                # Orkestrasi multi-container (app + airflow)
├── environment.yml                   # Reproduksi environment Conda
├── Makefile                          # Perintah shortcut untuk operasi umum
├── pyproject.toml                    # Konfigurasi black, isort, pytest
├── requirements.txt                  # Dependensi pip (auto-generate dari conda)
├── setup.py                          # Instalasi package src/ sebagai editable
└── README.md                         # Dokumentasi utama proyek
```

---

# 3. Penjelasan Setiap Komponen

## Direktori `data/` — Lapisan Penyimpanan Data

Direktori ini mengikuti prinsip **immutability pada data mentah**: folder `raw/` diperlakukan sebagai *read-only* secara permanen. Tidak ada skrip yang boleh menulis ke `raw/` setelah file sumber ditempatkan di sana. Pembersihan dan transformasi selalu menghasilkan file baru di `interim/` dan `processed/`, sehingga seluruh lineage data dapat ditelusuri ke belakang.

Folder `processed/` menyimpan file dalam format **Parquet** (bukan CSV) karena Parquet menyimpan tipe data secara eksplisit, mendukung kompresi kolumnar yang efisien untuk dataset ratusan ribu baris, dan jauh lebih cepat dibaca oleh Pandas maupun Spark jika proyek berkembang ke skala yang lebih besar.

## Direktori `notebooks/` — Zona Eksplorasi

Setiap notebook diberi awalan nomor dua digit yang merepresentasikan urutan tahapan CRISP-DM, diikuti nomor sub-tahap. Penamaan ini memastikan bahwa siapa pun yang membuka folder dapat memahami alur analisis tanpa perlu membaca isi file terlebih dahulu. Notebook di folder ini **tidak digunakan di lingkungan produksi** — fungsinya murni untuk eksplorasi, dokumentasi keputusan analitik, dan komunikasi antar anggota tim.

## Direktori `src/` — Kode Produksi

Folder ini berisi seluruh kode yang telah melewati code review dan siap dijalankan secara otomatis. Setiap subfolder merepresentasikan satu lapisan dalam pipeline: `data/` untuk ingestion dan cleaning, `features/` untuk transformasi, `models/` untuk training dan evaluasi, `visualization/` untuk output visual, `monitoring/` untuk operasional pasca-deploy. Seluruh fungsi publik di folder ini wajib memiliki docstring dan unit test yang sesuai.

## Direktori `tests/` — Jaminan Kualitas Kode

Folder `unit/` berisi test yang menguji satu fungsi secara terisolasi dengan data sintetis minimal. Folder `integration/` berisi test yang menjalankan beberapa komponen secara bersamaan untuk memverifikasi bahwa antarmuka antar modul berfungsi dengan benar. Dataset `tests/data/sample_data.csv` berisi 200–500 baris yang cukup representatif untuk menjalankan seluruh pipeline dalam hitungan detik, tanpa perlu mengakses data produksi.

## Direktori `models/` — Artefak Model

Setiap file model diberi versi semantik (`v1.0`, `v1.1`, dst.) sehingga rollback ke versi sebelumnya dapat dilakukan dengan mengubah satu baris di `config.yaml`. File `scaler_v1.0.pkl` selalu disimpan berdampingan dengan model karena StandardScaler harus identik antara waktu training dan waktu prediksi — menyimpannya secara terpisah adalah salah satu sumber bug tersembunyi yang paling umum dalam proyek ML.

---

# 4. Contoh Isi File Kunci

## 4.1 `README.md`

```markdown
# Inventory Segmentation — Segmentasi Produk Berbasis RFM & K-Means

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CI Status](https://github.com/org/inventory-segmentation/actions/workflows/ci.yml/badge.svg)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
![CRISP-DM](https://img.shields.io/badge/methodology-CRISP--DM-orange)

Proyek ini mengimplementasikan segmentasi produk berbasis **fitur RFM (Recency,
Frequency, Monetary)** menggunakan **K-Means Clustering** untuk mengoptimalkan
manajemen inventory ritel. Produk dikelompokkan ke dalam 4 segmen strategis:
⭐ Bintang, 🔄 Andalan Harian, ⚠️ Evaluasi, dan 💤 Tidur Panjang.

**Target bisnis:** Menurunkan biaya penyimpanan 15% dan meningkatkan inventory
turnover 10% dalam 12 bulan.

---

## Hasil Utama

| Metrik | Nilai |
|---|---|
| Silhouette Score (K=4) | **0,543** |
| Jumlah Produk Tersegmentasi | 4.070 SKU |
| Varians Explained (PCA 2D) | 84,5% |
| Periode Data | Des 2010 – Des 2011 |

![Cluster Visualization](docs/diagrams/cluster_profiles.png)

---

## Struktur Repositori

\`\`\`
inventory-segmentation/
├── config/          # Parameter terpusat
├── data/            # Data mentah, interim, dan processed
├── notebooks/       # Eksplorasi CRISP-DM per tahap
├── src/             # Kode produksi tervalidasi
├── tests/           # Unit & integration tests
├── scripts/         # Entry point pipeline
└── airflow/         # DAG penjadwalan
\`\`\`

---

## Instalasi

### Menggunakan Conda (Direkomendasikan)

\`\`\`bash
# 1. Clone repositori
git clone https://github.com/org/inventory-segmentation.git
cd inventory-segmentation

# 2. Buat environment dari file yang telah terkunci
conda env create -f environment.yml

# 3. Aktifkan environment
conda activate inventory-seg

# 4. Install package src/ dalam mode editable
pip install -e .
\`\`\`

### Menggunakan pip (Alternatif)

\`\`\`bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
\`\`\`

---

## Menjalankan Pipeline

\`\`\`bash
# Jalankan seluruh pipeline dari data mentah hingga model terlatih
make all

# Atau jalankan tahap per tahap:
make data        # Cleaning + feature engineering
make train       # Training K-Means
make evaluate    # Laporan evaluasi + visualisasi
make deploy      # Luncurkan dashboard Streamlit

# Bersihkan artefak sementara
make clean
\`\`\`

---

## Menjalankan Tests

\`\`\`bash
# Jalankan seluruh test suite dengan coverage report
pytest tests/ -v --cov=src --cov-report=html

# Hanya unit test (lebih cepat saat development)
pytest tests/unit/ -v

# Hanya integration test
pytest tests/integration/ -v
\`\`\`

---

## Kontribusi

Baca [CONTRIBUTING.md](CONTRIBUTING.md) untuk panduan branching model,
konvensi commit, dan proses code review.

---

## Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE). Dataset sumber
(UCI Online Retail) dilisensikan di bawah CC BY 4.0.

---

## Tim

| Peran | Nama | Tanggung Jawab |
|---|---|---|
| Lead Data Scientist | [Nama] | Arsitektur pipeline, modeling |
| Data Engineer | [Nama] | Infrastruktur data, Airflow |
| ML Engineer | [Nama] | Deployment, monitoring |
| Business Analyst | [Nama] | Validasi bisnis, pelaporan |
```

---

## 4.2 `config/config.yaml`

```yaml
# =============================================================================
# config.yaml — Parameter Terpusat Proyek Segmentasi Inventory
# Versi: 1.0.0 | Diperbarui: 2026-06-01
# Semua path bersifat relatif terhadap root direktori proyek.
# =============================================================================

project:
  name: "inventory-segmentation"
  version: "1.0.0"
  description: "Segmentasi produk ritel berbasis RFM dan K-Means"
  random_state: 42
  snapshot_date: null          # null = T+1 dari tanggal transaksi terakhir

# -----------------------------------------------------------------------------
# PATH KONFIGURASI
# -----------------------------------------------------------------------------
paths:
  data:
    raw: "data/raw/online_retail.xlsx"
    interim: "data/interim/online_retail_cleaned.parquet"
    processed:
      rfm_features: "data/processed/rfm_features.parquet"
      rfm_scaled:   "data/processed/rfm_scaled.npy"
    external:
      country_map: "data/external/country_region_map.csv"

  models:
    kmeans:  "models/kmeans_v1.0.pkl"
    scaler:  "models/scaler_v1.0.pkl"
    metrics: "models/metrics/"

  outputs:
    reports:   "docs/reports/"
    figures:   "docs/diagrams/"
    logs:      "logs/"

# -----------------------------------------------------------------------------
# SUMBER DATA
# -----------------------------------------------------------------------------
data_source:
  url: "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
  license: "CC BY 4.0"
  provider: "UCI Machine Learning Repository"
  citation: "Chen, D. (2015). Online Retail. UCI Machine Learning Repository."

# -----------------------------------------------------------------------------
# PARAMETER PEMBERSIHAN DATA
# -----------------------------------------------------------------------------
cleaning:
  # Kode produk non-standar yang dihapus dari analisis
  non_product_codes:
    - "POST"           # Biaya pengiriman
    - "D"              # Diskon manual
    - "M"              # Penyesuaian manual
    - "BANK CHARGES"   # Biaya bank
    - "PADS"           # Pad internal
    - "DOT"            # Administrasi sistem
    - "CRUK"           # Donasi amal
    - "gift_0001"      # Kartu hadiah
    - "gift_0002"
    - "gift_0003"

  filters:
    remove_cancelled: true         # Hapus InvoiceNo dengan prefix 'C'
    min_quantity: 1                # Quantity harus > 0
    min_unit_price: 0.01           # UnitPrice harus > 0 (toleransi £0,01)
    require_description: true      # Hapus baris tanpa Description

# -----------------------------------------------------------------------------
# PARAMETER FEATURE ENGINEERING (RFM)
# -----------------------------------------------------------------------------
feature_engineering:
  rfm:
    recency_unit: "days"           # Satuan Recency (hari)
    monetary_metric: "quantity"    # "quantity" atau "revenue"
    frequency_metric: "unique_invoices"

  winsorization:
    lower_percentile: 0.01         # Persentil bawah (1%)
    upper_percentile: 0.99         # Persentil atas (99%)
    apply_to:
      - "Recency"
      - "Frequency"
      - "Monetary"

  log_transform:
    apply_to:
      - "Frequency"                # Monetary dan Frequency ditransformasi log
      - "Monetary"
    method: "log1p"                # log(1+x) untuk menangani nilai 0

  scaling:
    method: "StandardScaler"       # Alternatif: "MinMaxScaler", "RobustScaler"

# -----------------------------------------------------------------------------
# PARAMETER MODEL K-MEANS
# -----------------------------------------------------------------------------
model:
  kmeans:
    n_clusters: 4                  # Jumlah klaster final (dari Elbow + Silhouette)
    init: "k-means++"             # Strategi inisialisasi sentroid
    n_init: 50                     # Jumlah run dengan inisialisasi berbeda
    max_iter: 1000                 # Iterasi maksimum per run
    tol: 1.0e-6                    # Toleransi konvergensi
    algorithm: "lloyd"             # "lloyd" atau "elkan"

  hyperparameter_search:
    k_range: [2, 3, 4, 5, 6, 7, 8, 9, 10]  # Rentang K untuk Elbow & Silhouette

# -----------------------------------------------------------------------------
# KRITERIA EVALUASI
# -----------------------------------------------------------------------------
evaluation:
  thresholds:
    min_silhouette_score: 0.50     # Model gagal jika di bawah nilai ini
    max_davies_bouldin: 1.00       # DB Index harus di bawah 1,0
    min_cluster_size_pct: 0.05     # Klaster terkecil harus ≥ 5% total produk
    max_cluster_size_pct: 0.50     # Klaster terbesar tidak boleh > 50%

  robustness:
    n_bootstrap_runs: 20           # Jumlah sub-sampel untuk uji stabilitas
    sample_fraction: 0.80          # Fraksi data per sub-sampel
    max_silhouette_std: 0.05       # Std dev skor Silhouette antar run

# -----------------------------------------------------------------------------
# KONFIGURASI CLUSTER LABEL
# -----------------------------------------------------------------------------
cluster_labels:
  # Urutan label berdasarkan ranking Monetary DESC, Recency ASC
  # (disesuaikan setelah melihat sentroid aktual pasca training)
  mapping:
    0: "Bintang"
    1: "Tidur Panjang"
    2: "Andalan Harian"
    3: "Evaluasi"

  business_rules:
    Bintang:
      reorder_policy: "CONTINUOUS_REVIEW"
      auto_reorder: true
      service_level: 0.975
      review_cycle_days: 1

    Andalan Harian:
      reorder_policy: "PERIODIC_REVIEW"
      auto_reorder: false
      service_level: 0.950
      review_cycle_days: 14

    Evaluasi:
      reorder_policy: "CONDITIONAL_REORDER"
      auto_reorder: false
      service_level: 0.900
      review_cycle_days: 30

    Tidur Panjang:
      reorder_policy: "HOLD"
      auto_reorder: false
      service_level: null
      review_cycle_days: null

# -----------------------------------------------------------------------------
# KONFIGURASI MONITORING & ALERTING
# -----------------------------------------------------------------------------
monitoring:
  dashboard:
    port: 8501
    host: "0.0.0.0"
    refresh_interval_seconds: 3600   # Refresh data dashboard setiap jam

  alerts:
    slack_webhook_env_var: "SLACK_WEBHOOK_URL"   # Baca dari env variable
    triggers:
      stockout_threshold_days: 3     # Alert jika stok Bintang < 3 hari
      recency_spike_days: 14         # Alert jika Recency Bintang naik > 14 hari
      cluster_migration_pct: 0.15    # Alert jika > 15% produk migrasi klaster

  refresh_schedule:
    rfm_features: "0 6 * * MON"     # Cron: setiap Senin 06.00
    model_retrain: "0 3 1 1,4,7,10 *"  # Cron: 1 Jan, Apr, Jul, Okt, 03.00
```

---

## 4.3 `environment.yml`

```yaml
name: inventory-seg
channels:
  - conda-forge
  - defaults

dependencies:
  # Python runtime
  - python=3.10.14

  # Data manipulation
  - pandas=2.2.2
  - numpy=1.26.4
  - openpyxl=3.1.2          # Membaca file .xlsx sumber data UCI
  - pyarrow=15.0.2           # Backend Parquet untuk pandas

  # Machine Learning
  - scikit-learn=1.4.2
  - scipy=1.13.0

  # Visualisasi
  - matplotlib=3.8.4
  - seaborn=0.13.2
  - plotly=5.22.0

  # Konfigurasi & utilitas
  - pyyaml=6.0.1
  - python-dotenv=1.0.1
  - tqdm=4.66.4

  # Pengujian
  - pytest=8.2.1
  - pytest-cov=5.0.0

  # Kualitas kode
  - black=24.4.2
  - flake8=7.0.0
  - isort=5.13.2
  - pre-commit=3.7.1

  # Notebook
  - jupyterlab=4.2.1
  - ipykernel=6.29.4
  - nbconvert=7.16.4

  # Airflow (diinstal via pip — versi conda tidak selalu ter-update)
  # Deployment
  - pip

  - pip:
    # Dashboard monitoring
    - streamlit==1.35.0

    # Validasi data
    - great-expectations==0.18.15

    # Data version control
    - dvc==3.51.2
    - dvc-s3==3.2.0            # Jika remote storage di AWS S3

    # Alerting
    - slack-sdk==3.30.0

    # Orkestrasi
    - apache-airflow==2.9.1

    # Package proyek sendiri (editable install)
    - -e .
```

---

## 4.4 `.gitignore`

```gitignore
# =============================================================================
# .gitignore — Inventory Segmentation Project
# Prinsip: Commit kode dan konfigurasi. Jangan commit data, model besar,
#          credential, atau artefak yang dapat dihasilkan ulang.
# =============================================================================

# -----------------------------------------------------------------------------
# DATA — Gunakan DVC untuk versi kontrol data
# -----------------------------------------------------------------------------
data/raw/
data/interim/
data/processed/
data/external/
# Pengecualian: file DVC tracking boleh di-commit
!data/raw/.gitkeep
!data/interim/.gitkeep
!data/processed/.gitkeep
!*.dvc

# -----------------------------------------------------------------------------
# MODEL — File .pkl bisa berukuran besar, lacak dengan DVC atau Git LFS
# -----------------------------------------------------------------------------
models/*.pkl
models/*.joblib
models/*.h5
# Pengecualian: file metrics JSON tetap di-commit untuk tracking performa
!models/metrics/

# -----------------------------------------------------------------------------
# LOG & OUTPUT SEMENTARA
# -----------------------------------------------------------------------------
logs/
*.log
docs/reports/tmp/

# -----------------------------------------------------------------------------
# PYTHON RUNTIME
# -----------------------------------------------------------------------------
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg
*.egg-info/
dist/
build/
.eggs/

# -----------------------------------------------------------------------------
# VIRTUAL ENVIRONMENT
# -----------------------------------------------------------------------------
.venv/
env/
venv/
ENV/
.conda/

# -----------------------------------------------------------------------------
# JUPYTER NOTEBOOK
# -----------------------------------------------------------------------------
.ipynb_checkpoints/
*/.ipynb_checkpoints/
# Catatan: Output cell notebook sengaja dikosongkan sebelum commit
# (dikonfigurasi via pre-commit hook nbstripout)

# -----------------------------------------------------------------------------
# CREDENTIAL & SECRETS
# -----------------------------------------------------------------------------
.env
.env.local
.env.production
secrets/
*.pem
*.key
config/secrets.yaml

# -----------------------------------------------------------------------------
# IDE & EDITOR
# -----------------------------------------------------------------------------
.vscode/
.idea/
*.swp
*.swo
.DS_Store
Thumbs.db

# -----------------------------------------------------------------------------
# COVERAGE & TESTING
# -----------------------------------------------------------------------------
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/

# -----------------------------------------------------------------------------
# AIRFLOW
# -----------------------------------------------------------------------------
airflow/logs/
airflow/plugins/__pycache__/

# -----------------------------------------------------------------------------
# DVC
# -----------------------------------------------------------------------------
.dvc/tmp/
.dvc/cache/
```

---

## 4.5 `Makefile`

```makefile
# =============================================================================
# Makefile — Inventory Segmentation Pipeline
# Penggunaan: make [target]
# Contoh:    make all        → jalankan pipeline lengkap
#            make data       → hanya preprocessing
#            make clean      → hapus artefak sementara
# =============================================================================

# Variabel konfigurasi
PYTHON        := python
CONFIG        := config/config.yaml
SRC_DIR       := src
SCRIPTS_DIR   := scripts
TEST_DIR      := tests
VENV          := .venv
COVERAGE_MIN  := 80

.PHONY: all data train evaluate deploy test lint format clean help

# -----------------------------------------------------------------------------
# TARGET UTAMA
# -----------------------------------------------------------------------------

## all: Jalankan pipeline lengkap dari data mentah hingga model ter-deploy
all: data train evaluate
	@echo "✅ Pipeline selesai. Jalankan 'make deploy' untuk meluncurkan dashboard."

## data: Jalankan preprocessing dan feature engineering
data:
	@echo "🔄 [1/3] Memuat dan membersihkan data mentah..."
	$(PYTHON) $(SCRIPTS_DIR)/run_pipeline.py --stage clean --config $(CONFIG)
	@echo "🔄 [2/3] Membangun fitur RFM..."
	$(PYTHON) $(SCRIPTS_DIR)/run_pipeline.py --stage features --config $(CONFIG)
	@echo "✅ Data siap di data/processed/"

## train: Latih model K-Means dengan parameter dari config
train:
	@echo "🤖 Melatih model K-Means (n_clusters=4, n_init=50)..."
	$(PYTHON) $(SCRIPTS_DIR)/run_training.py --config $(CONFIG)
	@echo "✅ Model tersimpan di models/"

## evaluate: Hasilkan laporan evaluasi lengkap dan visualisasi
evaluate:
	@echo "📊 Mengevaluasi model dan menghasilkan laporan..."
	$(PYTHON) $(SCRIPTS_DIR)/run_evaluation.py --config $(CONFIG)
	@echo "✅ Laporan tersimpan di docs/reports/"

## deploy: Luncurkan dashboard Streamlit monitoring segmen
deploy:
	@echo "🚀 Meluncurkan dashboard monitoring segmen produk..."
	$(PYTHON) $(SCRIPTS_DIR)/run_monitoring.py --config $(CONFIG)

## refresh: Jalankan re-training model (untuk jadwal kuartalan)
refresh:
	@echo "🔁 Re-training model dengan data terbaru..."
	$(PYTHON) $(SCRIPTS_DIR)/refresh_model.py --config $(CONFIG)
	@echo "✅ Model baru tersimpan. Lakukan validasi sebelum deploy ke produksi."

# -----------------------------------------------------------------------------
# TESTING & KUALITAS KODE
# -----------------------------------------------------------------------------

## test: Jalankan seluruh test suite dengan coverage report
test:
	@echo "🧪 Menjalankan test suite..."
	pytest $(TEST_DIR)/ -v \
		--cov=$(SRC_DIR) \
		--cov-report=html:htmlcov \
		--cov-report=term-missing \
		--cov-fail-under=$(COVERAGE_MIN)
	@echo "✅ Coverage report di htmlcov/index.html"

## test-unit: Jalankan hanya unit test (lebih cepat saat development)
test-unit:
	pytest $(TEST_DIR)/unit/ -v

## test-integration: Jalankan hanya integration test
test-integration:
	pytest $(TEST_DIR)/integration/ -v

## lint: Periksa kode dengan flake8
lint:
	@echo "🔍 Memeriksa style kode..."
	flake8 $(SRC_DIR)/ $(SCRIPTS_DIR)/ --config=.flake8
	@echo "✅ Tidak ada pelanggaran style."

## format: Format kode otomatis dengan black dan isort
format:
	@echo "✨ Memformat kode dengan black dan isort..."
	black $(SRC_DIR)/ $(SCRIPTS_DIR)/ $(TEST_DIR)/
	isort $(SRC_DIR)/ $(SCRIPTS_DIR)/ $(TEST_DIR)/
	@echo "✅ Kode telah diformat."

## format-check: Periksa formatting tanpa mengubah file (untuk CI)
format-check:
	black --check $(SRC_DIR)/ $(SCRIPTS_DIR)/
	isort --check-only $(SRC_DIR)/ $(SCRIPTS_DIR)/

# -----------------------------------------------------------------------------
# DATA VERSION CONTROL (DVC)
# -----------------------------------------------------------------------------

## dvc-pull: Unduh data dan model dari remote storage DVC
dvc-pull:
	@echo "⬇️  Mengunduh data dari DVC remote..."
	dvc pull

## dvc-push: Unggah data dan model terbaru ke remote storage DVC
dvc-push:
	@echo "⬆️  Mengunggah data ke DVC remote..."
	dvc push

# -----------------------------------------------------------------------------
# PEMELIHARAAN
# -----------------------------------------------------------------------------

## clean: Hapus artefak sementara (log, cache, output test)
clean:
	@echo "🧹 Membersihkan artefak sementara..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage logs/*.log
	@echo "✅ Direktori bersih."

## clean-models: Hapus model lama (HATI-HATI: tidak dapat dipulihkan tanpa DVC)
clean-models:
	@echo "⚠️  Menghapus file model. Pastikan sudah di-push ke DVC remote."
	rm -f models/*.pkl models/*.joblib

## install-hooks: Pasang pre-commit hooks ke repository lokal
install-hooks:
	pre-commit install
	@echo "✅ Pre-commit hooks terpasang."

## help: Tampilkan daftar target yang tersedia
help:
	@grep -E '^## ' Makefile | sed 's/## //' | column -t -s ':'
```

---

## 4.6 `Dockerfile`

```dockerfile
# =============================================================================
# Dockerfile — Inventory Segmentation Monitoring Dashboard
# Base: python:3.10-slim-bookworm (image resmi, slim untuk ukuran minimal)
# Entrypoint: Dashboard Streamlit monitoring segmen produk
# Build: docker build -t inventory-seg:1.0 .
# Run:   docker run -p 8501:8501 --env-file .env inventory-seg:1.0
# =============================================================================

FROM python:3.10-slim-bookworm

# -----------------------------------------------------------------------------
# Metadata image
# -----------------------------------------------------------------------------
LABEL maintainer="data-science-team@company.com"
LABEL version="1.0.0"
LABEL description="Inventory Segmentation — Dashboard Monitoring Segmen Produk"

# -----------------------------------------------------------------------------
# Variabel environment untuk konfigurasi runtime
# -----------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1    \
    PYTHONUNBUFFERED=1           \
    PIP_NO_CACHE_DIR=1           \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app                \
    STREAMLIT_PORT=8501          \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# -----------------------------------------------------------------------------
# Sistem dependensi minimal
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Setup direktori aplikasi
# -----------------------------------------------------------------------------
WORKDIR ${APP_HOME}

# -----------------------------------------------------------------------------
# Instal dependensi Python terlebih dahulu (manfaatkan Docker layer cache)
# Layer ini hanya di-rebuild jika requirements.txt berubah
# -----------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Copy source code aplikasi
# Urutan ini memaksimalkan cache: dependensi jarang berubah, kode sering berubah
# -----------------------------------------------------------------------------
COPY config/       ${APP_HOME}/config/
COPY src/          ${APP_HOME}/src/
COPY models/       ${APP_HOME}/models/
COPY scripts/      ${APP_HOME}/scripts/
COPY setup.py      ${APP_HOME}/
COPY README.md     ${APP_HOME}/

# -----------------------------------------------------------------------------
# Instal package src/ sebagai editable dalam container
# -----------------------------------------------------------------------------
RUN pip install -e .

# -----------------------------------------------------------------------------
# Buat user non-root untuk keamanan (jangan jalankan sebagai root)
# -----------------------------------------------------------------------------
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser ${APP_HOME}
USER appuser

# -----------------------------------------------------------------------------
# Expose port dashboard Streamlit
# -----------------------------------------------------------------------------
EXPOSE ${STREAMLIT_PORT}

# -----------------------------------------------------------------------------
# Health check — verifikasi bahwa dashboard responsif
# -----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${STREAMLIT_PORT}/_stcore/health || exit 1

# -----------------------------------------------------------------------------
# Entrypoint — jalankan dashboard monitoring saat container start
# -----------------------------------------------------------------------------
ENTRYPOINT ["streamlit", "run", "src/monitoring/dashboard.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
```

---

## 4.7 `.github/workflows/ci.yml`

```yaml
# =============================================================================
# ci.yml — Pipeline CI/CD untuk Proyek Inventory Segmentation
# Trigger: Setiap push ke branch main atau pull request ke branch main
# Jobs:
#   1. lint-and-format  — flake8, black check, isort check
#   2. test             — pytest dengan coverage report
#   3. validate-config  — validasi config.yaml dapat dimuat tanpa error
# =============================================================================

name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: "3.10"
  MIN_COVERAGE: 80

jobs:
  # ---------------------------------------------------------------------------
  # Job 1: Pemeriksaan Kualitas Kode
  # ---------------------------------------------------------------------------
  lint-and-format:
    name: "🔍 Lint & Format Check"
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repositori
        uses: actions/checkout@v4

      - name: Setup Python ${{ env.PYTHON_VERSION }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: "pip"

      - name: Install dependensi linting
        run: |
          pip install flake8 black isort

      - name: Jalankan flake8 (PEP8 compliance)
        run: |
          flake8 src/ scripts/ tests/ \
            --max-line-length=100 \
            --exclude=__pycache__,.venv \
            --statistics
        # Gagal jika ada pelanggaran style

      - name: Periksa formatting black
        run: |
          black --check --diff src/ scripts/ tests/
        # Gagal jika kode belum diformat dengan black

      - name: Periksa urutan import dengan isort
        run: |
          isort --check-only --diff src/ scripts/ tests/
        # Gagal jika urutan import tidak konsisten

  # ---------------------------------------------------------------------------
  # Job 2: Testing & Coverage
  # ---------------------------------------------------------------------------
  test:
    name: "🧪 Unit & Integration Tests"
    runs-on: ubuntu-latest
    needs: lint-and-format   # Hanya jalankan jika lint berhasil

    steps:
      - name: Checkout repositori
        uses: actions/checkout@v4

      - name: Setup Python ${{ env.PYTHON_VERSION }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: "pip"

      - name: Install dependensi proyek
        run: |
          pip install -e ".[dev]"

      - name: Jalankan test suite dengan coverage
        run: |
          pytest tests/ \
            -v \
            --cov=src \
            --cov-report=xml:coverage.xml \
            --cov-report=term-missing \
            --cov-fail-under=${{ env.MIN_COVERAGE }} \
            --tb=short
        # Pipeline gagal jika coverage di bawah MIN_COVERAGE

      - name: Upload coverage ke Codecov
        uses: codecov/codecov-action@v4
        if: success()
        with:
          file: coverage.xml
          flags: unittests
          fail_ci_if_error: false   # Jangan gagalkan CI karena masalah upload

      - name: Upload test results sebagai artifact
        uses: actions/upload-artifact@v4
        if: always()               # Upload bahkan jika test gagal
        with:
          name: test-results
          path: |
            coverage.xml
            htmlcov/

  # ---------------------------------------------------------------------------
  # Job 3: Validasi Konfigurasi
  # ---------------------------------------------------------------------------
  validate-config:
    name: "⚙️ Validasi config.yaml"
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repositori
        uses: actions/checkout@v4

      - name: Setup Python ${{ env.PYTHON_VERSION }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install pyyaml dan dependensi minimal
        run: pip install pyyaml

      - name: Validasi config.yaml dapat dimuat
        run: |
          python -c "
          import yaml, sys
          with open('config/config.yaml') as f:
              cfg = yaml.safe_load(f)
          # Periksa kunci wajib ada
          required_keys = ['project', 'paths', 'model', 'evaluation']
          missing = [k for k in required_keys if k not in cfg]
          if missing:
              print(f'ERROR: Kunci wajib tidak ditemukan: {missing}')
              sys.exit(1)
          # Periksa n_clusters valid
          n_clusters = cfg['model']['kmeans']['n_clusters']
          if not (2 <= n_clusters <= 20):
              print(f'ERROR: n_clusters={n_clusters} di luar rentang valid [2, 20]')
              sys.exit(1)
          print('✅ config.yaml valid.')
          "

  # ---------------------------------------------------------------------------
  # Job 4: Build Docker Image (hanya pada merge ke main)
  # ---------------------------------------------------------------------------
  build-docker:
    name: "🐳 Build Docker Image"
    runs-on: ubuntu-latest
    needs: [lint-and-format, test]
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'

    steps:
      - name: Checkout repositori
        uses: actions/checkout@v4

      - name: Setup Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login ke GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build dan push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

# 5. Praktik Terbaik

## 5.1 Version Control: Git Flow & Conventional Commits

**Branching Model (Git Flow yang Disederhanakan)**

Repositori ini menggunakan tiga jenis branch permanen dan dua jenis branch sementara:

- `main` adalah branch produksi yang hanya menerima merge dari `develop` setelah semua CI pass. Setiap commit di `main` merepresentasikan versi yang siap digunakan.
- `develop` adalah branch integrasi di mana seluruh fitur baru digabungkan sebelum naik ke `main`. Pipeline CI berjalan pada setiap push ke `develop`.
- `feature/[nama-fitur]` dibuat dari `develop` untuk setiap tugas baru. Nama branch menggunakan kebab-case deskriptif: `feature/rfm-winsorization`, `feature/silhouette-dashboard`, bukan `feature/update1`.
- `hotfix/[nama-fix]` dibuat dari `main` untuk perbaikan bug kritis di produksi dan di-merge kembali ke `main` dan `develop` secara bersamaan.
- `release/[versi]` dibuat dari `develop` saat mempersiapkan rilis baru untuk staging review.

**Conventional Commits**

Setiap pesan commit mengikuti format `<type>(<scope>): <description>` untuk memungkinkan changelog otomatis dan pencarian riwayat yang efisien:

```
feat(rfm): tambah fitur TotalRevenue sebagai alternatif Monetary
fix(cleaner): perbaiki filter kode non-produk yang melewatkan prefix 'CRUK'
docs(readme): perbarui instruksi instalasi untuk Python 3.10
test(rfm_builder): tambah test untuk dataset dengan satu produk
refactor(trainer): ekstrak logika Elbow ke fungsi terpisah
ci(github-actions): tambah job validasi config.yaml
```

**Aturan Code Review**

Setiap Pull Request ke `develop` memerlukan minimal satu approval dari anggota tim lain. Review wajib memeriksa: apakah semua fungsi baru memiliki docstring, apakah ada unit test yang sesuai, apakah perubahan konfigurasi didokumentasikan, dan apakah tidak ada data atau credential yang tidak sengaja di-commit.

---

## 5.2 Kolaborasi: Notebook vs. Skrip Produksi

**Pemisahan yang Ketat antara Eksplorasi dan Produksi**

Kode di `notebooks/` dan kode di `src/` memiliki tujuan fundamental yang berbeda dan tidak boleh dicampur. Notebook adalah dokumen hidup tempat analis berpikir keras, membuat kesalahan, dan mendokumentasikan keputusan analitik — kode di dalamnya boleh tidak rapi, berulang, dan penuh komentar eksploratif. Skrip di `src/` adalah kode yang bertanggung jawab terhadap hasil bisnis — harus bersih, teruji, dan terdokumentasi.

**Aturan Penamaan Notebook**

Setiap notebook mengikuti format: `[nomor_tahap].[nomor_sub]_[deskripsi_singkat].ipynb`. Nomor tahap mengikuti urutan CRISP-DM: `01` untuk Data Understanding, `02` untuk Preparation, `03` untuk Modeling, `04` untuk Evaluation, `05` untuk Reporting. Deskripsi singkat menggunakan snake_case. Contoh yang benar: `03.2_elbow_silhouette_analysis.ipynb`. Contoh yang salah: `analisis final REVISI3.ipynb`.

**Konversi Notebook ke Skrip Produksi**

Saat logika dalam notebook siap diproduksikan, prosesnya adalah: (1) ekstrak fungsi-fungsi inti dari cell notebook ke file `.py` di `src/`; (2) tulis unit test untuk setiap fungsi tersebut; (3) pastikan fungsi hanya bergantung pada input parameter dan tidak pada variabel global cell notebook; (4) jalankan `nbconvert` untuk menyimpan versi final notebook sebagai PDF dokumentasi. Notebook asli tetap dipertahankan sebagai rekam jejak eksplorasi.

**Pembersihan Output Sebelum Commit**

Seluruh output cell notebook (grafik, tabel, hasil print) dihapus sebelum commit menggunakan hook `nbstripout`. Ini mencegah diff Git yang besar akibat perubahan output dan menghindari risiko data sensitif tersimpan dalam output.

---

## 5.3 Manajemen Data: DVC dan Kebijakan Akses

**Mengapa DVC Diperlukan**

Git tidak dirancang untuk melacak file biner besar seperti dataset Excel, file Parquet terproses, atau model `.pkl`. DVC mengisi celah ini: file data disimpan di remote storage (AWS S3, Google Cloud Storage, atau server lokal), sementara Git hanya melacak file `.dvc` kecil yang merupakan pointer ke versi data yang tepat. Ini berarti perintah `git checkout v1.0` diikuti `dvc pull` akan mereproduksi secara persis lingkungan data yang digunakan pada versi tersebut.

**Hierarki Penyimpanan Data**

Data mentah (`data/raw/`) disimpan di bucket S3 dengan akses read-only untuk seluruh tim dan write-access terbatas hanya untuk Data Engineer. Data processed (`data/processed/`) dapat ditulis oleh pipeline otomatis. Model (`models/`) hanya ditulis oleh pipeline training yang berjalan di environment CI/CD, bukan di laptop lokal anggota tim.

**Kebijakan Akses**

Tidak ada anggota tim yang mengakses data produksi langsung dari laptop mereka untuk keperluan eksplorasi. Eksplorasi dilakukan menggunakan sampel data (`tests/data/sample_data.csv` atau subset 10% dari data processed). Akses ke data lengkap hanya terjadi melalui pipeline otomatis yang berjalan di server terisolasi.

---

## 5.4 Pengujian: Strategi Multi-Layer

**Unit Test untuk Fungsi Preprocessing**

Setiap fungsi di `src/data/cleaner.py` memiliki minimal tiga test: (1) test kasus normal dengan input valid, (2) test kasus tepi (*edge case*) seperti dataset kosong atau semua nilai nol, dan (3) test yang memverifikasi bahwa fungsi melempar exception yang tepat untuk input yang tidak valid. Gunakan `pytest.fixture` untuk mendefinisikan dataset sampel yang dapat digunakan oleh banyak test tanpa duplikasi.

**Integration Test untuk Pipeline**

Test di `tests/integration/test_pipeline_e2e.py` menjalankan seluruh pipeline dari file CSV sampel hingga output klaster dalam satu test function. Test ini tidak menguji detail implementasi — hanya memverifikasi bahwa (1) pipeline berjalan tanpa error, (2) output memiliki skema yang benar, dan (3) jumlah klaster sesuai konfigurasi.

**Great Expectations untuk Validasi Data**

Great Expectations digunakan sebagai *data contract*: seperangkat aturan yang mendefinisikan apa yang dimaksud dengan "data valid". Contoh ekspektasi yang didefinisikan untuk data mentah: kolom `Quantity` tidak boleh memiliki nilai NULL, nilai `UnitPrice` harus antara 0,01 dan 10.000, dan kolom `InvoiceDate` harus dalam rentang 2010-12-01 hingga 2012-01-01. Validasi dijalankan otomatis pada awal setiap pipeline run, dan pipeline akan berhenti dengan pesan error yang jelas jika ada ekspektasi yang dilanggar.

**Target Coverage**

Seluruh fungsi di `src/data/`, `src/features/`, dan `src/models/` harus mencapai test coverage minimal 80% (dikonfigurasi dalam `Makefile` dan `ci.yml`). Fungsi di `src/visualization/` dan `src/monitoring/` dikecualikan dari requirement ini karena melibatkan rendering grafis yang sulit diuji secara otomatis.

---

## 5.5 Deployment dan Monitoring

**Strategi CI/CD untuk Refresh Model Kuartalan**

Re-training model kuartalan dijalankan sebagai GitHub Actions workflow terpisah (`model-refresh.yml`) yang dijadwalkan dengan cron. Workflow ini menjalankan: (1) `dvc pull` untuk mendapatkan data terbaru, (2) `make data` dan `make train`, (3) protokol validasi otomatis (Silhouette Score, distribusi klaster), dan (4) jika semua validasi lulus, model baru otomatis di-push ke DVC remote dan tagged di Git sebagai `model-v1.x`. Jika validasi gagal, workflow berhenti dan mengirim notifikasi ke Slack.

**Penjadwalan dengan Airflow**

Airflow mengelola dua DAG: `weekly_rfm_refresh` yang memperbarui nilai RFM setiap Senin pagi dari database transaksi, dan `quarterly_retrain` yang menjalankan pipeline training penuh setiap kuartal. Setiap task dalam DAG memiliki `max_retries=2` dan `retry_delay=timedelta(minutes=5)` untuk menangani kegagalan sementara jaringan atau penyimpanan.

**Dashboard dengan Streamlit**

Dashboard Streamlit di `src/monitoring/dashboard.py` dirancang sebagai aplikasi single-page dengan empat panel: ringkasan segmen, pemantauan stok Bintang, tracking likuidasi Tidur Panjang, dan matriks migrasi klaster antar periode. Dashboard membaca data dari file Parquet terbaru di `data/processed/` dan secara otomatis menampilkan versi data kapan file tersebut terakhir diperbarui.

**Alerting ke Slack**

Modul `src/monitoring/alerting.py` mengirim notifikasi ke channel Slack yang ditentukan ketika: (a) Silhouette Score re-training baru di bawah threshold, (b) lebih dari 15% produk Bintang mengalami kenaikan Recency > 14 hari dalam satu minggu, atau (c) stok salah satu produk Bintang mendekati stockout. URL Slack webhook disimpan sebagai environment variable (`SLACK_WEBHOOK_URL`) dan tidak pernah hardcoded dalam kode atau konfigurasi.

---

## 5.6 Portofolio: Repositori yang Menarik bagi Rekruter

**README sebagai Halaman Muka Proyek**

README yang efektif menjawab tiga pertanyaan dalam 30 detik pertama: apa yang dilakukan proyek ini, mengapa hasilnya penting, dan bagaimana cara menjalankannya. Gunakan badge untuk menampilkan CI status, Python version, dan coverage secara visual. Sertakan satu gambar visualisasi hasil klaster di atas fold — rekruter teknis langsung tahu apakah proyek ini relevan dengan kebutuhan mereka.

**Dokumentasi yang Menunjukkan Proses Berpikir**

Rekruter data science berpengalaman tidak hanya menilai kode akhir, tetapi juga proses pengambilan keputusan. Notebook di `notebooks/` yang terdokumentasi dengan baik — termasuk percobaan yang gagal dan alasan mengapa pendekatan tertentu ditinggalkan — menunjukkan kemampuan analitis yang lebih dalam daripada kode produksi yang sempurna tanpa konteks.

**Laporan PDF yang Dapat Dibagikan**

File `docs/reports/final_report.pdf` memungkinkan rekruter non-teknis (HR, manajer bisnis) memahami dampak bisnis proyek tanpa perlu membaca kode. Laporan ini harus menyertakan satu halaman ringkasan eksekutif, visualisasi profil klaster yang bersih, dan proyeksi dampak finansial yang dikuantifikasi.

**Dashboard Live sebagai Diferensiasi**

Mendeploy dashboard Streamlit ke Streamlit Community Cloud (gratis untuk repositori publik) dan menyertakan link-nya di README secara dramatis meningkatkan daya tarik portofolio. Rekruter dapat berinteraksi langsung dengan hasil kerja tanpa perlu setup environment lokal — ini menghilangkan hambatan terbesar untuk mengevaluasi proyek data science.

**Conventional Commits sebagai Bukti Profesionalisme**

Riwayat commit yang bersih dengan conventional commits menunjukkan kedisiplinan engineering. Rekruter yang mengklik tab "Commits" dan melihat pesan seperti `fix: hapus NaN sebelum clustering biar error hilang` versus `fix(cleaner): tangani nilai NaN pada kolom CustomerID sebelum agregasi RFM` mendapatkan sinyal yang sangat berbeda tentang standar kerja kandidat.

---

# 6. Alur Kerja Tim: Dari Kickoff hingga Deploy

## Phase 0 — Setup (Hari 1–2)

**Data Engineer** membuat repositori, mengkonfigurasi branch protection pada `main`, memasang pre-commit hooks, dan mengatur DVC remote. **Lead Data Scientist** menulis `config.yaml` dengan seluruh parameter awal dan mendefinisikan ekspektasi Great Expectations untuk data mentah. Seluruh tim melakukan `conda env create -f environment.yml` dan memverifikasi bahwa `make test` berjalan tanpa error pada sampel data.

Folder yang aktif: `.github/`, `config/`, `environment.yml`, `tests/data/`, `.gitignore`

---

## Phase 1 — Business & Data Understanding (Minggu 1–2)

**Business Analyst** mendokumentasikan latar belakang, pemangku kepentingan, dan kriteria sukses di `docs/business/`. **Data Engineer** mengunduh dataset sumber, menempatkannya di `data/raw/`, menjalankan `dvc add data/raw/` untuk memulai tracking, dan mem-push ke remote storage.

**Lead Data Scientist** memimpin eksplorasi di `notebooks/01_eda/`, mendokumentasikan temuan distribusi, missing value, dan pola tren bulanan. Setiap temuan kritis dikomunikasikan ke tim via PR yang berisi update notebook — bukan via chat — sehingga seluruh diskusi terdokumentasi dalam riwayat repositori.

Folder yang aktif: `data/raw/`, `notebooks/01_eda/`, `docs/business/`, `docs/technical/`

---

## Phase 2 — Data Preparation (Minggu 3–4)

**Lead Data Scientist** mengeksplorasi strategi pembersihan di `notebooks/02_preprocessing/`. Setelah pendekatan divalidasi dalam notebook, **ML Engineer** mengimplementasikan logika yang sama ke dalam fungsi-fungsi di `src/data/cleaner.py` — dengan docstring lengkap dan unit test di `tests/unit/test_cleaner.py`.

**Branching:** ML Engineer membuat `feature/data-cleaning-pipeline` dari `develop`, mengimplementasikan kode, memastikan `make test` pass, lalu membuka PR. Lead Data Scientist mereview PR — memverifikasi bahwa implementasi konsisten dengan keputusan analitik di notebook.

Setelah merge, `make data` dapat dijalankan oleh siapa pun di tim dan menghasilkan output yang identik di `data/interim/` dan `data/processed/`.

Folder yang aktif: `src/data/`, `tests/unit/`, `notebooks/02_preprocessing/`

---

## Phase 3 — Feature Engineering (Minggu 4–5)

**ML Engineer** mengimplementasikan `src/features/rfm_builder.py` dan `src/features/transformer.py` berdasarkan eksplorasi di `notebooks/03_modeling/03.1_rfm_construction.ipynb`. Fungsi Winsorization, log transform, dan StandardScaler diimplementasikan sebagai kelas yang menerima parameter dari `config.yaml` — tidak ada angka hardcoded dalam kode.

**Lead Data Scientist** memvalidasi distribusi fitur RFM hasil transformasi dalam notebook dan mendokumentasikan keputusan (mengapa Frequency di-log transform tetapi Recency tidak) di `docs/technical/modeling_report.md`.

Folder yang aktif: `src/features/`, `notebooks/03_modeling/`, `data/processed/`

---

## Phase 4 — Modeling (Minggu 5–7)

**Lead Data Scientist** menentukan K optimal melalui eksplorasi di `notebooks/03_modeling/03.2_elbow_silhouette.ipynb`, mendokumentasikan seluruh rentang K yang diuji beserta skor yang dihasilkan. Nilai K=4 dikonfirmasi dan dicatat di `config.yaml`.

**ML Engineer** mengimplementasikan `src/models/trainer.py` dengan parameter dari `config.yaml`, menjalankan `make train`, dan memverifikasi bahwa `models/kmeans_v1.0.pkl` dan `models/scaler_v1.0.pkl` tersimpan dengan benar. Model di-push ke DVC remote dan file `.dvc` yang dihasilkan di-commit ke Git.

Folder yang aktif: `src/models/`, `models/`, `notebooks/03_modeling/`

---

## Phase 5 — Evaluation (Minggu 7–8)

**Lead Data Scientist** menjalankan `make evaluate`, memeriksa laporan Silhouette Score, dan melakukan profiling klaster di `notebooks/04_evaluation/`. Empat label bisnis (Bintang, Andalan Harian, Evaluasi, Tidur Panjang) dikonfirmasi bersama **Business Analyst** melalui sesi review bersama.

**Business Analyst** menyusun `docs/reports/final_report.pdf` dan `docs/reports/executive_summary.pdf`, yang disertai visualisasi dari `docs/diagrams/` yang dihasilkan oleh `make evaluate`.

Folder yang aktif: `notebooks/04_evaluation/`, `docs/reports/`, `docs/diagrams/`

---

## Phase 6 — Deployment (Minggu 9–10)

**ML Engineer** mengimplementasikan `src/monitoring/dashboard.py` (Streamlit) dan `src/monitoring/alerting.py` (Slack), membangun Docker image dengan `docker build`, dan mengkonfigurasi Airflow DAGs di `airflow/dags/`.

**Data Engineer** mengkonfigurasi environment produksi: menetapkan environment variables (`SLACK_WEBHOOK_URL`, credential DVC remote), mendeploy container ke server, dan memverifikasi bahwa kedua Airflow DAG terjadwal dengan benar.

**Lead Data Scientist** melakukan final review seluruh dokumentasi, memastikan `README.md` mencerminkan status proyek terkini, dan mem-publish versi final repositori.

Folder yang aktif: `src/monitoring/`, `airflow/`, `scripts/`, `Dockerfile`

---

## Matriks Tanggung Jawab Tim

| Artefak | Lead DS | Data Eng | ML Eng | BA |
|---|---|---|---|---|
| `config/config.yaml` | ✏️ Tulis | 👀 Review | 👀 Review | — |
| `data/raw/` | — | ✏️ Kelola | — | — |
| `notebooks/` | ✏️ Tulis | — | — | 👀 Baca |
| `src/data/` | 📋 Spesifikasi | — | ✏️ Implementasi | — |
| `src/features/` | 📋 Spesifikasi | — | ✏️ Implementasi | — |
| `src/models/` | ✏️ Tulis | — | 👀 Review | — |
| `src/monitoring/` | 📋 Spesifikasi | — | ✏️ Implementasi | — |
| `tests/` | 👀 Review | — | ✏️ Tulis | — |
| `airflow/dags/` | — | ✏️ Tulis | 👀 Review | — |
| `docs/reports/` | 👀 Review | — | — | ✏️ Tulis |
| `Dockerfile` | — | 👀 Review | ✏️ Tulis | — |
| `.github/workflows/` | — | ✏️ Tulis | 👀 Review | — |

**Legenda:** ✏️ Penulis utama | 👀 Wajib review sebelum merge | 📋 Penyedia spesifikasi | — Tidak terlibat langsung

---

*Dokumen ini merupakan panduan hidup yang harus diperbarui seiring perkembangan proyek. Setiap keputusan arsitektur yang berbeda dari panduan ini harus didokumentasikan dalam ADR (Architecture Decision Record) di `docs/technical/`.*

---
**Disusun oleh:** Lead Data Scientist  
**Tanggal:** Juni 2026  
**Versi:** 1.0.0
