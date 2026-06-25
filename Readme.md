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