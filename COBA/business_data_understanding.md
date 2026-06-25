# Laporan Analisis Segmentasi Produk untuk Optimalisasi Inventory Ritel
## Proyek: Peningkatan Efisiensi Inventory melalui Segmentasi Produk Berbasis Data Historis Penjualan

---

# 1. Business Understanding

## 1.1 Latar Belakang

Industri ritel beroperasi di bawah tekanan ganda yang saling bertentangan: kelebihan stok (*overstock*) mengunci modal kerja dalam bentuk barang yang tidak bergerak, sementara kekurangan stok (*understock*) mengakibatkan *lost sales* dan erosi kepercayaan pelanggan. Riset industri menunjukkan bahwa biaya penyimpanan (*holding cost*) rata-rata menyerap 20–30% dari nilai stok per tahun, mencakup biaya gudang, asuransi, risiko keusangan, dan biaya modal tersembunyi. Di sisi lain, *stockout* yang tidak terkelola dapat menekan revenue hingga 4–8% per kuartal, terutama pada kategori produk dengan permintaan musiman tinggi.

Permasalahan ini berakar pada pendekatan inventory yang seragam (*one-size-fits-all*): kebijakan pemesanan yang sama diterapkan pada semua produk, tanpa mempertimbangkan perbedaan fundamental dalam pola permintaan, siklus hidup produk, dan kontribusi margin. Produk dengan volume tinggi dan frekuensi pembelian stabil membutuhkan strategi *continuous replenishment*, sedangkan produk dengan permintaan sporadic dan nilai tinggi lebih cocok dikelola dengan pendekatan *just-in-time*. Tanpa segmentasi yang tepat, keputusan pembelian bersifat reaktif dan sub-optimal.

Segmentasi produk berbasis data historis penjualan menyediakan fondasi kuantitatif untuk membedakan perilaku setiap SKU, mengelompokkannya ke dalam klaster yang bermakna secara bisnis, dan merumuskan kebijakan inventory yang spesifik per segmen. Proyek ini dirancang untuk mentransformasi pengelolaan inventory dari pendekatan intuitif menuju keputusan berbasis bukti.

---

## 1.2 Pemangku Kepentingan Utama

**Inventory Manager** adalah pemangku kepentingan primer yang paling langsung terdampak oleh hasil analisis ini. Tim inventory membutuhkan panduan operasional yang jelas: berapa *safety stock* yang diperlukan per segmen, kapan titik pemesanan ulang (*reorder point*) harus dipicu, dan produk mana yang layak menjadi prioritas pemantauan harian. Output klasterisasi harus dapat diterjemahkan langsung ke dalam parameter sistem WMS (*Warehouse Management System*) yang mereka gunakan.

**Tim Purchasing** memanfaatkan segmentasi ini sebagai dasar negosiasi dengan pemasok. Produk dalam klaster permintaan tinggi dan stabil menjadi kandidat untuk kontrak *blanket order* dengan harga lebih rendah, sedangkan produk di klaster permintaan tidak menentu membutuhkan fleksibilitas volume dalam kesepakatan pengadaan. Dengan visibilitas berbasis data, tim purchasing dapat mengalihkan leverage negosiasi dari asumsi ke angka nyata.

**Tim Finance** berkepentingan pada dampak langsung terhadap arus kas dan profitabilitas. Biaya penyimpanan yang dikurangi melalui optimasi stok akan tercermin dalam perbaikan *working capital ratio*, sementara peningkatan *inventory turnover* menandakan efisiensi aset yang lebih tinggi. Tim finance memerlukan proyeksi dampak finansial dari setiap skenario klaster sebagai dasar persetujuan anggaran dan evaluasi ROI proyek ini.

---

## 1.3 Tujuan Bisnis

Proyek ini ditetapkan untuk mencapai dua sasaran bisnis yang terukur dan terikat waktu dalam horizon 12 bulan pasca-implementasi:

1. **Penurunan biaya penyimpanan sebesar 15%** dibandingkan baseline tahun berjalan, dicapai melalui pengurangan *average days inventory outstanding* pada klaster produk lambat bergerak, dan penerapan batas stok maksimum berbasis data pada SKU bernilai tinggi namun permintaan rendah.

2. **Peningkatan *inventory turnover ratio* sebesar 10%** secara keseluruhan, dengan target utama pada klaster produk bervolume tinggi yang saat ini mengalami akumulasi stok berlebih akibat siklus pemesanan yang tidak tersinkronisasi dengan pola permintaan aktual.

Kedua target ini bukan sekadar aspirasi operasional — keduanya berdampak langsung pada kemampuan perusahaan untuk mengalokasikan modal kerja ke peluang pertumbuhan lain, termasuk ekspansi kategori produk baru dan pengembangan pasar geografis yang lebih potensial.

---

## 1.4 Kriteria Sukses Teknis

Validitas model klasterisasi diukur melalui dua dimensi yang saling melengkapi:

**Dimensi kuantitatif** mensyaratkan *Silhouette Score* > 0,5 pada hasil akhir klasterisasi. Nilai ini mengindikasikan bahwa setiap produk secara signifikan lebih mirip dengan anggota klasternya dibandingkan dengan klaster tetangga — suatu prasyarat agar kebijakan inventory per klaster dapat dirumuskan dengan keyakinan statistik yang memadai. Selain itu, stabilitas klaster diverifikasi melalui *robustness testing* dengan variasi parameter inisialisasi dan sub-sampling data, sehingga hasil yang diperoleh bukan artefak dari kondisi awal algoritma.

**Dimensi kualitatif** mensyaratkan bahwa klaster yang dihasilkan bersifat *actionable*: setiap klaster harus dapat diterjemahkan menjadi paling sedikit satu rekomendasi kebijakan inventory yang konkret, dapat dipahami oleh manajer non-teknis, dan realistis untuk diimplementasikan dalam sistem operasional yang ada. Klaster yang secara matematis valid tetapi tidak memiliki interpretasi bisnis yang jelas dianggap gagal memenuhi kriteria sukses proyek ini.

---

# 2. Data Understanding

## 2.1 Sumber Data, Lisensi, dan Deskripsi Dataset

Dataset yang digunakan adalah **Online Retail Dataset** yang dipublikasikan oleh *UCI Machine Learning Repository* (Dr. Daqing Chen, London South Bank University). Dataset ini tersedia di bawah lisensi **Creative Commons Attribution 4.0 (CC BY 4.0)**, yang memungkinkan penggunaan komersial maupun akademis selama atribusi sumber diberikan. Data mencakup seluruh transaksi penjualan dari sebuah peritel e-commerce berbasis di Inggris yang bergerak di segmen *gift and home décor*, untuk periode **1 Desember 2010 hingga 9 Desember 2011** — rentang satu tahun penuh yang memungkinkan analisis siklus musiman secara komprehensif.

### Data Dictionary

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `InvoiceNo` | String | Nomor faktur unik per transaksi; prefix `C` menandai transaksi pembatalan (*cancellation*) |
| `StockCode` | String | Kode unik identifikasi produk (SKU); beberapa kode bersifat non-produk (biaya pengiriman, penyesuaian manual) |
| `Description` | String | Nama deskriptif produk dalam bahasa Inggris; memiliki inkonsistensi penulisan antar transaksi |
| `Quantity` | Integer | Jumlah unit produk dalam satu baris transaksi; nilai negatif merepresentasikan retur atau pembatalan |
| `InvoiceDate` | Datetime | Tanggal dan waktu transaksi dicatat dalam sistem (format: MM/DD/YYYY HH:MM) |
| `UnitPrice` | Float | Harga satuan produk dalam mata uang Pound Sterling (GBP); beberapa entri bernilai nol atau negatif |
| `CustomerID` | Float | Identifikasi numerik pelanggan; bersifat opsional sehingga terdapat entri kosong untuk transaksi anonim |
| `Country` | String | Negara domisili pelanggan berdasarkan alamat pengiriman |

---

## 2.2 Kualitas Data

### Missing Values

Kolom `CustomerID` mengandung *missing value* yang signifikan — sekitar **24,93% dari total baris** tidak memiliki identifikasi pelanggan. Ini merupakan masalah struktural, bukan acak: transaksi *guest checkout* dan penjualan B2B tanpa akun terdaftar secara sistematis tidak merekam CustomerID. Implikasinya, analisis berbasis perilaku pelanggan individu (seperti RFM) memerlukan subset data yang lebih kecil, sementara analisis berbasis produk dapat menggunakan seluruh dataset. Kolom `Description` juga memiliki sejumlah kecil nilai kosong yang umumnya berkorespondensi dengan kode produk non-standar atau entri kesalahan sistem.

### Duplikasi

Terdapat baris duplikat yang identik di seluruh atribut, diduga berasal dari kesalahan pencatatan ganda atau migrasi sistem. Duplikat ini wajib dihapus sebelum agregasi karena akan menggandakan volume transaksi secara artifisial dan mendistorsi metrik frekuensi pembelian per produk.

### Transaksi Pembatalan dan Retur

Transaksi dengan prefix `C` pada `InvoiceNo` dan nilai `Quantity` negatif merepresentasikan pembatalan pesanan atau retur barang. Transaksi ini tidak boleh diperlakukan setara dengan transaksi penjualan normal dalam proses klasterisasi. Strategi penanganan yang tepat adalah memisahkan transaksi batal dan menganalisisnya secara terpisah sebagai *return rate* per produk — sebuah sinyal penting tentang kualitas produk atau kesesuaian ekspektasi pelanggan yang justru bernilai tinggi untuk segmentasi.

### Outlier

Distribusi `Quantity` dan `UnitPrice` menunjukkan *right-skew* ekstrem dengan kehadiran nilai pencilan yang substansial. Pesanan dalam jumlah ribuan unit pada satu faktur kemungkinan besar merepresentasikan transaksi B2B *bulk order*, bukan pembelian eceran biasa — perilaku yang secara fundamental berbeda dari pembelian konsumen individual. Demikian pula, beberapa entri `UnitPrice` bernilai nol (kemungkinan produk sampel atau hadiah promosi) dan beberapa bernilai di atas £500 (produk premium atau set koleksi). Outlier ini memerlukan penanganan kontekstual: bukan dihapus secara membabi buta, melainkan diidentifikasi mekanismenya dan diputuskan secara eksplisit apakah akan dipertahankan, di-*cap*, atau dipisahkan ke segmen analisis tersendiri.

---

## 2.3 Eksplorasi Data Kunci

### Distribusi Quantity

Distribusi volume pembelian per baris transaksi bersifat sangat *right-skewed*, di mana mayoritas transaksi (>75%) mencakup pembelian kurang dari 12 unit, namun terdapat ekor panjang hingga ribuan unit per faktur. Median quantity jauh di bawah rata-rata, mengkonfirmasi dominasi transaksi eceran skala kecil dengan adanya segmen minoritas *bulk buyer* yang mendominasi volume total. Pola ini menunjukkan perlunya transformasi logaritmik sebelum klasterisasi untuk menghindari dominasi outlier dalam perhitungan jarak antar klaster.

### Produk Paling Laris

Analisis frekuensi pembelian per `StockCode` mengungkap konsentrasi yang tajam: sebagian kecil SKU — sekitar 5–10% dari total kode produk — menyumbang lebih dari 50% volume transaksi dan revenue. Produk-produk ini umumnya adalah item *gift* dengan harga satuan rendah hingga menengah yang memiliki daya tarik universal. Identifikasi SKU *high-runner* ini menjadi fondasi untuk klaster pertama yang membutuhkan prioritas *continuous replenishment* dengan *safety stock* yang memadai.

### Tren Bulanan

Data menunjukkan pola musiman yang kuat dengan puncak permintaan pada kuartal keempat (Oktober–Desember), sejalan dengan siklus belanja akhir tahun dan musim hadiah di pasar Eropa. Periode November–Desember mencatat volume transaksi rata-rata 2–3 kali lebih tinggi dibandingkan bulan-bulan tenang di awal tahun. Tren ini memiliki implikasi langsung pada timing pemesanan: klaster produk musiman memerlukan strategi *pre-positioning* stok yang berbeda secara fundamental dari produk dengan permintaan stabil sepanjang tahun.

### Kontribusi Geografis

Inggris (*United Kingdom*) mendominasi transaksi dengan kontribusi lebih dari 85% dari total volume penjualan. Negara-negara Eropa seperti Jerman, Prancis, Belanda, dan EIRE (Irlandia) menjadi pasar ekspor utama dengan kontribusi kolektif sekitar 10–12%. Distribusi geografis ini relevan untuk segmentasi karena produk tertentu mungkin memiliki permintaan yang sangat terlokalisasi — sebuah dimensi yang dapat memperkaya profil klaster produk dengan karakteristik pasar.

### Korelasi Antar Variabel Numerik

Analisis korelasi antara `Quantity` dan `UnitPrice` menunjukkan hubungan negatif moderat yang mengkonfirmasi pola umum: produk bervolume tinggi cenderung memiliki harga satuan lebih rendah, sementara produk premium terjual dalam kuantitas lebih kecil. Korelasi ini tidak kuat secara linear, mengindikasikan bahwa segmentasi akan lebih efektif menggunakan fitur turunan (seperti total revenue per SKU, frekuensi transaksi, dan koefisien variasi demand) daripada sekadar mengandalkan variabel mentah.

---

## 2.4 Representativitas Sampel dan Potensi Bias

Dataset ini merepresentasikan satu peritel tunggal di segmen *gift and home décor* berbasis di Inggris, sehingga temuan analisis tidak dapat digeneralisasi secara langsung ke industri ritel lain atau geografi berbeda. Terdapat beberapa dimensi bias yang perlu diakui secara eksplisit:

**Bias Temporal:** Periode data (2010–2011) mencerminkan kondisi pasar pasca-krisis finansial global 2008, di mana perilaku konsumen Eropa mungkin berbeda signifikan dari kondisi normal. Pola musiman dan preferensi produk yang teridentifikasi harus divalidasi terhadap data yang lebih baru sebelum digunakan sebagai basis kebijakan jangka panjang.

**Bias Segmen Pelanggan:** Tidak adanya CustomerID pada ~25% transaksi menciptakan sampel yang tidak lengkap untuk analisis berbasis pelanggan. Segmen pelanggan anonim — yang mungkin memiliki perilaku pembelian berbeda dari pelanggan terdaftar — hanya dapat dianalisis secara agregat, bukan individual.

**Bias Platform:** Seluruh data berasal dari kanal e-commerce, sehingga produk yang memiliki performa kuat di kanal offline atau toko fisik tidak terwakili. Klaster yang dihasilkan mencerminkan perilaku pasar digital, bukan gambaran permintaan total lintas kanal.

**Bias Produk Non-Standar:** Kehadiran `StockCode` non-produk seperti kode untuk biaya pengiriman, penyesuaian manual, dan pemberian sampel dapat mencemari analisis jika tidak difilter dengan cermat. Kode-kode ini secara teknis memiliki `Quantity` dan `UnitPrice`, tetapi tidak merepresentasikan perilaku penjualan produk nyata.

Pemahaman atas keterbatasan ini bukan melemahkan nilai analisis, melainkan mendefinisikan batas inferensi yang valid dan menjadi bagian dari transparansi ilmiah yang bertanggung jawab.

---

*Dokumen ini merupakan bagian dari laporan lengkap proyek segmentasi produk. Bagian selanjutnya akan mencakup Data Preparation, Modeling, Evaluation, dan Deployment.*

---
**Disusun oleh:** Lead Data Scientist  
**Tanggal:** Juni 2026  
**Versi Dokumen:** 1.0
