# 🏎️ F1 Bayesian Predictor & Live Tracker 2026

[![Streamlit App](https://static.streamlit.io/badge_svg.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-red.svg)](https://www.python.org/)
[![FastF1](https://img.shields.io/badge/API-FastF1-orange.svg)](https://github.com/theOehrly/FastF1)

Platform analitik prediktif canggih untuk mensimulasikan kualifikasi dan balapan utama Formula 1 secara real-time pada era regulasi mesin & sasis baru 2026. Platform ini menggabungkan model pembelajaran mesin (*Learning-To-Rank*), pembaruan performa Bayesian ELO dinamis, dan simulator fisik *NumPy-Vectorized Monte Carlo* 10.000+ iterasi.

---

## 🌟 Fitur Utama

### 1. ⏱️ ML-Based Qualifying Grid Predictor
* **Algoritma**: `LightGBM LTR (Learning-To-Rank)` berbasis *LambdaMART* untuk mengurutkan posisi starting grid secara akurat.
* **Bayesian Credible Intervals**: Menggunakan *Quantile Regression* untuk memberikan rentang estimasi waktu lap terbaik, median, dan terburuk pada selang kepercayaan 90% yang dinamis terhadap suhu sirkuit dan intensitas hujan.
* **SHAP Interpretability**: Visualisasi kontribusi performa latihan bebas (FP3/SQ) dan faktor prior pembalap terhadap hasil prediksi.

### 2. 🏁 Vectorized Monte Carlo Race Simulator (10,000+ Runs)
* **Simulasi Fisik Lap-demi-Lap**: Mensimulasikan ribuan balapan alternatif secara stokastik (acak) kurang dari 1 detik menggunakan operasi matriks ter-vektorisasi di NumPy.
* **Pemodelan Balap Realistis**: Mengintegrasikan model degradasi ban fisik (Soft, Medium, Hard), efek udara kotor (*dirty air*), probabilitas keluarnya Safety Car berdasarkan sejarah sirkuit, tabrakan acak (DNF), dan strategi pergantian ban (pit stop).
* **DNF Enforcement**: Menjamin pembalap yang sudah terkonfirmasi pensiun/DNF pada lap tertentu memiliki probabilitas 100% DNF di hasil simulasi.

### 3. 🔴 Live Status Awareness & Auto-Refresh
* **Deteksi Status GP Dinamis**: Secara otomatis membagi balapan menjadi 3 status berdasarkan waktu UTC rill jadwal resmi FastF1 2026:
  * `✅ RACE COMPLETED`: Balapan telah selesai (pengguna mendapat akses slider replay lap penuh P1-Finish).
  * `🔴 LIVE — LAP X/Y`: Balapan sedang berjalan secara real-time.
  * `📅 UPCOMING`: Balapan belum dimulai (menampilkan tanggal & jam tayang).
* **Auto-Refresh Non-Blocking**: Halaman dashboard akan melakukan refresh otomatis non-blocking menggunakan `@st.fragment` setiap 30 detik untuk menarik data posisi berjalan dari FastF1 API hanya jika status balapan sedang `ONGOING`.

### 4. 🔄 Sprint Weekend Dynamic Fallbacks
* **Penanganan Cerdas**: Jika sirkuit yang dipilih menggunakan format *Sprint Weekend* (yang tidak memiliki sesi FP3), sistem secara otomatis mengalihkan penarikan data performa kualifikasi ke **Sprint Qualifying (SQ)** atau **Practice 1 (FP1)** agar pipeline tidak crash.

---

## 🏛️ Arsitektur Matematika Bayesian

Sistem ini mengaplikasikan **Teorema Bayes** untuk memperbarui ekspektasi performa pembalap secara dinamis:

$$\text{Posterior } P(A|B) \propto \text{Likelihood } P(B|A) \times \text{Prior } P(A)$$

```
+---------------------------------------+
|        FastF1 API / Live Data         |
+-------------------+-------------------+
                    |
                    v
+---------------------------------------+             +-----------------------------+
| FP3 / SQ Lap Times & Speed Traps      | ----------> | Bayesian ELO Updates (Form) |
+---------------------------------------+             +--------------+--------------+
                                                                     |
                                                                     v (Posterior ELO)
+---------------------------------------+             +--------------+--------------+
| Track Temperature & Rain Intensity    | ----------> | LightGBM Ranker (Qualifying)|
+---------------------------------------+             +--------------+--------------+
                                                                     |
                                                                     v (ML Predicted Grid)
+---------------------------------------+             +--------------+--------------+
| Starting Grid Order & Live Standings  | ----------> | Monte Carlo Race Simulator  |
+---------------------------------------+             +--------------+--------------+
                                                                     |
                                                                     v
                                                      +--------------+--------------+
                                                      | Win, Podium & DNF Prob %    |
                                                      +-----------------------------+
```

### A. Prior — $P(A)$ (Ekspektasi Historis)
* **Data**: Basis rating ELO awal pembalap (`base_elo` di `src/utils.py`). Dikalibrasi berdasarkan rangkuman statistik karier riil pembalap hingga akhir musim 2025 (persentase dominasi teampair) digabung dengan proyeksi performa mesin regulasi baru 2026 (mesin Mercedes superior, Red Bull terhambat reliabilitas).

### B. Likelihood — $P(B|A)$ (Performa Terkini)
* **Data**: Selisih (*delta*) waktu rata-rata lap FP3 atau Sprint Qualifying antara pembalap dengan rekan satu timnya (head-to-head teampair). Jika pembalap dengan ELO prior rendah mengalahkan rekannya secara dominan, hal ini memberikan nilai *Likelihood* tinggi bahwa performanya saat ini sedang meningkat melampaui statistik historisnya.

### C. Posterior — $P(A|B)$ (Rating Performa Terkini)
* **Data**: ELO Terupdate (`updated_elo`). Rating ini akan bergeser naik jika pembalap berkinerja lebih baik daripada dugaan awal (Prior) dan turun jika sebaliknya. ELO Posterior ini diumpankan ke model kualifikasi (ML Ranker) dan simulator balapan (baseline fisik pembalap).

---

## 📂 Struktur Direktori

```text
├── .streamlit/
│   └── config.toml          # Konfigurasi visual tema Premium Asphalt & Neon Red
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py   # API Parser FastF1, Penentu Status GP & Sprint Fallbacks
│   ├── models.py           # Model LGBM Ranker, Regresi Kuantil, & MC Simulator
│   └── utils.py            # Basis Data Sirkuit 2026, ELO Calibrator & Utility Matematika
├── app.py                  # Aplikasi Dashboard Utama Streamlit
├── run_pipeline.py         # Pipeline CLI End-to-End Prediksi
├── fastf1_tutorial.ipynb   # Jupyter Notebook interaktif arsitektur dan tutorial end-to-end
├── requirements.txt        # Daftar dependensi modul Python
└── README.md               # Dokumentasi Proyek
```

---

## 🚀 Instalasi & Cara Menjalankan

### 1. Kloning Repositori
```bash
git clone https://github.com/fbudimannn/F1_PREDICTION.git
cd F1_PREDICTION
```

### 2. Buat Virtual Environment & Install Dependensi
```bash
python -m venv venv
source venv/bin/activate  # Untuk Linux/macOS
# atau
venv\Scripts\activate     # Untuk Windows

pip install -r requirements.txt
```

### 3. Jalankan Aplikasi Dashboard Streamlit
```bash
streamlit run app.py
```

### 4. Jalankan Pipeline Prediksi lewat CLI (Terminal)
Anda bisa mengeksekusi pipeline prediksi kualifikasi dan balapan sirkuit mana pun secara langsung via command-line:
```bash
python run_pipeline.py --circuit canada --temp 28.0 --sims 10000
```

---

## ☁️ Panduan Deployment (Streamlit Community Cloud)

1. Pastikan seluruh kode terbaru di cabang `main` sudah terdorong ke repositori GitHub Anda.
2. Masuk ke **[share.streamlit.io](https://share.streamlit.io/)** menggunakan akun GitHub Anda.
3. Klik **"New app"** di dashboard Streamlit Cloud Anda.
4. Setel konfigurasi berikut:
   * **Repository**: `fbudimannn/F1_PREDICTION`
   * **Branch**: `main`
   * **Main file path**: `app.py`
5. Klik **"Deploy!"** dan aplikasi dashboard analitik premium F1 Anda akan aktif secara publik dalam beberapa saat!
