# Prediksi Kematangan Sawit - Machine Learning Pipeline

Proyek ini berisi *pipeline* pembelajaran mesin (Machine Learning) berbasis Deep Learning untuk mendeteksi dan mengklasifikasikan tingkat kematangan Tandan Buah Segar (TBS) kelapa sawit. Terdapat 4 kelas klasifikasi utama:
1. **Mentah**
2. **Mengkal**
3. **Matang**
4. **Lewat Matang**

Model dirancang siap untuk *deployment* di platform Android (Mobile) menggunakan TensorFlow Lite, guna mencapai target akurasi tinggi namun dengan komputasi dan ukuran file yang efisien.

---

## ⚙️ Persyaratan Sistem & Instalasi

Proyek ini dikembangkan menggunakan **Python 3.10** dan pustaka **TensorFlow**. Kami merekomendasikan **Anaconda** untuk mengisolasi *environment*.

1. Buat environment baru:
   ```bash
   conda create -n sawit_env python=3.10 -y
   conda activate sawit_env
   ```
2. Instal dependensi dari `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📁 Persiapan Dataset

Sebelum mulai menjalankan kode pelatihan, tempatkan gambar dataset mentah di dalam folder `dataset_raw/` dengan struktur subfolder sebagai berikut:

```text
dataset_raw/
├── Mentah/       (min. 300 gambar)
├── Mengkal/      (min. 300 gambar)
├── Matang/       (min. 300 gambar)
└── Lewat Matang/ (min. 300 gambar)
```
*Pipeline ini akan secara otomatis melakukan split dataset menjadi rasio Latih(70%), Validasi(15%), dan Uji(15%) ke dalam folder `dataset_split/`.*

---

## 🚀 Pelatihan Model (Jupyter Notebook)

Seluruh alur input-proses-output dan visualisasi evaluasi dijalankan pada satu notebook interaktif, yaitu file **`pelatihan_mobilenetv2.ipynb`**.

### Pipeline Utama Terdiri Atas:
1. **Augmentasi Data**: Memanfaatkan `Albumentations` untuk variasi rotasi, flip, variasi kecerahan (0.6-1.4), zoom ±20%, dan Gaussian Noise agar memperkaya data latih.
2. **Transfer Learning (MobileNetV2)**: 
   - Tahap 1 (Frozen): Membekukan seluruh *base layer*, fokus ekstraksi fitur (15 Epochs, LR: 0.001).
   - Tahap 2 (Fine-Tuning): Membuka 30 *layer* teratas untuk penyesuaian khusus bentuk sawit (30 Epochs, LR: 0.00001).
3. **Kuantisasi ke TFLite**: Mengonversi hasil latihan dengan target `Float16` agar ukuran output menjadi sekitar **7 MB** dan sangat ringan untuk aplikasi *Mobile Android*.
4. **Evaluasi Berstandar Akademik**: Menampilkan *Training Curves* (Accuracy vs Loss) serta kalkulasi detail *Confusion Matrix* yang mengukur Akurasi, Precision, Recall, dan F1-Score.

### Target Objektif (Sesuai PRD):
- Akurasi keseluruhan model **Minimal 88%**.
- Skor *Recall* (Sensitivitas) khusus pada kelas "Matang" **Minimal 90%** untuk mengurangi risiko kerugian ekonomi dari melesetnya prediksi buah matang.

---

## 📊 Hasil Output Akhir
Jika pelatihan berjalan dengan sukses, Anda akan mendapatkan file **`mobilenetv2_sawit_fp16.tflite`** di dalam *root directory*, siap diintegrasikan ke Flutter Anda.
