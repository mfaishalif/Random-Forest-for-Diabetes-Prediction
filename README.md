# Data Visualization and Prediction for Healthcare Purpose Using Random Forest Algorithm

**Proyek UAS Mata Kuliah Analisis dan Visualisasi Data**

> [!NOTE]
> Proyek ini merupakan implementasi dan peningkatan dari naskah ilmiah berjudul **"Data Visualization and Prediction for Healthcare Purpose Using Random Forest Algorithm"** karya **Evgenija Gjurovska, Snezana Savoska, Natasha Blazheska-Tabakovska, dan Kostandina Veljanovska**.
>
> 🔗 **Tautan Naskah Asli:** [IEEE Xplore](https://ieeexplore.ieee.org/document/11098229)

## 👥 Tim Penyusun (Kelompok 11)

- **Austhey Nikolaus Kris Prafena** (187231061)
- **Muhammad Faishal Ishlahuddin Fikri** (187231108)

---

## 📋 Deskripsi Proyek

Diabetes Mellitus adalah penyakit kronis yang terjadi ketika tubuh tidak dapat memproduksi atau menggunakan insulin selaiknya, menyebabkan kadar glukosa darah meningkat. Hal ini bisa berujung pada komplikasi serius. Deteksi dini sangat krusial namun sering terhambat karena gejala awal yang tidak spesifik.

Proyek ini bertujuan untuk membangun model Machine Learning untuk memprediksi kemungkinan diabetes secara dini berdasarkan data diagnostik pasien **Pima Indians Diabetes Dataset** (sumber: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)).

Kami menggunakan algoritma **Random Forest Classifier** yang dioptimalkan dengan teknik **SMOTE (Synthetic Minority Over-sampling Technique)** untuk menangani ketidakseimbangan data, serta menerapkan **imputasi data** untuk menangani nilai yang tidak valid (missing values).

Selain akurasi, proyek ini menekankan pada **Explainable AI (XAI)** menggunakan **SHAP (SHapley Additive exPlanations)** agar keputusan model transparan dan dapat dipahami secara medis.

## 🚀 Peningkatan dari Naskah Asli

Dibandingkan dengan metode baseline pada naskah referensi, proyek ini melakukan beberapa peningkatan teknis untuk mengatasi kelemahan seperti *recall* yang rendah dan bias pada kelas mayoritas:

- **Handling Imbalanced Data**: Menggunakan **SMOTE** untuk menyeimbangkan kelas Diabetes dan Non-Diabetes (rasio 1:1 pada data latih).
- **Advanced Preprocessing**: Melakukan imputasi nilai '0' (tidak valid) pada kolom Glukosa, Tekanan Darah, dll, dengan nilai rata-rata, alih-alih membiarkannya.
- **Optimasi Hyperparameter**: Menggunakan `GridSearchCV` untuk mencari parameter model yang paling optimal, bukan menggunakan parameter default.

## 🛠️ Metodologi & Alur Kerja

1.  **Data Preprocessing (Pembersihan)**:
    -   Mengidentifikasi nilai `0` pada kolom vital (Glucose, BloodPressure, SkinThickness, Insulin, BMI) sebagai *missing value*.
    -   Mengganti nilai `0` dengan `NaN`, kemudian mengisi `NaN` dengan nilai **mean (rata-rata)** kolom tersebut.

2.  **Data Splitting (Pembagian Data)**:
    -   Membagi dataset menjadi **80% Training** dan **20% Testing**.
    -   Menggunakan `random_state=42` untuk hasil yang konsisten.

3.  **Handling Imbalance (SMOTE)**:
    -   Menerapkan SMOTE **hanya pada Data Training**. Data Testing dibiarkan asli untuk evaluasi yang jujur.

4.  **Hyperparameter Tuning**:
    -   Mencari kombinasi terbaik menggunakan `GridSearchCV` dengan ruang pencarian:
        -   `n_estimators`: [100, 200]
        -   `max_depth`: [None, 10, 20]
        -   `min_samples_split`: [2, 5]
        -   `criterion`: ['gini', 'entropy']

5.  **Explainable AI**:
    -   Menggunakan SHAP values untuk interpretasi model (Faktor risiko global & diagnosis per pasien).

## 📊 Hasil Analisis

### Temuan Parameter Terbaik
Berdasarkan Grid Search, konfigurasi model terbaik adalah:
-   `criterion`: **'gini'**
-   `max_depth`: **10** (Mencegah overfitting)
-   `n_estimators`: **100**
-   `min_samples_split`: **2**

### Performa Model
Peningkatan signifikan dicapai dibandingkan baseline:

| Metrik | Hasil Proyek Ini | Keterangan |
| :--- | :--- | :--- |
| **Akurasi** | **74.68%** | Lebih tinggi dari baseline paper (68.75%). |
| **Recall (Diabetes)** | **82.00%** | Kemampuan model mendeteksi pasien positif diabetes sangat baik (minim False Negative). |

### Visualisasi yang Dihasilkan
Script akan menghasilkan file gambar berikut untuk analisis:

| File | Deskripsi |
|------|-----------|
| `visualization.png` | Statistik deskriptif & Heatmap korelasi. |
| `confusion_matrix_smote.png` | Evaluasi kesalahan prediksi model. |
| `feature_importance_tuned.png` | Ranking fitur paling berpengaruh. |
| `shap_summary_plot.png` | Faktor risiko global (Glukosa terbukti faktor utama). |
| `shap_dependence_glucose.png` | Hubungan detail Glukosa vs Usia. |
| `shap_individual_patient.png` | Penjelasan diagnosis untuk satu pasien tertentu. |

## 📦 Cara Instalasi & Penggunaan

1. **Instal Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Library utama: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, shap.*

2. **Jalankan Program**
   ```bash
   python random_forest.py
   ```
