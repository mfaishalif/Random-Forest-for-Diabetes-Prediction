import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import shap


# 1. LOAD DATASET
# Membaca file csv
df = pd.read_csv('diabetes.csv')

print("--- Data Info ---")
print(df.info())
print("\n--- Statistik Deskriptif Awal ---")
print(df.describe())

# 2. PREPROCESSING & CLEANING (EXPERT STEP)
# Dalam dataset Pima, nilai 0 pada kolom berikut adalah missing value (tidak mungkin Glukosa atau BMI = 0)
cols_missing_vals = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Mengganti 0 dengan NaN lalu mengisi dengan mean (imputasi)
# Ini penting agar Random Forest tidak belajar dari data sampah (noise)
for col in cols_missing_vals:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].mean())

# 3. VISUALISASI DATA (Sesuai Judul Paper)
plt.figure(figsize=(12, 10))

# A. Correlation Heatmap
# Tujuannya melihat hubungan antar variabel (misal: Glucose vs Outcome)
plt.subplot(2, 1, 1)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')

# B. Distribusi Outcome
plt.subplot(2, 1, 2)
# Simpan plot distribusi kelas sebelum SMOTE untuk referensi
sns.countplot(x='Outcome', data=df, palette='viridis')
plt.title('Distribusi Kelas ASLI (Sebelum SMOTE)')
plt.tight_layout()
plt.savefig('visualization.png')
plt.close()

# 4. DATA SPLITTING
# Memisahkan Fitur (X) dan Target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split Data (80% Training, 20% Testing) - Standar umum
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n--- Distribusi Training Sebelum SMOTE ---")
print(y_train.value_counts())

# 5. IMPLEMENTASI SMOTE (Synthetic Minority Over-sampling Technique)
# PENTING: SMOTE hanya diterapkan pada X_train, BUKAN X_test.
# Kita tidak boleh menyentuh data testing dengan data sintesis (Data Leakage).
print("\n... Sedang menerapkan SMOTE pada data Training ...")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"--- Distribusi Training Setelah SMOTE ---")
print(y_train_resampled.value_counts())
print("(Sekarang jumlah kelas 0 dan 1 seimbang)")

# 6. TRAINING MODEL & HYPERPARAMETER TUNING
# Menentukan kombinasi parameter yang akan diadu
param_grid = {
    'n_estimators': [100, 200],       # Coba 100 pohon atau 200 pohon
    'max_depth': [None, 10, 20],      # Coba pohon tanpa batas, atau dibatasi
    'min_samples_split': [2, 5],      # Syarat membelah node
    'criterion': ['gini', 'entropy']  # Cara hitung kualitas split
}

# Inisialisasi Random Forest
# n_estimators=100 artinya menggunakan 100 pohon keputusan
# A. Bikin model kosongan (tanpa set n_estimators dulu)
rf_base = RandomForestClassifier(random_state=42)

# B. Masukkan ke dalam "Mesin Pencari" (GridSearch)
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, 
                           cv=5, n_jobs=-1, scoring='accuracy')

# C. Latih GridSearch-nya (Dia akan looping training berkali-kali)
grid_search.fit(X_train_resampled, y_train_resampled)

# D. Ambil pemenangnya (Model Terbaik)
best_rf_model = grid_search.best_estimator_

# Prediksi
y_pred = best_rf_model.predict(X_test)

# 7. EVALUASI MODEL
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n--- Evaluasi Model Random Forest (dengan SMOTE) ---")
print(f"Akurasi: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(report)

# Visualisasi Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No Diabetes', 'Diabetes'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (with SMOTE)')
plt.savefig('confusion_matrix_smote.png')
plt.close()

# 8. FEATURE IMPORTANCE (Analisa Tambahan)
# Melihat fitur apa yang paling berpengaruh
feature_importances = pd.Series(best_rf_model.feature_importances_, index=X.columns)

feature_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importance (Tuned + SMOTE)')
plt.xlabel('Importance Score')
plt.savefig('feature_importance_tuned.png')
plt.close()

print(f"Parameter Terbaik yang ditemukan: {grid_search.best_params_}")


# 9. IMPLEMENTASI SHAP VALUES
print("\n--- Menghitung SHAP Values (Explainable AI) ---")

# A. Inisialisasi Explainer
explainer = shap.TreeExplainer(best_rf_model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    print("Deteksi Format SHAP: List of Arrays")
    # Format: [Array(Class 0), Array(Class 1)]
    shap_vals_target = shap_values[1]
    base_value_target = explainer.expected_value[1]
elif len(np.shape(shap_values)) == 3:
    print("Deteksi Format SHAP: 3D Array")
    # Format: (Samples, Features, Classes) -> Kita ambil slice index 1 (Class 1)
    shap_vals_target = shap_values[:, :, 1]
    base_value_target = explainer.expected_value[1]
else:
    print("Deteksi Format SHAP: 2D Array (Default)")
    shap_vals_target = shap_values
    base_value_target = explainer.expected_value

print(f"Shape values target: {np.shape(shap_vals_target)}")

# C. VISUALISASI 1: SUMMARY PLOT
plt.figure(figsize=(10, 8))
# Gunakan variabel 'shap_vals_target' yang sudah distandarisasi
shap.summary_plot(shap_vals_target, X_test, show=False) 
plt.title('SHAP Summary Plot: Faktor Risiko Diabetes', fontsize=14)
plt.tight_layout()
plt.savefig('shap_summary_plot.png')
plt.close()
print("-> 'shap_summary_plot.png' berhasil disimpan.")

# D. VISUALISASI 2: DEPENDENCE PLOT
plt.figure(figsize=(8, 6))
shap.dependence_plot("Glucose", shap_vals_target, X_test, show=False, interaction_index="Age")
plt.title('SHAP Dependence: Glucose vs Age Impact')
plt.tight_layout()
plt.savefig('shap_dependence_glucose.png')
plt.close()
print("-> 'shap_dependence_glucose.png' berhasil disimpan.")

# E. VISUALISASI 3: INDIVIDUAL FORCE PLOT
# Analisa satu pasien (index 0)
patient_idx = 0
patient_data = X_test.iloc[patient_idx]

print(f"\nContoh Analisa Pasien ke-{patient_idx}:")
# Trik visualisasi: Force plot butuh matplotlib=True untuk disimpan sebagai gambar statis
plt.figure(figsize=(12, 4))
shap.plots.force(base_value_target, 
                 shap_vals_target[patient_idx, :], 
                 patient_data, 
                 matplotlib=True, 
                 show=False)
plt.savefig('shap_individual_patient.png')
plt.close()
print("-> 'shap_individual_patient.png' berhasil disimpan.")

print("\nSelesai! Cek file PNG untuk melihat hasil interpretasi.")