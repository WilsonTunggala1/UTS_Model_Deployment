Proyek ini adalah sistem prediksi berbasis Machine Learning yang dirancang untuk memprediksi apakah seorang mahasiswa akan mendapatkan pekerjaan (Placement) dan mengestimasi besaran gaji yang akan diterima (Salary in LPA). 

Proyek in idikembangkan menggunakan MLFlow, client-server dengan FastAPI sebagai backend dan Streamlit untuk UI pengguna

Two-Stage Prediction:
- Klasifikasi: Memprediksi status kelulusan kerja (Placed/Not Placed).
- Regresi: Jika diprediksi "Placed", sistem akan menghitung estimasi gaji menggunakan model regresi terbaik.

Pipeline yang Solid: Menggunakan ColumnTransformer untuk menangani data nominal (OneHot), ordinal, dan numerik secara otomatis.

Antarmuka Intuitif: Menggunakan sistem Tabs dan Forms pada Streamlit untuk input 20 fitur mahasiswa secara rapi.

Tracking Eksperimen: Integrasi dengan MLflow untuk mencatat parameter dan metrik model.

Visualisasi Data: Menampilkan grafik keahlian (coding, komunikasi, aptitude) secara real-time setelah prediksi.