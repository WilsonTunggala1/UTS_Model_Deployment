import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="wide")

# Load the model
@st.cache_resource
def load_models():
    clf_model = joblib.load('artifacts/placement_clf_pipeline.pkl')
    reg_model = joblib.load('artifacts/placement_reg_pipeline.pkl')
    return clf_model, reg_model

try:
    clf_pipeline, reg_pipeline = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Gagal memuat model. Pastikan file .pkl ada di folder 'artifacts'. Error: {e}")

def main():
    st.title('Student Placement & Salary Predictor')
    st.markdown("""
    Aplikasi ini memprediksi apakah seorang mahasiswa akan mendapatkan pekerjaan (Placed) atau tidak. 
    Jika diprediksi mendapatkan pekerjaan, sistem akan secara otomatis memprediksi estimasi gaji (Salary LPA) yang akan didapatkan.
    """)

    if not models_loaded:
        st.stop()

    # Sidebar untuk Informasi Aplikasi
    with st.sidebar:
        st.header("Informasi Aplikasi")
        st.write("Aplikasi ini menggunakan model Machine Learning Two-Stage:")
        st.info("Tahap 1: LightGBM Classifier untuk memprediksi kelulusan penempatan.")
        st.success("Tahap 2: LightGBM Regressor untuk memprediksi gaji (khusus yang lolos).")
        st.divider()
        st.write("Dibuat untuk keperluan MLOps Pipeline.")

    # Menggunakan form agar aplikasi tidak me-reload setiap kali user mengubah input
    with st.form("prediction_form"):
        st.subheader("Masukkan Data Mahasiswa")
        
        # Menggunakan Tabs untuk mengelompokkan 20 input fitur agar tidak terlihat menumpuk
        tab1, tab2, tab3 = st.tabs(["📚 Akademik & Keahlian", "🚀 Pengalaman & Proyek", "👤 Profil Pribadi"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                cgpa = st.number_input("CGPA (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
                attendance_percentage = st.slider("Persentase Kehadiran (%)", 0, 100, 85)
                study_hours_per_day = st.number_input("Jam Belajar Per Hari", 0, 24, 4)
                backlogs = st.number_input("Jumlah Backlog (Mata kuliah mengulang)", 0, 10, 0)
                branch = st.selectbox("Jurusan (Branch)", ["Computer Science", "Information Technology", "Electronics", "Mechanical", "Civil", "Other"])
            with col2:
                coding_skill_rating = st.slider("Rating Skill Coding (1-100)", 1, 100, 50)
                communication_skill_rating = st.slider("Rating Skill Komunikasi (1-100)", 1, 100, 50)
                aptitude_skill_rating = st.slider("Rating Skill Aptitude (1-100)", 1, 100, 50)

        with tab2:
            col3, col4 = st.columns(2)
            with col3:
                projects_completed = st.number_input("Jumlah Proyek Diselesaikan", 0, 50, 2)
                internships_completed = st.number_input("Jumlah Magang Diselesaikan", 0, 10, 1)
                hackathons_participated = st.number_input("Jumlah Hackathon Diikuti", 0, 50, 0)
            with col4:
                certifications_count = st.number_input("Jumlah Sertifikasi", 0, 50, 1)
                extracurricular_involvement = st.select_slider("Keterlibatan Ekstrakurikuler", options=["Low", "Medium", "High"], value="Medium")

        with tab3:
            col5, col6 = st.columns(2)
            with col5:
                gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
                city_tier = st.selectbox("Asal Kota", ["Tier 1", "Tier 2", "Tier 3"])
                family_income_level = st.selectbox("Tingkat Pendapatan Keluarga", ["Low", "Medium", "High"])
            with col6:
                part_time_job = st.radio("Memiliki Kerja Part-Time?", ["Yes", "No"])
                internet_access = st.radio("Akses Internet di Rumah?", ["Yes", "No"])
                sleep_hours = st.number_input("Jam Tidur Rata-rata", 0, 24, 7)
                stress_level = st.slider("Tingkat Stres (1-10)", 1, 10, 2)

        # Tombol Submit di dalam form
        submit_button = st.form_submit_button(label="Prediksi Sekarang")

    if submit_button:
        input_data = {
            'gender': gender,
            'branch': branch,
            'part_time_job': part_time_job,
            'internet_access': internet_access,
            'cgpa': cgpa,
            'backlogs': backlogs,
            'study_hours_per_day': study_hours_per_day,
            'attendance_percentage': attendance_percentage,
            'projects_completed': projects_completed,
            'internships_completed': internships_completed,
            'coding_skill_rating': coding_skill_rating,
            'communication_skill_rating': communication_skill_rating,
            'aptitude_skill_rating': aptitude_skill_rating,
            'hackathons_participated': hackathons_participated,
            'certifications_count': certifications_count,
            'sleep_hours': sleep_hours,
            'stress_level': stress_level,
            'family_income_level': family_income_level,
            'city_tier': city_tier,
            'extracurricular_involvement': extracurricular_involvement
        }

        # Ubah ke DataFrame
        df_input = pd.DataFrame([input_data])

        st.divider()
        st.subheader("Hasil Prediksi")

        # Animasi Loading
        with st.spinner("Menganalisis data mahasiswa."):
            
            placement_prediction = clf_pipeline.predict(df_input)[0]
            
            # Mendapatkan persentase probabilitas (confidence score)
            placement_proba = clf_pipeline.predict_proba(df_input)[0]
            confidence = placement_proba[1] if placement_prediction == 1 else placement_proba[0]

            col_res1, col_res2 = st.columns(2)

            if placement_prediction == 1:
                # Jika Lolos Penempatan
                with col_res1:
                    st.success(f"Status: PLACED (Diterima Bekerja)*")
                    st.write(f"Tingkat Kepercayaan Model: {confidence*100:.2f}%")
                
                salary_prediction = reg_pipeline.predict(df_input)[0]
                
                with col_res2:
                    st.info("Estimasi Gaji (Salary):")
                    st.metric(label="Lakhs Per Annum (LPA)", value=f"{salary_prediction:.2f} LPA")
                    
                # Visualisasi Sederhana: Membandingkan Skill
                st.write("Analisis Profil Keahlian Anda:")
                chart_data = pd.DataFrame({
                    "Keahlian": ["Coding", "Komunikasi", "Aptitude"],
                    "Skor Anda": [coding_skill_rating, communication_skill_rating, aptitude_skill_rating]
                }).set_index("Keahlian")
                st.bar_chart(chart_data, color="#2ECC71")

            else:
                # Jika Tidak Lolos Penempatan
                with col_res1:
                    st.error(f"Status: NOT PLACED (Belum Mendapatkan Pekerjaan)")
                    st.write(f"Tingkat Kepercayaan Model: {confidence*100:.2f}%")
                
                with col_res2:
                    st.warning("Estimasi Gaji (Salary):")
                    st.metric(label="Lakhs Per Annum (LPA)", value="0.00 LPA")

if __name__ == "__main__":
    main()