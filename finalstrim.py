import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, dan fitur
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')  # list kolom saat training

st.set_page_config(page_title="Prediksi Hiring Kandidat", page_icon="💼")
st.title("💼 Prediksi Keputusan Hiring Kandidat")
st.markdown("Sistem ini membantu memprediksi apakah seorang kandidat layak diterima berdasarkan data dan model machine learning yang telah dilatih.")

st.sidebar.header("🔍 Input Kandidat")
mode = st.sidebar.radio("Pilih Mode Input:", ["Input Manual", "Upload CSV"])

if mode == "Input Manual":
    st.subheader("📋 Form Input Data Kandidat")

    recruitment_strategy = st.selectbox("Strategi Rekrutmen", ["1", "2", "3"])
    education_level = st.selectbox("Tingkat Pendidikan", ["1", "2", "3", "4"])
    skill_score = st.slider("Skor Keterampilan (0-10)", 0.0, 10.0, 5.0)
    interview_score = st.slider("Skor Interview (0-10)", 0.0, 10.0, 5.0)
    personality_score = st.slider("Skor Kepribadian (0-10)", 0.0, 10.0, 5.0)
    experience_years = st.slider("Pengalaman Kerja (tahun)", 0, 20, 2)

    # Buat dataframe input manual
    input_data = {
        'SkillScore': skill_score,
        'ExperienceYears': experience_years,
        'InterviewScore': interview_score,
        'PersonalityScore': personality_score,
    }

    for level in ["1", "2", "3", "4"]:
        input_data[f'EducationLevel_{level}'] = 1 if education_level == level else 0

    for strategy in ["1", "2", "3"]:
        input_data[f'RecruitmentStrategy_{strategy}'] = 1 if recruitment_strategy == strategy else 0

    input_df = pd.DataFrame([input_data])

else:
    st.subheader("📂 Upload File CSV Kandidat")
    uploaded_file = st.file_uploader("Upload file CSV berisi data kandidat", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("📄 Data yang Diupload:")
        st.dataframe(raw_df.head())

        input_df = raw_df.copy()

        # Tambahkan one-hot jika belum ada
        for level in ["1", "2", "3", "4"]:
            if f'EducationLevel_{level}' not in input_df.columns:
                input_df[f'EducationLevel_{level}'] = 0

        for strategy in ["1", "2", "3"]:
            if f'RecruitmentStrategy_{strategy}' not in input_df.columns:
                input_df[f'RecruitmentStrategy_{strategy}'] = 0

        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_names]

# Proses prediksi jika data tersedia
if 'input_df' in locals():
    try:
        # Pastikan urutan & nama kolom sesuai saat scaler dilatih
        input_df = input_df[scaler.feature_names_in_]
        input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

        st.subheader("📊 Hasil Prediksi")
        predictions = model.predict(input_df_scaled)

        if mode == "Input Manual":
            if st.button("🔍 Prediksi Sekarang"):
                if predictions[0] == 1:
                    st.success("✅ Kandidat kemungkinan **DITERIMA**")
                else:
                    st.error("❌ Kandidat kemungkinan **TIDAK DITERIMA**")
        else:
            input_df['Prediksi'] = ["DITERIMA" if p == 1 else "TIDAK DITERIMA" for p in predictions]
            st.write("📋 Tabel Hasil Prediksi:")
            st.dataframe(input_df)

            st.download_button("📥 Download Hasil Prediksi", data=input_df.to_csv(index=False), file_name="hasil_prediksi.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses data: {e}")
