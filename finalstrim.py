import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model, scaler, dan fitur
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

st.set_page_config(page_title="Prediksi Hiring Kandidat", page_icon="💼", layout="wide")

# --- HEADER ---
st.markdown("""
<div style='text-align: center; padding: 10px 0'>
    <h1 style='color:#4A90E2;'>💼 Prediksi Keputusan Hiring Kandidat</h1>
    <p style='font-size:16px'>Gunakan sistem ini untuk membantu keputusan rekrutmen berbasis data & AI.</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR INFO ---
with st.sidebar:
    st.header("📘️ Panduan Penggunaan")
    st.markdown("""
    - Pilih **mode input** data: manual atau CSV
    - Masukkan data kandidat secara lengkap
    - Centang validasi manual sebelum prediksi
    - Unduh hasil jika dibutuhkan
    """)
    st.divider()
    mode = st.radio("🔧 Mode Input", ["Input Manual", "Upload CSV"])

# ======================== INPUT MANUAL ========================
if mode == "Input Manual":
    st.subheader("📋 Form Input Data Kandidat")

    with st.expander("ℹ️ Penjelasan Setiap Variabel"):
        st.markdown("""
        - **Strategi Rekrutmen**: 1=LinkedIn, 2=Job Fair, 3=Referensi
        - **Tingkat Pendidikan**: 1=SMA, 2=D3, 3=S1, 4=S2/S3
        - **Skor**: Diisi nilai 0–10 untuk Skill, Interview, Personality
        - **Pengalaman**: Tahun pengalaman kerja
        """)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        recruitment_strategy = st.selectbox(
            "📌 Strategi Rekrutmen", ["1", "2", "3"],
            help="1=LinkedIn, 2=Job Fair, 3=Referensi"
        )
        education_level = st.selectbox(
            "🎓 Tingkat Pendidikan", ["1", "2", "3", "4"],
            help="1=SMA, 2=D3, 3=S1, 4=S2/S3"
        )

    with col2:
        skill_score = st.slider("🛠️ Skor Keterampilan", 0.0, 10.0, 5.0, help="Skill teknikal")
        interview_score = st.slider("🗣️ Skor Interview", 0.0, 10.0, 5.0, help="Wawancara")

    with col3:
        personality_score = st.slider("🤝 Skor Kepribadian", 0.0, 10.0, 5.0, help="Softskill & attitude")
        experience_years = st.slider("📅 Pengalaman (tahun)", 0, 20, 2)

    # One-hot encode
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

# ======================== CSV MODE ========================
else:
    st.subheader("📂 Upload CSV Data Kandidat")
    uploaded_file = st.file_uploader("📄 Upload CSV berisi data kandidat", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.success("✅ File berhasil dibaca. Menampilkan beberapa data pertama:")
        st.dataframe(raw_df.head())

        input_df = raw_df.copy()

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

# ======================== PREDIKSI ========================
if 'input_df' in locals():
    st.markdown("---")
    st.subheader("🔍 Proses Prediksi")

    try:
        input_df = input_df[scaler.feature_names_in_]
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

        predictions = model.predict(input_scaled)

        if mode == "Input Manual":
            is_valid = st.checkbox("Saya sudah memastikan data di atas benar dan siap diprediksi.")

            if st.button("🚀 Jalankan Prediksi"):
                if is_valid:
                    result = predictions[0]
                    if result == 1:
                        st.markdown("""
                        <div style='background-color:#DFF2BF;padding:20px;border-radius:10px'>
                            <h3 style='color:#4F8A10'>✅ Kandidat kemungkinan <u>DITERIMA</u></h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='background-color:#FFBABA;padding:20px;border-radius:10px'>
                            <h3 style='color:#D8000C'>❌ Kandidat kemungkinan <u>TIDAK DITERIMA</u></h3>
                        </div>
                        """, unsafe_allow_html=True)

                    # Visualisasi skor kandidat
                    st.markdown("### 📈 Visualisasi Skor Kandidat")
                    score_data = {
                        'Skill': skill_score,
                        'Interview': interview_score,
                        'Personality': personality_score,
                        'Experience': experience_years
                    }

                    fig, ax = plt.subplots()
                    ax.bar(score_data.keys(), score_data.values(), color=["#3498db", "#2ecc71", "#9b59b6", "#f39c12"])
                    ax.set_ylim(0, max(score_data.values()) + 2)
                    ax.set_ylabel("Skor")
                    st.pyplot(fig)

                else:
                    st.warning("⚠️ Silakan centang validasi sebelum melanjutkan.")

        else:
            input_df['Prediksi'] = ["DITERIMA" if p == 1 else "TIDAK DITERIMA" for p in predictions]
            st.success("✅ Hasil prediksi siap ditinjau:")
            st.dataframe(input_df)

            st.download_button("📥 Download Hasil Prediksi", data=input_df.to_csv(index=False),
                               file_name="hasil_prediksi.csv", mime="text/csv")

            # Statistik
            st.markdown("### 📊 Statistik Hasil Prediksi")
            accepted = input_df['Prediksi'].value_counts().get("DITERIMA", 0)
            rejected = input_df['Prediksi'].value_counts().get("TIDAK DITERIMA", 0)

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("✅ Diterima", accepted)
            with col_b:
                st.metric("❌ Tidak Diterima", rejected)

            fig, ax = plt.subplots()
            sns.countplot(data=input_df, x='Prediksi', palette=["#27ae60", "#e74c3c"], ax=ax)
            ax.set_title("Distribusi Hasil Prediksi")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses data: {e}")
