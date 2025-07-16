import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Konfigurasi Awal
st.set_page_config(page_title="Prediksi Hiring Kandidat", page_icon="💼", layout="wide")

# --- KONFIGURASI MODEL & FILE ---
MODEL_PATH = 'random_forest_model.pkl'
SCALER_PATH = 'scaler.pkl'
FEATURES_PATH = 'feature_names.pkl'
HISTORY_FILE = 'riwayat_prediksi.csv'

# Fungsi untuk memuat model dan komponen
@st.cache_resource
def load_components():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Gagal memuat komponen model: {e}")
        st.stop()

model, scaler, feature_names = load_components()

# Fungsi untuk manajemen riwayat
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            history = pd.read_csv(HISTORY_FILE)
            # Konversi ke dictionary dan pastikan CandidateName ada
            history['CandidateName'] = history.get('CandidateName', 'Unknown')
            return history.to_dict('records')
        except Exception as e:
            st.error(f"Error loading history: {e}")
            return []
    return []

def save_history(history):
    try:
        history_df = pd.DataFrame(history)
        os.makedirs(os.path.dirname(HISTORY_FILE) or '.', exist_ok=True)
        history_df.to_csv(HISTORY_FILE, index=False)
    except Exception as e:
        st.error(f"Gagal menyimpan riwayat: {e}")

# Inisialisasi session state
if 'history' not in st.session_state:
    st.session_state.history = load_history()

if 'show_history' not in st.session_state:
    st.session_state.show_history = False

# --- UI HEADER ---
st.markdown("""
<div style='text-align: center; padding: 10px 0'>
    <h1 style='color:#4A90E2;'>💼 Prediksi Keputusan Hiring Kandidat</h1>
    <p style='font-size:16px'>Gunakan sistem ini untuk membantu keputusan rekrutmen berbasis data & AI</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📘 Panduan Penggunaan")
    st.markdown("""
    1. Pilih **mode input** data
    2. Isi form atau upload file CSV
    3. Validasi data sebelum prediksi
    4. Simpan hasil jika diperlukan
    5. Cek riwayat untuk analisis
    """)
    st.divider()
    
    mode = st.radio("🔧 Mode Input", ["Input Manual", "Upload CSV"])
    
    if st.button("🕐 Lihat Riwayat Prediksi"):
        st.session_state.show_history = not st.session_state.show_history
    
    st.divider()
    st.markdown("**Kolom Wajib CSV:**")
    st.markdown("- Nama (name/nama/CandidateName)")
    st.markdown("- SkillScore, InterviewScore")
    st.markdown("- PersonalityScore, ExperienceYears")

# ======================== INPUT MANUAL ========================
if mode == "Input Manual":
    st.subheader("📋 Form Input Data Kandidat")
    
    with st.expander("ℹ️ Panduan Input", expanded=True):
        st.markdown("""
        - **Nama Kandidat**: Wajib diisi
        - **Strategi Rekrutmen**: 
          - 1 = Referensi Karyawan
          - 2 = Sosial Media  
          - 3 = Aplikasi Pencari Kerja
        - **Tingkat Pendidikan**:
          - 1 = SMA
          - 2 = D3  
          - 3 = S1
          - 4 = S2/S3
        - **Skor**: Rentang 0-10
        """)
    
    # Input Data
    col1, col2 = st.columns(2)
    
    with col1:
        candidate_name = st.text_input("🧑‍💼 Nama Kandidat*", "")
        recruitment_strategy = st.selectbox("📌 Strategi Rekrutmen*", ["1", "2", "3"], index=1)
        education_level = st.selectbox("🎓 Tingkat Pendidikan*", ["1", "2", "3", "4"], index=2)
    
    with col2:
        skill_score = st.slider("🛠️ Skor Keterampilan (0-10)*", 0.0, 10.0, 7.0)
        interview_score = st.slider("🗣️ Skor Interview (0-10)*", 0.0, 10.0, 6.5)
        personality_score = st.slider("🤝 Skor Kepribadian (0-10)*", 0.0, 10.0, 8.0)
        experience_years = st.slider("🗓️ Pengalaman (tahun)*", 0, 20, 3)
    
    # Validasi input
    if not candidate_name:
        st.warning("⚠️ Harap isi nama kandidat")
        st.stop()
    
    # Format data untuk prediksi
    input_data = {
        'CandidateName': candidate_name,
        'SkillScore': skill_score,
        'ExperienceYears': experience_years,
        'InterviewScore': interview_score,
        'PersonalityScore': personality_score,
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Encoding variabel kategori
    for level in ["1", "2", "3", "4"]:
        input_data[f'EducationLevel_{level}'] = 1 if education_level == level else 0
    
    for strategy in ["1", "2", "3"]:
        input_data[f'RecruitmentStrategy_{strategy}'] = 1 if recruitment_strategy == strategy else 0
    
    input_df = pd.DataFrame([input_data])

# ======================== CSV MODE ========================
else:
    st.subheader("📂 Upload CSV Data Kandidat")
    uploaded_file = st.file_uploader("Pilih file CSV*", type=["csv"], accept_multiple_files=False)
    
    if uploaded_file is not None:
        try:
            # Baca file CSV
            raw_df = pd.read_csv(uploaded_file)
            
            # Standarisasi nama kolom
            column_mapping = {
                'name': 'CandidateName',
                'nama': 'CandidateName',
                'Nama': 'CandidateName',
                'nama_kandidat': 'CandidateName',
                'NamaKandidat': 'CandidateName'
            }
            raw_df.rename(columns=column_mapping, inplace=True)
            
            # Cek kolom wajib
            required_columns = ['SkillScore', 'InterviewScore', 'PersonalityScore', 'ExperienceYears']
            missing_cols = [col for col in required_columns if col not in raw_df.columns]
            
            if missing_cols:
                st.error(f"❌ Kolom wajib tidak ditemukan: {', '.join(missing_cols)}")
                st.stop()
            
            # Handle kolom nama
            if 'CandidateName' not in raw_df.columns:
                raw_df['CandidateName'] = [f"Kandidat_{i+1}" for i in range(len(raw_df))]
            
            # Tambahkan timestamp
            raw_df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Handle kolom pendidikan
            for level in ["1", "2", "3", "4"]:
                if f'EducationLevel_{level}' not in raw_df.columns:
                    raw_df[f'EducationLevel_{level}'] = 0
            
            # Handle kolom strategi rekrutmen
            for strategy in ["1", "2", "3"]:
                if f'RecruitmentStrategy_{strategy}' not in raw_df.columns:
                    raw_df[f'RecruitmentStrategy_{strategy}'] = 0
            
            # Pastikan semua fitur model ada
            for col in feature_names:
                if col not in raw_df.columns and col != 'Timestamp':
                    raw_df[col] = 0
            
            input_df = raw_df[feature_names + ['Timestamp', 'CandidateName']]
            
            st.success(f"✅ Berhasil memproses {len(input_df)} kandidat")
            st.dataframe(input_df.head())
            
        except Exception as e:
            st.error(f"❌ Gagal memproses file: {str(e)}")
            st.stop()

# ======================== PROSES PREDIKSI ========================
# ======================== PROSES PREDIKSI ========================
if 'input_df' in locals():
    st.markdown("---")
    st.subheader("🔍 Hasil Prediksi")

    try:
        # Simpan kolom non-fitur
        meta_df = input_df[['CandidateName', 'Timestamp']].copy()

        # Scaling hanya kolom fitur
        X_input = input_df[scaler.feature_names_in_]
        X_scaled = pd.DataFrame(scaler.transform(X_input), columns=scaler.feature_names_in_)

        # Lakukan prediksi
        predictions = model.predict(X_scaled)

        # Gabungkan hasil
        input_df = input_df.copy()
        input_df['Prediction'] = ["DITERIMA" if p == 1 else "TIDAK DITERIMA" for p in predictions]
        input_df['TotalScore'] = input_df[['SkillScore', 'InterviewScore',
                                           'PersonalityScore', 'ExperienceYears']].sum(axis=1)
        input_df['CandidateName'] = meta_df['CandidateName']
        input_df['Timestamp'] = meta_df['Timestamp']

        # Urutkan berdasarkan kelayakan
        input_df = input_df.sort_values(by=['Prediction', 'TotalScore'], ascending=[False, False])
        input_df.insert(0, 'No', range(1, len(input_df)+1))

        # Tampilkan hasil
        st.dataframe(input_df.style.applymap(
            lambda x: 'color: green' if x == "DITERIMA" else 'color: red',
            subset=['Prediction']
        ))

        # Tombol aksi
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="📅 Download Hasil Prediksi",
                data=input_df.to_csv(index=False),
                file_name="hasil_prediksi.csv",
                mime="text/csv"
            )

        with col2:
            if st.button("💲 Simpan Semua ke Riwayat"):
                records_to_save = []
                for _, row in input_df.iterrows():
                    record = {
                        'CandidateName': row['CandidateName'],
                        'Timestamp': row['Timestamp'],
                        'Prediction': row['Prediction'],
                        'TotalScore': row['TotalScore'],
                        'SkillScore': row['SkillScore'],
                        'InterviewScore': row['InterviewScore'],
                        'PersonalityScore': row['PersonalityScore'],
                        'ExperienceYears': row['ExperienceYears']
                    }
                    records_to_save.append(record)

                st.session_state.history.extend(records_to_save)
                save_history(st.session_state.history)
                st.success(f"✔️ {len(records_to_save)} prediksi telah disimpan")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")

# ======================== TAMPILKAN RIWAYAT ========================
if st.session_state.show_history:
    st.markdown("---")
    st.subheader("📜 Riwayat Prediksi")
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        
        # Pastikan kolom yang diperlukan ada
        if 'TotalScore' not in history_df.columns:
            history_df['TotalScore'] = (
                history_df.get('SkillScore', 0) + 
                history_df.get('InterviewScore', 0) + 
                history_df.get('PersonalityScore', 0) + 
                history_df.get('ExperienceYears', 0)
            )
        
        # Filter UI
        st.markdown("### 🔍 Filter Riwayat")
        col1, col2 = st.columns(2)
        
        with col1:
            sort_by = st.selectbox("Urutkan berdasarkan", 
                                 ["Terbaru", "Total Skor", "Kelayakan"])
        
        with col2:
            filter_by = st.selectbox("Tampilkan", 
                                   ["Semua", "DITERIMA", "TIDAK DITERIMA"])
        
        # Apply filters
        if filter_by == "DITERIMA":
            history_df = history_df[history_df['Prediction'] == "DITERIMA"]
        elif filter_by == "TIDAK DITERIMA":
            history_df = history_df[history_df['Prediction'] == "TIDAK DITERIMA"]
        
        # Apply sorting
        if sort_by == "Total Skor":
            history_df = history_df.sort_values('TotalScore', ascending=False)
        elif sort_by == "Kelayakan":
            history_df = history_df.sort_values(['Prediction', 'TotalScore'], 
                                             ascending=[False, False])
        else:  # Terbaru
            if 'Timestamp' in history_df.columns:
                history_df = history_df.sort_values('Timestamp', ascending=False)
            else:
                history_df = history_df.sort_index(ascending=False)
        
        # Tampilkan data
        history_df = history_df.reset_index(drop=True)
        history_df.insert(0, 'No', range(1, len(history_df)+1))
        
        # Pilih kolom untuk ditampilkan
        display_cols = ['No', 'CandidateName', 'Prediction', 'TotalScore']
        if 'Timestamp' in history_df.columns:
            display_cols.insert(1, 'Timestamp')
        
        # Tombol aksi
        st.dataframe(
            history_df[display_cols].style.applymap(
                lambda x: 'color: green' if x == "DITERIMA" else 'color: red',
                subset=['Prediction']
            )
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.download_button(
                label="📤 Ekspor Riwayat Lengkap",
                data=history_df.to_csv(index=False),
                file_name="riwayat_prediksi_lengkap.csv",
                mime="text/csv"
            )
        
        with col3:
            if st.button("🗑️ Hapus Semua Riwayat", type="secondary"):
                st.session_state.history = []
                save_history(st.session_state.history)
                st.rerun()
    
    else:
        st.info("Belum ada riwayat prediksi yang tersimpan.")
