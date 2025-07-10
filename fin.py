import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, dan fitur
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')  # list kolom saat training

st.title("💼 Prediksi Keputusan Hiring Kandidat")

# Input user
recruitment_strategy = st.selectbox("Strategi Rekrutmen", ["1", "2", "3"])
education_level = st.selectbox("Tingkat Pendidikan", ["1", "2", "3", "4"])

skill_score = st.slider("Skor Keterampilan (0-10)", 0.0, 10.0, 5.0)
interview_score = st.slider("Skor Interview (0-10)", 0.0, 10.0, 5.0)
personality_score = st.slider("Skor Kepribadian (0-10)", 0.0, 10.0, 5.0)
experience_years = st.slider("Pengalaman Kerja (tahun)", 0, 20, 2)

# Buat dataframe kosong dulu
input_data = {
    'SkillScore': skill_score,
    'ExperienceYears': experience_years,
    'InterviewScore': interview_score,
    'PersonalityScore': personality_score,
}

# Tambahkan semua kemungkinan one-hot
for level in ["1", "2", "3", "4"]:
    input_data[f'EducationLevel_{level}'] = 1 if education_level == level else 0

for strategy in ["1", "2", "3"]:
    input_data[f'RecruitmentStrategy_{strategy}'] = 1 if recruitment_strategy == strategy else 0

input_df = pd.DataFrame([input_data])

# Tambahkan kolom kosong (0) jika ada yang hilang dari feature_names
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

# Urutkan sesuai urutan training
input_df = input_df[feature_names]

# Scaling numerik
numerical_cols = ['SkillScore', 'ExperienceYears', 'InterviewScore', 'PersonalityScore']
input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

# Prediksi
# if st.button("🔍 Prediksi Sekarang"):
#     prediction = model.predict(input_df)[0]
#     prob = model.predict_proba(input_df)[0][prediction]

#     if prediction == 1:
#         st.success(f"✅ Kandidat kemungkinan **DITERIMA** dengan probabilitas {prob:.2f}")
#     else:
#         st.error(f"❌ Kandidat kemungkinan **TIDAK diterima** dengan probabilitas {prob:.2f}")

if st.button("🔍 Prediksi Sekarang"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ Kandidat kemungkinan **DITERIMA**")
    else:
        st.error("❌ Kandidat kemungkinan **TIDAK DITERIMA**")


# st.write("Kolom input ke scaler:", input_df.columns.tolist())
# st.write("Feature names di scaler:", scaler.feature_names_in_.tolist())