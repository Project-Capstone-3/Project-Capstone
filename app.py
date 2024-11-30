import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="SugarGuard - Kenali Risikomu, Prediksi Diabetesmu",
    page_icon="https://github.com/Project-Capstone-3/Project-Capstone/blob/master/SugarGuard.png?raw=true",
    layout="wide",
)

# --- Fungsi Memuat Model dan Scaler ---
@st.cache_resource
def load_model_and_scaler():
    # Muat model dan scaler
    model = joblib.load("svm_model .pkl")  # Sesuaikan nama file model
    scaler = joblib.load("scaler.pkl")   # Pastikan scaler disimpan saat pelatihan
    return model, scaler

model, scaler = load_model_and_scaler()

# --- Sidebar ---
st.sidebar.image(
    "https://github.com/Project-Capstone-3/Project-Capstone/blob/master/SugarGuard.png?raw=true",
)
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Home", "About Us", "Prediksi Penyakit Diabetes"])

# --- Page: Home ---
if page == "Home":
    st.title("Selamat Datang di SugarGuard!")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://github.com/Project-Capstone-3/Project-Capstone/blob/master/SugarGuard.png?raw=true", use_column_width=True)
    with col2:
        st.markdown("""
        ### Apa itu SugarGuard?
        SugarGuard adalah aplikasi berbasis **Machine Learning** yang dirancang untuk membantu Anda
        memprediksi risiko terkena diabetes. Kami menyediakan alat yang mudah digunakan dan
        akurat untuk membantu pengambilan keputusan kesehatan Anda.
        """)

# --- Page: About Us ---
elif page == "About Us":
    st.title("Tentang Kami 👩‍💻👨‍💻")
    st.markdown("""
    Kami adalah tim ahli yang memiliki tujuan untuk menggabungkan **teknologi** dan **kesehatan** 
    demi masa depan yang lebih baik.
    """)

# --- Page: Prediksi Penyakit Diabetes ---
elif page == "Prediksi Penyakit Diabetes":
    st.title("Prediksi Risiko Diabetes 🩺")

    # Input data pengguna (harus sesuai dengan fitur model)
    input_data = pd.DataFrame({
        "Pregnancies": [st.sidebar.slider("Kehamilan (Pregnancies)", 0, 17, 3)],
        "Glucose": [st.sidebar.slider("Glukosa (Glucose)", 0, 200, 120)],
        "BloodPressure": [st.sidebar.slider("Tekanan Darah (BloodPressure)", 0, 122, 70)],
        "SkinThickness": [st.sidebar.slider("Ketebalan Kulit (SkinThickness)", 0, 99, 20)],
        "Insulin": [st.sidebar.slider("Insulin", 0.0, 846.0, 79.0)],
        "BMI": [st.sidebar.slider("BMI", 0.0, 67.1, 20.0)],
        "DiabetesPedigreeFunction": [st.sidebar.slider("Fungsi Keturunan Diabetes", 0.0, 2.5, 0.5)],
        "Age": [st.sidebar.slider("Usia", 21, 81, 33)],
    })

    # Tampilkan data input
    st.subheader("Data Anda")
    st.dataframe(input_data)

    # Skalakan data input menggunakan scaler yang telah dilatih
    input_scaled = scaler.transform(input_data)

    # Prediksi menggunakan model
    try:
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        diabetes_labels = ['Tidak Diabetes', 'Diabetes']

        st.subheader("Hasil Prediksi")
        st.markdown(f"""
        ### Anda berisiko: **{diabetes_labels[prediction[0]]}** 🩺
        - **Probabilitas Tidak Diabetes**: {prediction_proba[0][0]*100:.2f}%
        - **Probabilitas Diabetes**: {prediction_proba[0][1]*100:.2f}%
        """)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
