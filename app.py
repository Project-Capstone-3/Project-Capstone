import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from sklearn.preprocessing import StandardScaler

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="SugarGuard - Kenali Risikomu, Prediksi Diabetesmu",
    page_icon="https://github.com/Project-Capstone-3/Project-Capstone/blob/master/SugarGuard.png?raw=true",
    layout="wide",
)

# --- Fungsi untuk Memuat Model ---
@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/Project-Capstone-3/Project-Capstone/master/svm_model.pkl"  # Sesuaikan URL model
    local_filename = "svm_model.pkl"
    
    # Unduh file dari URL
    response = requests.get(url)
    with open(local_filename, 'wb') as f:
        f.write(response.content)
    
    # Muat model dari file lokal
    model = joblib.load(local_filename)
    return model

# Memuat model
try:
    model = load_model()
    st.success("Model berhasil dimuat.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")

# --- Sidebar ---
st.sidebar.image(
    "https://github.com/Project-Capstone-3/Project-Capstone/blob/master/SugarGuard.png?raw=true",
    use_column_width=True,
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

        ### Fitur Utama:
        - **Prediksi Risiko Diabetes** dengan hasil akurat.
        - **Visualisasi Data** untuk memahami kesehatan Anda.
        - **Akurasi Tinggi** dari model Machine Learning terbaik.

        **Coba sekarang dan pastikan kesehatan Anda!**
        """)

# --- Page: About Us ---
elif page == "About Us":
    st.title("Tentang Kami 👩‍💻👨‍💻")
    st.markdown("""
    Kami adalah tim ahli yang memiliki tujuan untuk menggabungkan **teknologi** dan **kesehatan** 
    demi masa depan yang lebih baik. Dengan menggunakan algoritma pembelajaran mesin, kami ingin
    membantu lebih banyak orang memahami risiko kesehatan mereka.

    ### Kontak Kami
    - **Email**: sugarguard@support.com
    - **Instagram**: [@SugarGuard](https://instagram.com)
    """)

# --- Page: Prediksi Penyakit Diabetes ---
elif page == "Prediksi Penyakit Diabetes":
    st.title("Prediksi Risiko Diabetes 🩺")

    # Input data pengguna
    input_data = pd.DataFrame({
        "Pregnancies": [st.sidebar.slider("Kehamilan (Pregnancies)", 0, 17, 3)],
        "Glucose": [st.sidebar.slider("Glukosa (Glucose)", 0, 200, 120)],
        "BloodPressure": [st.sidebar.slider("Tekanan Darah (BloodPressure)", 0, 122, 70)],
        "SkinThickness": [st.sidebar.slider("Ketebalan Kulit (SkinThickness)", 0, 99, 20)],
        "Insulin": [st.sidebar.slider("Insulin", 0.0, 846.0, 79.0)],
        "BMI": [st.sidebar.slider("BMI", 0.0, 67.1, 20.0)],
        "DiabetesPedigreeFunction": [st.sidebar.slider("Fungsi Pedigree Diabetes", 0.0, 2.42, 0.47)],
        "Age": [st.sidebar.slider("Usia", 21, 81, 33)],
    })

    st.subheader("Data Anda")
    st.dataframe(input_data)

    # Skalakan data pengguna
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)

    # Prediksi menggunakan model
    try:
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        diabetes = np.array(['Tidak Diabetes', 'Diabetes'])
        st.subheader("Hasil Prediksi")
        st.markdown(f"""
        ### Anda berisiko: **{diabetes[prediction][0]}** 🩺
        - **Probabilitas Tidak Diabetes**: {prediction_proba[0][0]*100:.2f}%
        - **Probabilitas Diabetes**: {prediction_proba[0][1]*100:.2f}%
        """)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
